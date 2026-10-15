"""体检主检 summary-ref 写回客户端 — 体检胸片所见草稿外发主路径之二。

设计原则（spec.md §5 维度 4 正式对接层 + plan.md §6 风险缓解）：

1. **真实外发（httpx 异步）**：``POST {HEALTHCHECK_API_URL}/api/v1/healthcheck/{visitId}/summary-ref``
   Bearer Token ``HEALTHCHECK_API_TOKEN`` 凭据位全部从 env 读（``CxrTriageConfig``），
   **不**硬编码、不进 ``tests/fixtures/``。
2. **payload 对齐 spec.md §4 体检主检输出**::

       {
         "visit_id": "<体检流水号>",
         "summary_snippet": "胸片：右下肺气胸可能（0.87），建议进一步 CT。",
         "source_study_uids": ["<StudyInstanceUID>"],
         "model_version": "<git-sha-or-semver>"
       }

3. **summary_snippet 形态**：由 caller 把 ``finding_text`` 包装成"胸片：<所见>"短句
   + 概率（max_prob 数值）+ "建议进一步 …"等中性建议；本模块**不**
   自动生成 snippet 文本，只校验收到的 snippet 不空。
4. **source_study_uids**：可含多条（同一 visitId 下可能有多张胸片 + 其它
   检查）；这里只放本张 DICOM 的 StudyInstanceUID；后续 visitId 合并
   由体检主检侧处理。
5. **失败入 outbox**（spec.md §5 维度 6 失败恢复 + plan.md §6 风险缓解）：
   与 ``RISWriter`` **同一** outbox 表，但 ``target_endpoint`` 用 ``hc://``
   前缀区分；retry_loop 由 ``HealthcheckWriter`` 独立协程驱动。
   - ``200 <= status < 300`` → 成功，``audit_log_insert(event_type="healthcheck.written")``
   - 超时 / 网络错 / 4xx-5xx → 失败，``Outbox.enqueue(target_endpoint=hc://...)`` +
     ``audit_log_insert(event_type="healthcheck.queued_for_retry")``
6. **不静默丢数据**：失败**必**入 outbox + 写 audit；HTTP 错抛
   ``HealthcheckWriterError`` 由 caller 决策，但不因网络异常中断主流程
   （sr 已经签发，状态机推进优先；写回后台异步重试）。
7. **不耦合业务层**：本模块只 import ``config``/``persistence``/``audit``，
   不 import ``state``/``engine``；与 ``RISWriter`` 共享 ``Outbox`` 表
   但 ``target_endpoint`` 区分（前缀 + URL 字符串）。
8. **仅 sr_signed 触发**：本模块自身**不**校验状态机，状态守门由 caller
   （FastAPI router / 状态机 sr_sign 路径 / 双写协调）负责；本模块做
   的是"把这一段 snippet 用正确格式发到体检主检"。
9. **患者 PHI 不外发**：明文 ``PatientID`` / ``AccessionNumber`` 永不
   出现在 payload 中；只带 ``visit_id``（体检流水号，由 caller 决定如何
   映射患者）+ ``source_study_uids``（DICOM 标识符，不直接是可识别 PHI）。

参考：源仓库 ``medrax`` **不涉及** 体检主检对接，本模块是工程支撑（医院
对接层）；spec.md §7 开源对照已注明 ``**新建**：体检主检 summary-ref
HTTP 写回 + outbox 重试 → 真实外发``。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from ..config import CxrTriageConfig, load_config
from ..persistence.outbox import Outbox
from .audit import AuditLogger

logger = logging.getLogger(__name__)


# =====================================================================
# 常量
# =====================================================================

# outbox target_endpoint 前缀；与 RISWriter 区分（同一 outbox 表）
HEALTHCHECK_TARGET_PREFIX: str = "hc://summary-ref/"

# 端点路径模板（spec.md §4 健康体检主检端点）
_SUMMARY_REF_PATH_TEMPLATE: str = "/api/v1/healthcheck/{visit_id}/summary-ref"

# 重试行为配置（与 RISWriter 同步数值）
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# 超时（与 RISWriter 同步：医院内网通常 2s 内，5s 兜底）
DEFAULT_TIMEOUT_SECONDS: float = 5.0

# retry_loop 配置
DEFAULT_RETRY_POLL_SECONDS: float = 30.0
DEFAULT_RETRY_BATCH_SIZE: int = 10

# audit_log event 命名
EVENT_HEALTHCHECK_WRITTEN: str = "healthcheck.written"
EVENT_HEALTHCHECK_QUEUED: str = "healthcheck.queued_for_retry"
EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED: str = "healthcheck.outbox_enqueue_failed"
EVENT_HEALTHCHECK_RETRY_SENT: str = "healthcheck.retry_sent"
EVENT_HEALTHCHECK_RETRY_FAILED: str = "healthcheck.retry_failed"
EVENT_HEALTHCHECK_RETRY_ALERTING: str = "healthcheck.retry_alerting"
EVENT_HEALTHCHECK_RETRY_SKIPPED: str = "healthcheck.retry_skipped"

# summary_snippet 前缀约定：caller 拼 snippet 时参照此写法
SUMMARY_SNIPPET_PREFIX: str = "胸片："

# 默认 model_version（与 RISWriter 同步为 0.1.0；工程支撑层）
DEFAULT_MODEL_VERSION: str = "0.1.0"

# source_study_uids 的硬上限（防止异常 case 把大量 UIDs 推到体检主检）
MAX_SOURCE_STUDY_UIDS: int = 50


# =====================================================================
# 异常
# =====================================================================
class HealthcheckWriterError(Exception):
    """体检主检 writer 异常基类。"""


class HealthcheckPayloadError(HealthcheckWriterError):
    """体检主检 payload 构造/校验失败。"""


class HealthcheckWritebackError(HealthcheckWriterError):
    """体检主检 HTTP 调用失败（含超时 / 4xx / 5xx / 网络错）。"""


# =====================================================================
# 工具
# =====================================================================
def build_healthcheck_summary_ref_url(
    base_url: str, visit_id: str
) -> str:
    """根据 ``HEALTHCHECK_API_URL`` 拼出 summary-ref 完整 URL。

    Parameters
    ----------
    base_url:
        ``HEALTHCHECK_API_URL`` 形如 ``http://127.0.0.1:8082``（不带尾斜杠）。
    visit_id:
        体检流水号。

    Returns
    -------
    str
        形如 ``http://127.0.0.1:8082/api/v1/healthcheck/<visitId>/summary-ref``。
    """
    if not base_url:
        raise HealthcheckPayloadError("Healthcheck base_url 为空")
    if not visit_id:
        raise HealthcheckPayloadError("visit_id 为空")
    return (
        f"{base_url.rstrip('/')}"
        f"{_SUMMARY_REF_PATH_TEMPLATE.format(visit_id=visit_id)}"
    )


def build_healthcheck_target_endpoint(visit_id: str) -> str:
    """outbox 用的 ``target_endpoint`` 标识符（用于 retry_loop 区分 RIS vs 主检）。

    形如 ``hc://summary-ref/<visitId>``；retry_loop 据此前缀路由。
    """
    return f"{HEALTHCHECK_TARGET_PREFIX}{visit_id}"


def _extract_visit_id_from_target_endpoint(target_endpoint: str) -> str:
    """从 outbox ``target_endpoint`` 反解 ``visit_id``。

    形如 ``hc://summary-ref/<visit_id>`` → ``<visit_id>``；空 / 非 hc 前缀
    抛 ``HealthcheckPayloadError``。retry_loop 据此重建 summary-ref URL。
    """
    if not target_endpoint or not isinstance(target_endpoint, str):
        raise HealthcheckPayloadError("target_endpoint 非法")
    if not target_endpoint.startswith(HEALTHCHECK_TARGET_PREFIX):
        raise HealthcheckPayloadError(
            f"target_endpoint 非体检主检前缀: {target_endpoint!r}"
        )
    visit_id = target_endpoint[len(HEALTHCHECK_TARGET_PREFIX):]
    if not visit_id:
        raise HealthcheckPayloadError(
            f"target_endpoint 缺 visit_id: {target_endpoint!r}"
        )
    return visit_id


def build_healthcheck_payload(
    visit_id: str,
    summary_snippet: str,
    source_study_uids: list[str],
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Any]:
    """构造体检主检 summary-ref payload（与 spec.md §4 一致）。

    Parameters
    ----------
    visit_id:
        体检流水号；与 ``patient_id_hash`` 同属间接标识。
    summary_snippet:
        已生成的所见短句（含概率与中性建议）。**不**自动拼接 —— 本函数
        只校验非空 + 字段类型；具体文本由 caller（双写协调逻辑 / 包装
        服务）决定。
    source_study_uids:
        本张 DICOM 的 ``StudyInstanceUID``（或同一 visitId 下多张）。
    model_version:
        算法版本（默认 ``0.1.0``）；飞检追溯用。

    Returns
    -------
    dict
        4 字段 JSON-ready dict。

    Raises
    ------
    HealthcheckPayloadError
        任一必填字段缺失或类型错。
    """
    if not visit_id or not isinstance(visit_id, str):
        raise HealthcheckPayloadError(f"visit_id 非法: {visit_id!r}")
    if summary_snippet is None or not isinstance(summary_snippet, str):
        raise HealthcheckPayloadError(
            f"summary_snippet 非法: {type(summary_snippet).__name__}"
        )
    if not summary_snippet.strip():
        # 体检主检拒绝空 snippet（信息科对接规范）
        raise HealthcheckPayloadError("summary_snippet 为空")
    if not isinstance(source_study_uids, list):
        raise HealthcheckPayloadError(
            f"source_study_uids 必须为 list: "
            f"{type(source_study_uids).__name__}"
        )
    if len(source_study_uids) == 0:
        raise HealthcheckPayloadError("source_study_uids 为空")
    if len(source_study_uids) > MAX_SOURCE_STUDY_UIDS:
        raise HealthcheckPayloadError(
            f"source_study_uids 超上限 {MAX_SOURCE_STUDY_UIDS}"
        )
    for uid in source_study_uids:
        if not isinstance(uid, str) or not uid:
            raise HealthcheckPayloadError(
                f"source_study_uids 含非法元素: {uid!r}"
            )
    if not isinstance(model_version, str) or not model_version:
        raise HealthcheckPayloadError(f"model_version 非法: {model_version!r}")

    return {
        "visit_id": visit_id,
        "summary_snippet": summary_snippet.strip(),
        "source_study_uids": list(source_study_uids),
        "model_version": model_version,
    }


def make_summary_snippet(
    finding_text: str,
    max_prob: float,
    positive_flag: bool,
    extra_suggestion: str = "",
) -> str:
    """包装所见成体检主检 snippet（caller 辅助函数）。

    本函数**不**自动生成 — caller 可选用，也可自己拼。"中性建议"措辞
    限定为 ``建议进一步 <检查>`` 或 ``建议随访``；**不**写诊断与
    治疗建议（CLAUDE.md 安全边界）。

    Parameters
    ----------
    finding_text:
        已生成的中文所见（已剥 DRAFT_MARKER 最好，但本函数再做一遍兜底）。
    max_prob:
        classifier max_prob，用于在 snippet 中标注概率。
    positive_flag:
        True → "可能"语气（不写诊断）；False → "未见明显异常"。
    extra_suggestion:
        额外建议；默认空串。

    Returns
    -------
    str
        形如 ``"胸片：右下肺野见气胸征象（0.87），建议进一步 CT。"``。
    """
    if not finding_text:
        # 兜底（caller 勿传空）
        finding_text = "未见明显异常" if not positive_flag else "请人工复核"
    # 兜底剥 DRAFT_MARKER，避免体检主检收到"草稿"字样
    cleaned = finding_text.replace(
        "（草稿 / 未经医师确认）", ""
    ).replace("（草稿）", "").strip()
    prob_str = f"（{max_prob:.2f}）" if positive_flag else ""
    tone = "可能" if positive_flag else "未见明显异常"
    head = f"{SUMMARY_SNIPPET_PREFIX}{cleaned}{prob_str}"
    tail = extra_suggestion.strip() if extra_suggestion else ""
    # 拼接：去掉多余逗号
    if tail:
        return f"{head}，{tail}。".replace("，。", "。").rstrip("，") + (
            "" if head.endswith("。") else "。"
        )
    if head.endswith("。"):
        return f"{head}"
    return f"{head}。"


# =====================================================================
# HealthcheckWriter
# =====================================================================
class HealthcheckWriter:
    """体检主检 summary-ref 写回客户端 + outbox 异步重试兜底。

    Parameters
    ----------
    config:
        ``CxrTriageConfig`` 实例；凭据位 ``healthcheck_api_url`` /
        ``healthcheck_api_token`` 必填。
    outbox:
        ``Outbox`` 实例；失败入条。
    audit_logger:
        ``AuditLogger`` 实例；写 audit_log 记录外发结果。
    http_client:
        可选 ``httpx.AsyncClient``；缺省懒加载。
    timeout_seconds:
        HTTP 调用超时；默认 5s。
    """

    def __init__(
        self,
        config: CxrTriageConfig,
        outbox: Outbox,
        audit_logger: AuditLogger,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry_poll_seconds: float = DEFAULT_RETRY_POLL_SECONDS,
        retry_batch_size: int = DEFAULT_RETRY_BATCH_SIZE,
    ) -> None:
        self._config = config
        self._outbox = outbox
        self._audit = audit_logger
        self._http_client = http_client
        self._owns_client = http_client is None
        self._timeout = float(timeout_seconds)
        # retry_loop 配置（与 RISWriter 同形但独立通道）
        self._retry_poll_seconds = float(retry_poll_seconds)
        self._retry_batch_size = int(retry_batch_size)
        # retry_loop 生命周期（asyncio.Event + Task）
        self._retry_stop_event: Optional[asyncio.Event] = None
        self._retry_task: Optional[asyncio.Task[None]] = None
        self._stats = {
            "write_attempts": 0,
            "write_successes": 0,
            "write_failures": 0,
            "outbox_enqueues": 0,
            "retry_cycles": 0,
            "retry_attempts": 0,
            "retry_successes": 0,
            "retry_failures": 0,
            "retry_alerts": 0,
            "retry_skipped": 0,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """关闭 HTTP client + 优雅停止 retry_loop（若已起）。"""
        # 1) 先停 retry_loop
        await self.stop_retry_loop()
        # 2) 关 client
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def retry_running(self) -> bool:
        """retry_loop 是否仍在后台运行。"""
        return self._retry_task is not None and not self._retry_task.done()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": "cxr-triage-draft/0.1.0"},
            )
        return self._http_client

    @property
    def stats(self) -> dict[str, int]:
        """只读副本。"""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    async def write_summary_ref(
        self,
        visit_id: str,
        summary_snippet: str,
        source_study_uids: list[str],
        model_version: str = DEFAULT_MODEL_VERSION,
        *,
        study_uid: Optional[str] = None,
    ) -> bool:
        """写一份 summary-ref 到体检主检。

        Parameters
        ----------
        visit_id:
            体检流水号。
        summary_snippet:
            已生成的中性短句（含概率 + 建议）。
        source_study_uids:
            同一 visitId 下的 DICOM UIDs。
        model_version:
            算法版本；飞检追溯用。
        study_uid:
            （可选）当前触发本写的 StudyInstanceUID；写入 audit 时附带，
            不进 payload（payload 仅含 visit_id 间接标识）。

        Returns
        -------
        bool
            ``True`` = 体检主检 200 OK；``False`` = 失败（已入 outbox）。

        Notes
        -----
        - 本方法**不抛**网络异常；失败路径必入 outbox + 写 audit_log。
        - payload 校验失败仍抛 ``HealthcheckPayloadError``（这是 caller
          bug，不可救）。
        - 仅 ``sr_signed`` 状态触发写回；本方法**不**校验状态机，由
          caller 决定调用时机（FastAPI router / state_machine /
          双写协调）。
        """
        payload = build_healthcheck_payload(
            visit_id=visit_id,
            summary_snippet=summary_snippet,
            source_study_uids=source_study_uids,
            model_version=model_version,
        )
        url = build_healthcheck_summary_ref_url(
            self._config.healthcheck_api_url, visit_id
        )
        target_endpoint = build_healthcheck_target_endpoint(visit_id)

        self._stats["write_attempts"] += 1

        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.healthcheck_api_token}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            await self._handle_failure(
                visit_id=visit_id,
                study_uid=study_uid,
                target_endpoint=target_endpoint,
                payload=payload,
                error_message=(
                    f"Healthcheck 网络失败: {type(exc).__name__}: {exc}"
                ),
            )
            return False
        except httpx.HTTPError as exc:
            # 其他 httpx 错误（如 protocol error）
            await self._handle_failure(
                visit_id=visit_id,
                study_uid=study_uid,
                target_endpoint=target_endpoint,
                payload=payload,
                error_message=(
                    f"Healthcheck HTTP 错: {type(exc).__name__}: {exc}"
                ),
            )
            return False

        status = response.status_code
        if 200 <= status < 300:
            self._stats["write_successes"] += 1
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid or visit_id,
                event_type=EVENT_HEALTHCHECK_WRITTEN,
                payload={
                    "url": url,
                    "status_code": status,
                    "visit_id": visit_id,
                    "snippet_chars": len(payload["summary_snippet"]),
                    "source_study_count": len(payload["source_study_uids"]),
                    "model_version": payload["model_version"],
                    "tool_version": payload["model_version"],
                },
            )
            logger.info(
                "Healthcheck summary-ref written visit=%s status=%s",
                visit_id,
                status,
            )
            return True

        # 4xx-5xx：入 outbox + audit
        body_excerpt = response.text[:200] if response.text else ""
        await self._handle_failure(
            visit_id=visit_id,
            study_uid=study_uid,
            target_endpoint=target_endpoint,
            payload=payload,
            error_message=f"Healthcheck HTTP {status}: {body_excerpt}",
        )
        return False

    # ------------------------------------------------------------------
    # 内部：失败兜底
    # ------------------------------------------------------------------
    async def _handle_failure(
        self,
        visit_id: str,
        study_uid: Optional[str],
        target_endpoint: str,
        payload: dict[str, Any],
        error_message: str,
    ) -> None:
        """失败入 outbox + 写 audit；不抛。"""
        self._stats["write_failures"] += 1
        try:
            outbox_id = await self._outbox.enqueue(
                study_uid=study_uid or visit_id,
                target_endpoint=target_endpoint,
                payload=payload,
            )
            self._stats["outbox_enqueues"] += 1
            event_type = EVENT_HEALTHCHECK_QUEUED
            extra = {
                "outbox_id": outbox_id,
                "target_endpoint": target_endpoint,
                "visit_id": visit_id,
                "error": error_message[:200],
            }
        except Exception as exc:  # noqa: BLE001
            event_type = EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED
            extra = {
                "target_endpoint": target_endpoint,
                "visit_id": visit_id,
                "outbox_error": f"{type(exc).__name__}: {exc}",
                "primary_error": error_message[:200],
            }
            logger.error(
                "Healthcheck outbox enqueue FAILED visit=%s err=%s",
                visit_id,
                exc,
            )

        try:
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid or visit_id,
                event_type=event_type,
                payload=extra,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Healthcheck 失败 audit 写入失败 visit=%s err=%s",
                visit_id,
                exc,
            )

    # ------------------------------------------------------------------
    # retry_loop 生命周期（asyncio.Event + Task 模式）
    # ------------------------------------------------------------------
    async def start_retry_loop(self) -> None:
        """启动后台 retry_loop 协程；幂等（重复启动 noop）。

        Notes
        -----
        - 由 FastAPI lifespan 启动；正常不重复调用。
        - 与 ``RISWriter.start_retry_loop`` **并行**，不互锁——两 writer
          各自跑各自的 retry_loop，``Outbox`` 表由 ``target_endpoint``
          前缀（``ris://`` vs ``hc://``）路由。
        - ``stop_event`` 用于优雅退出（先唤醒再 cancel）。
        """
        if self.retry_running:
            return
        # asyncio.Event 必须在 event loop 内构造；此处仅 start() 时建
        self._retry_stop_event = asyncio.Event()
        self._retry_task = asyncio.create_task(
            self.retry_loop(),
            name="cxr-healthcheck-retry-loop",
        )
        logger.info(
            "Healthcheck retry_loop started poll=%ss batch=%s",
            self._retry_poll_seconds,
            self._retry_batch_size,
        )

    async def stop_retry_loop(self) -> None:
        """优雅停止 retry_loop；幂等。

        设置 stop_event → 唤醒 wait_for → task 自然结束；若 task 已结束
        或未起则 noop。
        """
        if self._retry_stop_event is not None:
            self._retry_stop_event.set()
        if self._retry_task is not None:
            try:
                # 给 task 一个 timeout 等待自然退出；cancel 作兜底
                await asyncio.wait_for(
                    asyncio.shield(self._retry_task),
                    timeout=self._retry_poll_seconds + 5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Healthcheck retry_loop 优雅退出超时，强制 cancel"
                )
                self._retry_task.cancel()
                try:
                    await self._retry_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._retry_task = None
        self._retry_stop_event = None

    async def retry_loop(self) -> None:
        """体检主检 outbox 后台重试协程。

        流程（每轮）：

        1. ``asyncio.wait_for`` ``stop_event`` 在 ``retry_poll_seconds`` 内被
           set → 跳出循环（优雅退出）。
        2. ``Outbox.dequeue(retry_batch_size)`` 拉 pending 行；过滤
           ``target_endpoint.startswith(HEALTHCHECK_TARGET_PREFIX)`` 的
           体检主检行。
        3. 对每条体检主检行 POST 主检；成功 → ``Outbox.mark_sent`` + 写
           ``healthcheck.retry_sent`` audit；失败 → ``Outbox.mark_retry`` +
           写 ``healthcheck.retry_failed`` audit。
        4. ``mark_retry`` 返 ``alerting`` → 写 ``healthcheck.retry_alerting``
           audit（超 MAX_RETRY_BEFORE_ALERT=3 触发浏览器顶部告警 banner）。

        Notes
        -----
        - 不抛：任何单条异常被本轮捕获 + 写 ``healthcheck.retry_failed``，
          下一轮继续。
        - 与 ``RISWriter.retry_loop`` **并行独立**：本协程不锁 RIS writer，
          outbox 由 ``target_endpoint`` 前缀自然路由。
        - ``visit_id`` 从 ``target_endpoint`` 解析（``hc://summary-ref/<visit_id>``）。
        """
        stop_event = self._retry_stop_event
        if stop_event is None:
            # 直接调用 retry_loop（没经 start_retry_loop）→ 懒建 event
            self._retry_stop_event = asyncio.Event()
            stop_event = self._retry_stop_event

        while not stop_event.is_set():
            # 1) 等下轮（或被 stop 唤醒）
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._retry_poll_seconds,
                )
                # 没异常 → 被 set 唤醒
                break
            except asyncio.TimeoutError:
                # 正常轮询间隔到了
                pass
            except asyncio.CancelledError:
                raise

            # 2) 拉取本轮待发
            self._stats["retry_cycles"] += 1
            try:
                rows = await self._outbox.dequeue(limit=self._retry_batch_size)
            except Exception as exc:  # noqa: BLE
                logger.error(
                    "Healthcheck retry_loop dequeue FAILED err=%s", exc
                )
                continue

            # 3) 过滤体检主检行（非 hc 行 = RIS 或其他 writer 的，留给
            #    对应 writer 跑）
            hc_rows = [
                r
                for r in rows
                if r["target_endpoint"].startswith(HEALTHCHECK_TARGET_PREFIX)
            ]
            self._stats["retry_skipped"] += len(rows) - len(hc_rows)

            # 4) 单条 POST（不并发：单台医院内网体检主检通常 ≤ 2-3 个研究
            #    在队，并发价值不大）
            for row in hc_rows:
                try:
                    await self._process_retry_row(row)
                except Exception as exc:  # noqa: BLE001
                    # 单条异常兜底：不静默丢；mark_retry + audit
                    outbox_id = int(row["id"])
                    error_message = (
                        f"retry_loop exception: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.error(
                        "Healthcheck retry_loop row=%s FAILED err=%s",
                        outbox_id,
                        exc,
                    )
                    new_status = "pending"
                    try:
                        new_status = await self._outbox.mark_retry(
                            outbox_id=outbox_id,
                            error_message=error_message,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    self._stats["retry_failures"] += 1
                    # 写 retry_failed audit（异常路径也要留痕，飞检可查）
                    try:
                        await self._audit.audit_log_insert(
                            actor_role="system",
                            study_uid=row["study_uid"],
                            event_type=EVENT_HEALTHCHECK_RETRY_FAILED,
                            payload={
                                "outbox_id": outbox_id,
                                "target_endpoint": row["target_endpoint"],
                                "error": error_message[:200],
                                "new_status": new_status,
                                "exception_class": type(exc).__name__,
                            },
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    # alerting 也写一行 retry_alerting（与 _on_retry_failure 一致）
                    if new_status == "alerting":
                        self._stats["retry_alerts"] += 1
                        try:
                            await self._audit.audit_log_insert(
                                actor_role="system",
                                study_uid=row["study_uid"],
                                event_type=EVENT_HEALTHCHECK_RETRY_ALERTING,
                                payload={
                                    "outbox_id": outbox_id,
                                    "target_endpoint": row["target_endpoint"],
                                    "last_error": error_message[:200],
                                    "alert_message": (
                                        "体检主检写回连续失败超阈值，"
                                        "请联系信息科"
                                    ),
                                },
                            )
                        except Exception:  # noqa: BLE001
                            pass

        logger.info("Healthcheck retry_loop exited")

    async def _process_retry_row(self, row: dict[str, Any]) -> None:
        """处理单条 outbox 行：POST 体检主检 → mark_sent / mark_retry + audit。

        从 ``target_endpoint``（``hc://summary-ref/<visit_id>``）解析
        ``visit_id`` 重建 URL；payload 含 ``summary_snippet`` / ``source_study_uids``
        等已发格式。
        """
        outbox_id = int(row["id"])
        study_uid = row["study_uid"]
        payload = row["payload"]
        retry_count = int(row["retry_count"])
        visit_id = _extract_visit_id_from_target_endpoint(
            row["target_endpoint"]
        )

        self._stats["retry_attempts"] += 1
        url = build_healthcheck_summary_ref_url(
            self._config.healthcheck_api_url, visit_id
        )

        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": (
                        f"Bearer {self._config.healthcheck_api_token}"
                    ),
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
            await self._on_retry_failure(
                row=row,
                error_message=(
                    f"retry network error: {type(exc).__name__}: {exc}"
                ),
            )
            return

        status = response.status_code
        if 200 <= status < 300:
            await self._outbox.mark_sent(outbox_id)
            self._stats["retry_successes"] += 1
            try:
                await self._audit.audit_log_insert(
                    actor_role="system",
                    study_uid=study_uid,
                    event_type=EVENT_HEALTHCHECK_RETRY_SENT,
                    payload={
                        "outbox_id": outbox_id,
                        "retry_count": retry_count,
                        "status_code": status,
                        "target_endpoint": row["target_endpoint"],
                        "visit_id": visit_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Healthcheck retry_sent audit FAILED err=%s", exc
                )
            logger.info(
                "Healthcheck retry sent visit=%s outbox_id=%s retry_count=%s",
                visit_id,
                outbox_id,
                retry_count,
            )
            return

        # 4xx/5xx → mark_retry
        body_excerpt = response.text[:200] if response.text else ""
        await self._on_retry_failure(
            row=row,
            error_message=f"retry HTTP {status}: {body_excerpt}",
        )

    async def _on_retry_failure(
        self, row: dict[str, Any], error_message: str
    ) -> None:
        """retry 失败兜底：mark_retry → 写 retry_failed；状态切 alerting
        再写 retry_alerting 触发浏览器 banner。"""
        outbox_id = int(row["id"])
        study_uid = row["study_uid"]
        new_status = await self._outbox.mark_retry(
            outbox_id=outbox_id, error_message=error_message
        )
        self._stats["retry_failures"] += 1

        try:
            await self._audit.audit_log_insert(
                actor_role="system",
                study_uid=study_uid,
                event_type=EVENT_HEALTHCHECK_RETRY_FAILED,
                payload={
                    "outbox_id": outbox_id,
                    "target_endpoint": row["target_endpoint"],
                    "error": error_message[:200],
                    "new_status": new_status,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Healthcheck retry_failed audit FAILED err=%s", exc
            )

        # alerting：连续失败超阈值 → 写 alert audit（供浏览器面板 SSE 渲染）
        if new_status == "alerting":
            self._stats["retry_alerts"] += 1
            try:
                await self._audit.audit_log_insert(
                    actor_role="system",
                    study_uid=study_uid,
                    event_type=EVENT_HEALTHCHECK_RETRY_ALERTING,
                    payload={
                        "outbox_id": outbox_id,
                        "target_endpoint": row["target_endpoint"],
                        "last_error": error_message[:200],
                        "alert_message": (
                            "体检主检写回连续失败超阈值，请联系信息科"
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Healthcheck retry_alerting audit FAILED err=%s", exc
                )
            logger.error(
                "Healthcheck retry ALERTING outbox_id=%s", outbox_id
            )

    # ------------------------------------------------------------------
    # 调试 / 测试辅助
    # ------------------------------------------------------------------
    @staticmethod
    def encode_payload_for_outbox(payload: dict[str, Any]) -> str:
        """outbox 落盘前 JSON 序列化（ensure_ascii=False 保留中文 snippet）。"""
        return json.dumps(payload, ensure_ascii=False, default=str)


# =====================================================================
# 工厂
# =====================================================================
def build_healthcheck_writer(
    config: Optional[CxrTriageConfig] = None,
    outbox: Optional[Outbox] = None,
    audit_logger: Optional[AuditLogger] = None,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> HealthcheckWriter:
    """工厂：创建 ``HealthcheckWriter``。

    显式参数优先；缺省从 ``load_config()`` 读 + 调用方需自行注入
    ``outbox`` / ``audit_logger``。
    """
    cfg = config or load_config()
    if outbox is None:
        raise HealthcheckWriterError("build_healthcheck_writer 必须显式传入 outbox")
    if audit_logger is None:
        raise HealthcheckWriterError(
            "build_healthcheck_writer 必须显式传入 audit_logger"
        )
    return HealthcheckWriter(
        config=cfg,
        outbox=outbox,
        audit_logger=audit_logger,
        http_client=http_client,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "HealthcheckWriter",
    "HealthcheckWriterError",
    "HealthcheckPayloadError",
    "HealthcheckWritebackError",
    "HEALTHCHECK_TARGET_PREFIX",
    "RETRYABLE_HTTP_STATUSES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_RETRY_POLL_SECONDS",
    "DEFAULT_RETRY_BATCH_SIZE",
    "DEFAULT_MODEL_VERSION",
    "MAX_SOURCE_STUDY_UIDS",
    "SUMMARY_SNIPPET_PREFIX",
    "EVENT_HEALTHCHECK_WRITTEN",
    "EVENT_HEALTHCHECK_QUEUED",
    "EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED",
    "EVENT_HEALTHCHECK_RETRY_SENT",
    "EVENT_HEALTHCHECK_RETRY_FAILED",
    "EVENT_HEALTHCHECK_RETRY_ALERTING",
    "EVENT_HEALTHCHECK_RETRY_SKIPPED",
    "build_healthcheck_summary_ref_url",
    "build_healthcheck_target_endpoint",
    "build_healthcheck_payload",
    "make_summary_snippet",
    "_extract_visit_id_from_target_endpoint",
    "build_healthcheck_writer",
]
