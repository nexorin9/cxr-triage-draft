"""RIS finding-draft 写回客户端 — 体检胸片所见草稿外发主路径。

设计原则（spec.md §5 维度 4 正式对接层 + plan.md §6 风险缓解）：

1. **真实外发（httpx 异步）**：``POST {RIS_API_URL}/api/v1/studies/{study_uid}/finding-draft``
   Bearer Token ``RIS_API_TOKEN`` 凭据位全部从 env 读（``CxrTriageConfig``），
   **不**硬编码、不进 ``tests/fixtures/``。
2. **payload 对齐 spec.md §4 RIS 输出**::

       {
         "study_uid": "<StudyInstanceUID>",
         "patient_id_hash": "<SHA256(PatientID+salt)>",
         "finding_struct": {"pathologies": [...], "key_qa": [...], "grounding_boxes": [...]},
         "finding_text": "<中文所见，DRAFT_MARKER 已剥>",
         "positive_flag": true,
         "tool_version": "<git-sha-or-semver>",
         "draft_ts": "<ISO8601>"
       }

3. **草稿标识剥离**：``finding_text`` 落到 RIS 前必须把 ``DRAFT_MARKER``
   （"（草稿 / 未经医师确认）"）去掉——RIS 是医疗记录系统，"草稿"两字
   不应进正式报告；保留 ``draft_ts`` / ``tool_version`` 便于飞检追溯
   哪一版算法出的草稿。
4. **失败入 outbox**（spec.md §5 维度 6 失败恢复 + plan.md §6 风险缓解）：
   - ``200 <= status < 300`` → 成功，``audit_log_insert(event_type="ris.written")``
   - 超时 / 网络错 / 4xx-5xx → 失败，``Outbox.enqueue(target_endpoint)`` + ``audit_log_insert(event_type="ris.queued_for_retry")``
5. **不静默丢数据**：失败**必**入 outbox + 写 audit；HTTP 错抛 ``RISWriterError``
   由 caller 决策，但**不**因网络异常中断主流程（sr 已经签发，状态机
   推进优先；写回后台异步重试）。
6. **不耦合业务层**：本模块只 import ``config``/``persistence``/``audit``，
   不 import ``state``/``engine``；与 ``HealthcheckWriter`` 共享 ``Outbox``
   表但 ``target_endpoint`` 区分（URL 字符串 + 写入字段 ``type=ris_xxx``）。
7. **仅 sr_signed 触发**：本模块自身**不**校验状态机，状态守门由 caller
   （FastAPI router / 状态机 sr_sign 路径）负责；本模块做的是
   "把这一份 draft 用正确格式发到 RIS"。
8. **患者 PHI 不外发**：明文 ``PatientID`` 永远不出现在 payload 中；只
   带 ``patient_id_hash``。``AccessionNumber`` / ``StudyDate`` 等也不进
   payload（RIS 端可由 ``study_uid`` 反查）。

参考：源仓库 ``medrax`` **不涉及** RIS 写回，本模块是工程支撑（医院
对接层）；spec.md §7 开源对照已注明 ``**新建**：RIS finding-draft HTTP
写回 + outbox 重试 → 真实外发``。
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

# DRAFT_MARKER 与 finding_draft.DRAFT_MARKER 同源；写回 RIS 前必须剥掉。
# RIS 是医疗记录系统，不应出现"草稿"两字；保留 draft_ts / tool_version
# 便于飞检追溯算法版本。
DRAFT_MARKER: str = "（草稿 / 未经医师确认）"

# outbox target_endpoint 前缀；与 HealthcheckWriter 区分
RIS_TARGET_PREFIX: str = "ris://finding-draft/"

# 4xx-5xx 都视为可重试的瞬时错误（RIS 端可能短暂 503）；死信由
# MAX_RETRY_BEFORE_ALERT=3 触发 alerting
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# 超时（与 ModelGateway 同步；RIS 是医院内网通常 2s 内返回，5s 兜底）
DEFAULT_TIMEOUT_SECONDS: float = 5.0

# 端点路径模板
_FINDING_DRAFT_PATH_TEMPLATE: str = "/api/v1/studies/{study_uid}/finding-draft"

# retry_loop 轮询间隔；RIS 是医院内网，通常 30s 一次足够重试
DEFAULT_RETRY_POLL_SECONDS: float = 30.0

# retry_loop 单轮最多处理行数（避免一次拉空表）
DEFAULT_RETRY_BATCH_SIZE: int = 10

# audit_log event 命名
EVENT_RIS_RETRY_SENT: str = "ris.retry_sent"
EVENT_RIS_RETRY_FAILED: str = "ris.retry_failed"
EVENT_RIS_RETRY_ALERTING: str = "ris.retry_alerting"
EVENT_RIS_RETRY_SKIPPED: str = "ris.retry_skipped"


# =====================================================================
# 异常
# =====================================================================
class RISWriterError(Exception):
    """RIS writer 异常基类。"""


class RISPayloadError(RISWriterError):
    """RIS payload 构造/校验失败。"""


class RISWritebackError(RISWriterError):
    """RIS HTTP 调用失败（含超时 / 4xx / 5xx / 网络错）。"""


# =====================================================================
# 工具
# =====================================================================
def _strip_draft_marker(text: str) -> str:
    """把 ``DRAFT_MARKER`` 从中文所见中剥掉。

    RIS 是医疗记录系统，不应出现"草稿"两字；保留 draft_ts / tool_version
    便于飞检追溯算法版本。

    Examples
    --------
    >>> _strip_draft_marker("右下肺野见气胸征象（草稿 / 未经医师确认），余肺野清晰。")
    '右下肺野见气胸征象，余肺野清晰。'
    """
    if not text:
        return ""
    # DRAFT_MARKER 前后可能有逗号/句号；统一处理
    return text.replace(DRAFT_MARKER, "").strip()


def build_ris_finding_draft_url(base_url: str, study_uid: str) -> str:
    """根据 ``RIS_API_URL`` 拼出 finding-draft 完整 URL。

    Parameters
    ----------
    base_url:
        ``RIS_API_URL`` 形如 ``http://127.0.0.1:8081``（不带尾斜杠）。
    study_uid:
        DICOM ``StudyInstanceUID``。

    Returns
    -------
    str
        形如 ``http://127.0.0.1:8081/api/v1/studies/<uid>/finding-draft``。
    """
    if not base_url:
        raise RISPayloadError("RIS base_url 为空")
    if not study_uid:
        raise RISPayloadError("study_uid 为空")
    return (
        f"{base_url.rstrip('/')}"
        f"{_FINDING_DRAFT_PATH_TEMPLATE.format(study_uid=study_uid)}"
    )


def build_ris_target_endpoint(study_uid: str) -> str:
    """outbox 用的 ``target_endpoint`` 标识符（用于 retry_loop 区分 RIS vs 主检）。

    形如 ``ris://finding-draft/<uid>``；retry_loop 据此选客户端。
    """
    return f"{RIS_TARGET_PREFIX}{study_uid}"


def build_ris_payload(
    study_uid: str,
    patient_id_hash: str,
    finding_text: str,
    finding_struct: dict[str, Any],
    positive_flag: bool,
    draft_ts: str,
    tool_version: str = "0.1.0",
) -> dict[str, Any]:
    """构造 RIS finding-draft payload（与 spec.md §4 一致）。

    Parameters
    ----------
    study_uid:
        DICOM ``StudyInstanceUID``。
    patient_id_hash:
        ``SHA256(PatientID + salt)``；明文 PatientID 不进 payload。
    finding_text:
        中文所见；本函数会**自动**剥掉 ``DRAFT_MARKER``。
    finding_struct:
        ``{pathologies, key_qa, grounding_boxes}`` 完整结构。
    positive_flag:
        阳性 flag（max_prob ≥ 阈值）。
    draft_ts:
        UTC ISO8601 草稿时间戳。
    tool_version:
        算法版本（默认 ``__version__``）；飞检追溯用。

    Returns
    -------
    dict
        7 字段 JSON-ready dict。

    Raises
    ------
    RISPayloadError
        任一必填字段缺失或类型错。
    """
    if not study_uid or not isinstance(study_uid, str):
        raise RISPayloadError(f"study_uid 非法: {study_uid!r}")
    if not isinstance(patient_id_hash, str):
        # 允许空串（hash 在某些脱敏路径下可能为空）；但类型必须 str
        raise RISPayloadError(
            f"patient_id_hash 类型错: {type(patient_id_hash).__name__}"
        )
    if finding_text is None or not isinstance(finding_text, str):
        raise RISPayloadError(f"finding_text 非法: {type(finding_text).__name__}")
    if not isinstance(finding_struct, dict):
        raise RISPayloadError(
            f"finding_struct 必须为 dict: {type(finding_struct).__name__}"
        )
    if not isinstance(draft_ts, str) or not draft_ts:
        raise RISPayloadError(f"draft_ts 非法: {draft_ts!r}")
    if not isinstance(tool_version, str) or not tool_version:
        raise RISPayloadError(f"tool_version 非法: {tool_version!r}")

    return {
        "study_uid": study_uid,
        "patient_id_hash": patient_id_hash,
        "finding_struct": finding_struct,
        "finding_text": _strip_draft_marker(finding_text),
        "positive_flag": bool(positive_flag),
        "tool_version": tool_version,
        "draft_ts": draft_ts,
    }


# =====================================================================
# RISWriter
# =====================================================================
class RISWriter:
    """RIS finding-draft 写回客户端 + outbox 异步重试兜底。

    Parameters
    ----------
    config:
        ``CxrTriageConfig`` 实例；凭据位 ``ris_api_url`` / ``ris_api_token``
        必填。
    outbox:
        ``Outbox`` 实例；失败入条。
    audit_logger:
        ``AuditLogger`` 实例；写 audit_log 记录外发结果。
    http_client:
        可选 ``httpx.AsyncClient``；缺省懒加载（30s keepalive）。
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
        # retry_loop 配置
        self._retry_poll_seconds = float(retry_poll_seconds)
        self._retry_batch_size = int(retry_batch_size)
        # retry_loop 生命周期（asyncio.Event + Task）
        self._retry_stop_event: Optional[asyncio.Event] = None
        self._retry_task: Optional[asyncio.Task[None]] = None
        # 简单统计；面板 /api/health 可读
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

    @property
    def retry_running(self) -> bool:
        """retry_loop 是否仍在后台运行。"""
        return self._retry_task is not None and not self._retry_task.done()

    # ------------------------------------------------------------------
    # retry_loop 生命周期（asyncio.Event + asyncio.Task）
    # ------------------------------------------------------------------
    async def start_retry_loop(self) -> None:
        """启动后台 retry_loop 协程；幂等（重复启动 noop）。

        Notes
        -----
        - 由 FastAPI lifespan / Task 28 启动；正常不重复调用。
        - ``retry_poll_seconds`` 控制轮询间隔；30s 是 RIS 内网默认值。
        - ``stop_event`` 用于优雅退出（先唤醒再 cancel）。
        """
        if self.retry_running:
            return
        # asyncio.Event 必须在 event loop 内构造；此处仅 start() 时建
        self._retry_stop_event = asyncio.Event()
        self._retry_task = asyncio.create_task(
            self.retry_loop(),
            name="cxr-ris-retry-loop",
        )
        logger.info(
            "RIS retry_loop started poll=%ss batch=%s",
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
                logger.warning("RIS retry_loop 优雅退出超时，强制 cancel")
                self._retry_task.cancel()
                try:
                    await self._retry_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._retry_task = None
        self._retry_stop_event = None

    # ------------------------------------------------------------------
    # retry_loop 主循环
    # ------------------------------------------------------------------
    async def retry_loop(self) -> None:
        """RIS outbox 后台重试协程。

        流程（每轮）：

        1. ``asyncio.wait_for`` ``stop_event`` 在 ``retry_poll_seconds`` 内被
           set → 跳出循环（优雅退出）。
        2. ``Outbox.dequeue(retry_batch_size)`` 拉 pending 行；过滤
           ``target_endpoint.startswith(RIS_TARGET_PREFIX)`` 的 RIS 行。
        3. 对每条 RIS 行 POST RIS；成功 → ``Outbox.mark_sent`` + 写
           ``ris.retry_sent`` audit；失败 → ``Outbox.mark_retry`` + 写
           ``ris.retry_failed`` audit。
        4. ``mark_retry`` 返 ``alerting`` → 写 ``ris.retry_alerting`` audit
           （超 MAX_RETRY_BEFORE_ALERT=3 触发浏览器顶部告警 banner）。

        Notes
        -----
        - 不抛：任何单条异常被本轮捕获 + 写 ``ris.retry_failed``，
          下一轮继续。
        - 不主动 pause：RIS 端完全不可达时，30s/次轮询不会冲垮医院
          内网；可由 /api/health 关停。
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
                logger.error("RIS retry_loop dequeue FAILED err=%s", exc)
                continue

            # 3) 过滤 RIS 行（非 RIS 行 = 体检主检或其他 writer 的，留给
            #    对应 writer 跑）
            ris_rows = [
                r
                for r in rows
                if r["target_endpoint"].startswith(RIS_TARGET_PREFIX)
            ]
            self._stats["retry_skipped"] += len(rows) - len(ris_rows)

            # 4) 单条 POST（不并发：单台医院内网 RIS 通常 ≤ 2-3 个研究
            #    在队，并发价值不大；如未来高并发场景改 gather 即可）
            for row in ris_rows:
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
                        "RIS retry_loop row=%s FAILED err=%s",
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
                            event_type=EVENT_RIS_RETRY_FAILED,
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
                                event_type=EVENT_RIS_RETRY_ALERTING,
                                payload={
                                    "outbox_id": outbox_id,
                                    "target_endpoint": row["target_endpoint"],
                                    "last_error": error_message[:200],
                                    "alert_message": (
                                        "RIS 写回连续失败超阈值，"
                                        "请联系信息科"
                                    ),
                                },
                            )
                        except Exception:  # noqa: BLE001
                            pass

        logger.info("RIS retry_loop exited")

    async def _process_retry_row(self, row: dict[str, Any]) -> None:
        """处理单条 outbox 行：POST RIS → mark_sent / mark_retry + audit。"""
        outbox_id = int(row["id"])
        study_uid = row["study_uid"]
        payload = row["payload"]
        retry_count = int(row["retry_count"])

        self._stats["retry_attempts"] += 1
        url = build_ris_finding_draft_url(self._config.ris_api_url, study_uid)

        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.ris_api_token}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
            await self._on_retry_failure(
                row=row,
                error_message=f"retry network error: {type(exc).__name__}: {exc}",
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
                    event_type=EVENT_RIS_RETRY_SENT,
                    payload={
                        "outbox_id": outbox_id,
                        "retry_count": retry_count,
                        "status_code": status,
                        "target_endpoint": row["target_endpoint"],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RIS retry_sent audit FAILED err=%s", exc)
            logger.info(
                "RIS retry sent study=%s outbox_id=%s retry_count=%s",
                study_uid,
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
                event_type=EVENT_RIS_RETRY_FAILED,
                payload={
                    "outbox_id": outbox_id,
                    "target_endpoint": row["target_endpoint"],
                    "error": error_message[:200],
                    "new_status": new_status,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RIS retry_failed audit FAILED=%s", exc)

        # alerting：连续失败超阈值 → 写 alert audit（供浏览器面板 SSE 渲染）
        if new_status == "alerting":
            self._stats["retry_alerts"] += 1
            try:
                await self._audit.audit_log_insert(
                    actor_role="system",
                    study_uid=study_uid,
                    event_type=EVENT_RIS_RETRY_ALERTING,
                    payload={
                        "outbox_id": outbox_id,
                        "target_endpoint": row["target_endpoint"],
                        "last_error": error_message[:200],
                        "alert_message": (
                            "RIS 写回连续失败超阈值，请联系信息科"
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RIS retry_alerting audit FAILED err=%s", exc)
            logger.error(
                "RIS retry ALERTING study=%s outbox_id=%s",
                study_uid,
                outbox_id,
            )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    async def write_finding_draft(
        self,
        study_uid: str,
        patient_id_hash: str,
        finding_text: str,
        finding_struct: dict[str, Any],
        positive_flag: bool,
        draft_ts: str,
        tool_version: str = "0.1.0",
    ) -> bool:
        """写一份 finding-draft 到 RIS。

        Returns
        -------
        bool
            ``True`` = RIS 200 OK；``False`` = 失败（已入 outbox）。

        Notes
        -----
        - 本方法**不抛**网络异常；失败路径必入 outbox + 写 audit_log。
        - payload 校验失败仍抛 ``RISPayloadError``（这是 caller bug，
          不可救）。
        - 仅 ``sr_signed`` 状态触发写回；本方法**不**校验状态机，由
          caller 决定调用时机（FastAPI router / state_machine）。
        """
        payload = build_ris_payload(
            study_uid=study_uid,
            patient_id_hash=patient_id_hash,
            finding_text=finding_text,
            finding_struct=finding_struct,
            positive_flag=positive_flag,
            draft_ts=draft_ts,
            tool_version=tool_version,
        )
        url = build_ris_finding_draft_url(self._config.ris_api_url, study_uid)
        target_endpoint = build_ris_target_endpoint(study_uid)

        self._stats["write_attempts"] += 1

        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.ris_api_token}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            # 网络层失败：入 outbox + audit
            await self._handle_failure(
                study_uid=study_uid,
                target_endpoint=target_endpoint,
                payload=payload,
                error_message=f"RIS 网络失败: {type(exc).__name__}: {exc}",
            )
            return False
        except httpx.HTTPError as exc:
            # 其他 httpx 错误（如 protocol error）
            await self._handle_failure(
                study_uid=study_uid,
                target_endpoint=target_endpoint,
                payload=payload,
                error_message=f"RIS HTTP 错: {type(exc).__name__}: {exc}",
            )
            return False

        status = response.status_code
        if 200 <= status < 300:
            self._stats["write_successes"] += 1
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid,
                event_type="ris.written",
                payload={
                    "url": url,
                    "status_code": status,
                    "positive_flag": payload["positive_flag"],
                    "tool_version": payload["tool_version"],
                },
            )
            logger.info(
                "RIS finding-draft written study=%s status=%s",
                study_uid,
                status,
            )
            return True

        # 4xx-5xx：入 outbox + audit
        body_excerpt = response.text[:200] if response.text else ""
        await self._handle_failure(
            study_uid=study_uid,
            target_endpoint=target_endpoint,
            payload=payload,
            error_message=(
                f"RIS HTTP {status}: {body_excerpt}"
            ),
        )
        return False

    # ------------------------------------------------------------------
    # 内部：失败兜底
    # ------------------------------------------------------------------
    async def _handle_failure(
        self,
        study_uid: str,
        target_endpoint: str,
        payload: dict[str, Any],
        error_message: str,
    ) -> None:
        """失败入 outbox + 写 audit；不抛。"""
        self._stats["write_failures"] += 1
        try:
            outbox_id = await self._outbox.enqueue(
                study_uid=study_uid,
                target_endpoint=target_endpoint,
                payload=payload,
            )
            self._stats["outbox_enqueues"] += 1
            event_type = "ris.queued_for_retry"
            extra = {
                "outbox_id": outbox_id,
                "target_endpoint": target_endpoint,
                "error": error_message[:200],
            }
        except Exception as exc:  # noqa: BLE001
            # outbox 写失败（极罕见，DB 异常）；只写 audit，强制把责
            # 任留给信息科
            event_type = "ris.outbox_enqueue_failed"
            extra = {
                "target_endpoint": target_endpoint,
                "outbox_error": f"{type(exc).__name__}: {exc}",
                "primary_error": error_message[:200],
            }
            logger.error(
                "RIS outbox enqueue FAILED study=%s err=%s",
                study_uid,
                exc,
            )

        try:
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid,
                event_type=event_type,
                payload=extra,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RIS 失败 audit 写入失败 study=%s err=%s", study_uid, exc)

    # ------------------------------------------------------------------
    # 调试 / 测试辅助
    # ------------------------------------------------------------------
    @staticmethod
    def encode_payload_for_outbox(payload: dict[str, Any]) -> str:
        """outbox 落盘前 JSON 序列化（ensure_ascii=False 保留中文所见）。"""
        return json.dumps(payload, ensure_ascii=False, default=str)


# =====================================================================
# 工厂
# =====================================================================
def build_ris_writer(
    config: Optional[CxrTriageConfig] = None,
    outbox: Optional[Outbox] = None,
    audit_logger: Optional[AuditLogger] = None,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> RISWriter:
    """工厂：创建 ``RISWriter``。

    显式参数优先；缺省从 ``load_config()`` 读 + 调用方需自行注入
    ``outbox`` / ``audit_logger``。
    """
    cfg = config or load_config()
    if outbox is None:
        raise RISWriterError("build_ris_writer 必须显式传入 outbox")
    if audit_logger is None:
        raise RISWriterError("build_ris_writer 必须显式传入 audit_logger")
    return RISWriter(
        config=cfg,
        outbox=outbox,
        audit_logger=audit_logger,
        http_client=http_client,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "RISWriter",
    "RISWriterError",
    "RISPayloadError",
    "RISWritebackError",
    "DRAFT_MARKER",
    "RIS_TARGET_PREFIX",
    "RETRYABLE_HTTP_STATUSES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_RETRY_POLL_SECONDS",
    "DEFAULT_RETRY_BATCH_SIZE",
    "EVENT_RIS_RETRY_SENT",
    "EVENT_RIS_RETRY_FAILED",
    "EVENT_RIS_RETRY_ALERTING",
    "EVENT_RIS_RETRY_SKIPPED",
    "build_ris_finding_draft_url",
    "build_ris_target_endpoint",
    "build_ris_payload",
    "_strip_draft_marker",
    "build_ris_writer",
]
