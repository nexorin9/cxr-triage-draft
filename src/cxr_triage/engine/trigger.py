"""trigger — 阳性分诊触发器（PACS 入站 StudyInstanceUID → StateGraph）。

业务定位（spec.md §2 工作闭环 + §5 维度 1 + plan.md §5 深度 (a)）：

体检 PACS 落片 → ``incoming_studies`` 入条（status=pending）→ 本触发器

1. **拉一条 status=pending 行**：
   - 同 study 短时间二次入站 → ``StateIdempotencyStore.lookup`` 命中
     → 复用旧 draft（5 分钟 TTL），不重跑 5 件工具 + 不重调 LLM。
2. **未命中** → WADO 拉图 → ``DicomProcessor`` 调窗 8-bit PNG →
   ``CxrTriageWorkflow.run`` 跑 5 件工具 → ``TriageQueue.enqueue`` 入
   阳性队列 + ``StateIdempotencyStore.cache_draft`` 缓存。
3. **推送 SSE 事件**（``SsePublisher.publish_draft_ready``）：浏览器
   面板即时刷新。
4. **失败语义**：
   - DICOM 拉图失败（WadoFetchError）→ 标 ``status=failed_retry`` +
     audit ``trigger_wado_failed``，不静默丢数据；
   - 调窗失败（DicomProcessError）→ ``status=failed_retry`` +
     audit ``trigger_dicom_process_failed``；
   - 5 件工具全部跑出 ``degraded=True`` → 仍入队（带降级 banner），
     audit ``trigger_processed_degraded``；
   - 异常抛出 → ``status=failed_retry`` + audit ``trigger_exception``
     保留入站数据不删，运维可重放。

实现取舍（plan.md §5 深度 (a) 真实算法 + spec.md §5 维度 3 状态闭环）：

- **不删除 incoming_studies 行**：状态字段 ``status`` 推进 ``pending
  → processed / failed_retry``，便于飞检回放「哪条曾入站 / 哪条
  跑成功 / 哪条需重试」。
- **幂等与重入**：同 study 二次入站 → idempotency 命中复用，旧行不
  重跑；与 Task 8 ``StateIdempotencyStore`` 共享 ``draft_idempotency``
  表。
- **降级但仍入队**：模型不可达 / 工具抛异常 / keyqa 失败 → 产物结构
  完整、finding_text 含「（草稿 / 未经医师确认）」+ degradation_reasons；
  入队后由医师在浏览器面板看到降级 banner 自行处理（spec.md §10 安
  全边界 + §5 维度 4 协议兼容）。
- **不接临床状态推进**：本类只把入站 → 草稿；不接 ``state_machine``
  状态跃迁（找 ``CxrStateMachine.transition``），不写 RIS / 体检主检
  （找 ``writers.RISWriter`` / ``writers.HealthcheckWriter``）。
- **不接 SSE 协议**：本类只通过 ``SsePublisher`` 接口回调
  ``publish_draft_ready``；不直接连 ``/api/sse`` 端点（避免 web 层
  反向耦合）。

参考：
- 源仓库 ``medrax/agent/agent.py`` 多工具协同 + 日志模式
- 源仓库 ``medrax/interface.py:135-176`` Gradio 流式推送
  → 本类替换为 SSE publish 回调
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from ..config import CxrTriageConfig, load_config
from ..pacs.dicom_processor import (
    DicomProcessError,
    DicomProcessor,
)
from ..pacs.wado import WadoClient, WadoFetchError
from ..persistence.sqlite import Database
from ..state.idempotency import (
    DEFAULT_TOOL_VERSION,
    StateIdempotencyStore,
)
from ..writers.audit import (
    VALID_AUDIT_ACTOR_ROLES,
    AuditLogger,
)
from .triage_queue import TriageQueue
from .workflow import CxrTriageWorkflow, WorkflowError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 协程轮询间隔（秒）：5s 内未消费新 study 触发器仍能看到（spec.md §6
# 部署：单张 30 秒内出结构化产物；5s 轮询给到 6 个周期内消化）
DEFAULT_POLL_INTERVAL_SECONDS: float = 5.0

# 单次拉取 pending 行上限：避免一次拉太多阻塞事件循环
DEFAULT_PENDING_BATCH_SIZE: int = 50

# incoming_studies.status 状态机
STATUS_PENDING: str = "pending"
STATUS_PROCESSED: str = "processed"
STATUS_FAILED_RETRY: str = "failed_retry"

# 允许的状态值白名单（写库时校验）
ALLOWED_INCOMING_STATUSES: frozenset[str] = frozenset(
    {STATUS_PENDING, STATUS_PROCESSED, STATUS_FAILED_RETRY}
)

# audit 事件类型（点分 snake_case，便于飞检按 event_type 拉流）
EVENT_TRIGGER_STARTED: str = "trigger.process_started"
EVENT_TRIGGER_PROCESSED: str = "trigger.processed"
EVENT_TRIGGER_PROCESSED_DEGRADED: str = "trigger.processed_degraded"
EVENT_TRIGGER_IDEMPOTENT_HIT: str = "trigger.idempotent_hit"
EVENT_TRIGGER_WADO_FAILED: str = "trigger.wado_failed"
EVENT_TRIGGER_DICOM_PROCESS_FAILED: str = "trigger.dicom_process_failed"
EVENT_TRIGGER_WORKFLOW_FAILED: str = "trigger.workflow_failed"
EVENT_TRIGGER_EXCEPTION: str = "trigger.exception"
EVENT_TRIGGER_INCOMING_MARKED: str = "trigger.incoming_marked"


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class TriageTriggerError(RuntimeError):
    """TriageTrigger 启动 / 关闭 / 单条处理不可恢复错误。"""


# ---------------------------------------------------------------------------
# 协议（duck-typed）
# ---------------------------------------------------------------------------

# SSE 推送回调（Task 29 SsePublisher.publish_draft_ready 同形）
SsePublishCallback = Callable[[str, dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# 触发器主类
# ---------------------------------------------------------------------------

class TriageTrigger:
    """阳性分诊触发器：watch ``incoming_studies.status='pending'`` 行 → 跑 StateGraph。

    业务定位（spec.md §2 工作闭环 + plan.md §4 核心引擎深度 (a)）：

    1. ``process_pending()`` — 单轮扫描 + 处理；可被外部定时任务或
       ``run_pending_loop()`` 协程循环调用。
    2. ``run_pending_loop()`` — 后台协程循环（每 ``poll_interval_seconds``
       跑一次 ``process_pending``），由 FastAPI lifespan 启动 / 关闭。
    3. ``process_one(study_uid)`` — 处理单条 study（手工重放 / 测试
       / 驳回后重跑用）。

    Parameters
    ----------
    config:
        业务配置（pacs_wado_url / pacs_wado_token / positive_threshold
        等）。测试可显式注入；生产由 ``load_config()`` 默认。
    db:
        已 ``await db.connect()`` 打开的 ``Database`` 实例。
    workflow:
        编译后的 ``CxrTriageWorkflow``（含 5 件工具）。测试可注入
        mock workflow。
    triage_queue:
        阳性优先级队列；enqueue 是触发器主路径的最后一站。
    idempotency:
        草稿幂等缓存（Task 8）；同 study 同 tool_version 5 分钟内
        二次入站直接复用。
    audit_logger:
        审计日志双层落盘（Task 9）；可为 ``None``，表示不落 audit
        （仅单测场景）。
    wado_client:
        WADO-RS 拉图客户端；测试可注入 mock。
    dicom_processor:
        DICOM 调窗工具；测试可注入 mock。
    sse_publish:
        可选 ``async def publish(study_uid, payload)`` 回调；触发
        器把 ``draft_ready`` 事件推给浏览器面板（Task 29 接
        SsePublisher）。None 表示不推（与已有逻辑解耦）。
    poll_interval_seconds:
        ``run_pending_loop`` 轮询间隔（默认 5s）。
    pending_batch_size:
        单轮 ``process_pending`` 拉取 pending 行的上限。
    patient_id_salt:
        PatientID hash 用的盐；缺省从 ``CXR_TRIAGE_PATIENT_ID_SALT``
        env 读，与 PacsListener / DicomFolderWatcher 一致。
    """

    def __init__(
        self,
        config: CxrTriageConfig,
        db: Database,
        workflow: CxrTriageWorkflow,
        triage_queue: TriageQueue,
        idempotency: StateIdempotencyStore,
        *,
        audit_logger: Optional[AuditLogger] = None,
        wado_client: Optional[WadoClient] = None,
        dicom_processor: Optional[DicomProcessor] = None,
        sse_publish: Optional[SsePublishCallback] = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        pending_batch_size: int = DEFAULT_PENDING_BATCH_SIZE,
        patient_id_salt: Optional[str] = None,
    ) -> None:
        # ---- 入参校验 ----
        if not isinstance(config, CxrTriageConfig):
            raise TriageTriggerError(
                f"config 必须是 CxrTriageConfig；收到 {type(config).__name__}"
            )
        if not isinstance(db, Database):
            raise TriageTriggerError(
                f"db 必须是 Database；收到 {type(db).__name__}"
            )
        if not isinstance(workflow, CxrTriageWorkflow) and not (
            hasattr(workflow, "run") and callable(getattr(workflow, "run"))
        ):
            raise TriageTriggerError(
                "workflow 必须是 CxrTriageWorkflow 或暴露 async run(study_uid, "
                "png_bytes, metadata) 方法"
            )
        # ---- duck-typed 校验：生产是 TriageQueue / StateIdempotencyStore
        # 实例；测试可注入 FakeTriageQueue / FakeIdempotency —— 只需暴露
        # ``enqueue`` / ``lookup`` / ``cache_draft`` 异步方法。
        if not (
            hasattr(triage_queue, "enqueue") and callable(getattr(triage_queue, "enqueue"))
        ):
            raise TriageTriggerError(
                "triage_queue 必须暴露 async enqueue() 方法或为 TriageQueue 实例"
            )
        if not (
            (hasattr(idempotency, "lookup") and callable(getattr(idempotency, "lookup")))
            and (
                hasattr(idempotency, "cache_draft")
                and callable(getattr(idempotency, "cache_draft"))
            )
        ):
            raise TriageTriggerError(
                "idempotency 必须暴露 async lookup() + cache_draft() 方法或为 "
                "StateIdempotencyStore 实例"
            )
        if audit_logger is not None and not isinstance(audit_logger, AuditLogger):
            raise TriageTriggerError(
                f"audit_logger 必须是 AuditLogger 或 None；收到 {type(audit_logger).__name__}"
            )
        # wado_client / dicom_processor 走 duck-typed 校验（生产是 WadoClient
        # / DicomProcessor 实例；测试可注入 FakeWado / FakeDicomProcessor 替身
        # ——只需暴露 ``fetch`` / ``process`` 方法）
        if wado_client is not None and not (
            hasattr(wado_client, "fetch") and callable(getattr(wado_client, "fetch"))
        ):
            raise TriageTriggerError(
                "wado_client 必须暴露 async fetch(study_uid) 方法或为 None"
            )
        if dicom_processor is not None and not (
            hasattr(dicom_processor, "process")
            and callable(getattr(dicom_processor, "process"))
        ):
            raise TriageTriggerError(
                "dicom_processor 必须暴露 process(dcm_bytes) 方法或为 None"
            )
        if sse_publish is not None and not callable(sse_publish):
            raise TriageTriggerError("sse_publish 必须是可调用对象或 None")
        if not (0.1 <= float(poll_interval_seconds) <= 600.0):
            raise TriageTriggerError(
                f"poll_interval_seconds 必须在 [0.1, 600.0] 区间；收到 {poll_interval_seconds}"
            )
        if not isinstance(pending_batch_size, int) or pending_batch_size < 1:
            raise TriageTriggerError(
                f"pending_batch_size 必须是正整数；收到 {pending_batch_size!r}"
            )

        self.config = config
        self.db = db
        self.workflow = workflow
        self.triage_queue = triage_queue
        self.idempotency = idempotency
        self.audit_logger = audit_logger
        self.wado_client = wado_client or WadoClient(self.config)
        self.dicom_processor = dicom_processor or DicomProcessor()
        self.sse_publish = sse_publish
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.pending_batch_size = int(pending_batch_size)
        self._salt = patient_id_salt or os.getenv(
            "CXR_TRIAGE_PATIENT_ID_SALT", "cxr-triage-salt"
        )

        # 协程生命周期
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._stats = {
            "scanned": 0,
            "idempotent_hits": 0,
            "processed": 0,
            "degraded": 0,
            "wado_failed": 0,
            "dicom_process_failed": 0,
            "workflow_failed": 0,
            "exceptions": 0,
        }

    # ------------------------------------------------------------------
    # 生命周期（run_pending_loop 协程）
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动 ``run_pending_loop`` 后台协程。"""
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("TriageTrigger 已在运行，跳过重复 start()")
            return
        self._stop_event = asyncio.Event()
        self._loop_task = asyncio.create_task(
            self.run_pending_loop(), name="cxr-triage-trigger-loop"
        )
        logger.info(
            "TriageTrigger 已启动: poll_interval=%.1fs batch_size=%d",
            self.poll_interval_seconds,
            self.pending_batch_size,
        )

    async def stop(self) -> None:
        """优雅关闭后台协程。"""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(
                    self._loop_task, timeout=self.poll_interval_seconds * 2 + 5.0
                )
            except asyncio.TimeoutError:
                logger.warning("TriageTrigger 关闭超时，强制取消")
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                logger.exception("TriageTrigger 后台协程异常退出")
        self._loop_task = None
        self._stop_event = None
        # 关闭 wado_client（如果是我们自己 new 出来的）
        try:
            await self.wado_client.close()
        except Exception:  # noqa: BLE001
            logger.exception("关闭 WadoClient 失败（不阻断 trigger 关闭）")
        logger.info("TriageTrigger 已停止")

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    async def run_pending_loop(self) -> None:
        """后台协程循环：每 ``poll_interval_seconds`` 跑一次 ``process_pending``。"""
        assert self._stop_event is not None
        try:
            while not self._stop_event.is_set():
                try:
                    await self.process_pending()
                except Exception:  # noqa: BLE001
                    self._stats["exceptions"] += 1
                    logger.exception("process_pending 单轮异常（不退出循环）")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("TriageTrigger run_pending_loop 被取消")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("TriageTrigger run_pending_loop 异常退出")

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # 主路径
    # ------------------------------------------------------------------
    async def process_pending(self) -> int:
        """拉取 status=pending 行并逐条处理。返回本轮实际处理的条数。

        与 ``process_one`` 不同：批量、单条失败不影响其它条；用于
        定时轮询 / 启动时回放。
        """
        rows = await self._list_pending_rows(limit=self.pending_batch_size)
        processed = 0
        for row in rows:
            try:
                ok = await self.process_one(row["study_uid"])
            except Exception:  # noqa: BLE001 —— 单条失败兜底
                self._stats["exceptions"] += 1
                logger.exception(
                    "process_one 未捕获异常 study=%s（已落 failed_retry）",
                    row["study_uid"],
                )
                # 异常路径已在 process_one 内部尽量捕获；这里只兜底
                try:
                    await self._mark_incoming_status(
                        row["study_uid"], STATUS_FAILED_RETRY
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("入站 status 兜底标记失败")
                continue
            if ok:
                processed += 1
        return processed

    async def process_one(self, study_uid: str) -> bool:
        """处理单条 study（手工重放 / 驳回后重跑 / 单测入口）。

        流程：

        1. 校验 study_uid → 拉 ``incoming_studies`` 行（顺便取 metadata）。
        2. 幂等查 → 命中复用旧 draft（5 分钟 TTL）→ 入队 + 推 SSE +
           标 ``processed`` + audit ``trigger_idempotent_hit``。
        3. 未命中 → 标 ``pending→processing``（软锁）+ audit
           ``trigger_process_started`` → WADO 拉图 → DicomProcessor 调
           窗 → StateGraph.run → TriageQueue.enqueue → idempotency.cache
           → 推 SSE → 标 ``processed`` + audit ``trigger_processed``。
        4. 任意中间步骤失败：标 ``failed_retry`` + 对应 audit event，不
           抛；返回 ``False``。

        Returns
        -------
        bool
            True = 本条 study 已被消费（processed / idempotent_hit），
            False = 失败（failed_retry / 异常），调用方可重试。
        """
        if not isinstance(study_uid, str) or not study_uid:
            raise TriageTriggerError(
                f"study_uid 必须是字符串；收到 {study_uid!r}"
            )
        uid = study_uid.strip()
        if not uid:
            raise TriageTriggerError("study_uid 不能为空")

        # 0. 入站行存在性
        row = await self._fetch_incoming_row(uid)
        if row is None:
            logger.warning("process_one: study 不在 incoming_studies；忽略 %s", uid)
            return False

        # 1. 幂等查
        try:
            cached = await self.idempotency.lookup(uid, DEFAULT_TOOL_VERSION)
        except Exception:  # noqa: BLE001
            logger.exception("idempotency.lookup 失败 study=%s（降级继续）", uid)
            cached = None

        if cached:
            self._stats["idempotent_hits"] += 1
            return await self._handle_idempotent_hit(uid, row, cached)

        # 2. 处理（入站行 → 草稿）
        return await self._process_fresh(uid, row)

    # ------------------------------------------------------------------
    # 内部：幂等命中分支
    # ------------------------------------------------------------------
    async def _handle_idempotent_hit(
        self, uid: str, row: dict, cached: dict
    ) -> bool:
        """幂等命中 → 复用旧 draft → 入队 + 推 SSE + 标 processed。"""
        try:
            max_prob = float(cached.get("classifier_max_prob") or 0.0)
            finding_text = cached.get("finding_text") or ""
            positive_flag = bool(cached.get("positive_flag"))

            # 入队（走 INSERT OR REPLACE；同 study 二次入站也走"覆盖更新"）
            await self.triage_queue.enqueue(
                uid,
                max_prob,
                cached,
                positive_flag=positive_flag,
            )
            # 标 processed
            await self._mark_incoming_status(uid, STATUS_PROCESSED)
            # 推 SSE
            await self._publish_draft(
                uid,
                {
                    "study_uid": uid,
                    "max_prob": max_prob,
                    "positive_flag": positive_flag,
                    "finding_text": finding_text,
                    "reused_from_idempotency": True,
                },
            )
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_IDEMPOTENT_HIT,
                {
                    "max_prob": max_prob,
                    "positive_flag": positive_flag,
                    "tool_version": cached.get("tool_version", DEFAULT_TOOL_VERSION),
                    "source": row.get("source", ""),
                },
            )
            logger.info(
                "TriageTrigger 幂等命中 study=%s max_prob=%.4f positive=%s",
                uid,
                max_prob,
                positive_flag,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._stats["exceptions"] += 1
            logger.exception("幂等命中分支异常 study=%s", uid)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "idempotent_hit",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

    # ------------------------------------------------------------------
    # 内部：主路径（拉图 → 跑 StateGraph → 入队 → 推 SSE）
    # ------------------------------------------------------------------
    async def _process_fresh(self, uid: str, row: dict) -> bool:
        """从 incoming_studies → draft → triage_queue + SSE。"""
        # ---- 2a. WADO 拉图 ----
        try:
            dcm_bytes = await self.wado_client.fetch(uid)
        except WadoFetchError as exc:
            self._stats["wado_failed"] += 1
            logger.warning("WADO 拉图失败 study=%s err=%s", uid, exc)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_WADO_FAILED,
                {
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False
        except Exception as exc:  # noqa: BLE001
            self._stats["wado_failed"] += 1
            logger.exception("WADO 拉图未知异常 study=%s", uid)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "wado_fetch",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

        # ---- 2b. DicomProcessor 调窗 ----
        try:
            dcm_result = self.dicom_processor.process(dcm_bytes)
            png_bytes = dcm_result.png_bytes
            dcm_metadata = dcm_result.metadata
        except DicomProcessError as exc:
            self._stats["dicom_process_failed"] += 1
            logger.warning("DicomProcessor 调窗失败 study=%s err=%s", uid, exc)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_DICOM_PROCESS_FAILED,
                {
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False
        except Exception as exc:  # noqa: BLE001
            self._stats["dicom_process_failed"] += 1
            logger.exception("DicomProcessor 调窗未知异常 study=%s", uid)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "dicom_process",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

        # ---- 2c. 拼 metadata（incoming_studies 行 + DicomProcessor 抽出）----
        workflow_metadata = self._build_workflow_metadata(row, dcm_metadata)

        # ---- 2d. 写 process_started audit ----
        await self._audit(
            "system",
            uid,
            EVENT_TRIGGER_STARTED,
            {
                "source": row.get("source", ""),
                "png_bytes": len(png_bytes),
            },
        )

        # ---- 2e. StateGraph 跑 5 件工具 ----
        try:
            final_state = await self.workflow.run(uid, png_bytes, workflow_metadata)
        except WorkflowError as exc:
            self._stats["workflow_failed"] += 1
            logger.warning(
                "StateGraph 拒绝运行 study=%s err=%s（构造/入参非法）", uid, exc
            )
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_WORKFLOW_FAILED,
                {
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False
        except Exception as exc:  # noqa: BLE001
            self._stats["workflow_failed"] += 1
            logger.exception("StateGraph.run 未知异常 study=%s", uid)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "workflow_run",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

        # ---- 2f. 提取产物 + 写 idempotency 缓存 ----
        # 补 tool_version 字段（StateIdempotencyStore 用其做复合主键）
        final_state = dict(final_state)
        final_state.setdefault("tool_version", DEFAULT_TOOL_VERSION)
        final_state.setdefault("study_uid", uid)
        try:
            await self.idempotency.cache_draft(uid, final_state, DEFAULT_TOOL_VERSION)
        except Exception as exc:  # noqa: BLE001
            # 幂等缓存失败不阻断主路径；log 后继续（飞检按 audit 还原）
            self._stats["exceptions"] += 1
            logger.exception("idempotency.cache_draft 失败 study=%s", uid)
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "idempotency_cache",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )

        # ---- 2g. 入队 ----
        max_prob = float(final_state.get("classifier_max_prob") or 0.0)
        positive_flag = bool(final_state.get("positive_flag"))
        finding_text = final_state.get("finding_text") or ""
        try:
            await self.triage_queue.enqueue(
                uid, max_prob, final_state, positive_flag=positive_flag
            )
        except Exception as exc:  # noqa: BLE001
            self._stats["exceptions"] += 1
            logger.exception("TriageQueue.enqueue 失败 study=%s", uid)
            try:
                await self._mark_incoming_status(uid, STATUS_FAILED_RETRY)
            except Exception:  # noqa: BLE001
                logger.exception("mark_incoming_status 失败")
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "triage_enqueue",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

        # ---- 2h. 标 processed + 推 SSE + audit ----
        try:
            await self._mark_incoming_status(uid, STATUS_PROCESSED)
        except Exception as exc:  # noqa: BLE001
            logger.exception("mark_incoming_status(processed) 失败 study=%s", uid)
            await self._audit(
                "system",
                uid,
                EVENT_TRIGGER_EXCEPTION,
                {
                    "stage": "mark_processed",
                    "error_class": type(exc).__name__,
                    "error": str(exc),
                },
            )

        degraded = bool(final_state.get("degraded"))
        if degraded:
            self._stats["degraded"] += 1
        self._stats["processed"] += 1

        event_type = (
            EVENT_TRIGGER_PROCESSED_DEGRADED if degraded else EVENT_TRIGGER_PROCESSED
        )
        await self._audit(
            "system",
            uid,
            event_type,
            {
                "max_prob": max_prob,
                "positive_flag": positive_flag,
                "tools_executed": list(final_state.get("tools_executed") or []),
                "degradation_reasons": list(
                    final_state.get("degradation_reasons") or []
                ),
                "tool_version": final_state.get("tool_version", DEFAULT_TOOL_VERSION),
                "source": row.get("source", ""),
            },
        )

        # 推 SSE
        await self._publish_draft(
            uid,
            {
                "study_uid": uid,
                "max_prob": max_prob,
                "positive_flag": positive_flag,
                "finding_text": finding_text,
                "finding_struct": final_state.get("finding_struct"),
                "grounding_boxes": final_state.get("grounding_boxes"),
                "png_preview_b64": final_state.get("png_preview_b64"),
                "tools_executed": list(final_state.get("tools_executed") or []),
                "degraded": degraded,
                "degradation_reasons": list(
                    final_state.get("degradation_reasons") or []
                ),
                "reused_from_idempotency": False,
            },
        )

        logger.info(
            "TriageTrigger processed study=%s max_prob=%.4f positive=%s "
            "degraded=%s tools=%s",
            uid,
            max_prob,
            positive_flag,
            degraded,
            list(final_state.get("tools_executed") or []),
        )
        return True

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    async def _list_pending_rows(self, limit: int) -> list[dict]:
        """拉取 status=pending 的入站行（按 received_at 升序，先入先出）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, patient_id, patient_id_hash, accession_number,
                       modality, study_date, patient_sex, patient_age,
                       source, status, max_prob, received_at, updated_at
                FROM incoming_studies
                WHERE status = ?
                ORDER BY received_at ASC
                LIMIT ?
                """,
                (STATUS_PENDING, int(limit)),
            )
            rows = await cur.fetchall()
            await cur.close()
        self._stats["scanned"] += len(rows)
        return [dict(r) for r in rows]

    async def _fetch_incoming_row(self, study_uid: str) -> Optional[dict]:
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, patient_id, patient_id_hash, accession_number,
                       modality, study_date, patient_sex, patient_age,
                       source, status, max_prob, received_at, updated_at
                FROM incoming_studies
                WHERE study_uid = ?
                """,
                (study_uid,),
            )
            row = await cur.fetchone()
            await cur.close()
        return dict(row) if row is not None else None

    async def _mark_incoming_status(self, study_uid: str, status: str) -> None:
        if status not in ALLOWED_INCOMING_STATUSES:
            raise TriageTriggerError(f"incoming_studies.status 非法: {status!r}")
        async with self.db.connection() as conn:
            await conn.execute(
                """
                UPDATE incoming_studies
                SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                WHERE study_uid = ?
                """,
                (status, study_uid),
            )
            await conn.commit()

    def _build_workflow_metadata(
        self, row: dict, dcm_metadata: dict
    ) -> dict[str, Any]:
        """拼 StateGraph.run 用的 metadata dict。

        incoming_studies 行（patient_id_hash / accession_number / patient_sex
        / patient_age / modality / study_date） + DicomProcessor 抽出的
        像素层信息（rows / columns / bits_stored / window_* / rescale_*），
        供 CXRKeyQA 等 LLM 工具读 PatientSex / PatientAge。
        """
        meta: dict[str, Any] = {
            "study_uid": row.get("study_uid", ""),
            "patient_id_hash": row.get("patient_id_hash", ""),
            "accession_number": row.get("accession_number", ""),
            "modality": row.get("modality", ""),
            "study_date": row.get("study_date", ""),
            "patient_sex": row.get("patient_sex", ""),
            "patient_age": row.get("patient_age", ""),
            "source": row.get("source", ""),
        }
        # DicomProcessor 抽出的纯像素层信息（rows / columns / bits_stored /
        # window_width / window_level / rescale_*）合并；DicomProcessor
        # 也含 study_uid / modality 但与 incoming_studies 一致，incoming
        # 优先（来自 PACS，更权威）。
        for k, v in (dcm_metadata or {}).items():
            if k in {"study_uid", "modality", "patient_id_hash"}:
                continue
            meta.setdefault(k, v)
        return meta

    async def _audit(
        self,
        actor_role: str,
        study_uid: str,
        event_type: str,
        payload: dict,
    ) -> None:
        if self.audit_logger is None:
            return
        if actor_role not in VALID_AUDIT_ACTOR_ROLES:
            logger.warning(
                "_audit: actor_role=%r 不在白名单；改用 system", actor_role
            )
            actor_role = "system"
        try:
            await self.audit_logger.audit_log_insert(
                actor_role=actor_role,
                study_uid=study_uid,
                event_type=event_type,
                payload=payload,
            )
        except Exception:  # noqa: BLE001
            # audit 失败不阻断主路径；只记 log
            logger.exception(
                "audit_log 写失败 event=%s study=%s", event_type, study_uid
            )

    async def _publish_draft(self, study_uid: str, payload: dict) -> None:
        if self.sse_publish is None:
            return
        try:
            await self.sse_publish(study_uid, payload)
        except Exception:  # noqa: BLE001
            # SSE 推送失败不阻断主路径；只记 log（飞检按 audit 还原）
            logger.exception("SSE publish 失败 study=%s", study_uid)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def build_triage_trigger(
    config: CxrTriageConfig,
    db: Database,
    workflow: CxrTriageWorkflow,
    triage_queue: TriageQueue,
    idempotency: StateIdempotencyStore,
    *,
    audit_logger: Optional[AuditLogger] = None,
    wado_client: Optional[WadoClient] = None,
    dicom_processor: Optional[DicomProcessor] = None,
    sse_publish: Optional[SsePublishCallback] = None,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    pending_batch_size: int = DEFAULT_PENDING_BATCH_SIZE,
) -> TriageTrigger:
    """工厂函数：从 ``load_config()`` 构造 ``TriageTrigger``。"""
    return TriageTrigger(
        config=config,
        db=db,
        workflow=workflow,
        triage_queue=triage_queue,
        idempotency=idempotency,
        audit_logger=audit_logger,
        wado_client=wado_client,
        dicom_processor=dicom_processor,
        sse_publish=sse_publish,
        poll_interval_seconds=poll_interval_seconds,
        pending_batch_size=pending_batch_size,
    )


def build_triage_trigger_from_env(
    db: Database,
    workflow: CxrTriageWorkflow,
    triage_queue: TriageQueue,
    idempotency: StateIdempotencyStore,
    *,
    audit_logger: Optional[AuditLogger] = None,
    sse_publish: Optional[SsePublishCallback] = None,
) -> TriageTrigger:
    """便捷工厂：从 env 读 config。"""
    return build_triage_trigger(
        config=load_config(),
        db=db,
        workflow=workflow,
        triage_queue=triage_queue,
        idempotency=idempotency,
        audit_logger=audit_logger,
        sse_publish=sse_publish,
    )


__all__ = [
    "TriageTrigger",
    "TriageTriggerError",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_PENDING_BATCH_SIZE",
    "STATUS_PENDING",
    "STATUS_PROCESSED",
    "STATUS_FAILED_RETRY",
    "ALLOWED_INCOMING_STATUSES",
    "EVENT_TRIGGER_STARTED",
    "EVENT_TRIGGER_PROCESSED",
    "EVENT_TRIGGER_PROCESSED_DEGRADED",
    "EVENT_TRIGGER_IDEMPOTENT_HIT",
    "EVENT_TRIGGER_WADO_FAILED",
    "EVENT_TRIGGER_DICOM_PROCESS_FAILED",
    "EVENT_TRIGGER_WORKFLOW_FAILED",
    "EVENT_TRIGGER_EXCEPTION",
    "EVENT_TRIGGER_INCOMING_MARKED",
    "SsePublishCallback",
    "build_triage_trigger",
    "build_triage_trigger_from_env",
]
