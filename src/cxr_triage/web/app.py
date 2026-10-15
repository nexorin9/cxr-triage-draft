"""FastAPI app + routers — 体检胸片所见草稿系统的对外入口。

设计原则（spec.md §5 维度 4 + CLAUDE.md GUI 优先 · 文档分层）：

1. **常驻 FastAPI 服务**：通过 ``uvicorn cxr_triage.web.app:app`` 拉起；浏览器
   三栏面板（Task 30-31）+ SSE 推送（Task 29）合于此。
2. **lifespan 一站式组装**：lifespan 内构造 ``Database`` / ``AuditLogger`` /
   ``Outbox`` / ``RISWriter`` / ``HealthcheckWriter`` / ``StateIdempotencyStore`` /
   ``TriageQueue`` / ``TriageTrigger`` / ``CxrStateMachine`` / ``AlertPublisher`` /
   ``PacsListener`` / ``DicomFolderWatcher``，全部挂在 ``app.state``。
3. **显式注入**：生产代码全部用 ``load_config()`` 读 env；测试代码允许显式
   注入 fake 模块（仅在 lifespan 缺省路径之外）。
4. **同步 Web 入口 + 异步主路径**：FastAPI async route 调 ``async`` 模块
   （StateMachine / Queue / Trigger），路由层**不**做状态逻辑。
5. **sr_signed 双写**：``/api/studies/{uid}/confirm`` 推到 ``sr_signed`` 后
   自动触发 RISWriter + HealthcheckWriter 双写（spec.md §3 工作闭环），
   任一失败入 outbox 重试 + 写 ``dual_write`` audit。
6. **不静默写临床状态**：所有跳转必须经 ``CxrStateMachine.transition()``
   走 CAS；路由层**不**直接 UPDATE triage_states。
7. **降级策略**：模型不可达 / 5 件工具异常 → ``degraded=True`` 仍入队，
   banner 由前端 SSE listener 推（``/api/health.degraded`` 字段同步反映）。

参考：本模块**新建**（源仓库 ``MedRAX`` 含 Gradio interface 但**不涉及**
DICOM 状态机 + 双对外接）。FastAPI 入口 + routers 由本任务落地。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..config import CxrTriageConfig, load_config
from ..engine.cxr_keyqa import CXRKeyQA
from ..engine.finding_draft import FindingDraft
from ..engine.image_preview import ImagePreview
from ..engine.positive_grounding import PositiveGrounding
from ..engine.triage_classifier import TriageClassifier
from ..engine.triage_queue import DEFAULT_DISPATCH_LIMIT, TriageQueue
from ..engine.trigger import TriageTrigger, build_triage_trigger
from ..engine.workflow import CxrTriageWorkflow, ToolRegistry, build_workflow
from ..gateway import ModelGateway, build_model_gateway
from ..pacs.folder_watcher import DicomFolderWatcher
from ..pacs.listener import PacsListener
from ..persistence.idempotency import IdempotencyStore
from ..persistence.migrations import init_db
from ..persistence.outbox import Outbox
from ..persistence.sqlite import Database, DEFAULT_DB_PATH as DEFAULT_SQLITE_PATH
from ..state.idempotency import StateIdempotencyStore
from ..state.state_machine import (
    ActorRole,
    CxrState,
    CxrStateMachine,
    StateTransitionError,
)
from ..writers.alert_banner import (
    ALERT_EVENT_NAME,
    AlertPublisher,
    SsePublishCallback as AlertSsePublishCallback,
    build_alert_publisher,
)
from ..writers.audit import AuditLogger, build_audit_logger
from ..writers.healthcheck_writer import (
    HealthcheckWriter,
    build_healthcheck_writer,
    make_summary_snippet,
)
from ..writers.ris_writer import (
    RISWriter,
    build_ris_writer,
)

from . import (
    ALERT_BANNER_CSS_FILENAME,
    ALERT_BANNER_JS_FILENAME,
    DEFAULT_DEGRADATION_BANNER_TEXT,
    INDEX_DEFAULT_SERVICE_NAME,
    INDEX_DEFAULT_VERSION,
    INDEX_ENDPOINT,
    INDEX_TEMPLATE_NAME,
    SSE_ENDPOINT,
    STATIC_DIR,
    STATIC_MOUNT_PREFIX,
    TEMPLATES_DIR,
)
from .sse import SsePublisher, build_sse_publisher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量（路由名、Pydantic 限值）
# ---------------------------------------------------------------------------

# 默认 SQLite 路径（生产建议通过 CXR_TRIAGE_DB_PATH env 注入）
DEFAULT_SQLITE_DB_PATH: str = DEFAULT_SQLITE_PATH

# 默认 tool_calls JSONL 目录（生产建议通过 CXR_TRIAGE_TOOL_CALLS_DIR env 注入）
DEFAULT_TOOL_CALLS_DIR: str = "/tmp/cxr-triage/tool_calls"

# 默认 patient_id_salt（生产建议通过 CXR_TRIAGE_PATIENT_ID_SALT env 注入）
DEFAULT_PATIENT_ID_SALT: str = "cxr-triage-salt"

# 默认 SSE publisher 若未启用则是一个 noop 回调（不阻断主路径）
_NULL_SSE_PUBLISH: AlertSsePublishCallback = lambda study_uid, payload: asyncio.sleep(0)  # type: ignore


# SSE 媒体类型（application/event-stream 也合法；text/event-stream 最常见）
SSE_MEDIA_TYPE: str = "text/event-stream"


def _noop_sse_publisher() -> SsePublisher:
    """测试/无 SSE 模式下的占位 publisher（不接客户端，broadcast 仍记 history）。"""
    return build_sse_publisher(
        max_history_seconds=1.0,
        client_queue_maxsize=1,
        keepalive_interval_seconds=600.0,
    )


# ---------------------------------------------------------------------------
# Jinja2 templates（Task 30：浏览器三栏面板）
# ---------------------------------------------------------------------------

# 模块顶层 Jinja2Templates 实例；测试 / 生产 / 单 render 共用一份 env，
# 避免 create_app 每次重建。TEMPLATES_DIR 由 ``web.__init__`` 暴露。
JINJA_TEMPLATES: Jinja2Templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# 端点表（首页 HTML 在 body 里直接渲染；模块顶层常量便于单测断言）
INDEX_ENDPOINTS: list[str] = [
    "GET  /api/health",
    "GET  /api/studies",
    "GET  /api/studies/{study_uid}",
    "POST /api/studies/{study_uid}/confirm",
    "POST /api/studies/{study_uid}/reject",
    "GET  /api/queue/positive",
    "GET  /api/queue/negative",
    "GET  /api/outbox",
    "GET  /api/sse",
    "GET  /",
]


# ---------------------------------------------------------------------------
# Pydantic 请求 / 响应模型
# ---------------------------------------------------------------------------


class ConfirmRequest(BaseModel):
    """``/api/studies/{uid}/confirm`` body。"""

    actor_role: str = Field(
        ...,
        description="操作者角色：jr / reviewer / sr / system",
    )
    target_state: str = Field(
        ...,
        description=(
            "期望目标状态：jr_confirmed / sr_reviewed / sr_signed（驳回用 /reject 端点）"
        ),
    )
    expected_from_state: Optional[str] = Field(
        default=None,
        description="期望源状态（CAS 用）；不传则取当前状态",
    )


class RejectRequest(BaseModel):
    """``/api/studies/{uid}/reject`` body。"""

    actor_role: str = Field(..., description="驳回者角色：jr / reviewer / sr")
    from_state: Optional[str] = Field(
        default=None,
        description="期望源状态；不传则取当前状态",
    )
    reason: str = Field(default="", description="驳回原因（写入 audit_log）")


class ConfirmResponse(BaseModel):
    """``/api/studies/{uid}/confirm`` 响应。"""

    success: bool
    study_uid: str
    from_state: str
    to_state: str
    new_state: str
    actor_role: str
    audit_log_id: Optional[int] = None
    from_state_actual: Optional[str] = None
    dual_write: Optional[dict[str, Any]] = Field(
        default=None,
        description="sr_signed 双写结果；非 sr_signed 跃迁时为 null",
    )


class RejectResponse(BaseModel):
    """``/api/studies/{uid}/reject`` 响应。"""

    success: bool
    study_uid: str
    from_state: str
    to_state: str  # final = draft_pending
    new_state: str
    actor_role: str
    audit_log_id: Optional[int] = None
    reason: str = ""


class HealthResponse(BaseModel):
    """``/api/health`` 响应。"""

    status: str
    pacs: str  # connected | watch_folder | fallback | disconnected
    model: str  # ok | degraded | unknown
    outbox_alerts: int
    outbox_pending: int
    outbox_alerting: int
    db_ok: bool
    trigger_running: bool
    pacs_listener_running: bool
    folder_watcher_running: bool
    ris_retry_running: bool
    healthcheck_retry_running: bool
    state_count: dict[str, int]
    queue_count: dict[str, int]
    uptime_seconds: float
    config: dict[str, Any]


class QueueItem(BaseModel):
    """队列单条 / 列表响应里的 study 行。"""

    study_uid: str
    max_prob: float
    positive_flag: bool
    finding_text: str
    inserted_at: Optional[str] = None
    dispatched_at: Optional[str] = None
    triage_state: Optional[str] = None


class StudyDetailResponse(BaseModel):
    """``/api/studies/{uid}`` 详情（draft + state + triage_queue 行）。"""

    study_uid: str
    triage_state: Optional[str]
    incoming: Optional[dict[str, Any]]
    triage_queue: Optional[dict[str, Any]]
    draft: Optional[dict[str, Any]] = Field(
        default=None,
        description="从 draft_idempotency 缓存拉的产物（5 分钟内）",
    )
    audit_trail: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="最近 10 条 state_change / state_reject 审计行",
    )


# ---------------------------------------------------------------------------
# 双写协调（sr_signed 触发）
# ---------------------------------------------------------------------------


class DualWriteResult(BaseModel):
    """sr_signed 双写 RIS + 体检主检结果聚合。"""

    ris_status: str  # ok | fail
    healthcheck_status: str  # ok | fail
    audit_log_id: Optional[int] = None
    visit_id: Optional[str] = None


async def perform_sr_signed_dual_write(
    *,
    study_uid: str,
    ris_writer: RISWriter,
    healthcheck_writer: HealthcheckWriter,
    audit_logger: AuditLogger,
    draft: dict[str, Any],
    visit_id: Optional[str] = None,
) -> DualWriteResult:
    """sr_signed 后调用：RIS + 体检主检双写。

    - RIS: ``write_finding_draft(study_uid=..., patient_id_hash=..., finding_text=...,
      finding_struct=..., positive_flag=..., draft_ts=..., tool_version=...)``
    - 体检主检: ``write_summary_ref(visit_id=..., summary_snippet=..., source_study_uids=[uid], ...)``

    任一失败 ``ris_status=fail`` 或 ``healthcheck_status=fail``，但**不抛**
    （双写兜底由 outbox + alert 接管）。``audit_logger.write_dual_write``
    双写联合落 audit（spec.md §5 维度 6 双写对账）。

    Returns
    -------
    DualWriteResult 含两边 status + 统一 audit id。
    """
    if not draft:
        # 没有 draft 不写（避免空 payload）
        return DualWriteResult(ris_status="fail", healthcheck_status="fail")
    ris_ok = False
    hc_ok = False
    try:
        ris_ok = await ris_writer.write_finding_draft(
            study_uid=study_uid,
            patient_id_hash=str(draft.get("patient_id_hash") or ""),
            finding_text=str(draft.get("finding_text") or ""),
            finding_struct=dict(draft.get("finding_struct") or {}),
            positive_flag=bool(draft.get("positive_flag")),
            draft_ts=str(draft.get("draft_ts") or _utc_now_iso()),
            tool_version=str(
                draft.get("tool_version") or "0.1.0"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "RIS dual_write exception study=%s err=%s", study_uid, exc
        )
    if visit_id:
        snippet = make_summary_snippet(
            finding_text=str(draft.get("finding_text") or ""),
            max_prob=float(draft.get("classifier_max_prob") or 0.0),
            positive_flag=bool(draft.get("positive_flag")),
            extra_suggestion="建议进一步 CT",
        )
        try:
            hc_ok = await healthcheck_writer.write_summary_ref(
                visit_id=visit_id,
                summary_snippet=snippet,
                source_study_uids=[study_uid],
                model_version=str(
                    draft.get("tool_version") or "0.1.0"
                ),
                study_uid=study_uid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Healthcheck dual_write exception study=%s err=%s",
                study_uid,
                exc,
            )

    ris_status = "ok" if ris_ok else "fail"
    hc_status = "ok" if hc_ok else "fail"

    audit_id: Optional[int] = None
    try:
        # AuditLogger.write_dual_write 返回 (ris_id, healthcheck_id)；
        # DualWriteResult.audit_log_id 字段只暴露 RIS 那条（飞检按
        # pair_ts + 双 event_type 关联两条 audit_log 行）。
        dual_audit_ids = await audit_logger.write_dual_write(
            study_uid=study_uid,
            ris_status=ris_status,
            healthcheck_status=hc_status,
        )
        if isinstance(dual_audit_ids, tuple) and len(dual_audit_ids) >= 1:
            audit_id = int(dual_audit_ids[0])
        else:
            audit_id = int(dual_audit_ids) if dual_audit_ids is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "audit_logger.write_dual_write FAILED study=%s err=%s",
            study_uid,
            exc,
        )
    return DualWriteResult(
        ris_status=ris_status,
        healthcheck_status=hc_status,
        audit_log_id=audit_id,
        visit_id=visit_id,
    )


# ---------------------------------------------------------------------------
# AppState 容器：放在 app.state 上，方便路由 + 测试访问
# ---------------------------------------------------------------------------


class AppState:
    """FastAPI ``app.state`` 上的业务组件容器。

    所有 lifespan 内构造的对象集中存放；单测可通过 fake 全替换。
    """

    def __init__(
        self,
        *,
        config: CxrTriageConfig,
        db: Database,
        audit_logger: AuditLogger,
        outbox: Outbox,
        ris_writer: RISWriter,
        healthcheck_writer: HealthcheckWriter,
        idempotency: StateIdempotencyStore,
        triage_queue: TriageQueue,
        workflow: CxrTriageWorkflow,
        triage_trigger: TriageTrigger,
        state_machine: CxrStateMachine,
        alert_publisher: Optional[AlertPublisher],
        pacs_listener: Optional[PacsListener],
        folder_watcher: Optional[DicomFolderWatcher],
        sse_publish: Optional[Any],
        sse_publisher: Optional[SsePublisher] = None,
        startup_at: float = 0.0,
    ) -> None:
        self.config = config
        self.db = db
        self.audit_logger = audit_logger
        self.outbox = outbox
        self.ris_writer = ris_writer
        self.healthcheck_writer = healthcheck_writer
        self.idempotency = idempotency
        self.triage_queue = triage_queue
        self.workflow = workflow
        self.triage_trigger = triage_trigger
        self.state_machine = state_machine
        self.alert_publisher = alert_publisher
        self.pacs_listener = pacs_listener
        self.folder_watcher = folder_watcher
        self.sse_publish = sse_publish or _NULL_SSE_PUBLISH
        self.sse_publisher = sse_publisher or _noop_sse_publisher()
        self.startup_at = startup_at or time.time()

    # ---- 测试辅助：注入一个不依赖真 db 的 state ----
    @classmethod
    def for_testing(
        cls,
        *,
        config: Optional[CxrTriageConfig] = None,
        db: Optional[Database] = None,
        audit_logger: Optional[AuditLogger] = None,
        outbox: Optional[Outbox] = None,
        ris_writer: Optional[RISWriter] = None,
        healthcheck_writer: Optional[HealthcheckWriter] = None,
        idempotency: Optional[StateIdempotencyStore] = None,
        triage_queue: Optional[TriageQueue] = None,
        workflow: Optional[CxrTriageWorkflow] = None,
        triage_trigger: Optional[TriageTrigger] = None,
        state_machine: Optional[CxrStateMachine] = None,
        alert_publisher: Optional[AlertPublisher] = None,
        pacs_listener: Optional[PacsListener] = None,
        folder_watcher: Optional[DicomFolderWatcher] = None,
        sse_publish: Optional[Any] = None,
        sse_publisher: Optional[SsePublisher] = None,
    ) -> "AppState":
        cfg = config or load_config()
        # 真实 workflow 构造路径（与 _build_lifespan_state 一致）：
        # 注册 5 件工具 + build_workflow。测试通常会显式 inject workflow，
        # 这条 fallback 仅在没有 inject 时兜底（推荐 inject，避免拉模型权重）。
        _registry = ToolRegistry()
        _registry.register(TriageClassifier())
        _registry.register(CXRKeyQA())
        _registry.register(FindingDraft(positive_threshold=cfg.positive_threshold))
        _registry.register(PositiveGrounding())
        _registry.register(ImagePreview())
        _real_workflow = build_workflow(
            _registry, positive_threshold=cfg.positive_threshold
        )
        return cls(
            config=cfg,
            db=db or Database(),
            audit_logger=audit_logger
            or build_audit_logger(db=db or Database(), tool_calls_dir=DEFAULT_TOOL_CALLS_DIR),
            outbox=outbox or Outbox(db or Database()),
            ris_writer=ris_writer
            or build_ris_writer(
                config=cfg, outbox=Outbox(db or Database()), audit_logger=AuditLogger(db or Database())
            ),
            healthcheck_writer=healthcheck_writer
            or build_healthcheck_writer(
                config=cfg, outbox=Outbox(db or Database()), audit_logger=AuditLogger(db or Database())
            ),
            idempotency=idempotency or StateIdempotencyStore(db or Database()),
            triage_queue=triage_queue or TriageQueue(db or Database()),
            workflow=workflow or _real_workflow,
            triage_trigger=triage_trigger or build_triage_trigger(
                config=cfg,
                db=db or Database(),
                workflow=workflow or _real_workflow,
                triage_queue=triage_queue or TriageQueue(db or Database()),
                idempotency=idempotency or StateIdempotencyStore(db or Database()),
            ),
            state_machine=state_machine
            or CxrStateMachine(db or Database()),
            alert_publisher=alert_publisher,
            pacs_listener=pacs_listener,
            folder_watcher=folder_watcher,
            sse_publish=sse_publish,
            sse_publisher=sse_publisher,
        )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """UTC ISO8601（与 persistence / audit 同源）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _coerce_state(s: Optional[str]) -> Optional[CxrState]:
    """``None`` / ``CxrState`` → ``Optional[CxrState]``。"""
    if s is None:
        return None
    if isinstance(s, CxrState):
        return s
    raw = str(s).strip().lower()
    for st in CxrState:
        if st.value == raw:
            return st
    return None


def _coerce_role(s: str) -> ActorRole:
    """``str`` → ``ActorRole``；非法 → StateTransitionError。"""
    if isinstance(s, ActorRole):
        return s
    raw = str(s or "").strip().lower()
    for r in ActorRole:
        if r.value == raw:
            return r
    raise StateTransitionError(f"未知角色: {s!r}")


# ---------------------------------------------------------------------------
# Lifespan：构造完整组件图
# ---------------------------------------------------------------------------


async def _build_lifespan_state(
    *,
    config: Optional[CxrTriageConfig] = None,
    db_path: Optional[str] = None,
    tool_calls_dir: Optional[str] = None,
    enable_pacs_listener: bool = True,
    enable_folder_watcher: bool = True,
    sse_publish: Optional[AlertSsePublishCallback] = None,
    sse_publisher: Optional[SsePublisher] = None,
) -> AppState:
    """从 env / 默认值构造一份 AppState（lifespan 与测试均可调用）。

    Parameters
    ----------
    enable_pacs_listener:
        是否起 ``PacsListener``（生产 True；测试可关避免端口占用）。
    enable_folder_watcher:
        是否起 ``DicomFolderWatcher``（同上）。
    sse_publish:
        Task 29 ``SsePublisher.publish`` 注入点；未实现传 ``None`` 时
        降级为 noop callback（主路径仍跑）。
    sse_publisher:
        Task 29 ``SsePublisher`` 实例；未提供则新建一个，并把
        ``publisher.publish`` 接到 AlertPublisher / TriageTrigger 上。
    """
    cfg = config or load_config()
    db = Database(db_path or os.getenv("CXR_TRIAGE_DB_PATH", DEFAULT_SQLITE_DB_PATH))
    await db.connect()
    await init_db(db)

    audit_logger = build_audit_logger(
        db=db, tool_calls_dir=tool_calls_dir or DEFAULT_TOOL_CALLS_DIR
    )
    outbox = Outbox(db)
    ris_writer = build_ris_writer(config=cfg, outbox=outbox, audit_logger=audit_logger)
    healthcheck_writer = build_healthcheck_writer(
        config=cfg, outbox=outbox, audit_logger=audit_logger
    )
    idempotency = StateIdempotencyStore(db)
    triage_queue = TriageQueue(db)

    # ---- 构建 ModelGateway + 5 件工具（lifespan 真实启动路径）----
    # Tools 需要 model_gateway + audit_logger；构造 ToolRegistry + build_workflow。
    # 这一段在 for_testing 里以 fake workflow 短路，**仅 lifespan 真实路径**走。
    model_gateway = build_model_gateway(audit_logger=audit_logger)
    registry = ToolRegistry()
    registry.register(TriageClassifier())
    registry.register(CXRKeyQA(model_gateway=model_gateway, audit_logger=audit_logger))
    registry.register(
        FindingDraft(positive_threshold=cfg.positive_threshold)
    )
    registry.register(
        PositiveGrounding(
            model_gateway=model_gateway, audit_logger=audit_logger
        )
    )
    registry.register(ImagePreview())
    workflow = build_workflow(
        registry, positive_threshold=cfg.positive_threshold
    )

    # ---- SsePublisher (Task 29) ----
    sse_publisher_instance = sse_publisher or _noop_sse_publisher()
    resolved_sse_publish = sse_publish or sse_publisher_instance.publish

    triage_trigger = build_triage_trigger(
        config=cfg,
        db=db,
        workflow=workflow,
        triage_queue=triage_queue,
        idempotency=idempotency,
        audit_logger=audit_logger,
        sse_publish=resolved_sse_publish,
    )
    state_machine = CxrStateMachine(db)

    # Alert publisher
    alert_publisher = build_alert_publisher(
        outbox=outbox,
        audit=audit_logger,
        sse_publish=resolved_sse_publish,
    )
    outbox.register_alert_callback(alert_publisher.on_outbox_alert_transition)

    pacs_listener: Optional[PacsListener] = None
    if enable_pacs_listener:
        try:
            pacs_listener = PacsListener(
                config=cfg,
                db=db,
                audit_hook=None,
            )
            await pacs_listener.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PacsListener 启动失败（不阻断），降级到 watch_folder: %s", exc
            )
            pacs_listener = None

    folder_watcher: Optional[DicomFolderWatcher] = None
    if enable_folder_watcher:
        try:
            folder_watcher = DicomFolderWatcher(config=cfg, db=db)
            await folder_watcher.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DicomFolderWatcher 启动失败: %s", exc)
            folder_watcher = None

    return AppState(
        config=cfg,
        db=db,
        audit_logger=audit_logger,
        outbox=outbox,
        ris_writer=ris_writer,
        healthcheck_writer=healthcheck_writer,
        idempotency=idempotency,
        triage_queue=triage_queue,
        workflow=workflow,
        triage_trigger=triage_trigger,
        state_machine=state_machine,
        alert_publisher=alert_publisher,
        pacs_listener=pacs_listener,
        folder_watcher=folder_watcher,
        sse_publish=resolved_sse_publish,
        sse_publisher=sse_publisher_instance,
        startup_at=time.time(),
    )


async def _teardown_lifespan_state(state: AppState) -> None:
    """反向关闭 ``_build_lifespan_state``。"""
    try:
        await state.triage_trigger.stop()
    except Exception:  # noqa: BLE001
        logger.exception("TriageTrigger 关闭异常")
    try:
        await state.ris_writer.aclose()
    except Exception:  # noqa: BLE001
        logger.exception("RISWriter 关闭异常")
    try:
        await state.healthcheck_writer.aclose()
    except Exception:  # noqa: BLE001
        logger.exception("HealthcheckWriter 关闭异常")
    try:
        if state.folder_watcher is not None:
            await state.folder_watcher.stop()
    except Exception:  # noqa: BLE001
        logger.exception("DicomFolderWatcher 关闭异常")
    try:
        if state.pacs_listener is not None:
            await state.pacs_listener.stop()
    except Exception:  # noqa: BLE001
        logger.exception("PacsListener 关闭异常")
    try:
        await state.db.close()
    except Exception:  # noqa: BLE001
        logger.exception("Database 关闭异常")


def create_app(
    *,
    config: Optional[CxrTriageConfig] = None,
    db_path: Optional[str] = None,
    tool_calls_dir: Optional[str] = None,
    enable_pacs_listener: bool = True,
    enable_folder_watcher: bool = True,
    sse_publish: Optional[AlertSsePublishCallback] = None,
    sse_publisher: Optional[SsePublisher] = None,
) -> FastAPI:
    """构造 FastAPI app（含 lifespan + routers）。

    生产调用::

        app = create_app()

    测试可注入 ``config`` / ``db_path`` / 关闭 PACS 与 folder watcher
    避免端口占用 + 隔离副作用。
    """
    state_holder: dict[str, Optional[AppState]] = {"state": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: D401
        state = await _build_lifespan_state(
            config=config,
            db_path=db_path,
            tool_calls_dir=tool_calls_dir,
            enable_pacs_listener=enable_pacs_listener,
            enable_folder_watcher=enable_folder_watcher,
            sse_publish=sse_publish,
            sse_publisher=sse_publisher,
        )
        state_holder["state"] = state
        app.state.cxr = state

        # 启动 retry_loop 协程（RIS + 主检；与 TriageTrigger 并行）
        try:
            await state.ris_writer.start_retry_loop()
        except Exception:  # noqa: BLE001
            logger.exception("RIS retry_loop 启动失败（不阻断）")
        try:
            await state.healthcheck_writer.start_retry_loop()
        except Exception:  # noqa: BLE001
            logger.exception("Healthcheck retry_loop 启动失败（不阻断）")
        # 启动 TriageTrigger 轮询协程
        try:
            await state.triage_trigger.start()
        except Exception:  # noqa: BLE001
            logger.exception("TriageTrigger.start 失败（不阻断）")

        try:
            yield
        finally:
            await _teardown_lifespan_state(state)
            state_holder["state"] = None
            app.state.cxr = None

    app = FastAPI(
        title="体检胸片所见草稿系统 / CXR-Triage-Draft",
        description="体检 PACS 入站 → 多工具协同 → 三角色流转 → RIS + 体检主检双写",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.cxr_holder = state_holder  # 测试可在 create_app 后注入 fake

    _register_routers(app)
    _mount_static(app)
    return app


def _register_routers(app: FastAPI) -> None:
    """注册所有路由（与 lifespan 解耦，单测可直接 on_event 注册）。"""

    # ----------------------------------------------------------------------
    # /api/health
    # ----------------------------------------------------------------------
    @app.get("/api/health", response_model=HealthResponse)
    async def api_health() -> HealthResponse:
        state: AppState = app.state.cxr  # type: ignore[assignment]
        db_ok = await state.db.ping()
        outbox_counts = await state.outbox.count_by_status()
        state_counts = await state.state_machine.count_by_state()
        pacs_status = (
            "connected" if state.pacs_listener is not None and state.pacs_listener.is_running
            else "watch_folder" if state.folder_watcher is not None and state.folder_watcher.is_running
            else "fallback"
        )
        model_status = "ok"
        if state.triage_trigger.stats().get("degraded", 0) > 0:
            model_status = "degraded"

        # config 安全摘要（不含 token / key）
        cfg = state.config
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            pacs=pacs_status,
            model=model_status,
            outbox_alerts=int(outbox_counts.get("alerting", 0)),
            outbox_pending=int(outbox_counts.get("pending", 0)),
            outbox_alerting=int(outbox_counts.get("alerting", 0)),
            db_ok=bool(db_ok),
            trigger_running=state.triage_trigger.is_running,
            pacs_listener_running=bool(
                state.pacs_listener is not None and state.pacs_listener.is_running
            ),
            folder_watcher_running=bool(
                state.folder_watcher is not None and state.folder_watcher.is_running
            ),
            ris_retry_running=state.ris_writer.retry_running,
            healthcheck_retry_running=state.healthcheck_writer.retry_running,
            state_count=state_counts,
            queue_count={
                "pending": int(await state.triage_queue.count_pending() or 0),
                "dispatched": int(await state.triage_queue.count_dispatched() or 0),
            },
            uptime_seconds=float(time.time() - state.startup_at),
            config={
                "pacs_ae_title": cfg.pacs_ae_title,
                "pacs_endpoint": cfg.pacs_endpoint,
                "pacs_wado_url": cfg.pacs_wado_url,
                "ris_api_url": cfg.ris_api_url,
                "healthcheck_api_url": cfg.healthcheck_api_url,
                "openai_base_url": cfg.openai_base_url,
                "openai_model": cfg.openai_model,
                "positive_threshold": cfg.positive_threshold,
                "watch_folder": str(cfg.watch_folder),
            },
        )

    # ----------------------------------------------------------------------
    # /api/studies — 列表
    # ----------------------------------------------------------------------
    @app.get("/api/studies")
    async def api_list_studies(
        state_filter: Optional[str] = Query(
            default=None,
            description="状态过滤（draft_pending / jr_confirmed / ...）",
        ),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        target = _coerce_state(state_filter)
        if state_filter and target is None:
            raise HTTPException(
                status_code=400,
                detail=f"未知 state_filter: {state_filter}",
            )
        items = await st.state_machine.list_by_state(target or CxrState.DRAFT_PENDING, limit=limit) if target else []
        # 同步拉 triage_queue 行（用于面板首屏）
        for it in items:
            queue_row = await st.triage_queue.get_by_study(it["study_uid"])
            if queue_row:
                it["max_prob"] = float(queue_row.get("max_prob") or 0.0)
                it["positive_flag"] = bool(queue_row.get("positive_flag"))
                it["finding_text"] = queue_row.get("finding_text") or ""
                it["dispatched_at"] = queue_row.get("dispatched_at")
            else:
                it["max_prob"] = 0.0
                it["positive_flag"] = False
                it["finding_text"] = ""
        return {"items": items, "count": len(items)}

    # ----------------------------------------------------------------------
    # /api/studies/{uid} — 详情
    # ----------------------------------------------------------------------
    @app.get("/api/studies/{study_uid}", response_model=StudyDetailResponse)
    async def api_study_detail(study_uid: str) -> StudyDetailResponse:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        if not study_uid or not study_uid.strip():
            raise HTTPException(status_code=400, detail="study_uid 不能为空")
        # 状态
        triage_state = await st.state_machine.query_state(study_uid)
        # incoming 行
        async with st.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, patient_id_hash, accession_number, modality,
                       study_date, patient_sex, patient_age, source, status,
                       max_prob, received_at, updated_at
                FROM incoming_studies WHERE study_uid = ?
                """,
                (study_uid,),
            )
            row = await cur.fetchone()
            await cur.close()
        incoming = dict(row) if row is not None else None
        # queue 行
        queue_row = await st.triage_queue.get_by_study(study_uid)
        # draft (idempotency)
        draft = await st.idempotency.lookup_with_meta(study_uid)
        # audit_trail（最近 10 条 state_*）
        audit_trail = await st.state_machine.list_audit_for_study(study_uid, limit=10)
        return StudyDetailResponse(
            study_uid=study_uid,
            triage_state=triage_state.value if triage_state else None,
            incoming=incoming,
            triage_queue=queue_row,
            draft=draft,
            audit_trail=audit_trail,
        )

    # ----------------------------------------------------------------------
    # /api/studies/{uid}/confirm — 状态确认 + sr_signed 双写
    # ----------------------------------------------------------------------
    @app.post("/api/studies/{study_uid}/confirm", response_model=ConfirmResponse)
    async def api_confirm(study_uid: str, body: ConfirmRequest) -> ConfirmResponse:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        if not study_uid or not study_uid.strip():
            raise HTTPException(status_code=400, detail="study_uid 不能为空")
        if not body.actor_role or not body.target_state:
            raise HTTPException(status_code=400, detail="actor_role 与 target_state 必填")

        # 入参校验
        try:
            role = _coerce_role(body.actor_role)
        except StateTransitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        target = _coerce_state(body.target_state)
        if target is None:
            raise HTTPException(status_code=400, detail=f"未知 target_state: {body.target_state}")

        # 若不传 expected_from_state → 实际查当前状态
        cur = await st.state_machine.query_state(study_uid)
        if cur is None:
            await st.state_machine.get_or_init_state(study_uid)
            cur = CxrState.DRAFT_PENDING
        from_state = _coerce_state(body.expected_from_state) or cur

        try:
            result = await st.state_machine.transition(
                study_uid=study_uid,
                from_state=from_state,
                to_state=target,
                actor_role=role,
            )
        except StateTransitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        dual: Optional[DualWriteResult] = None
        # sr_signed 成功 → 双写 RIS + 体检主检；非 sr_signed 仍写队列分派
        if result.success and result.new_state == CxrState.SR_SIGNED:
            draft_meta = await st.idempotency.lookup(study_uid) or {}
            # 默认 visit_id 同 study_uid（无 external visit 映射时）；
            # 真实系统应按 AccessionNumber / 健康体检流水号 映射。
            visit_id = (draft_meta.get("visit_id") if isinstance(draft_meta, dict) else None) or study_uid
            dual = await perform_sr_signed_dual_write(
                study_uid=study_uid,
                ris_writer=st.ris_writer,
                healthcheck_writer=st.healthcheck_writer,
                audit_logger=st.audit_logger,
                draft=draft_meta if isinstance(draft_meta, dict) else {},
                visit_id=visit_id,
            )

        # 推进队列派发（非 rejected；disposition 不在此处删除，便于驳回后重跑）
        return ConfirmResponse(
            success=result.success,
            study_uid=study_uid,
            from_state=from_state.value,
            to_state=target.value,
            new_state=result.new_state.value,
            actor_role=role.value,
            audit_log_id=result.audit_log_id,
            from_state_actual=result.from_state_actual.value
            if result.from_state_actual
            else None,
            dual_write=dual.model_dump() if dual is not None else None,
        )

    # ----------------------------------------------------------------------
    # /api/studies/{uid}/reject — 驳回（一并回 draft_pending + 清幂等）
    # ----------------------------------------------------------------------
    @app.post("/api/studies/{study_uid}/reject", response_model=RejectResponse)
    async def api_reject(study_uid: str, body: RejectRequest) -> RejectResponse:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        if not study_uid or not study_uid.strip():
            raise HTTPException(status_code=400, detail="study_uid 不能为空")
        if not body.actor_role:
            raise HTTPException(status_code=400, detail="actor_role 必填")
        try:
            role = _coerce_role(body.actor_role)
        except StateTransitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        cur = await st.state_machine.query_state(study_uid)
        if cur is None:
            await st.state_machine.get_or_init_state(study_uid)
            cur = CxrState.DRAFT_PENDING
        from_state = _coerce_state(body.from_state) or cur

        try:
            result = await st.state_machine.reject(
                study_uid=study_uid,
                from_state=from_state,
                actor_role=role,
                reason=body.reason or "",
            )
        except StateTransitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # 驳回成功 → 清幂等缓存，下一轮 trigger 重新跑 5 件工具
        if result.success:
            try:
                await st.idempotency.invalidate_study(study_uid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "驳回后清幂等缓存失败 study=%s err=%s", study_uid, exc
                )

        return RejectResponse(
            success=result.success,
            study_uid=study_uid,
            from_state=from_state.value,
            to_state=result.new_state.value,  # 回退到 draft_pending
            new_state=result.new_state.value,
            actor_role=role.value,
            audit_log_id=result.audit_log_id,
            reason=body.reason or "",
        )

    # ----------------------------------------------------------------------
    # /api/queue/positive + /api/queue/negative — 浏览器首屏数据源
    # ----------------------------------------------------------------------
    @app.get("/api/queue/positive", response_model=list[QueueItem])
    async def api_queue_positive(
        limit: int = Query(default=DEFAULT_DISPATCH_LIMIT, ge=1, le=200),
    ) -> list[QueueItem]:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        # positive_flag=True 通过 get_by_study 在 list_pending 之后过滤；
        # list_pending 已按 max_prob DESC 排序，直接过滤
        pending = await st.triage_queue.list_pending(limit=max(limit * 2, limit))
        items = [p for p in pending if bool(p.get("positive_flag"))]
        items = items[:limit]
        return [_queue_row_to_item(it) for it in items]

    @app.get("/api/queue/negative", response_model=list[QueueItem])
    async def api_queue_negative(
        limit: int = Query(default=DEFAULT_DISPATCH_LIMIT, ge=1, le=200),
    ) -> list[QueueItem]:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        pending = await st.triage_queue.list_pending(limit=max(limit * 2, limit))
        items = [p for p in pending if not bool(p.get("positive_flag"))]
        items = items[:limit]
        return [_queue_row_to_item(it) for it in items]

    # ----------------------------------------------------------------------
    # /api/outbox — outbox 监控（飞检 / 信息科）
    # ----------------------------------------------------------------------
    @app.get("/api/outbox")
    async def api_outbox_summary() -> dict[str, Any]:
        st: AppState = app.state.cxr  # type: ignore[assignment]
        return {
            "counts": await st.outbox.count_by_status(),
            "alerting": await st.outbox.count_alerting(),
            "stats": {
                "ris": st.ris_writer.stats,
                "healthcheck": st.healthcheck_writer.stats,
            },
        }

    # ----------------------------------------------------------------------
    # / — 浏览器三栏面板首页（Jinja2 模板，Task 30）
    # ----------------------------------------------------------------------

    async def _build_index_context() -> dict[str, Any]:
        """构建模板渲染上下文（端点表 + 服务信息 + 计数）。"""
        ctx: dict[str, Any] = {
            "service_name": INDEX_DEFAULT_SERVICE_NAME,
            "version": INDEX_DEFAULT_VERSION,
            "endpoint_count": len(INDEX_ENDPOINTS),
            "endpoints": INDEX_ENDPOINTS,
            "degradation_banner_text": DEFAULT_DEGRADATION_BANNER_TEXT,
            "sse_endpoint": SSE_ENDPOINT,
            "static_prefix": STATIC_MOUNT_PREFIX,
            "queue_count": {"positive": 0, "negative": 0},
            "outbox_alerting": 0,
            "degraded": False,
        }
        try:
            st: AppState = app.state.cxr  # type: ignore[assignment]
            if st is not None and hasattr(st, "triage_queue"):
                pending = await st.triage_queue.list_pending(
                    limit=DEFAULT_DISPATCH_LIMIT,
                )
                ctx["queue_count"] = {
                    "positive": sum(
                        1 for it in pending if it.get("positive_flag")
                    ),
                    "negative": sum(
                        1 for it in pending if not it.get("positive_flag")
                    ),
                }
                ctx["outbox_alerting"] = await st.outbox.count_alerting()
                # 模型状态：通过 audit_log / health 字段推断（避免硬耦合）
                ctx["degraded"] = bool(getattr(st, "model_degraded", False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("index context 拉取失败（不阻断）：%s", exc)
        return ctx

    @app.get(INDEX_ENDPOINT)
    async def api_index(request: Request) -> Any:
        """浏览器三栏分诊面板（阳性队列 / 阴性队列 / 草稿所见）。

        渲染 ``templates/index.html``，传 ``service_name`` / ``endpoint_count``
        / ``degradation_banner_text`` 等上下文。客户端 Alpine.js 监听 SSE
        增量更新列表，三角色流转按钮走 ``/api/studies/{uid}/confirm`` /
        ``/api/studies/{uid}/reject``。
        """
        ctx = await _build_index_context()
        return JINJA_TEMPLATES.TemplateResponse(
            request, INDEX_TEMPLATE_NAME, ctx
        )

    # ----------------------------------------------------------------------
    # /api/sse — Server-Sent Events 实时推送（Task 29）
    # ----------------------------------------------------------------------
    @app.get(SSE_ENDPOINT)
    async def api_sse(last_seen_ts: Optional[str] = Query(default=None)) -> StreamingResponse:
        """SSE 流式端点：浏览器 EventSource 客户端连这一路。

        Query 参数：
        - ``last_seen_ts``：可选 ISO8601 字符串；客户端断线重连时
          把最后收到的事件 ts 传上来，先 replay ``max_history_seconds``
          内历史再切到 live feed。
        """
        st: AppState = app.state.cxr  # type: ignore[assignment]
        return StreamingResponse(
            st.sse_publisher.event_stream(last_seen_ts=last_seen_ts),
            media_type=SSE_MEDIA_TYPE,
        )


def _queue_row_to_item(it: dict[str, Any]) -> QueueItem:
    """triage_queue 行 → QueueItem。"""
    return QueueItem(
        study_uid=it.get("study_uid", ""),
        max_prob=float(it.get("max_prob") or 0.0),
        positive_flag=bool(it.get("positive_flag")),
        finding_text=str(it.get("finding_text") or ""),
        inserted_at=it.get("inserted_at"),
        dispatched_at=it.get("dispatched_at"),
        triage_state=None,
    )


def _mount_static(app: FastAPI) -> None:
    """挂载 ``/static`` 目录（含 Task 27 banner CSS / JS）。"""
    if STATIC_DIR.exists():
        app.mount(
            STATIC_MOUNT_PREFIX,
            StaticFiles(directory=str(STATIC_DIR)),
            name="cxr-static",
        )


# ---------------------------------------------------------------------------
# 模块顶层 app（uvicorn cxr_triage.web.app:app 直接起）
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


__all__ = [
    "AppState",
    "ConfirmRequest",
    "ConfirmResponse",
    "DualWriteResult",
    "HealthResponse",
    "INDEX_ENDPOINTS",
    "JINJA_TEMPLATES",
    "QueueItem",
    "RejectRequest",
    "RejectResponse",
    "SSE_MEDIA_TYPE",
    "StudyDetailResponse",
    "create_app",
    "perform_sr_signed_dual_write",
    "app",
]
