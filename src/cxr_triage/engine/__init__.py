"""engine — 多工具协同推理 + 阳性分诊触发 + 核心算法层。

公开入口：
- ``TriageQueue``              阳性优先级队列（按 max_prob 降序），浏览器
                              第一栏「阳性队列」的数据源；同时承载
                              finding_text 摘要。
- ``TriageQueueError``         入参 / 状态非法异常
- ``DEFAULT_DISPATCH_LIMIT``   ``list_pending`` 默认拉取上限

业务定位（spec.md §5 维度 1 + 3 + plan.md §5 深度 (a) 真实算法）：

- PACS / watch_folder 入站 → StateGraph 跑完 5 件工具 → 入 ``TriageQueue``
  → 浏览器面板 SSE 推送 ``new_study`` 事件 → 放射初诊医师 (jr) 点
  ``confirm`` → 走 ``CxrStateMachine.transition`` 推进 ``draft_pending → jr_confirmed``
  → 主检复核 (reviewer) → 终审签发 (sr) → RIS finding-draft + 体检主检
  summary-ref 真实外发。
- 本类**只**管 max_prob 排序 + finding_text 缓存；**不**做状态机流转
  （找 ``state.CxrStateMachine``），也**不**写 RIS / 主检外发（找
  ``writers.RISWriter`` / ``writers.HealthcheckWriter``）。
- 同 ``StudyInstanceUID`` 二次入站 → ``INSERT OR REPLACE`` 刷新
  ``max_prob / positive_flag / finding_text / inserted_at``（draft 由
  完整 StateGraph 重跑出，避免用旧草稿冲掉新发现）。
- 与 ``outbox`` 同形：所有事件（enqueue / dispatch）都落 ``audit_log``，
  便于飞检追溯「哪条草稿进队列、何时被取走」。
"""
from __future__ import annotations

from .cxr_keyqa import (
    DEFAULT_KEY_QA_TEMPLATES,
    DEFAULT_TIMEOUT_SECONDS,
    KEY_QA_JSON_SCHEMA,
    CXRKeyQA,
    KeyQAResult,
    KeyQAToolError,
    ModelUnreachableError,
    build_cxr_keyqa,
)
from .image_preview import (
    DEFAULT_WINDOW_LEVEL as IMAGE_PREVIEW_DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH as IMAGE_PREVIEW_DEFAULT_WINDOW_WIDTH,
    PNG_DATA_URL_PREFIX,
    PNG_MAGIC,
    PREVIEW_MAX_BASE64_BYTES,
    ImagePreview,
    ImagePreviewError,
    ImagePreviewResult,
    build_data_url,
    build_image_preview,
    encode_png_to_b64,
)
from .finding_draft import (
    DEFAULT_LOCALIZATION_BY_PATHOLOGY,
    DEFAULT_POSITIVE_THRESHOLD,
    DRAFT_MARKER,
    FindingDraft,
    FindingDraftError,
    FindingDraftResult,
    MAX_PATHOLOGIES_IN_STRUCT,
    NO_ABNORMALITY_TEMPLATE,
    POSITIVE_TEMPLATE,
    TOP_N_PATHOLOGIES,
    build_finding_draft,
)
from .positive_grounding import (
    DEFAULT_BBOX_BY_REGION,
    DEFAULT_FALLBACK_BBOX,
    MAX_GROUNDING_BOXES,
    PATHOLOGY_NAMES as POSITIVE_GROUNDING_PATHOLOGY_NAMES,
    POSITIVE_GROUNDING_PROB_HEURISTIC,
    PositiveGrounding,
    PositiveGroundingError,
    PositiveGroundingResult,
    build_positive_grounding,
)
from .triage_classifier import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    PATHOLOGY_NAMES,
    POSITIVE_PROB_HEURISTIC,
    TriageClassifier,
    TriageClassifierError,
    TriageClassifierResult,
    build_triage_classifier,
)
from .triage_queue import (
    DEFAULT_DISPATCH_LIMIT,
    TriageQueue,
    TriageQueueError,
)
from .trigger import (
    ALLOWED_INCOMING_STATUSES,
    DEFAULT_PENDING_BATCH_SIZE,
    DEFAULT_POLL_INTERVAL_SECONDS,
    EVENT_TRIGGER_DICOM_PROCESS_FAILED,
    EVENT_TRIGGER_EXCEPTION,
    EVENT_TRIGGER_IDEMPOTENT_HIT,
    EVENT_TRIGGER_INCOMING_MARKED,
    EVENT_TRIGGER_PROCESSED,
    EVENT_TRIGGER_PROCESSED_DEGRADED,
    EVENT_TRIGGER_STARTED,
    EVENT_TRIGGER_WADO_FAILED,
    EVENT_TRIGGER_WORKFLOW_FAILED,
    STATUS_FAILED_RETRY,
    STATUS_PENDING,
    STATUS_PROCESSED,
    SsePublishCallback,
    TriageTrigger,
    TriageTriggerError,
    build_triage_trigger,
    build_triage_trigger_from_env,
)
from .workflow import (
    DEFAULT_TOOL_NAMES,
    CxrTriageWorkflow,
    ToolRegistry,
    WorkflowError,
    build_workflow,
)

__all__ = [
    # queue
    "TriageQueue",
    "TriageQueueError",
    "DEFAULT_DISPATCH_LIMIT",
    # trigger
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
    # classifier
    "TriageClassifier",
    "TriageClassifierError",
    "TriageClassifierResult",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "POSITIVE_PROB_HEURISTIC",
    "PATHOLOGY_NAMES",
    "build_triage_classifier",
    # workflow
    "CxrTriageWorkflow",
    "ToolRegistry",
    "WorkflowError",
    "DEFAULT_TOOL_NAMES",
    "build_workflow",
    # keyqa
    "CXRKeyQA",
    "KeyQAToolError",
    "ModelUnreachableError",
    "KeyQAResult",
    "DEFAULT_KEY_QA_TEMPLATES",
    "KEY_QA_JSON_SCHEMA",
    "DEFAULT_TIMEOUT_SECONDS",
    "build_cxr_keyqa",
    # image_preview
    "ImagePreview",
    "ImagePreviewError",
    "ImagePreviewResult",
    "IMAGE_PREVIEW_DEFAULT_WINDOW_WIDTH",
    "IMAGE_PREVIEW_DEFAULT_WINDOW_LEVEL",
    "PNG_DATA_URL_PREFIX",
    "PNG_MAGIC",
    "PREVIEW_MAX_BASE64_BYTES",
    "encode_png_to_b64",
    "build_data_url",
    "build_image_preview",
    # finding_draft
    "FindingDraft",
    "FindingDraftError",
    "FindingDraftResult",
    "DRAFT_MARKER",
    "NO_ABNORMALITY_TEMPLATE",
    "POSITIVE_TEMPLATE",
    "DEFAULT_POSITIVE_THRESHOLD",
    "TOP_N_PATHOLOGIES",
    "MAX_PATHOLOGIES_IN_STRUCT",
    "DEFAULT_LOCALIZATION_BY_PATHOLOGY",
    "build_finding_draft",
    # positive_grounding
    "PositiveGrounding",
    "PositiveGroundingError",
    "PositiveGroundingResult",
    "DEFAULT_BBOX_BY_REGION",
    "DEFAULT_FALLBACK_BBOX",
    "MAX_GROUNDING_BOXES",
    "POSITIVE_GROUNDING_PROB_HEURISTIC",
    "POSITIVE_GROUNDING_PATHOLOGY_NAMES",
    "build_positive_grounding",
]
