"""gateway — 单一 ModelGateway（OpenAI-compatible chat completions）。

源产品能力 / 代码证据（spec.md §7 开源对照 + plan.md §5 深度 (a)）：

源仓库 ``github_ref/MedRAX/main.py:113-117`` 把 ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL`` 作为 ``ChatOpenAI(openai_kwargs=...)`` 透传 —— 形态
是「**业务层直接耦合 langchain ChatOpenAI**」。本系统按 spec.md §5 维度 4
「凭据从 env 读、不绑定厂商 SDK」要求，做一层手写的 OpenAI-compatible
HTTP 网关：

  ModelGateway.chat_completion(prompt, study_uid, json_schema, timeout)
    → httpx POST {base_url}/chat/completions + Bearer {api_key}
    → 指数退避 2 次 + 30s timeout
    → 解析 OpenAI 2024-08+ chat.completion 形态
    → 写 tool_calls_<date>.jsonl 一行（model / tokens / latency / status）
    → 失败 → ModelUnreachableError（业务层走 soft-degrade + audit）

业务层（cxr_keyqa / finding_draft / 后续工具）只 import 本包，不直接
接触 httpx / openai / langchain_openai 任何厂商 SDK。

公开入口：

- ``ModelGateway``             单一模型网关（OpenAI-compatible）
- ``ModelGatewayError``        通用异常（构造 / 入参 / 协议错）
- ``ModelUnreachableError``    重试用尽（与 cxr_keyqa 同名，跨模块可捕获）
- ``ModelResponseError``       响应解析失败（JSON 损坏 / schema 校验失败）
- ``ChatRequest`` / ``ChatResponse`` / ``ChatMessage`` pydantic 模型
- ``ResponseFormat`` / ``build_json_schema_response_format``  schema 包装
- ``DEFAULT_TIMEOUT_SECONDS``  / ``DEFAULT_MAX_RETRIES``  默认配置
- ``RETRYABLE_HTTP_STATUSES`` / ``NON_RETRYABLE_HTTP_STATUSES``  重试白名单
- ``build_model_gateway``      工厂

实现取舍：

- **手写协议 vs SDK**：手写 httpx + JSON；保留 service_url / api_key /
  model 完全可控；不锁任何 OpenAI SDK 版本。
- **指数退避无 jitter**：本系统为内部自建网，jitter 不必要；保留 cap 30s
  封顶。
- **4xx 不重试**：业务错误立刻抛 ``ModelResponseError``，避免无效重试。
- **审计不阻断主路径**：``_record_call`` 抛异常时只 warn，不抛。
- **协议对齐 KeyQAProtocol**：``chat_completion`` 签名与
  ``cxr_keyqa.KeyQAProtocol`` 一致；可单测用 ``MockGateway`` 替换。
"""
from __future__ import annotations

from .gateway_schema import (
    RESPONSE_FORMAT_JSON_OBJECT,
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
    MessageRole,
    ResponseFormat,
    build_json_schema_response_format,
)
from .model_gateway import (
    DEGRADATION_BANNER_ZH,
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_CAP_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ModelGateway,
    ModelGatewayError,
    ModelResponseError,
    ModelUnreachableError,
    NON_RETRYABLE_HTTP_STATUSES,
    RETRYABLE_HTTP_STATUSES,
    build_model_gateway,
)

__all__ = [
    # main
    "ModelGateway",
    "ModelGatewayError",
    "ModelUnreachableError",
    "ModelResponseError",
    "build_model_gateway",
    # schema
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "ChatChoice",
    "ChatUsage",
    "ResponseFormat",
    "MessageRole",
    "build_json_schema_response_format",
    "RESPONSE_FORMAT_JSON_OBJECT",
    # constants
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_BASE_SECONDS",
    "DEFAULT_BACKOFF_CAP_SECONDS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEGRADATION_BANNER_ZH",
    "RETRYABLE_HTTP_STATUSES",
    "NON_RETRYABLE_HTTP_STATUSES",
]
