"""model_gateway — 单一 ModelGateway（OpenAI-compatible chat completions）。

设计原则（spec.md §5 维度 4 正式对接层 + 维度 6 审计留痕 + CLAUDE.md
「按需模型网关 / OpenAI-compatible 一种协议」+ plan.md §5 深度 (a) 边界
与异常用例）：

1. **单一入口**：业务层（cxr_keyqa / finding_draft / 后续工具）只调
   ``ModelGateway.chat_completion``，不直接 import 任何厂商 SDK。
2. **OpenAI-compatible 协议**：出站 ``POST {base_url}/chat/completions``，
   Bearer token 鉴权；入站按 OpenAI 2024-08+ ``chat.completion`` 形态解析。
3. **结构化输出**：``response_format=json_schema`` 强制 schema 约束（与
   ``cxr_keyqa.KEY_QA_JSON_SCHEMA`` 同形）。
4. **配置化**（不硬编码）：``base_url`` / ``api_key`` / ``model`` /
   ``timeout_seconds`` / ``max_retries`` 全部从 ``CxrTriageConfig`` 或构造
   参数注入；任何代码路径都不得写死凭据。
5. **超时 + 指数退避 + 限次重试**：
   - 单次请求 ``timeout_seconds``（默认 30s）；
   - 退避 ``retry_count`` 次（默认 2 次重试 → 总尝试 3 次）；
   - 退避曲线 ``1s, 2s, 4s, 8s ...``（最多 30s 封顶）；
   - 重试仅对网络错误 / 5xx / 408 / 429 触发；4xx 业务错误不重试。
6. **降级语义**：连续失败 → 抛 ``ModelUnreachableError``；调用方
   （``CXRKeyQA`` 等）走 soft-degrade + audit_log。
7. **可审计调用元数据**：每次调用落 ``tool_calls_<date>.jsonl`` 一行
   （``model`` / ``prompt_tokens`` / ``completion_tokens`` / ``latency_ms`` /
   ``status`` / ``study_uid`` / ``tool`` / ``extra``），便于飞检与 AI
   治理审查。
8. **不耦合业务侧**：本类不 import ``cxr_triage.engine`` / ``state`` /
   ``writers.audit``；通过构造期注入 ``audit_logger`` / ``config`` 实现
   依赖反转。
9. **协议而非耦合**：与 ``CXRKeyQA.KeyQAProtocol`` 对齐（``chat_completion``
   签名一致），可单测用 ``MockGateway`` 替代。

公开入口：

- ``ModelGateway``        单一模型网关
- ``ModelGatewayError``   通用异常（构造 / 入参错）
- ``ModelUnreachableError`` 重试用尽后仍失败（与 cxr_keyqa 同名异常）
- ``ModelResponseError``  响应解析失败（结构化失败 / JSON 损坏 / schema 校验失败）
- ``DEFAULT_TIMEOUT_SECONDS``  30s
- ``DEFAULT_MAX_RETRIES``      2（总尝试 3 次）
- ``DEFAULT_BACKOFF_BASE_SECONDS``  1.0
- ``DEFAULT_BACKOFF_CAP_SECONDS``   30.0
- ``RETRYABLE_HTTP_STATUSES``  408 / 429 / 5xx → 重试
- ``NON_RETRYABLE_HTTP_STATUSES``  4xx 业务错误（除 408/429）→ 不重试
- ``build_model_gateway``  工厂函数

实现取舍：

- **协议层 vs SDK 层**：本类**不**直接依赖 ``openai`` SDK；只走 ``httpx``
  + 手写 OpenAI 协议 JSON。理由：spec §5 维度 4 要求「凭据从 env 读、不
  绑定厂商 SDK」；手写协议能保证 service_url / api_key / model 完全可控。
- **指数退避 + 抖动**：避免重试风暴；抖动在 ``_compute_backoff_seconds``
  内简单 ``base * 2**attempt``（无 jitter；本系统为内部自建网，抖动不
  必要，但保留 cap 封顶 30s）。
- **4xx 不重试**：业务错误（如 ``400 bad request``、``401 unauthorized``、
  ``403 forbidden``）立刻抛错，避免无效重试。
- **降级 banner 由调用方负责**：本类只抛 ``ModelUnreachableError``；是否
  soft-degrade、是否写 audit 留调用方决定（保持网关层职责清晰）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import httpx

from ..config import CxrTriageConfig
from .gateway_schema import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ResponseFormat,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_BACKOFF_BASE_SECONDS: float = 1.0
DEFAULT_BACKOFF_CAP_SECONDS: float = 30.0

# 默认探测超时（is_available 轻量探活用，避免 30s 长尾）
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 5.0

# 重试的 HTTP 状态码（网络层 / 临时不可用 / 限流）
RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

# 4xx 业务错误（除 408/429）→ 不重试
NON_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 410, 415, 422})

# 降级时给医师看的固定 banner（spec.md §10 安全边界 + Task 20 强制）
DEGRADATION_BANNER_ZH: str = "模型不可达，仅出概率与定位，需医师人工复核"


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class ModelGatewayError(RuntimeError):
    """``ModelGateway`` 通用异常（构造 / 入参 / 协议错）。"""


class ModelUnreachableError(RuntimeError):
    """重试用尽后仍无法访问（连接拒绝 / 超时 / 5xx）。

    与 ``cxr_triage.engine.cxr_keyqa.ModelUnreachableError`` 同名异类；调用方
    可 ``isinstance`` 检测（两者无继承关系——跨模块用 ``isinstance`` 容易脆，
    业务层捕获时建议 ``except ModelUnreachableError`` 时既可捕本类也可捕
    ``cxr_keyqa.ModelUnreachableError``，因 ``RuntimeError`` 同根）。
    """


class ModelResponseError(RuntimeError):
    """模型返回的响应无法解析（JSON 损坏 / schema 校验失败 / 缺字段）。"""


# ---------------------------------------------------------------------------
# 审计 logger 协议（避免反向依赖 writers.audit）
# ---------------------------------------------------------------------------


class _AuditLoggerLike:
    """``ModelGateway`` 期望 ``audit_logger`` 实现的最小协议（task_call_record）。

    不强制 ``AuditLoggerProtocol`` 形式继承；只要对象有同名 async 方法即可。
    """

    async def tool_call_record(  # pragma: no cover - 协议占位
        self,
        tool: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status: str,
        study_uid: str = "",
        ts: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Any:
        ...


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class ModelGateway:
    """OpenAI-compatible 单一模型网关。

    Parameters
    ----------
    base_url:
        ``POST {base_url}/chat/completions`` 端点 base（不带尾部 / ）；
        默认从 ``CxrTriageConfig.openai_base_url`` 读。
    api_key:
        Bearer token；默认从 ``CxrTriageConfig.openai_api_key`` 读。
    model:
        模型名；默认从 ``CxrTriageConfig.openai_model`` 读。
    timeout_seconds:
        单次 HTTP 超时（覆盖 ``httpx.AsyncClient.timeout``）。
    max_retries:
        重试次数（不含首次尝试）；默认 ``DEFAULT_MAX_RETRIES=2`` → 总
        尝试 3 次。
    backoff_base_seconds / backoff_cap_seconds:
        指数退避基数与封顶；退避序列 ``base * 2**attempt``，封顶 ``cap``。
    audit_logger:
        期望实现 ``_AuditLoggerLike``；缺省时跳过 ``tool_call_record`` 落
        盘（便于单测与本地调试）。
    client_factory:
        可选 ``Callable[[float], httpx.AsyncClient]`` 工厂；测试时用
        ``respx`` 替换。默认直接构造 ``httpx.AsyncClient(timeout=...)``。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS,
        probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
        audit_logger: Optional[_AuditLoggerLike] = None,
        client_factory: Optional[
            Callable[[float], httpx.AsyncClient]
        ] = None,
    ) -> None:
        # 解析配置（base_url/api_key/model 缺省时从 CxrTriageConfig 读）
        if base_url is None or api_key is None or model is None:
            cfg = CxrTriageConfig()
            base_url = base_url or cfg.openai_base_url
            api_key = api_key or cfg.openai_api_key
            model = model or cfg.openai_model

        if not isinstance(base_url, str) or not base_url.strip():
            raise ModelGatewayError("base_url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelGatewayError(
                "api_key must be a non-empty string "
                "(从环境变量 OPENAI_API_KEY 注入)"
            )
        if not isinstance(model, str) or not model.strip():
            raise ModelGatewayError("model must be a non-empty string")
        try:
            ts = float(timeout_seconds)
        except (TypeError, ValueError):
            raise ModelGatewayError(
                f"timeout_seconds must be numeric, got {timeout_seconds!r}"
            )
        if ts <= 0 or ts > 600:
            raise ModelGatewayError(
                f"timeout_seconds must be in (0, 600], got {timeout_seconds}"
            )
        if not isinstance(max_retries, int) or max_retries < 0 or max_retries > 10:
            raise ModelGatewayError(
                f"max_retries must be int in [0, 10], got {max_retries!r}"
            )
        try:
            pts = float(probe_timeout_seconds)
        except (TypeError, ValueError):
            raise ModelGatewayError(
                f"probe_timeout_seconds must be numeric, got {probe_timeout_seconds!r}"
            )
        if pts <= 0 or pts > 60:
            raise ModelGatewayError(
                f"probe_timeout_seconds must be in (0, 60], got {probe_timeout_seconds}"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.backoff_cap_seconds = float(backoff_cap_seconds)
        self.probe_timeout_seconds = float(probe_timeout_seconds)
        self.audit_logger = audit_logger
        self._client_factory = client_factory or self._default_client_factory
        self._client: Optional[httpx.AsyncClient] = None

    # ---- 客户端生命周期 ----

    @staticmethod
    def _default_client_factory(timeout: float) -> httpx.AsyncClient:
        """默认 httpx 异步客户端工厂。"""
        return httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "cxr-triage/0.1.0 (+model-gateway)"},
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._client_factory(self.timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        """关闭内部 httpx 客户端（幂等）。"""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # ---- 降级路径（Task 20）----

    async def is_available(self) -> bool:
        """轻量探测：模型服务是否可达（不抛，失败返回 False）。

        通过 ``GET {base_url}/models`` 做 OpenAI 标准的健康检查；
        任意外部异常（超时 / 网络 / 5xx / 4xx）一律返回 False，便于
        业务层（keyqa / finding_draft）提前判断是否走降级路径。

        与 ``chat_completion`` 不同：

        - **不重试**（探测只关心当下可达性）
        - **不审计落盘**（探测不是真实业务调用，避免 JSONL 被噪音淹没）
        - **不抛异常**（调用方无须 ``try/except``）

        Returns
        -------
        bool
            True → 服务端 2xx；False → 任何非 2xx / 超时 / 网络错 / 协议错
        """
        client = await self._get_client()
        url = f"{self.base_url}/models"
        try:
            response = await client.get(url, timeout=self.probe_timeout_seconds)
        except httpx.HTTPError:
            return False
        except Exception:  # noqa: BLE001 —— 探测永不抛
            return False
        return 200 <= response.status_code < 300

    async def chat_completion_with_degradation(
        self,
        prompt: str,
        *,
        study_uid: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        tool: str = "",
    ) -> Dict[str, Any]:
        """降级安全包装：失败时**不抛**，返回带 ``degraded=True`` 的结构。

        与 ``chat_completion`` 的区别：

        - **永不抛异常**（ModelUnreachableError / ModelResponseError /
          ModelGatewayError / TimeoutError 一律被捕获）
        - **成功路径**：返回 ``{content, raw, model}``（与 ``chat_completion``
          同形），并 ``degraded=False``
        - **失败路径**：返回 ``{content: None, raw: {}, model: <gateway.model>,
          banner: DEGRADATION_BANNER_ZH, degraded: True, error_class,
          error_message, attempts?}``

        调用方（``CXRKeyQA`` / ``FindingDraft``）只需 ``if result.get("degraded")``
        即可走确定性降级，无需 ``try/except``。``tool_calls_<date>.jsonl``
        仍由内层 ``chat_completion`` 写 fail / degraded，不重复落盘。

        Parameters
        ----------
        prompt / study_uid / json_schema / timeout_seconds / tool:
            与 ``chat_completion`` 同语义（透传到内部 chat_completion）

        Returns
        -------
        dict
            统一结构，``degraded`` 字段是关键布尔
        """
        try:
            result = await self.chat_completion(
                prompt,
                study_uid=study_uid,
                json_schema=json_schema,
                timeout_seconds=timeout_seconds,
                tool=tool,
            )
            # 成功路径：附加 degraded=False 标记，便于调用方统一处理
            result["degraded"] = False
            result["banner"] = ""
            return result
        except ModelUnreachableError as exc:
            return {
                "content": None,
                "raw": {},
                "model": self.model,
                "degraded": True,
                "banner": DEGRADATION_BANNER_ZH,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "attempts": self.max_retries + 1,
            }
        except ModelResponseError as exc:
            return {
                "content": None,
                "raw": {},
                "model": self.model,
                "degraded": True,
                "banner": DEGRADATION_BANNER_ZH,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "attempts": 1,
            }
        except ModelGatewayError as exc:
            return {
                "content": None,
                "raw": {},
                "model": self.model,
                "degraded": True,
                "banner": DEGRADATION_BANNER_ZH,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "attempts": 0,
            }
        except (asyncio.TimeoutError, httpx.HTTPError) as exc:
            # 网络 / 超时层异常（chat_completion 一般已转 ModelUnreachableError，
            # 但保险起见仍捕获原始异常）
            return {
                "content": None,
                "raw": {},
                "model": self.model,
                "degraded": True,
                "banner": DEGRADATION_BANNER_ZH,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "attempts": 0,
            }
        except Exception as exc:  # noqa: BLE001 —— 防御性兜底，绝不外抛
            return {
                "content": None,
                "raw": {},
                "model": self.model,
                "degraded": True,
                "banner": DEGRADATION_BANNER_ZH,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "attempts": 0,
            }

    async def __aenter__(self) -> "ModelGateway":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ---- 公开契约（KeyQAProtocol 对齐）----

    async def chat_completion(
        self,
        prompt: str,
        *,
        study_uid: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        tool: str = "",
    ) -> Dict[str, Any]:
        """单次 chat completion；返回 ``{content, raw, model}``。

        Parameters
        ----------
        prompt:
            用户侧 prompt（拼接成单条 ``user`` 消息）。
        study_uid:
            业务侧 DICOM UID；写 ``tool_calls_*.jsonl`` 与 audit 用。
        json_schema:
            可选 OpenAI-compatible ``response_format=json_schema`` envelope；
            与 ``cxr_keyqa.KEY_QA_JSON_SCHEMA`` 同形。
        timeout_seconds:
            覆盖构造时 ``timeout_seconds``（用于不同模型不同超时）。
        tool:
            业务侧工具名（keyqa / finding_draft 等）；audit 用。

        Returns
        -------
        dict
            形态::

                {
                    "content": <str or list[dict]>,
                    "raw": <OpenAI 风格完整响应 dict>,
                    "model": <model_id>,
                }

        Raises
        ------
        ModelGatewayError
            构造 / 入参 / 协议错误
        ModelUnreachableError
            重试用尽后仍失败（连接拒绝 / 超时 / 5xx）
        ModelResponseError
            响应 JSON 损坏或结构化失败
        """
        # ---- 入参校验 ----
        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelGatewayError("prompt must be a non-empty string")
        if not isinstance(study_uid, str):
            raise ModelGatewayError("study_uid must be a string")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise ModelGatewayError(
                f"json_schema must be dict or None, got {type(json_schema).__name__}"
            )
        if timeout_seconds is not None:
            try:
                override = float(timeout_seconds)
            except (TypeError, ValueError):
                raise ModelGatewayError(
                    f"timeout_seconds override must be numeric, got {timeout_seconds!r}"
                )
            if override <= 0 or override > 600:
                raise ModelGatewayError(
                    f"timeout_seconds override must be in (0, 600], got {override}"
                )

        # ---- 构造请求 ----
        response_format: Optional[ResponseFormat] = None
        if json_schema is not None:
            # 业务侧传的是 OpenAI envelope（type=json_schema + json_schema=...）
            # 也可能只传了 schema 字典 —— 兼容两种
            if "type" in json_schema and json_schema["type"] == "json_schema":
                envelope = json_schema["json_schema"]
            else:
                envelope = json_schema
            schema_body = envelope.get("schema", {}) if isinstance(envelope, dict) else {}
            response_format = ResponseFormat(
                type="json_schema",
                json_schema={
                    "name": envelope.get("name", "model_response"),
                    "strict": envelope.get("strict", True),
                    "schema": schema_body,
                },
            )

        req = ChatRequest(
            model=self.model,
            messages=[ChatMessage(role="user", content=prompt)],
            response_format=response_format,
            study_uid=study_uid,
            tool=tool,
        )
        body = req.to_openai_payload()
        effective_timeout = float(timeout_seconds) if timeout_seconds is not None else self.timeout_seconds

        # ---- 重试循环 ----
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[BaseException] = None
        last_status: Optional[int] = None
        total_start = time.monotonic()

        for attempt in range(self.max_retries + 1):
            attempt_start = time.monotonic()
            try:
                response, parsed = await self._send_once(url, body, effective_timeout)
                elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
                # ---- 解析成功 → audit + 返回 ----
                await self._record_call(
                    tool=tool or "model_gateway",
                    model=parsed.model or self.model,
                    prompt_tokens=parsed.usage.prompt_tokens,
                    completion_tokens=parsed.usage.completion_tokens,
                    latency_ms=elapsed_ms,
                    status="ok",
                    study_uid=study_uid,
                    extra={"http_status": response.status_code, "attempt": attempt},
                )
                return {
                    "content": self._coerce_content(parsed),
                    "raw": response.json() if self._is_json_response(response) else {},
                    "model": parsed.model or self.model,
                }
            except _RetryableHttpStatus as exc:
                last_exc = exc
                last_status = exc.status_code
                elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
                logger.warning(
                    "model_gateway retryable http status %s (attempt %s/%s) study=%s",
                    exc.status_code, attempt + 1, self.max_retries + 1, study_uid,
                )
                await self._record_call(
                    tool=tool or "model_gateway",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=elapsed_ms,
                    status="fail",
                    study_uid=study_uid,
                    extra={
                        "http_status": exc.status_code,
                        "attempt": attempt,
                        "error_class": "retryable_http_status",
                    },
                )
            except _NonRetryableHttpStatus as exc:
                # 4xx 业务错误 → 立即抛，不重试
                elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
                await self._record_call(
                    tool=tool or "model_gateway",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=elapsed_ms,
                    status="fail",
                    study_uid=study_uid,
                    extra={
                        "http_status": exc.status_code,
                        "error_class": "non_retryable_http_status",
                        "body_excerpt": exc.body_excerpt,
                    },
                )
                raise ModelResponseError(
                    f"non-retryable HTTP {exc.status_code} from {url}: {exc.body_excerpt}"
                ) from exc
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
                logger.warning(
                    "model_gateway network error %s (attempt %s/%s) study=%s",
                    type(exc).__name__, attempt + 1, self.max_retries + 1, study_uid,
                )
                await self._record_call(
                    tool=tool or "model_gateway",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=elapsed_ms,
                    status="fail",
                    study_uid=study_uid,
                    extra={
                        "error_class": type(exc).__name__,
                        "attempt": attempt,
                    },
                )
            except (ModelResponseError, ValueError) as exc:
                # 响应 JSON 损坏 / 缺字段 → 不重试（重试不会让 schema 变好）
                elapsed_ms = int((time.monotonic() - attempt_start) * 1000)
                await self._record_call(
                    tool=tool or "model_gateway",
                    model=self.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=elapsed_ms,
                    status="fail",
                    study_uid=study_uid,
                    extra={
                        "error_class": type(exc).__name__,
                        "attempt": attempt,
                        "error": str(exc)[:200],
                    },
                )
                raise

            # 是否要再退避
            if attempt < self.max_retries:
                backoff = self._compute_backoff_seconds(attempt)
                logger.debug("model_gateway backoff %.2fs before retry", backoff)
                await asyncio.sleep(backoff)

        # 重试用尽
        total_ms = int((time.monotonic() - total_start) * 1000)
        logger.error(
            "model_gateway exhausted %s attempts for study=%s last_exc=%s last_status=%s",
            self.max_retries + 1, study_uid, last_exc, last_status,
        )
        await self._record_call(
            tool=tool or "model_gateway",
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=total_ms,
            status="degraded",
            study_uid=study_uid,
            extra={
                "error_class": type(last_exc).__name__ if last_exc else "Unknown",
                "attempts": self.max_retries + 1,
                "last_status": last_status,
            },
        )
        raise ModelUnreachableError(
            f"model_gateway exhausted {self.max_retries + 1} attempts "
            f"for study={study_uid!r}: {type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}"
        )

    # ---- 内部：单次 HTTP ----

    async def _send_once(
        self,
        url: str,
        body: Dict[str, Any],
        timeout: float,
    ) -> tuple[httpx.Response, ChatResponse]:
        """单次出站 HTTP；返回 ``(原始 Response, 解析后的 ChatResponse)``。"""
        client = await self._get_client()
        try:
            response = await client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException:
            raise
        except httpx.HTTPError as exc:
            # 把其它 HTTP 错误（RemoteProtocolError / NetworkError 等）按
            # 网络层错处理（外层会重试）
            raise

        # 状态码分支
        if response.status_code in RETRYABLE_HTTP_STATUSES:
            raise _RetryableHttpStatus(
                response.status_code, body_excerpt=_excerpt(response)
            )
        if response.status_code in NON_RETRYABLE_HTTP_STATUSES or (
            response.status_code >= 400 and response.status_code < 500
        ):
            raise _NonRetryableHttpStatus(
                response.status_code, body_excerpt=_excerpt(response)
            )
        if response.status_code >= 500:
            # 未列入 RETRYABLE 的 5xx 同样按可重试处理
            raise _RetryableHttpStatus(
                response.status_code, body_excerpt=_excerpt(response)
            )

        # 2xx 解析
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelResponseError(
                f"model_gateway response JSON parse error: {exc}"
            ) from exc

        try:
            parsed = ChatResponse.model_validate(data)
        except Exception as exc:  # noqa: BLE001 —— pydantic ValidationError
            raise ModelResponseError(
                f"model_gateway response validation failed: {exc}"
            ) from exc

        # 检查 parsed 是否真有 choices
        if not parsed.choices:
            raise ModelResponseError(
                f"model_gateway response has empty choices: {data}"
            )

        return response, parsed

    @staticmethod
    def _is_json_response(response: httpx.Response) -> bool:
        ct = (response.headers.get("content-type") or "").lower()
        return "application/json" in ct

    @staticmethod
    def _coerce_content(parsed: ChatResponse) -> Union[str, Any]:
        """从 ``ChatResponse`` 抽 ``content``；尽量按 json_schema 解析为对象。"""
        raw = parsed.get_content()
        if not raw:
            return raw
        # 如果 response_format 是 json_schema，尝试解析
        stripped = raw.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return raw

    # ---- 内部：退避 + 审计 ----

    def _compute_backoff_seconds(self, attempt: int) -> float:
        """指数退避：``base * 2**attempt``，封顶 ``cap``。"""
        backoff = self.backoff_base_seconds * (2 ** attempt)
        return float(min(backoff, self.backoff_cap_seconds))

    async def _record_call(
        self,
        *,
        tool: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status: str,
        study_uid: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写 ``tool_calls_<date>.jsonl`` 一行（audit_logger 缺省时跳过）。"""
        if self.audit_logger is None:
            return
        try:
            await self.audit_logger.tool_call_record(
                tool=tool,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status=status,
                study_uid=study_uid,
                extra=extra,
            )
        except Exception as exc:  # noqa: BLE001 —— 审计失败不阻断主路径
            logger.warning(
                "model_gateway audit_log tool_call_record failed (%s): %s",
                type(exc).__name__, exc,
            )


# ---------------------------------------------------------------------------
# 内部异常（控制 _send_once 与 chat_completion 之间的信号）
# ---------------------------------------------------------------------------


class _RetryableHttpStatus(ModelGatewayError):
    def __init__(self, status_code: int, body_excerpt: str = ""):
        super().__init__(f"retryable HTTP {status_code}: {body_excerpt}")
        self.status_code = int(status_code)
        self.body_excerpt = body_excerpt


class _NonRetryableHttpStatus(ModelGatewayError):
    def __init__(self, status_code: int, body_excerpt: str = ""):
        super().__init__(f"non-retryable HTTP {status_code}: {body_excerpt}")
        self.status_code = int(status_code)
        self.body_excerpt = body_excerpt


def _excerpt(response: httpx.Response, max_chars: int = 200) -> str:
    """取 response body 截断（前 max_chars 字符），便于日志/audit。"""
    try:
        text = response.text
    except Exception:  # noqa: BLE001
        return "<body_unreadable>"
    return text[:max_chars]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_model_gateway(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    audit_logger: Optional[_AuditLoggerLike] = None,
    client_factory: Optional[Callable[[float], httpx.AsyncClient]] = None,
) -> ModelGateway:
    """工厂：构造 ``ModelGateway`` 一行完成。"""
    return ModelGateway(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        backoff_cap_seconds=backoff_cap_seconds,
        probe_timeout_seconds=probe_timeout_seconds,
        audit_logger=audit_logger,
        client_factory=client_factory,
    )


__all__ = [
    "ModelGateway",
    "ModelGatewayError",
    "ModelUnreachableError",
    "ModelResponseError",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_BASE_SECONDS",
    "DEFAULT_BACKOFF_CAP_SECONDS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEGRADATION_BANNER_ZH",
    "RETRYABLE_HTTP_STATUSES",
    "NON_RETRYABLE_HTTP_STATUSES",
    "build_model_gateway",
]
