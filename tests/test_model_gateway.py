"""tests/test_model_gateway.py — Task 19 ModelGateway（OpenAI-compatible）单测。

目标（spec.md §5 维度 4 正式对接层 + 维度 6 审计 + plan.md §5 深度 (a) 边界
与异常用例 + Task 19 + 20 + 21）：

1. **构造期校验**：base_url / api_key / model 缺省时从 CxrTriageConfig 读；
   空字符串 / 非字符串 / 越界 timeout / 越界 max_retries 一律抛 ModelGatewayError
2. **入参校验**：prompt 空 / 非字符串 → ModelGatewayError；study_uid 非字符串
   拒绝；json_schema 非 dict 拒绝；timeout_seconds 越界拒绝
3. **happy path**：mock 服务端 200 + 合法 OpenAI 响应 → 返回 ``{content, raw,
   model}``；结构化输出 json_schema 自动 parse 为 dict
4. **Bearer 鉴权**：Authorization header 含 Bearer token
5. **结构化输出**：传 ``json_schema=KEY_QA_JSON_SCHEMA`` → 出站 body 含
   ``response_format.type=json_schema + json_schema.name=key_qa_response``
6. **超时（retryable）**：服务端 504 → 重试 2 次后抛 ModelUnreachableError +
   audit 3 行 fail + 1 行 degraded
7. **可重试 HTTP 状态码**：408 / 429 / 500 / 502 / 503 / 504 全部触发重试
8. **不可重试 HTTP 状态码**：400 / 401 / 403 / 404 → 立即抛 ModelResponseError，
   不重试，audit 1 行 fail
9. **响应 JSON 损坏**：返回非 JSON 内容 → ModelResponseError（不重试）
10. **响应缺 choices**：空 choices → ModelResponseError（不重试）
11. **网络错误**：httpx.ConnectError → 重试 + ModelUnreachableError
12. **指数退避**：attempt=0 → 1s；attempt=1 → 2s；attempt=2 → 4s；cap 30s
13. **重试成功**：服务端 503 → 503 → 200 → 返回 ok；audit 2 行 fail + 1 行 ok
14. **审计落盘**：用真 AuditLogger（tmpdir）→ tool_calls_<date>.jsonl 写入
    完整 8 字段 + status ok/fail/degraded
15. **audit 失败不阻断**：audit_logger 抛异常 → chat_completion 仍正常返
16. **close / aclose 幂等**：重复 aclose 不抛
17. **client_factory 注入**：自定义 client_factory 替换 httpx（respx）
18. **默认从 CxrTriageConfig 读 base_url/api_key/model**：env 缺 OPENAI_BASE_URL
    → ModelGatewayError
19. **不耦合业务层**：ModelGateway 不 import cxr_triage.engine
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest
import respx

from cxr_triage.engine import KEY_QA_JSON_SCHEMA
from cxr_triage.gateway import (
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_CAP_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    ModelGateway,
    ModelGatewayError,
    ModelResponseError,
    ModelUnreachableError,
    NON_RETRYABLE_HTTP_STATUSES,
    RETRYABLE_HTTP_STATUSES,
    build_model_gateway,
)
from cxr_triage.gateway.gateway_schema import (
    build_json_schema_response_format,
    ChatRequest,
    ChatResponse,
    ResponseFormat,
)
from cxr_triage.writers.audit import AuditLogger, build_audit_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENAI_OK_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-test-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "medrax-llava-med-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    [
                        {"q": "心影", "a": "心影大小未见明显异常。"},
                        {"q": "纵隔", "a": "纵隔居中。"},
                        {"q": "胸膜", "a": "两侧胸膜未见异常。"},
                        {"q": "肺野", "a": "两肺野透亮度正常。"},
                    ],
                    ensure_ascii=False,
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 256, "completion_tokens": 128, "total_tokens": 384},
}


def _make_client_factory() -> Any:
    """helper for tests; respx already intercepts via base_url."""
    return ModelGateway._default_client_factory


@pytest.fixture
def openai_mock():
    """respx 拦截 http://test-model-gateway.local:8000/v1/chat/completions。"""
    with respx.mock(
        base_url="http://test-model-gateway.local:8000/v1",
        assert_all_called=False,
    ) as mock:
        yield mock


@pytest.fixture
def env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """提供测试用 env 凭据 + CXR_TRIAGE_ALLOW_PLACEHOLDER=1 放行占位。"""
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test-model-gateway.local:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "medrax-llava-med-7b")
    # 同时满足 CxrTriageConfig 其它必填项（与 ModelGateway 无关，仅让 config 加载不抛）
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "placeholder-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://healthcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "placeholder-healthcheck-token")


@pytest.fixture
def env_credentials_alt(monkeypatch: pytest.MonkeyPatch) -> None:
    """提供另一组 env 凭据（用于「默认从 config 读」测试）。"""
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-xyz")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "placeholder-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://healthcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "placeholder-healthcheck-token")


# ---------------------------------------------------------------------------
# 1. 构造期校验
# ---------------------------------------------------------------------------


def test_construction_with_explicit_params(env_credentials: None) -> None:
    """显式 base_url / api_key / model → 构造成功。"""
    gw = ModelGateway(
        base_url="http://test-model-gateway.local:8000/v1",
        api_key="test-key",
        model="custom-model",
    )
    assert gw.base_url == "http://test-model-gateway.local:8000/v1"
    assert gw.api_key == "test-key"
    assert gw.model == "custom-model"
    assert gw.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert gw.max_retries == DEFAULT_MAX_RETRIES


def test_construction_defaults_from_config(env_credentials: None) -> None:
    """缺省 base_url / api_key / model → 从 CxrTriageConfig 读。"""
    gw = ModelGateway()
    assert gw.base_url == "http://test-model-gateway.local:8000/v1"
    assert gw.api_key == "test-secret-key"
    assert gw.model == "medrax-llava-med-7b"


def test_construction_with_alt_env(env_credentials_alt: None) -> None:
    gw = ModelGateway()
    assert gw.base_url == "http://env-host:9999/v1"
    assert gw.api_key == "env-key-xyz"
    assert gw.model == "env-model"


def test_construction_empty_base_url_rejected() -> None:
    with pytest.raises(ModelGatewayError, match="base_url"):
        ModelGateway(base_url="", api_key="x", model="m")


def test_construction_empty_api_key_rejected() -> None:
    with pytest.raises(ModelGatewayError, match="api_key"):
        ModelGateway(base_url="http://x", api_key="", model="m")


def test_construction_empty_model_rejected() -> None:
    with pytest.raises(ModelGatewayError, match="model"):
        ModelGateway(base_url="http://x", api_key="k", model="")


def test_construction_invalid_timeout_rejected(env_credentials: None) -> None:
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        ModelGateway(timeout_seconds=0)
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        ModelGateway(timeout_seconds=-5)
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        ModelGateway(timeout_seconds=601)
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        ModelGateway(timeout_seconds="not-a-number")


def test_construction_invalid_max_retries_rejected(env_credentials: None) -> None:
    with pytest.raises(ModelGatewayError, match="max_retries"):
        ModelGateway(max_retries=-1)
    with pytest.raises(ModelGatewayError, match="max_retries"):
        ModelGateway(max_retries=11)
    with pytest.raises(ModelGatewayError, match="max_retries"):
        ModelGateway(max_retries=1.5)  # type: ignore[arg-type]


def test_construction_base_url_trailing_slash_stripped(env_credentials: None) -> None:
    gw = ModelGateway(
        base_url="http://test-model-gateway.local:8000/v1/",
        api_key="k",
        model="m",
    )
    assert gw.base_url == "http://test-model-gateway.local:8000/v1"


# ---------------------------------------------------------------------------
# 2. 入参校验
# ---------------------------------------------------------------------------


def test_chat_completion_empty_prompt_rejected(env_credentials: None) -> None:
    gw = ModelGateway()
    with pytest.raises(ModelGatewayError, match="prompt"):
        asyncio.run(gw.chat_completion(""))


def test_chat_completion_non_string_study_uid_rejected(env_credentials: None) -> None:
    gw = ModelGateway()
    with pytest.raises(ModelGatewayError, match="study_uid"):
        asyncio.run(gw.chat_completion("hi", study_uid=123))  # type: ignore[arg-type]


def test_chat_completion_invalid_json_schema_rejected(env_credentials: None) -> None:
    gw = ModelGateway()
    with pytest.raises(ModelGatewayError, match="json_schema"):
        asyncio.run(gw.chat_completion("hi", json_schema="not-a-dict"))  # type: ignore[arg-type]


def test_chat_completion_invalid_timeout_override_rejected(env_credentials: None) -> None:
    gw = ModelGateway()
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        asyncio.run(gw.chat_completion("hi", timeout_seconds=0))
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        asyncio.run(gw.chat_completion("hi", timeout_seconds=-1))
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        asyncio.run(gw.chat_completion("hi", timeout_seconds=601))
    with pytest.raises(ModelGatewayError, match="timeout_seconds"):
        asyncio.run(gw.chat_completion("hi", timeout_seconds="abc"))


# ---------------------------------------------------------------------------
# 3. happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_content_raw_model(openai_mock: Any, env_credentials: None) -> None:
    route = openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )
    gw = ModelGateway()
    result = asyncio.run(gw.chat_completion("Hello model", study_uid="1.2.826.0.1.3680043.8.498.X.001"))
    assert isinstance(result, dict)
    assert "content" in result
    assert "raw" in result
    assert "model" in result
    # json_schema 解析为 list[dict]
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 4
    assert result["content"][0]["q"] == "心影"
    assert result["model"] == "medrax-llava-med-7b"
    assert route.called


def test_bearer_token_in_authorization_header(openai_mock: Any, env_credentials: None) -> None:
    """出站请求必须带 ``Authorization: Bearer <key>``。"""
    route = openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )
    gw = ModelGateway()
    asyncio.run(gw.chat_completion("hi"))
    request = route.calls.last.request
    auth = request.headers.get("Authorization")
    assert auth == "Bearer test-secret-key"


def test_user_message_serialized(openai_mock: Any, env_credentials: None) -> None:
    """prompt 拼成单条 ``user`` 消息。"""
    route = openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )
    gw = ModelGateway()
    asyncio.run(gw.chat_completion("请按 4 模板回答"))
    body = json.loads(route.calls.last.request.content)
    assert "model" in body
    assert "messages" in body
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "请按 4 模板回答"
    assert body["model"] == "medrax-llava-med-7b"


# ---------------------------------------------------------------------------
# 5. 结构化输出（json_schema）
# ---------------------------------------------------------------------------


def test_json_schema_envelope_in_request(openai_mock: Any, env_credentials: None) -> None:
    """传 ``json_schema=KEY_QA_JSON_SCHEMA`` → 出站 body 含 ``response_format``。"""
    route = openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )
    gw = ModelGateway()
    asyncio.run(
        gw.chat_completion(
            "hi",
            json_schema=KEY_QA_JSON_SCHEMA,
            tool="keyqa",
        )
    )
    body = json.loads(route.calls.last.request.content)
    assert "response_format" in body
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "key_qa_response"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["type"] == "array"


def test_json_schema_parsed_content_as_object(openai_mock: Any, env_credentials: None) -> None:
    """response 内容是合法 JSON → 自动 parse 为 dict。"""
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"summary": "右下肺野见少量气胸征象。"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )
    )
    gw = ModelGateway()
    result = asyncio.run(gw.chat_completion("hi"))
    assert isinstance(result["content"], dict)
    assert "summary" in result["content"]


# ---------------------------------------------------------------------------
# 6. 超时 / 5xx 重试 → ModelUnreachableError
# ---------------------------------------------------------------------------


def test_retryable_5xx_exhausts_retries(openai_mock: Any, env_credentials: None) -> None:
    """服务端 504 持续 → 3 次尝试后抛 ModelUnreachableError。"""
    openai_mock.post("/chat/completions").mock(return_value=httpx.Response(504, text="bad gateway"))
    gw = ModelGateway(max_retries=2, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi", study_uid="study-1"))


def test_retryable_5xx_audit_log_status(openai_mock: Any, env_credentials: None, tmp_path) -> None:
    """5xx 重试用尽 → audit 2 行 fail + 1 行 degraded。"""
    openai_mock.post("/chat/completions").mock(return_value=httpx.Response(503, text="unavailable"))
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = ModelGateway(
        max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001,
        audit_logger=audit,
    )
    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi", study_uid="study-2", tool="keyqa"))
    # 1 次重试 = 2 次尝试 → 2 行 fail + 1 行 degraded
    files = audit.list_tool_call_files()
    assert files
    records = AuditLogger.read_tool_call_records(files[0])
    statuses = [r["status"] for r in records]
    assert statuses.count("fail") == 2
    assert statuses.count("degraded") == 1
    # fail 行带 http_status
    fails = [r for r in records if r["status"] == "fail"]
    for f in fails:
        assert f["extra"]["http_status"] == 503
        assert f["extra"]["error_class"] == "retryable_http_status"
    # degraded 行带 attempts
    deg = [r for r in records if r["status"] == "degraded"][0]
    assert deg["extra"]["attempts"] == 2
    assert deg["tool"] == "keyqa"
    assert deg["study_uid"] == "study-2"


def test_retry_eventually_succeeds(openai_mock: Any, env_credentials: None, tmp_path) -> None:
    """503 → 503 → 200：第 3 次成功。"""
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(503, text="transient 1"),
            httpx.Response(503, text="transient 2"),
            httpx.Response(200, json=OPENAI_OK_RESPONSE),
        ]
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = ModelGateway(
        max_retries=2, backoff_base_seconds=0.001, backoff_cap_seconds=0.001,
        audit_logger=audit,
    )
    result = asyncio.run(gw.chat_completion("hi", study_uid="study-3"))
    assert isinstance(result["content"], list)
    files = audit.list_tool_call_files()
    records = AuditLogger.read_tool_call_records(files[0])
    statuses = [r["status"] for r in records]
    # 2 fail + 1 ok
    assert statuses.count("fail") == 2
    assert statuses.count("ok") == 1


def test_429_rate_limit_retried(openai_mock: Any, env_credentials: None) -> None:
    """429 限流：可重试白名单内 → 重试。"""
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(429, text="too many"),
            httpx.Response(200, json=OPENAI_OK_RESPONSE),
        ]
    )
    gw = ModelGateway(max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    result = asyncio.run(gw.chat_completion("hi"))
    assert "content" in result


def test_408_request_timeout_retried(openai_mock: Any, env_credentials: None) -> None:
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(408, text="timeout"),
            httpx.Response(200, json=OPENAI_OK_RESPONSE),
        ]
    )
    gw = ModelGateway(max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    result = asyncio.run(gw.chat_completion("hi"))
    assert "content" in result


@pytest.mark.parametrize("status_code", sorted(RETRYABLE_HTTP_STATUSES))
def test_all_retryable_status_codes_are_retried(
    openai_mock: Any, env_credentials: None, status_code: int
) -> None:
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(status_code, text=f"err {status_code}"),
            httpx.Response(200, json=OPENAI_OK_RESPONSE),
        ]
    )
    gw = ModelGateway(max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    result = asyncio.run(gw.chat_completion("hi"))
    assert "content" in result


# ---------------------------------------------------------------------------
# 8. 不可重试 4xx → ModelResponseError（不重试）
# ---------------------------------------------------------------------------


def test_non_retryable_4xx_no_retry(openai_mock: Any, env_credentials: None, tmp_path) -> None:
    """401 → 立即抛 ModelResponseError，audit 1 行 fail。"""
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = ModelGateway(max_retries=2, audit_logger=audit)
    with pytest.raises(ModelResponseError, match="non-retryable HTTP 401"):
        asyncio.run(gw.chat_completion("hi", study_uid="study-4"))
    # 只调 1 次 → 1 行 fail
    files = audit.list_tool_call_files()
    records = AuditLogger.read_tool_call_records(files[0])
    assert len(records) == 1
    assert records[0]["status"] == "fail"
    assert records[0]["extra"]["http_status"] == 401
    assert records[0]["extra"]["error_class"] == "non_retryable_http_status"


@pytest.mark.parametrize("status_code", sorted({c for c in NON_RETRYABLE_HTTP_STATUSES if c not in RETRYABLE_HTTP_STATUSES}))
def test_all_non_retryable_4xx_status_codes_no_retry(
    openai_mock: Any, env_credentials: None, status_code: int
) -> None:
    openai_mock.post("/chat/completions").mock(return_value=httpx.Response(status_code, text="err"))
    gw = ModelGateway(max_retries=3, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    with pytest.raises(ModelResponseError):
        asyncio.run(gw.chat_completion("hi"))


# ---------------------------------------------------------------------------
# 9. 响应 JSON 损坏 / 缺 choices
# ---------------------------------------------------------------------------


def test_response_invalid_json_raises_response_error(openai_mock: Any, env_credentials: None) -> None:
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    gw = ModelGateway(max_retries=0, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    with pytest.raises(ModelResponseError, match="JSON parse error"):
        asyncio.run(gw.chat_completion("hi"))


def test_response_empty_choices_raises_response_error(openai_mock: Any, env_credentials: None) -> None:
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"id": "x", "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        )
    )
    gw = ModelGateway(max_retries=0)
    with pytest.raises(ModelResponseError, match="empty choices"):
        asyncio.run(gw.chat_completion("hi"))


def test_response_validation_fails_raises_response_error(
    openai_mock: Any, env_credentials: None
) -> None:
    """响应缺关键字段（如 usage 类型错）→ ModelResponseError。"""
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"id": "x", "choices": "not-a-list"},  # 错类型
        )
    )
    gw = ModelGateway(max_retries=0)
    with pytest.raises(ModelResponseError):
        asyncio.run(gw.chat_completion("hi"))


# ---------------------------------------------------------------------------
# 11. 网络错误（连接拒绝）→ ModelUnreachableError
# ---------------------------------------------------------------------------


def test_connect_error_retried_then_unreachable(openai_mock: Any, env_credentials: None) -> None:
    """ConnectError → 重试 + ModelUnreachableError。"""
    openai_mock.post("/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    gw = ModelGateway(max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi", study_uid="study-5"))


def test_connect_error_then_success(openai_mock: Any, env_credentials: None) -> None:
    """ConnectError 1 次后恢复 200。"""
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("transient"),
            httpx.Response(200, json=OPENAI_OK_RESPONSE),
        ]
    )
    gw = ModelGateway(max_retries=1, backoff_base_seconds=0.001, backoff_cap_seconds=0.001)
    result = asyncio.run(gw.chat_completion("hi"))
    assert "content" in result


# ---------------------------------------------------------------------------
# 12. 指数退避曲线
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attempt,expected",
    [
        (0, 1.0),
        (1, 2.0),
        (2, 4.0),
        (3, 8.0),
        (4, 16.0),
        (5, 30.0),  # cap
        (10, 30.0),  # cap
    ],
)
def test_backoff_curve(env_credentials: None, attempt: int, expected: float) -> None:
    gw = ModelGateway()
    backoff = gw._compute_backoff_seconds(attempt)
    assert backoff == pytest.approx(expected, abs=0.01)


def test_backoff_cap_custom(env_credentials: None) -> None:
    gw = ModelGateway(backoff_base_seconds=2.0, backoff_cap_seconds=10.0)
    assert gw._compute_backoff_seconds(0) == 2.0
    assert gw._compute_backoff_seconds(1) == 4.0
    assert gw._compute_backoff_seconds(2) == 8.0
    assert gw._compute_backoff_seconds(3) == 10.0  # cap
    assert gw._compute_backoff_seconds(10) == 10.0  # cap


# ---------------------------------------------------------------------------
# 14. 审计落盘
# ---------------------------------------------------------------------------


def test_audit_log_full_record_on_success(
    openai_mock: Any, env_credentials: None, tmp_path
) -> None:
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = ModelGateway(audit_logger=audit)
    asyncio.run(
        gw.chat_completion(
            "hi",
            study_uid="study-6",
            tool="keyqa",
            json_schema=KEY_QA_JSON_SCHEMA,
        )
    )
    files = audit.list_tool_call_files()
    assert files
    records = AuditLogger.read_tool_call_records(files[0])
    assert len(records) == 1
    r = records[0]
    # 8 必填字段
    for key in (
        "ts", "study_uid", "tool", "model", "prompt_tokens", "completion_tokens",
        "latency_ms", "status",
    ):
        assert key in r
    assert r["tool"] == "keyqa"
    assert r["study_uid"] == "study-6"
    assert r["model"] == "medrax-llava-med-7b"
    assert r["prompt_tokens"] == 256
    assert r["completion_tokens"] == 128
    assert r["status"] == "ok"
    assert isinstance(r["latency_ms"], int)
    assert r["latency_ms"] >= 0
    # extra 子键
    assert "extra" in r
    assert r["extra"]["http_status"] == 200


def test_audit_logger_failure_does_not_block(
    openai_mock: Any, env_credentials: None
) -> None:
    """audit_logger.tool_call_record 抛异常 → chat_completion 仍正常返。"""
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )

    class _BadAudit:
        async def tool_call_record(self, **kw: Any) -> Any:
            raise RuntimeError("audit down")

    gw = ModelGateway(audit_logger=_BadAudit())  # type: ignore[arg-type]
    result = asyncio.run(gw.chat_completion("hi"))
    assert "content" in result


# ---------------------------------------------------------------------------
# 16. close / aclose 幂等
# ---------------------------------------------------------------------------


def test_aclose_idempotent(env_credentials: None) -> None:
    gw = ModelGateway()
    asyncio.run(gw.aclose())
    asyncio.run(gw.aclose())  # 二次关闭不抛


def test_async_context_manager(openai_mock: Any, env_credentials: None) -> None:
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )

    async def _go() -> Dict[str, Any]:
        async with ModelGateway() as gw:
            return await gw.chat_completion("hi")

    result = asyncio.run(_go())
    assert "content" in result


# ---------------------------------------------------------------------------
# 17. client_factory 注入
# ---------------------------------------------------------------------------


def test_custom_client_factory(openai_mock: Any, env_credentials: None) -> None:
    """client_factory 可注入（respx 走的也是这个 factory 的 client）。"""
    openai_mock.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_OK_RESPONSE)
    )

    custom_called: List[float] = []

    def factory(timeout: float) -> httpx.AsyncClient:
        custom_called.append(timeout)
        return httpx.AsyncClient(timeout=timeout)

    gw = ModelGateway(client_factory=factory)
    asyncio.run(gw.chat_completion("hi"))
    assert len(custom_called) == 1
    assert custom_called[0] == DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# gateway_schema 层独立测试
# ---------------------------------------------------------------------------


def test_chat_request_to_openai_payload() -> None:
    """ChatRequest → OpenAI body 字段映射。"""
    rf = ResponseFormat(
        type="json_schema",
        json_schema={"name": "x", "strict": True, "schema": {"type": "object"}},
    )
    req = ChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],  # type: ignore[arg-type]
        response_format=rf,
    )
    body = req.to_openai_payload()
    assert body["model"] == "m"
    assert body["temperature"] == 0.0
    assert body["response_format"]["type"] == "json_schema"
    # extra 不污染主字段
    assert "study_uid" not in body
    assert "tool" not in body


def test_response_format_json_object() -> None:
    rf = ResponseFormat(type="json_object")
    body = rf.to_openai()
    assert body == {"type": "json_object"}


def test_response_format_text_default() -> None:
    rf = ResponseFormat(type="text")
    assert rf.to_openai() == {"type": "text"}


def test_response_format_json_schema_requires_envelope() -> None:
    rf = ResponseFormat(type="json_schema", json_schema=None)
    with pytest.raises(ValueError, match="json_schema field"):
        rf.to_openai()


def test_build_json_schema_response_format() -> None:
    fmt = build_json_schema_response_format(
        name="finding_draft",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "finding_draft"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["type"] == "object"


def test_chat_response_get_content() -> None:
    parsed = ChatResponse.model_validate(OPENAI_OK_RESPONSE)
    content = parsed.get_content()
    assert "心影" in content
    assert parsed.get_model_id() == "medrax-llava-med-7b"


def test_chat_response_empty_choices() -> None:
    parsed = ChatResponse.model_validate({"choices": [], "model": "m"})
    assert parsed.get_content() == ""
    assert parsed.get_model_id() == "m"


def test_model_gateway_does_not_import_engine() -> None:
    """ModelGateway 不应反向依赖 cxr_triage.engine —— 避免循环引用。"""
    import inspect

    from cxr_triage.gateway import model_gateway as mg_module

    src = inspect.getsource(mg_module)
    # 任何 import cxr_triage.engine（除注释）都失败
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert "from cxr_triage.engine" not in line, f"unexpected import: {line!r}"
        assert "import cxr_triage.engine" not in line, f"unexpected import: {line!r}"


# ---------------------------------------------------------------------------
# 集成：与 CXRKeyQA 协议兼容
# ---------------------------------------------------------------------------


def test_chat_completion_signature_matches_keyqa_protocol(env_credentials: None) -> None:
    """``ModelGateway.chat_completion`` 签名与 ``KeyQAProtocol`` 兼容。"""
    import inspect

    sig = inspect.signature(ModelGateway.chat_completion)
    params = sig.parameters
    assert "prompt" in params
    assert "study_uid" in params
    assert "json_schema" in params
    assert "timeout_seconds" in params
    # tool 不在 KeyQAProtocol，但 ModelGateway 扩展支持
    assert "tool" in params


def test_max_retries_zero_means_single_attempt(openai_mock: Any, env_credentials: None) -> None:
    """max_retries=0 → 只尝试 1 次（无重试）。"""
    openai_mock.post("/chat/completions").mock(return_value=httpx.Response(503, text="err"))
    gw = ModelGateway(max_retries=0)
    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi"))


# ---------------------------------------------------------------------------
# build_model_gateway 工厂
# ---------------------------------------------------------------------------


def test_build_model_gateway_factory(env_credentials: None) -> None:
    gw = build_model_gateway(
        timeout_seconds=15.0,
        max_retries=3,
    )
    assert gw.timeout_seconds == 15.0
    assert gw.max_retries == 3
    assert gw.base_url == "http://test-model-gateway.local:8000/v1"


# ---------------------------------------------------------------------------
# 实跑时间断言：真重试至少 2 次有 backoff 等待
# ---------------------------------------------------------------------------


def test_real_backoff_actually_sleeps(openai_mock: Any, env_credentials: None) -> None:
    """服务端持续 503，max_retries=2 → 总耗时 ≥ 2 次 backoff 等待。"""
    openai_mock.post("/chat/completions").mock(return_value=httpx.Response(503, text="err"))
    base = 0.1
    # 不加 cap，让指数曲线（0.1s / 0.2s）生效
    gw = ModelGateway(max_retries=2, backoff_base_seconds=base, backoff_cap_seconds=10.0)
    t0 = time.monotonic()
    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi"))
    elapsed = time.monotonic() - t0
    # 2 次重试：第 1 次 backoff=0.1s，第 2 次 backoff=0.2s → 总计 ≥ 0.3s
    assert elapsed >= 0.25
