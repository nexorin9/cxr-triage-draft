"""tests/test_protocol_compat.py — Task 21 ModelGateway 协议兼容性测试。

目标（spec.md §5 维度 4 正式对接层 + plan.md §5 深度 (a) 边界与异常用例 +
Task 21 强制要求）：

1. **超时（TimeoutException）**：服务端一直不响应 → ModelGateway 退避 2 次
   后抛 ``ModelUnreachableError`` + audit 落 ``fail × 3`` + ``degraded × 1``
2. **不可达（ConnectError）**：服务端拒绝连接 → ModelGateway 退避 2 次后
   抛 ``ModelUnreachableError`` + audit 落 ``fail × 3`` + ``degraded × 1``
3. **结构化失败（ResponseError）**：服务端返回 200 但 JSON 损坏 →
   ModelGateway 不重试，立即抛 ``ModelResponseError`` + audit 落 ``fail × 1``
   （结构化失败重试无意义）

每用例都用 ``respx`` 拦截 HTTP；用真 ``AuditLogger``（tmpdir）落
``tool_calls_<date>.jsonl`` 验证审计契约；用最小 ``backoff_base/cap=0.001``
减少测试时长。

协议层覆盖：

- 不可达 / 超时属于「网络层错」（httpx.TimeoutException / ConnectError /
  NetworkError），ModelGateway 走 **重试 → 退避** 路径；
- 结构化失败属于「响应层错」（JSON 损坏 / 缺 choices / schema 校验失败），
  ModelGateway 走 **不重试** 路径。

这是 plan.md §5 深度 (a) 边界与异常用例的具体落地；与 spec.md §5 维度 4
「凭据从 env 读、不绑定厂商 SDK、协议而非耦合」形成闭环。

注意：

- ``tests/test_model_gateway.py`` 已覆盖 happy path / 401 / 5xx 退避 /
  退避曲线，本任务**专注**三类**最低层语义**的协议错（不重试结构化失败、
  重试网络错），不重复测其它维度。
- ``tests/test_model_gateway_degradation.py`` 已覆盖 ``chat_completion_with_degradation``
  与 ``DEGRADATION_BANNER_ZH``，本任务**专注** ``chat_completion`` 自身的
  协议降级路径与 audit 契约，不混入 degradation 包装。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import httpx
import pytest
import respx

from cxr_triage.gateway import (
    ModelGateway,
    ModelResponseError,
    ModelUnreachableError,
)
from cxr_triage.writers.audit import AuditLogger, build_audit_logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


OPENAI_ENDPOINT_BASE = "http://test-protocol-compat.local:8765/v1"


@pytest.fixture
def openai_mock() -> Any:
    """respx 拦截同 host 全部出站 HTTP。"""
    with respx.mock(base_url=OPENAI_ENDPOINT_BASE, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def env_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """注入 ModelGateway 构造所需的 env；用占位符开关放行。"""
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_ENDPOINT_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", "protocol-compat-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "protocol-compat-test-model")
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "placeholder-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://healthcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "placeholder-healthcheck-token")


def _fast_backoff_gateway(audit_logger: Any = None) -> ModelGateway:
    """构造一个 backoff ≈ 0 的 ModelGateway（测试提速用）。"""
    return ModelGateway(
        base_url=OPENAI_ENDPOINT_BASE,
        api_key="protocol-compat-test-key",
        model="protocol-compat-test-model",
        timeout_seconds=10.0,
        max_retries=2,  # 总 3 次尝试
        backoff_base_seconds=0.001,
        backoff_cap_seconds=0.001,
        audit_logger=audit_logger,
    )


# ---------------------------------------------------------------------------
# 1. 超时（TimeoutException）→ ModelUnreachableError + 退避 × 2
# ---------------------------------------------------------------------------


def test_timeout_triggers_retry_then_unreachable(
    openai_mock: Any, env_credentials: None, tmp_path
) -> None:
    """**超时（timeout）协议兼容性测试**。

    服务端响应**永不返回**（``respx.side_effect=TimeoutException``）→ 触发
    ModelGateway 退避 2 次后抛 ``ModelUnreachableError``；audit 落 ``fail × 3``
    + ``degraded × 1`` 共 4 行；最后一次 ``degraded`` 行 ``attempts=3``。

    验收：
    - 总尝试 3 次（max_retries=2 + 首次 = 3）
    - ``ModelUnreachableError`` 抛出
    - audit JSONL 含 3 行 fail + 1 行 degraded
    - 每行 fail 带 ``error_class`` 为 ``TimeoutException``
    - degraded 行 ``attempts=3`` + ``error_class=TimeoutException``
    """
    openai_mock.post("/chat/completions").mock(
        side_effect=httpx.TimeoutException("simulated upstream timeout"),
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = _fast_backoff_gateway(audit_logger=audit)

    with pytest.raises(ModelUnreachableError, match="exhausted"):
        asyncio.run(
            gw.chat_completion(
                "请按 4 模板回答",
                study_uid="protocol-timeout-1",
                tool="keyqa",
            )
        )

    # audit 落盘校验
    files = audit.list_tool_call_files()
    assert files, "audit JSONL 应当至少有一个文件被创建"
    records = AuditLogger.read_tool_call_records(files[0])

    statuses = [r["status"] for r in records]
    assert statuses.count("fail") == 3, f"应当有 3 行 fail（3 次尝试），实际 {statuses}"
    assert statuses.count("degraded") == 1, f"应当有 1 行 degraded（重试用尽），实际 {statuses}"

    # 每行 fail 携带 http_status=None + error_class=TimeoutException
    fails = [r for r in records if r["status"] == "fail"]
    for f in fails:
        assert f["extra"].get("error_class") == "TimeoutException", (
            f"fail 行的 error_class 应为 TimeoutException，实际 {f['extra']}"
        )
        assert f["extra"].get("http_status") is None
        assert f["study_uid"] == "protocol-timeout-1"
        assert f["tool"] == "keyqa"

    # degraded 行带 attempts=3
    degraded_rows = [r for r in records if r["status"] == "degraded"]
    assert len(degraded_rows) == 1
    assert degraded_rows[0]["extra"]["attempts"] == 3
    assert degraded_rows[0]["extra"]["error_class"] == "TimeoutException"
    assert degraded_rows[0]["extra"]["last_status"] is None


# ---------------------------------------------------------------------------
# 2. 不可达（ConnectError）→ ModelUnreachableError + 退避 × 2
# ---------------------------------------------------------------------------


def test_connection_refused_triggers_retry_then_unreachable(
    openai_mock: Any, env_credentials: None, tmp_path
) -> None:
    """**不可达（connection refused）协议兼容性测试**。

    服务端**连接被拒绝**（``respx.side_effect=ConnectError``）→ 触发
    ModelGateway 退避 2 次后抛 ``ModelUnreachableError``；audit 落 ``fail × 3``
    + ``degraded × 1`` 共 4 行。

    验收：
    - 总尝试 3 次
    - ``ModelUnreachableError`` 抛出
    - audit JSONL 含 3 行 fail + 1 行 degraded
    - 每行 fail 带 ``error_class=ConnectError``
    - degraded 行 ``attempts=3``
    """
    openai_mock.post("/chat/completions").mock(
        side_effect=httpx.ConnectError("simulated connection refused"),
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = _fast_backoff_gateway(audit_logger=audit)

    with pytest.raises(ModelUnreachableError, match="exhausted"):
        asyncio.run(
            gw.chat_completion(
                "请按 4 模板回答",
                study_uid="protocol-unreachable-1",
                tool="keyqa",
            )
        )

    files = audit.list_tool_call_files()
    assert files
    records = AuditLogger.read_tool_call_records(files[0])

    statuses = [r["status"] for r in records]
    assert statuses.count("fail") == 3
    assert statuses.count("degraded") == 1

    # 错误类应当是 ConnectError
    fails = [r for r in records if r["status"] == "fail"]
    for f in fails:
        assert f["extra"].get("error_class") == "ConnectError", (
            f"fail 行 error_class 应为 ConnectError，实际 {f['extra']}"
        )
        assert f["study_uid"] == "protocol-unreachable-1"

    degraded_rows = [r for r in records if r["status"] == "degraded"]
    assert len(degraded_rows) == 1
    assert degraded_rows[0]["extra"]["attempts"] == 3
    assert degraded_rows[0]["extra"]["error_class"] == "ConnectError"


# ---------------------------------------------------------------------------
# 3. 结构化失败（响应 200 但 JSON 损坏）→ ModelResponseError + 不重试
# ---------------------------------------------------------------------------


def test_structured_response_failure_no_retry(
    openai_mock: Any, env_credentials: None, tmp_path
) -> None:
    """**结构化失败协议兼容性测试**。

    服务端返回 **HTTP 200** 但 **body 不是合法 JSON**（``text=<html>``）→
    ModelGateway **不重试**（重试不会让 schema 变好），立即抛
    ``ModelResponseError``；audit 落 ``fail × 1``。

    验收：
    - 仅调 1 次（max_retries=2 也只跑首次）
    - ``ModelResponseError`` 抛出
    - audit JSONL 含 1 行 fail
    - fail 行 ``error_class=JSONDecodeError``（httpx.json() 失败为
      ``json.JSONDecodeError``，ModelGateway 归类为 ``JSONDecodeError`` /
      ``ValueError`` 均可，本测试要求存在）
    """
    openai_mock.post("/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                text="<html>this is not JSON</html>",
                headers={"Content-Type": "text/html"},
            ),
            httpx.Response(200, text="<html>should not reach here</html>",
                                       headers={"Content-Type": "text/html"}),
        ]
    )
    audit = build_audit_logger(tool_calls_dir=str(tmp_path / "tool_calls"))
    gw = _fast_backoff_gateway(audit_logger=audit)

    with pytest.raises(ModelResponseError, match="JSON parse error"):
        asyncio.run(
            gw.chat_completion(
                "请按 4 模板回答",
                study_uid="protocol-structural-1",
                tool="keyqa",
            )
        )

    # 结构化失败不重试 → 只调 1 次（mock 配的第二次 Response 不应被消费）
    files = audit.list_tool_call_files()
    assert files
    records = AuditLogger.read_tool_call_records(files[0])

    statuses = [r["status"] for r in records]
    assert statuses.count("fail") == 1, (
        f"结构化失败不应重试；应当只有 1 行 fail，实际 {statuses}"
    )
    assert statuses.count("degraded") == 0
    assert statuses.count("ok") == 0

    # fail 行 error_class 应是 JSON 解析错或 ModelResponseError 本身
    fail_row = records[0]
    assert fail_row["status"] == "fail"
    error_class = fail_row["extra"].get("error_class", "")
    assert error_class in {"JSONDecodeError", "ModelResponseError", "ValueError"}, (
        f"fail 行 error_class 应当是 JSON 解析相关类型，实际 {error_class!r}"
    )
    assert fail_row["study_uid"] == "protocol-structural-1"
    assert fail_row["tool"] == "keyqa"


# ---------------------------------------------------------------------------
# 4. 协议契约附加：service_url 拼装正确 + Bearer token 注入
# ---------------------------------------------------------------------------


def test_protocol_endpoint_path_and_bearer_token_injected(
    openai_mock: Any, env_credentials: None
) -> None:
    """**协议契约附加测试**：确认出站 URL 与 Bearer token 拼装正确。

    这是「不可达 / 超时 / 结构化失败」三类之外的协议层契约；放在
    ``test_protocol_compat.py`` 是因为：协议兼容性测试的核心是「ModelGateway
    出站的 HTTP 形态与目标 OpenAI-compatible 服务端约定一致」，URL 与
    Authorization 是其前置条件。

    验收：
    - 出站 URL = ``{base_url}/chat/completions``
    - Authorization = ``Bearer <api_key>``
    - 即使服务端返回 ConnectError，URL 与 Authorization 已被服务端（respx）
      正确接收（respx.mock 自身在 side_effect 触发前已捕获到请求元数据）
    """
    captured_requests = []

    def _capture_then_fail(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        raise httpx.ConnectError("captured then refused")

    openai_mock.post("/chat/completions").mock(side_effect=_capture_then_fail)
    gw = _fast_backoff_gateway()

    with pytest.raises(ModelUnreachableError):
        asyncio.run(gw.chat_completion("hi", study_uid="protocol-attach-1"))

    # 至少 1 次请求被捕获（实际是 3 次，但 side_effect 在首次触发即抛）
    assert captured_requests, "respx 未捕获到任何请求"
    req = captured_requests[0]

    # URL 路径正确（base_url 含 /v1 前缀，路径为 /v1/chat/completions）
    assert req.url.path == "/v1/chat/completions"
    assert str(req.url).startswith(OPENAI_ENDPOINT_BASE)

    # Bearer token 注入正确
    auth = req.headers.get("Authorization")
    assert auth == "Bearer protocol-compat-test-key"