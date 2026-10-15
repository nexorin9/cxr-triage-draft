"""tests/test_model_gateway_degradation.py — Task 20 ModelGateway 降级路径。

目标（spec.md §5 维度 4 正式对接层 + 维度 6 审计 + plan.md §5 深度 (a)
边界与异常用例 + Task 20）：

1. **DEGRADATION_BANNER_ZH 常量**：固定中文 banner，spec.md §10 安全边界
2. **is_available() 探测**：轻量 GET /models 探活；2xx → True；其它 → False；
   网络错 / 超时 / 4xx / 5xx 一律 False；永不抛
3. **chat_completion_with_degradation() 包装**：成功 → degraded=False；
   失败（503 重试用尽 / timeout / JSON 损坏 / 4xx）→ degraded=True +
   banner + error_class；**永不抛**
4. **3 类失败用例**（核心要求）：
   - 服务端 503：retry 2 次仍 503 → ModelUnreachableError → 包装 degraded=True
   - 服务端 timeout：httpx.TimeoutException → retry 用尽 → degraded=True
   - 结构化失败：服务端返回非 JSON 内容 → ModelResponseError → degraded=True
5. **tool_calls_<date>.jsonl 落盘**：degraded 用例写 status=fail/degraded 行；
   成功用例写 status=ok；banner 不重复落 JSONL
6. **不耦合业务层**：与 cxr_keyqa / finding_draft 解耦（仅依赖 ModelGateway
   公开契约 + gateway_schema）

设计原则：
- 降级语义由 ModelGateway 自己负责；调用方（cxr_keyqa / finding_draft）
  只看 ``degraded`` 布尔，不再写 ``try/except``
- banner 是给**医师**看的固定串，混在任何 retry error 之前——保证医师
  看到 banner 时**不读诊断也不出治疗建议**（spec.md §10）
- is_available() 是单查探活，不消耗 token / 不落 JSONL（避免探活日志
  把业务审计数据淹没）
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest
import respx

from cxr_triage.gateway import (
    DEGRADATION_BANNER_ZH,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    ModelGateway,
    ModelGatewayError,
    ModelResponseError,
    ModelUnreachableError,
    build_model_gateway,
)
from cxr_triage.writers.audit import AuditLogger, build_audit_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


OPENAI_OK_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-deg-test-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "medrax-llava-med-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    [{"q": "心影", "a": "心影正常。"}],
                    ensure_ascii=False,
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
}


def _build_gateway_with_audit(tmp_path, base_url: str = "https://api.example.com"):
    """用真 AuditLogger 构造 ModelGateway（落 JSONL 可读出验证）。"""
    audit_logger = build_audit_logger(db=None, tool_calls_dir=tmp_path / "tool_calls")
    # AuditLogger 构造不强制要 db，但 tool_call_record 需要 audit_log_insert
    # 走可选 db；我们这里只测 tool_calls JSONL 部分，audit_log 表不强求
    return build_model_gateway(
        base_url=base_url,
        api_key="sk-test-abc",
        model="medrax-llava-med-7b",
        timeout_seconds=2.0,  # 缩短单次 timeout 加速测试
        max_retries=1,  # 总尝试 2 次，加速
        backoff_base_seconds=0.01,
        backoff_cap_seconds=0.05,
        audit_logger=audit_logger,
    )


def _read_jsonl_records(tmp_path) -> List[Dict[str, Any]]:
    """读 tool_calls_<date>.jsonl 所有行返回 list[dict]。"""
    tool_dir = tmp_path / "tool_calls"
    if not tool_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for f in sorted(tool_dir.iterdir()):
        if not f.name.endswith(".jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ---------------------------------------------------------------------------
# 1. 常量层
# ---------------------------------------------------------------------------


def test_degradation_banner_zh_is_fixed_string():
    """DEGRADATION_BANNER_ZH 是固定中文 banner（spec.md §10 + Task 20 强制）。"""
    assert isinstance(DEGRADATION_BANNER_ZH, str)
    assert DEGRADATION_BANNER_ZH == "模型不可达，仅出概率与定位，需医师人工复核"


def test_default_probe_timeout_seconds():
    """DEFAULT_PROBE_TIMEOUT_SECONDS 默认 5.0s（避免 is_available 长尾）。"""
    assert DEFAULT_PROBE_TIMEOUT_SECONDS == 5.0
    assert DEFAULT_PROBE_TIMEOUT_SECONDS > 0
    assert DEFAULT_PROBE_TIMEOUT_SECONDS <= 30.0  # 不应超过 chat_completion


def test_gateway_stores_probe_timeout():
    """构造期 probe_timeout_seconds 入参落实例属性。"""
    gw = build_model_gateway(
        base_url="https://api.example.com",
        api_key="sk-test",
        model="m",
        probe_timeout_seconds=12.5,
    )
    assert gw.probe_timeout_seconds == 12.5


def test_gateway_probe_timeout_rejects_non_positive():
    """probe_timeout_seconds ≤ 0 拒绝；越界拒绝。"""
    with pytest.raises(ModelGatewayError):
        build_model_gateway(
            base_url="https://api.example.com",
            api_key="sk-test",
            model="m",
            probe_timeout_seconds=0.0,
        )
    with pytest.raises(ModelGatewayError):
        build_model_gateway(
            base_url="https://api.example.com",
            api_key="sk-test",
            model="m",
            probe_timeout_seconds=-1.0,
        )
    with pytest.raises(ModelGatewayError):
        build_model_gateway(
            base_url="https://api.example.com",
            api_key="sk-test",
            model="m",
            probe_timeout_seconds=120.0,  # 超过 60s 上限
        )


# ---------------------------------------------------------------------------
# 2. is_available() 探测
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_available_true_on_2xx(tmp_path):
    """服务端 2xx → True。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").respond(200, json={"data": []})
        ok = await gw.is_available()
    assert ok is True


@pytest.mark.asyncio
async def test_is_available_false_on_5xx(tmp_path):
    """服务端 5xx → False（不抛）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").respond(503, text="unavailable")
        ok = await gw.is_available()
    assert ok is False


@pytest.mark.asyncio
async def test_is_available_false_on_4xx(tmp_path):
    """服务端 4xx → False（不抛；401 表示凭据错也算不可用）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").respond(401, text="unauthorized")
        ok = await gw.is_available()
    assert ok is False


@pytest.mark.asyncio
async def test_is_available_false_on_network_error(tmp_path):
    """网络错误（连接拒绝）→ False（不抛）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        ok = await gw.is_available()
    assert ok is False


@pytest.mark.asyncio
async def test_is_available_false_on_timeout(tmp_path):
    """超时 → False（不抛）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        ok = await gw.is_available()
    assert ok is False


@pytest.mark.asyncio
async def test_is_available_false_on_unexpected_exception(tmp_path):
    """任何异常兜底 → False（探测永不抛）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://api.example.com/models").mock(
            side_effect=RuntimeError("weird bug")
        )
        ok = await gw.is_available()
    assert ok is False


@pytest.mark.asyncio
async def test_is_available_uses_probe_timeout(tmp_path):
    """is_available 用 probe_timeout_seconds（独立于 chat timeout）。"""
    gw = build_model_gateway(
        base_url="https://api.example.com",
        api_key="sk-test",
        model="m",
        probe_timeout_seconds=0.5,
        timeout_seconds=30.0,
        audit_logger=build_audit_logger(db=None, tool_calls_dir=tmp_path / "tc"),
    )
    captured: Dict[str, Any] = {}

    class _CaptureClient:
        def __init__(self, real_client):
            self._real = real_client

        async def get(self, url, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            captured["url"] = url
            return httpx.Response(200)

    with patch.object(gw, "_get_client", side_effect=lambda: _CaptureClient(None)):
        ok = await gw.is_available()
    assert ok is True
    # is_available 必须用 probe_timeout_seconds 作为 per-call timeout
    assert captured["timeout"] == 0.5
    assert captured["url"] == "https://api.example.com/models"


@pytest.mark.asyncio
async def test_is_available_does_not_write_jsonl(tmp_path):
    """is_available 探测**不**落 tool_calls JSONL（探活不算业务调用）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.example.com/models").respond(503)
        await gw.is_available()
        mock.get("https://api.example.com/models").respond(200)
        await gw.is_available()
    rows = _read_jsonl_records(tmp_path)
    assert rows == []  # 探测绝不落 JSONL


# ---------------------------------------------------------------------------
# 3. chat_completion_with_degradation() 包装
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_wrapper_happy_path(tmp_path):
    """happy path → degraded=False + banner 空 + content/raw/model 完整。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            200, json=OPENAI_OK_RESPONSE
        )
        result = await gw.chat_completion_with_degradation(
            "请描述心影", study_uid="1.2.3.4"
        )
    assert result["degraded"] is False
    assert result["banner"] == ""
    assert result["model"] == "medrax-llava-med-7b"
    # content 已被 _coerce_content 解析为 list
    assert isinstance(result["content"], list)
    assert result["content"][0]["q"] == "心影"
    assert result["raw"]["id"] == "chatcmpl-deg-test-1"
    # audit 写 1 行 ok
    rows = _read_jsonl_records(tmp)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["tool"] == "model_gateway"


# ---------------------------------------------------------------------------
# 4. 场景 A：服务端 503 重试用尽 → ModelUnreachableError → 降级包装
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_503_retry_exhausted(tmp_path):
    """场景 A：服务端连续 503 → retry 用尽 → ModelUnreachableError →
    chat_completion_with_degradation → degraded=True + banner + 3 行 fail + 1 行 degraded
    """
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            503, text="upstream busy"
        )
        result = await gw.chat_completion_with_degradation(
            "test prompt", study_uid="<EXAMPLE_STUDY_A>"
        )
    # 结构化降级返回
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["content"] is None
    assert result["raw"] == {}
    assert result["model"] == "medrax-llava-med-7b"
    assert result["error_class"] == "ModelUnreachableError"
    assert "exhausted" in result["error_message"].lower() or "503" in result["error_message"]
    assert result["attempts"] == 2  # max_retries=1 → 总尝试 2 次

    # tool_calls JSONL 验证：fail 行 + degraded 行
    rows = _read_jsonl_records(tmp_path)
    statuses = [r["status"] for r in rows]
    assert "fail" in statuses
    assert "degraded" in statuses
    # 至少有 1 行 fail（503 触发）+ 1 行 degraded（重试用尽）
    fail_rows = [r for r in rows if r["status"] == "fail"]
    degraded_rows = [r for r in rows if r["status"] == "degraded"]
    assert len(fail_rows) >= 1
    assert len(degraded_rows) == 1
    assert degraded_rows[0]["extra"]["attempts"] == 2


@pytest.mark.asyncio
async def test_degradation_on_503_call_to_inner_raises_unreachable():
    """场景 A 副：确认内层 chat_completion 抛 ModelUnreachableError（用于 Task 21 兼容）。"""
    gw = _build_gateway_with_audit(__import__("pathlib").Path("/tmp"))
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(503, text="x")
        with pytest.raises(ModelUnreachableError):
            await gw.chat_completion("test")


# ---------------------------------------------------------------------------
# 5. 场景 B：服务端 timeout → httpx.TimeoutException → 重试用尽 → 降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_timeout_retry_exhausted(tmp_path):
    """场景 B：服务端超时 → 重试用尽 → ModelUnreachableError → degraded=True。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        result = await gw.chat_completion_with_degradation(
            "test prompt", study_uid="<EXAMPLE_STUDY_B>"
        )
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["content"] is None
    assert result["model"] == "medrax-llava-med-7b"
    assert result["error_class"] == "ModelUnreachableError"
    assert result["attempts"] == 2

    rows = _read_jsonl_records(tmp_path)
    statuses = [r["status"] for r in rows]
    assert "fail" in statuses
    assert "degraded" in statuses


@pytest.mark.asyncio
async def test_degradation_on_connect_error(tmp_path):
    """ConnectError（连接拒绝）→ 重试用尽 → ModelUnreachableError → degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "ModelUnreachableError"
    assert result["attempts"] == 2


@pytest.mark.asyncio
async def test_degradation_on_remote_protocol_error(tmp_path):
    """RemoteProtocolError（协议错）→ 重试用尽 → ModelUnreachableError。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.example.com/chat/completions").mock(
            side_effect=httpx.RemoteProtocolError("protocol error")
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "ModelUnreachableError"


# ---------------------------------------------------------------------------
# 6. 场景 C：结构化失败（JSON 损坏 / 缺 choices）→ ModelResponseError → 降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_json_decode_error(tmp_path):
    """场景 C：服务端返回非 JSON 内容 → ModelResponseError → degraded=True。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            200, text="<html>500 Internal Server Error</html>",
            headers={"content-type": "text/html"},
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["content"] is None
    assert result["error_class"] == "ModelResponseError"
    assert result["attempts"] == 1  # 结构化失败不重试

    rows = _read_jsonl_records(tmp_path)
    statuses = [r["status"] for r in rows]
    # 结构化失败仅 1 行 fail；无 degraded 行（未耗尽重试）
    assert statuses.count("fail") == 1
    assert "degraded" not in statuses


@pytest.mark.asyncio
async def test_degradation_on_schema_validation_failure(tmp_path):
    """场景 C 副：返回缺 choices → ModelResponseError → degraded=True。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "m",
                "choices": [],  # 空 choices 触发 ModelResponseError
            },
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "ModelResponseError"
    assert "empty choices" in result["error_message"].lower()


# ---------------------------------------------------------------------------
# 7. 4xx 业务错误 → ModelResponseError → degraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_400_unauthorized(tmp_path):
    """400（业务错误）→ ModelResponseError → degraded=True + attempts=1。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            400, text='{"error": "bad request"}',
            headers={"content-type": "application/json"},
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "ModelResponseError"
    assert result["attempts"] == 1  # 业务错误不重试


@pytest.mark.asyncio
async def test_degradation_on_404(tmp_path):
    """404 → ModelResponseError → degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            404, text="not found"
        )
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["error_class"] == "ModelResponseError"


# ---------------------------------------------------------------------------
# 8. 边界：prompt 空 / study_uid 非字符串 → ModelGatewayError → degraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_invalid_prompt(tmp_path):
    """prompt 空字符串 → ModelGatewayError → 包装 degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    result = await gw.chat_completion_with_degradation("")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "ModelGatewayError"
    assert result["attempts"] == 0


@pytest.mark.asyncio
async def test_degradation_on_non_string_study_uid(tmp_path):
    """study_uid 非字符串 → ModelGatewayError → degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)
    result = await gw.chat_completion_with_degradation("test", study_uid=123)
    assert result["degraded"] is True
    assert result["error_class"] == "ModelGatewayError"


# ---------------------------------------------------------------------------
# 9. 防御性兜底：任何未知异常 → degraded=True（永不抛）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_on_unexpected_exception_in_chat(tmp_path):
    """内层 chat_completion 抛未预期异常（被 monkeypatch 模拟）→ 包装 degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)

    async def _boom(*args, **kwargs):
        raise RuntimeError("unexpected boom")

    with patch.object(gw, "chat_completion", side_effect=_boom):
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["banner"] == DEGRADATION_BANNER_ZH
    assert result["error_class"] == "RuntimeError"
    assert "unexpected boom" in result["error_message"]


@pytest.mark.asyncio
async def test_degradation_on_timeout_outside_chat(tmp_path):
    """asyncio.TimeoutError（外层 wait_for 包络）→ 包装 degraded=True。"""
    import pathlib
    tmp = tmp_path
    gw = _build_gateway_with_audit(tmp)

    async def _timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch.object(gw, "chat_completion", side_effect=_timeout):
        result = await gw.chat_completion_with_degradation("test")
    assert result["degraded"] is True
    assert result["error_class"] == "TimeoutError"


# ---------------------------------------------------------------------------
# 10. banner 一致性 + 内容约束
# ---------------------------------------------------------------------------


def test_degradation_banner_does_not_diagnose_or_prescribe():
    """banner 不出现诊断 / 治疗建议 / 处方等敏感词（spec.md §10 安全边界）。"""
    banner = DEGRADATION_BANNER_ZH
    # 不应包含诊断 / 治疗 / 处方 / 给药 / 剂量
    forbidden = ["诊断", "治疗", "处方", "给药", "剂量", "剂量调整"]
    for word in forbidden:
        assert word not in banner, f"banner 含敏感词 {word!r}"
    # 应包含"人工复核"语义
    assert "人工复核" in banner
    assert "模型不可达" in banner


def test_degradation_banner_is_immutable_string_constant():
    """banner 是 module-level str 常量；非运行时拼接。"""
    from cxr_triage.gateway import model_gateway as mg_module
    # banner 直接定义在模块顶层
    assert isinstance(mg_module.DEGRADATION_BANNER_ZH, str)
    assert mg_module.DEGRADATION_BANNER_ZH == DEGRADATION_BANNER_ZH


# ---------------------------------------------------------------------------
# 11. 与现有 chat_completion 兼容（keyqa / finding_draft 可平滑迁移）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_wrapper_compatible_with_keyqa_protocol():
    """包装方法签名与 chat_completion 一致（prompt / study_uid / json_schema /
    timeout_seconds / tool 五参数）。"""
    import inspect

    sig_chat = inspect.signature(ModelGateway.chat_completion)
    sig_deg = inspect.signature(ModelGateway.chat_completion_with_degradation)
    # 参数名集合一致
    assert sig_chat.parameters.keys() == sig_deg.parameters.keys()


@pytest.mark.asyncio
async def test_degradation_wrapper_passes_json_schema(tmp_path):
    """包装方法透传 json_schema 到内层 chat_completion（出站 envelope 完整）。"""
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "key_qa_response",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {"type": "object", "properties": {"q": {"type": "string"}, "a": {"type": "string"}}},
            },
        },
    }
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://api.example.com/chat/completions").respond(
            200, json=OPENAI_OK_RESPONSE
        )
        await gw.chat_completion_with_degradation(
            "test", study_uid="1.2.3", json_schema=schema, tool="keyqa"
        )
    # 出站 body 含 response_format
    request_body = route.calls.last.request.content
    body_dict = json.loads(request_body)
    assert body_dict["response_format"]["type"] == "json_schema"
    assert body_dict["response_format"]["json_schema"]["name"] == "key_qa_response"


# ---------------------------------------------------------------------------
# 12. tool_calls JSONL 字段契约
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_tool_calls_jsonl_fields_complete(tmp_path):
    """降级用例 JSONL 每行 8 字段全齐（model/prompt_tokens/completion_tokens/
    latency_ms/status/study_uid/tool/extra）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(503, text="x")
        await gw.chat_completion_with_degradation(
            "test", study_uid="<EXAMPLE_STUDY>", tool="finding_draft"
        )
    rows = _read_jsonl_records(tmp_path)
    assert rows
    for r in rows:
        # 8 字段
        assert "ts" in r
        assert "tool" in r and r["tool"] == "finding_draft"
        assert "model" in r
        assert "prompt_tokens" in r
        assert "completion_tokens" in r
        assert "latency_ms" in r and r["latency_ms"] >= 0
        assert "status" in r and r["status"] in ("ok", "fail", "degraded")
        assert "study_uid" in r and r["study_uid"] == "<EXAMPLE_STUDY>"


@pytest.mark.asyncio
async def test_degradation_does_not_write_extra_banner_to_jsonl(tmp_path):
    """banner 只在内存 dict 返给调用方；**不**落 tool_calls JSONL
    （避免探活日志把审计数据淹没）。"""
    gw = _build_gateway_with_audit(tmp_path)
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(503, text="x")
        await gw.chat_completion_with_degradation("test")
    rows = _read_jsonl_records(tmp_path)
    for r in rows:
        assert "banner" not in r
        assert "degraded" not in r


# ---------------------------------------------------------------------------
# 13. 端到端：3 场景连贯（503 / timeout / JSON 损坏）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_scenarios_end_to_end(tmp_path):
    """3 类降级场景连贯：503 / timeout / JSON 损坏 各跑一遍 →
    全部 degraded=True + banner 一致 + JSONL 各自 fail 行存在。"""
    gw = _build_gateway_with_audit(tmp_path)

    # 场景 A：503
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            503, text="busy"
        )
        r_a = await gw.chat_completion_with_degradation(
            "test", study_uid="<EXAMPLE_A>"
        )
    # 场景 B：timeout
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.example.com/chat/completions").mock(
            side_effect=httpx.TimeoutException("x")
        )
        r_b = await gw.chat_completion_with_degradation(
            "test", study_uid="<EXAMPLE_B>"
        )
    # 场景 C：JSON 损坏
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.example.com/chat/completions").respond(
            200, text="not json", headers={"content-type": "text/plain"}
        )
        r_c = await gw.chat_completion_with_degradation(
            "test", study_uid="<EXAMPLE_C>"
        )

    # 全部 degraded
    for r, label in [(r_a, "A"), (r_b, "B"), (r_c, "C")]:
        assert r["degraded"] is True, f"场景 {label} 应 degraded=True"
        assert r["banner"] == DEGRADATION_BANNER_ZH, f"场景 {label} banner 错"
        assert r["content"] is None, f"场景 {label} content 应 None"
        assert r["model"] == "medrax-llava-med-7b"

    # JSONL 落盘验证：3 场景至少 3 行 fail（A 2 fail + B 2 fail + C 1 fail = 5 fail）
    # + 至少 2 行 degraded（A + B 重试用尽；C 仅 1 次未触发 degraded）
    rows = _read_jsonl_records(tmp_path)
    statuses = [r["status"] for r in rows]
    assert statuses.count("fail") >= 5
    assert statuses.count("degraded") == 2  # 仅 A + B 触发重试用尽


# ---------------------------------------------------------------------------
# 14. 跨模块 import 隔离（不耦合业务层）
# ---------------------------------------------------------------------------


def test_model_gateway_does_not_import_engine_or_writers():
    """ModelGateway 模块顶层不 import engine / writers（保持依赖反转）。"""
    import cxr_triage.gateway.model_gateway as mg
    src = open(mg.__file__, encoding="utf-8").read()
    # 不允许直接 import cxr_triage.engine / cxr_triage.writers
    assert "from ..engine" not in src
    assert "from ..writers" not in src
    assert "from ..state" not in src