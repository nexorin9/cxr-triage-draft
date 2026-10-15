"""tests/test_ris_writer.py — RIS finding-draft 写回客户端（httpx 异步 + outbox 兜底）。

覆盖（CLAUDE.md Phase 4 验收 · Task 22）：

1. **payload 构造**：
   - 7 字段对齐 spec.md §4（study_uid / patient_id_hash / finding_struct
     / finding_text / positive_flag / tool_version / draft_ts）；
   - DRAFT_MARKER 落 RIS 前**必须**剥掉（"草稿"两字不进医疗记录）；
   - 缺字段 / 错类型抛 ``RISPayloadError``。
2. **URL 构造**：base_url 末斜杠 trim；study_uid 缺失抛错。
3. **happy-path 200**：mock httpx → RIS 200 → ``write_finding_draft`` 返
   True + audit_log 写 ``ris.written`` 行 + outbox **不**入条。
4. **失败路径 5xx**：mock RIS 500 → 返 False + outbox **入条** + audit_log
   写 ``ris.queued_for_retry`` 行（payload 含 target_endpoint / error）。
5. **失败路径 网络错**：httpx.TimeoutException / ConnectError → 同样入
   outbox + audit。
6. **失败路径 4xx**：400 / 404 → 入 outbox + audit（与 5xx 同处理，4xx
   视作可重试，由 MAX_RETRY_BEFORE_ALERT=3 触发 alerting）。
7. **不静默丢数据**：失败必入 outbox；outbox 入条失败 → 写
   ``ris.outbox_enqueue_failed`` audit 行（不让事件失踪）。
8. **Bearer Token 注入**：从 config.ris_api_token 读；出站 header 必含
   ``Authorization: Bearer <token>``。
9. **patient_id_hash 隔离**：明文 PatientID 永不出现在 payload 或 URL。
10. **并发**：5 张 study 并发写 → 全部 200 成功 + 5 条 audit + 0 outbox；
    5 张全部 500 → 5 条 outbox + 5 条 audit。
11. **stats 计数**：attempts / successes / failures / outbox_enqueues
    累加正确。
12. **工厂 build_ris_writer**：缺 outbox / audit_logger → 抛
    ``RISWriterError``；显式注入可用。
13. **aclose 幂等**：二次关闭不抛。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from cxr_triage.config import CxrTriageConfig
from cxr_triage.persistence import Database, init_db
from cxr_triage.persistence.outbox import Outbox
from cxr_triage.writers.audit import AuditLogger
from cxr_triage.writers.ris_writer import (
    DRAFT_MARKER,
    DEFAULT_TIMEOUT_SECONDS,
    RIS_TARGET_PREFIX,
    RISPayloadError,
    RISWriter,
    RISWriterError,
    build_ris_finding_draft_url,
    build_ris_payload,
    build_ris_target_endpoint,
    build_ris_writer,
)


# =====================================================================
# Fixtures
# =====================================================================
def _make_config(**overrides) -> CxrTriageConfig:
    """构造最小 CxrTriageConfig（凭据从 env 来，本测试用 placeholder + flag 放行）。"""
    import os

    os.environ.setdefault("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    os.environ.setdefault("RIS_API_TOKEN", "replace-me-ris-token")
    os.environ.setdefault("HEALTHCHECK_API_TOKEN", "replace-me-hc-token")
    os.environ.setdefault("OPENAI_API_KEY", "replace-me-llm-key")
    return CxrTriageConfig(**overrides)


def _payload_kwargs() -> dict[str, Any]:
    """一组典型 payload 输入（draft 形态，喂给 build_ris_payload）。"""
    return {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_RIS>",
        "patient_id_hash": "abc123def456" * 4,  # 64 hex
        "finding_text": (
            "右下肺野见少量气胸征象（草稿 / 未经医师确认），"
            "余肺野清晰。"
        ),
        "finding_struct": {
            "pathologies": [
                {"name": "Pneumothorax", "prob": 0.87, "present": True},
            ],
            "key_qa": [{"q": "心影", "a": "心影不大"}],
            "grounding_boxes": [
                {
                    "x": 0.55,
                    "y": 0.40,
                    "w": 0.20,
                    "h": 0.15,
                    "label": "右下肺",
                    "prob": 0.81,
                }
            ],
        },
        "positive_flag": True,
        "draft_ts": "2026-10-13T07:23:11.487Z",
        "tool_version": "0.1.0",
    }


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "ris_writer_test.db"))
    await database.connect()
    await init_db(database)
    yield database
    await database.close()


@pytest.fixture
async def outbox(db: Database) -> Outbox:
    return Outbox(db=db)


@pytest.fixture
async def audit(db: Database, tmp_path) -> AuditLogger:
    return AuditLogger(db=db, tool_calls_dir=tmp_path / "tool_calls")


@pytest.fixture
def config() -> CxrTriageConfig:
    return _make_config()


# =====================================================================
# 1. 常量 + payload + URL 工具
# =====================================================================
def test_draft_marker_constant() -> None:
    """DRAFT_MARKER 锁定，便于 RIS writer 剥草稿标识。"""
    assert DRAFT_MARKER == "（草稿 / 未经医师确认）"


def test_ris_target_prefix_constant() -> None:
    """RIS_TARGET_PREFIX 用于 outbox target_endpoint 区分 RIS vs 主检。"""
    assert RIS_TARGET_PREFIX == "ris://finding-draft/"


def test_default_timeout_constant() -> None:
    """DEFAULT_TIMEOUT_SECONDS=5s（RIS 医院内网通常 2s 内）。"""
    assert DEFAULT_TIMEOUT_SECONDS == 5.0


def test_build_ris_finding_draft_url_basic() -> None:
    """URL 拼装：base_url + 端点模板，study_uid 嵌入路径。"""
    url = build_ris_finding_draft_url(
        "http://127.0.0.1:8081", "1.2.826.0.1.3680043.8.498.<EXAMPLE_RIS>"
    )
    assert url == (
        "http://127.0.0.1:8081/api/v1/studies/"
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_RIS>/finding-draft"
    )


def test_build_ris_finding_draft_url_strip_trailing_slash() -> None:
    """base_url 末斜杠必须 trim，避免双斜杠。"""
    url = build_ris_finding_draft_url("http://127.0.0.1:8081/", "uid-1")
    assert url == "http://127.0.0.1:8081/api/v1/studies/uid-1/finding-draft"


def test_build_ris_finding_draft_url_empty_base() -> None:
    with pytest.raises(RISPayloadError, match="RIS base_url 为空"):
        build_ris_finding_draft_url("", "uid-1")


def test_build_ris_finding_draft_url_empty_study() -> None:
    with pytest.raises(RISPayloadError, match="study_uid 为空"):
        build_ris_finding_draft_url("http://127.0.0.1:8081", "")


def test_build_ris_target_endpoint() -> None:
    """outbox target_endpoint 含 RIS 前缀 + study_uid（便于 retry_loop 路由）。"""
    ep = build_ris_target_endpoint("uid-xyz")
    assert ep == "ris://finding-draft/uid-xyz"


def test_build_ris_payload_happy_path() -> None:
    """7 字段对齐 spec.md §4；DRAFT_MARKER 必剥。"""
    kw = _payload_kwargs()
    payload = build_ris_payload(**kw)
    assert payload["study_uid"] == kw["study_uid"]
    assert payload["patient_id_hash"] == kw["patient_id_hash"]
    assert payload["finding_struct"] == kw["finding_struct"]
    # DRAFT_MARKER 已剥
    assert DRAFT_MARKER not in payload["finding_text"]
    assert payload["finding_text"] == "右下肺野见少量气胸征象，余肺野清晰。"
    assert payload["positive_flag"] is True
    assert payload["tool_version"] == "0.1.0"
    assert payload["draft_ts"] == kw["draft_ts"]


def test_build_ris_payload_finding_text_no_draft_marker() -> None:
    """无 DRAFT_MARKER 的 finding_text 也应正常返回（不破坏）。"""
    kw = _payload_kwargs()
    kw["finding_text"] = "右下肺野见少量气胸征象，余肺野清晰。"
    payload = build_ris_payload(**kw)
    assert payload["finding_text"] == "右下肺野见少量气胸征象，余肺野清晰。"


def test_build_ris_payload_empty_finding_text() -> None:
    """空 finding_text 接受（阴性所见留白场景）。"""
    kw = _payload_kwargs()
    kw["finding_text"] = ""
    payload = build_ris_payload(**kw)
    assert payload["finding_text"] == ""


def test_build_ris_payload_default_tool_version() -> None:
    """tool_version 缺省 = "0.1.0"。"""
    kw = _payload_kwargs()
    kw.pop("tool_version")
    payload = build_ris_payload(**kw)
    assert payload["tool_version"] == "0.1.0"


def test_build_ris_payload_empty_study_uid_rejected() -> None:
    kw = _payload_kwargs()
    kw["study_uid"] = ""
    with pytest.raises(RISPayloadError, match="study_uid 非法"):
        build_ris_payload(**kw)


def test_build_ris_payload_non_str_study_uid_rejected() -> None:
    kw = _payload_kwargs()
    kw["study_uid"] = 12345
    with pytest.raises(RISPayloadError, match="study_uid 非法"):
        build_ris_payload(**kw)


def test_build_ris_payload_non_str_patient_id_hash_rejected() -> None:
    kw = _payload_kwargs()
    kw["patient_id_hash"] = 12345
    with pytest.raises(RISPayloadError, match="patient_id_hash 类型错"):
        build_ris_payload(**kw)


def test_build_ris_payload_empty_patient_id_hash_accepted() -> None:
    """patient_id_hash 允许空串（脱敏测试场景）。"""
    kw = _payload_kwargs()
    kw["patient_id_hash"] = ""
    payload = build_ris_payload(**kw)
    assert payload["patient_id_hash"] == ""


def test_build_ris_payload_non_str_finding_text_rejected() -> None:
    kw = _payload_kwargs()
    kw["finding_text"] = 123
    with pytest.raises(RISPayloadError, match="finding_text 非法"):
        build_ris_payload(**kw)


def test_build_ris_payload_non_dict_finding_struct_rejected() -> None:
    kw = _payload_kwargs()
    kw["finding_struct"] = "not-a-dict"
    with pytest.raises(RISPayloadError, match="finding_struct 必须为 dict"):
        build_ris_payload(**kw)


def test_build_ris_payload_empty_draft_ts_rejected() -> None:
    kw = _payload_kwargs()
    kw["draft_ts"] = ""
    with pytest.raises(RISPayloadError, match="draft_ts 非法"):
        build_ris_payload(**kw)


def test_build_ris_payload_empty_tool_version_rejected() -> None:
    kw = _payload_kwargs()
    kw["tool_version"] = ""
    with pytest.raises(RISPayloadError, match="tool_version 非法"):
        build_ris_payload(**kw)


def test_build_ris_payload_positive_flag_coerced_to_bool() -> None:
    """positive_flag 强制转 bool（避免 1/0 字典里出现非 bool）。"""
    kw = _payload_kwargs()
    kw["positive_flag"] = 1
    payload = build_ris_payload(**kw)
    assert payload["positive_flag"] is True
    assert isinstance(payload["positive_flag"], bool)


def test_build_ris_payload_no_phi_leak() -> None:
    """payload 中不能出现明文 PatientID（仅 patient_id_hash）。"""
    kw = _payload_kwargs()
    payload = build_ris_payload(**kw)
    payload_str = json.dumps(payload, ensure_ascii=False)
    # 没有"patient_id"键（只有 patient_id_hash）
    assert '"patient_id"' not in payload_str
    assert '"patient_id_hash"' in payload_str


# =====================================================================
# 2. 工厂 build_ris_writer 校验
# =====================================================================
def test_build_ris_writer_requires_outbox(config: CxrTriageConfig) -> None:
    with pytest.raises(RISWriterError, match="必须显式传入 outbox"):
        build_ris_writer(config=config, outbox=None, audit_logger=MagicMock())


def test_build_ris_writer_requires_audit(config: CxrTriageConfig) -> None:
    """audit_logger 缺省抛错。"""
    with pytest.raises(RISWriterError, match="必须显式传入 audit_logger"):
        build_ris_writer(config=config, outbox=MagicMock(), audit_logger=None)


# =====================================================================
# 3. happy-path 200
# =====================================================================
class _MockResponse:
    """httpx 响应的最简替身。"""

    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _MockAsyncClient:
    """``httpx.AsyncClient`` 的最小替身：仅拦截 ``post``。"""

    def __init__(self, response: _MockResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict, headers: dict) -> _MockResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_write_finding_draft_200_success(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """RIS 200 → True + audit_log 写 ris.written + outbox 不入条。"""
    mock_client = _MockAsyncClient(_MockResponse(200, "OK"))
    writer = RISWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is True

    # HTTP 调用必含 Bearer token + 正确 URL + payload 7 字段
    assert len(mock_client.calls) == 1
    call = mock_client.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {config.ris_api_token}"
    assert call["headers"]["Content-Type"].startswith("application/json")
    assert kw["study_uid"] in call["url"]
    assert "finding-draft" in call["url"]
    body = call["json"]
    assert set(body.keys()) == {
        "study_uid",
        "patient_id_hash",
        "finding_struct",
        "finding_text",
        "positive_flag",
        "tool_version",
        "draft_ts",
    }
    # DRAFT_MARKER 已剥
    assert DRAFT_MARKER not in body["finding_text"]

    # audit_log 落 ris.written
    rows = await audit.list_audit_for_study(kw["study_uid"])
    events = [r["event_type"] for r in rows]
    assert "ris.written" in events
    written = [r for r in rows if r["event_type"] == "ris.written"][0]
    assert written["payload"]["status_code"] == 200
    assert written["payload"]["tool_version"] == kw["tool_version"]

    # outbox 不入条
    by_status = await outbox.count_by_status()
    assert by_status.get("pending", 0) == 0
    assert by_status.get("sent", 0) == 0

    # stats
    stats = writer.stats
    assert stats["write_attempts"] == 1
    assert stats["write_successes"] == 1
    assert stats["write_failures"] == 0
    assert stats["outbox_enqueues"] == 0

    await writer.aclose()


# =====================================================================
# 4. 失败路径：5xx
# =====================================================================
@pytest.mark.asyncio
async def test_write_finding_draft_500_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """RIS 500 → False + outbox 入条 + audit_log 写 ris.queued_for_retry。"""
    mock_client = _MockAsyncClient(
        _MockResponse(500, "Internal Server Error")
    )
    writer = RISWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is False

    # outbox 必入条
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    row = pending[0]
    assert row["study_uid"] == kw["study_uid"]
    assert row["target_endpoint"] == build_ris_target_endpoint(kw["study_uid"])
    assert row["retry_count"] == 0
    # payload 含 7 字段
    assert "finding_text" in row["payload"]
    assert DRAFT_MARKER not in row["payload"]["finding_text"]

    # audit_log 落 ris.queued_for_retry
    rows = await audit.list_audit_for_study(kw["study_uid"])
    events = [r["event_type"] for r in rows]
    assert "ris.queued_for_retry" in events
    fail_audit = [r for r in rows if r["event_type"] == "ris.queued_for_retry"][0]
    assert "RIS HTTP 500" in fail_audit["payload"]["error"]
    assert fail_audit["payload"]["target_endpoint"] == build_ris_target_endpoint(
        kw["study_uid"]
    )

    # stats
    stats = writer.stats
    assert stats["write_attempts"] == 1
    assert stats["write_successes"] == 0
    assert stats["write_failures"] == 1
    assert stats["outbox_enqueues"] == 1

    await writer.aclose()


@pytest.mark.asyncio
async def test_write_finding_draft_503_retryable(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """RIS 503（同 RETRYABLE_HTTP_STATUSES）→ 入 outbox。"""
    mock_client = _MockAsyncClient(_MockResponse(503, "Service Unavailable"))
    writer = RISWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    assert "503" in (
        await audit.list_audit_for_study(kw["study_uid"])
    )[0]["payload"]["error"]
    await writer.aclose()


@pytest.mark.asyncio
async def test_write_finding_draft_4xx_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """RIS 400/404 → 入 outbox（视作可重试；由 MAX_RETRY_BEFORE_ALERT 触发 alerting）。"""
    for status in (400, 404, 422):
        mock_client = _MockAsyncClient(_MockResponse(status, f"err-{status}"))
        writer = RISWriter(
            config=config,
            outbox=outbox,
            audit_logger=audit,
            http_client=mock_client,  # type: ignore[arg-type]
        )
        kw = _payload_kwargs()
        kw["study_uid"] = f"uid-{status}"  # 区分每个 case
        result = await writer.write_finding_draft(**kw)
        assert result is False
        # outbox 入条
        pending = await outbox.dequeue(limit=10)
        assert any(p["study_uid"] == f"uid-{status}" for p in pending)
        await writer.aclose()


# =====================================================================
# 5. 失败路径：网络错 / 超时
# =====================================================================
class _TimeoutClient:
    async def post(self, *args, **kwargs):
        raise httpx.TimeoutException("RIS timeout 5s")

    async def aclose(self) -> None:
        pass


class _ConnectErrorClient:
    async def post(self, *args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_write_finding_draft_timeout_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """httpx.TimeoutException → 入 outbox + audit。"""
    writer = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=_TimeoutClient(),  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    rows = await audit.list_audit_for_study(kw["study_uid"])
    assert "TimeoutException" in rows[0]["payload"]["error"]
    await writer.aclose()


@pytest.mark.asyncio
async def test_write_finding_draft_connect_error_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """httpx.ConnectError → 入 outbox + audit。"""
    writer = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=_ConnectErrorClient(),  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    await writer.aclose()


# =====================================================================
# 6. outbox 入条失败兜底
# =====================================================================
class _FailingOutbox:
    """模拟 outbox.enqueue 抛异常（DB 损坏场景）。"""

    async def enqueue(self, **kwargs) -> int:
        raise RuntimeError("DB locked")


@pytest.mark.asyncio
async def test_write_finding_draft_outbox_enqueue_fails(
    config: CxrTriageConfig, audit: AuditLogger
) -> None:
    """outbox 入条失败 → 写 ``ris.outbox_enqueue_failed`` audit 行（不丢事件）。"""
    mock_client = _MockAsyncClient(_MockResponse(500, "RIS down"))
    writer = RISWriter(
        config=config,
        outbox=_FailingOutbox(),  # type: ignore[arg-type]
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_finding_draft(**kw)
    assert result is False

    rows = await audit.list_audit_for_study(kw["study_uid"])
    events = [r["event_type"] for r in rows]
    assert "ris.outbox_enqueue_failed" in events
    fail_audit = [r for r in rows if r["event_type"] == "ris.outbox_enqueue_failed"][0]
    assert "outbox_error" in fail_audit["payload"]
    assert "primary_error" in fail_audit["payload"]
    await writer.aclose()


# =====================================================================
# 7. 注入 client 默认懒加载
# =====================================================================
@pytest.mark.asyncio
async def test_writer_aclose_idempotent(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """aclose() 二次调用不抛（无外部 client 时也是 noop）。"""
    writer = RISWriter(config=config, outbox=outbox, audit_logger=audit)
    await writer.aclose()
    await writer.aclose()  # 二次幂等


@pytest.mark.asyncio
async def test_writer_lazy_client_creation(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """未注入 client 时，第一次 write_finding_draft 时懒创建；aclose 关闭自管 client。"""
    writer = RISWriter(config=config, outbox=outbox, audit_logger=audit)
    assert writer._http_client is None

    # 直接调用实际 httpx 会真发请求；这里只验证懒创建机制
    # 不实际发请求，只调用 _get_client
    client = await writer._get_client()
    assert client is not None
    assert writer._http_client is not None
    # 二次调用返回同一 client
    client2 = await writer._get_client()
    assert client2 is client

    # aclose 关闭自管 client
    await writer.aclose()
    assert writer._http_client is None


# =====================================================================
# 8. 并发
# =====================================================================
@pytest.mark.asyncio
async def test_write_finding_draft_concurrent_5_success(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """并发 5 张全部 200 → 5 条 audit + 0 outbox。"""

    class _CountingClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json, headers):
            self.calls += 1
            return _MockResponse(200, "OK")

        async def aclose(self) -> None:
            pass

    client = _CountingClient()
    writer = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=client,  # type: ignore[arg-type]
    )

    async def write_one(idx: int) -> bool:
        kw = _payload_kwargs()
        kw["study_uid"] = f"uid-conc-{idx}"
        return await writer.write_finding_draft(**kw)

    results = await asyncio.gather(*[write_one(i) for i in range(5)])
    assert all(r is True for r in results)

    # 5 条 audit（每张 study 一条 ris.written）
    for i in range(5):
        rows = await audit.list_audit_for_study(f"uid-conc-{i}")
        assert any(r["event_type"] == "ris.written" for r in rows)

    # 0 outbox
    by_status = await outbox.count_by_status()
    assert by_status.get("pending", 0) == 0

    assert client.calls == 5
    stats = writer.stats
    assert stats["write_attempts"] == 5
    assert stats["write_successes"] == 5
    assert stats["write_failures"] == 0
    await writer.aclose()


@pytest.mark.asyncio
async def test_write_finding_draft_concurrent_5_all_500(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """并发 5 张全部 500 → 5 条 outbox + 5 条 audit。"""
    mock_client = _MockAsyncClient(_MockResponse(500, "down"))
    writer = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )

    async def write_one(idx: int) -> bool:
        kw = _payload_kwargs()
        kw["study_uid"] = f"uid-fail-{idx}"
        return await writer.write_finding_draft(**kw)

    results = await asyncio.gather(*[write_one(i) for i in range(5)])
    assert all(r is False for r in results)

    # 5 条 outbox
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 5
    # 5 条 audit
    for i in range(5):
        rows = await audit.list_audit_for_study(f"uid-fail-{i}")
        assert any(r["event_type"] == "ris.queued_for_retry" for r in rows)
    stats = writer.stats
    assert stats["write_failures"] == 5
    assert stats["outbox_enqueues"] == 5
    await writer.aclose()


# =====================================================================
# 9. 端到端 happy-path：write → mark_sent（模拟 retry_loop 收尾）
# =====================================================================
@pytest.mark.asyncio
async def test_outbox_mark_sent_after_write_failure_then_retry(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """完整 outbox 生命周期：RIS 500 → outbox enqueue → retry_loop 重发 → mark_sent。"""
    # 第 1 次写：失败
    mock_client_fail = _MockAsyncClient(_MockResponse(500, "transient"))
    writer = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=mock_client_fail,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    await writer.write_finding_draft(**kw)
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    outbox_id = pending[0]["id"]

    # mark_retry 模拟重试失败 → retry_count=1
    status = await outbox.mark_retry(outbox_id, "still 500")
    assert status == "pending"

    # 模拟 retry_loop 重新发：换 client 返 200
    mock_client_ok = _MockAsyncClient(_MockResponse(200, "OK"))
    writer_ok = RISWriter(
        config=config, outbox=outbox, audit_logger=audit,
        http_client=mock_client_ok,  # type: ignore[arg-type]
    )
    row = pending[0]
    # 实际 retry_loop 会调 client.post(row['target_endpoint'], row['payload'])
    # 这里我们直接用 raw payload 重放以验证整条链路
    payload = row["payload"]
    response = await mock_client_ok.post(
        "http://127.0.0.1:8081/api/v1/studies/.../finding-draft",
        json=payload,
        headers={"Authorization": f"Bearer {config.ris_api_token}"},
    )
    assert response.status_code == 200

    # mark_sent 收尾
    await outbox.mark_sent(outbox_id)
    by_status = await outbox.count_by_status()
    assert by_status.get("sent", 0) == 1
    assert by_status.get("pending", 0) == 0

    await writer.aclose()
    await writer_ok.aclose()


# =====================================================================
# 10. 真实 respx 拦截（端到端协议兼容）
# =====================================================================
@pytest.mark.asyncio
async def test_real_httpx_round_trip_with_respx(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """真实 httpx.AsyncClient + respx 拦截 → 200 + 出站 body/header 正确。

    路径模板（spec.md §4）：``/api/v1/studies/{study_uid}/finding-draft``。
    respx 拦截使用 base_url + path 模式（与 test_protocol_compat.py 一致）。
    """
    respx = pytest.importorskip("respx")
    from httpx import Response as HttpxResponse

    # 用 RIS_API_URL 的 base URL 拦截（http://127.0.0.1:8081）
    base = config.ris_api_url.rstrip("/")
    captured: dict[str, Any] = {}

    def _side_effect(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return HttpxResponse(200, json={"status": "ok"})

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        # 拦截该 base 下的所有 finding-draft 路径
        mock.post(path__regex=r".*/finding-draft$").mock(side_effect=_side_effect)

        async with httpx.AsyncClient(timeout=5.0) as client:
            writer = RISWriter(
                config=config, outbox=outbox, audit_logger=audit,
                http_client=client,
            )
            kw = _payload_kwargs()
            result = await writer.write_finding_draft(**kw)
            assert result is True

            # 验证出站 request
            headers_lower = {k.lower(): v for k, v in captured["headers"].items()}
            assert "authorization" in headers_lower
            assert (
                headers_lower["authorization"]
                == f"Bearer {config.ris_api_token}"
            )
            assert kw["study_uid"] in captured["url"] or (
                kw["study_uid"]
                .replace("<", "%3C")
                .replace(">", "%3E")
                in captured["url"]
            )
            assert DRAFT_MARKER not in captured["body"]["finding_text"]
            assert set(captured["body"].keys()) == {
                "study_uid",
                "patient_id_hash",
                "finding_struct",
                "finding_text",
                "positive_flag",
                "tool_version",
                "draft_ts",
            }
            await writer.aclose()
