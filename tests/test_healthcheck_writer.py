"""tests/test_healthcheck_writer.py — 体检主检 summary-ref 写回客户端（httpx 异步 + outbox 兜底）。

覆盖（CLAUDE.md Phase 4 验收 · Task 24）：

1. **payload 构造**：
   - 4 字段对齐 spec.md §4（visit_id / summary_snippet / source_study_uids /
     model_version）；
   - summary_snippet 空字符串 / 全空白拒绝；
   - source_study_uids 必须非空 list + 元素必须为非空 str；
   - source_study_uids 超过 MAX_SOURCE_STUDY_UIDS 拒绝；
   - 缺字段 / 错类型抛 ``HealthcheckPayloadError``。
2. **URL 构造**：base_url 末斜杠 trim；visit_id 缺失抛错。
3. **make_summary_snippet**：自动剥 DRAFT_MARKER、阳性"可能"语气、
   阴性"未见明显异常"、空 finding_text 兜底、extra_suggestion 拼接。
4. **happy-path 200**：mock httpx → 体检主检 200 → ``write_summary_ref`` 返
   True + audit_log 写 ``healthcheck.written`` 行 + outbox **不**入条。
5. **失败路径 5xx**：mock 500 → 返 False + outbox **入条**（target_endpoint
   以 ``hc://`` 前缀区分 RIS）+ audit_log 写 ``healthcheck.queued_for_retry``。
6. **失败路径 网络错**：httpx.TimeoutException / ConnectError → 同样入
   outbox + audit。
7. **失败路径 4xx**：400 / 404 → 入 outbox + audit。
8. **不静默丢数据**：失败必入 outbox；outbox 入条失败 → 写
   ``healthcheck.outbox_enqueue_failed`` audit 行（不让事件失踪）。
9. **Bearer Token 注入**：从 config.healthcheck_api_token 读；出站 header
   必含 ``Authorization: Bearer <token>``。
10. **visit_id 隔离**：payload 中无明文 PatientID；只含 visit_id 间接标识。
11. **并发**：5 个 visitId 并发写 → 全部 200 成功 + 5 条 audit + 0 outbox；
    5 个全部 500 → 5 条 outbox + 5 条 audit。
12. **stats 计数**：attempts / successes / failures / outbox_enqueues
    累加正确。
13. **工厂 build_healthcheck_writer**：缺 outbox / audit_logger → 抛
    ``HealthcheckWriterError``；显式注入可用。
14. **aclose 幂等**：二次关闭不抛。
15. **target_endpoint 前缀区分**：与 RISWriter 共享同一 outbox 表，
    retry_loop 据此前缀路由。
16. **真实 respx 拦截**：端到端协议兼容 + 出站 body/header 正确。
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
from cxr_triage.writers.healthcheck_writer import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_TIMEOUT_SECONDS,
    EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED,
    EVENT_HEALTHCHECK_QUEUED,
    EVENT_HEALTHCHECK_WRITTEN,
    HEALTHCHECK_TARGET_PREFIX,
    MAX_SOURCE_STUDY_UIDS,
    SUMMARY_SNIPPET_PREFIX,
    HealthcheckPayloadError,
    HealthcheckWriter,
    HealthcheckWriterError,
    build_healthcheck_payload,
    build_healthcheck_summary_ref_url,
    build_healthcheck_target_endpoint,
    build_healthcheck_writer,
    make_summary_snippet,
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
    """一组典型 payload 输入（draft 形态，喂给 build_healthcheck_payload）。"""
    return {
        "visit_id": "HC-20261013-00001",
        "summary_snippet": (
            "胸片：右下肺野见少量气胸征象（0.87），建议进一步 CT。"
        ),
        "source_study_uids": [
            "1.2.826.0.1.3680043.8.498.<EXAMPLE_HC>",
        ],
        "model_version": "0.1.0",
    }


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "healthcheck_writer_test.db"))
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
def test_healthcheck_target_prefix_constant() -> None:
    """HEALTHCHECK_TARGET_PREFIX 用于 outbox target_endpoint 区分 RIS vs 主检。"""
    assert HEALTHCHECK_TARGET_PREFIX == "hc://summary-ref/"


def test_default_timeout_constant() -> None:
    """DEFAULT_TIMEOUT_SECONDS=5s（医院内网通常 2s 内，5s 兜底）。"""
    assert DEFAULT_TIMEOUT_SECONDS == 5.0


def test_default_model_version_constant() -> None:
    """DEFAULT_MODEL_VERSION=0.1.0（与 RIS 同步）。"""
    assert DEFAULT_MODEL_VERSION == "0.1.0"


def test_summary_snippet_prefix_constant() -> None:
    """SUMMARY_SNIPPET_PREFIX='胸片：' 锁定（caller 拼 snippet 参照）。"""
    assert SUMMARY_SNIPPET_PREFIX == "胸片："


def test_max_source_study_uids_constant() -> None:
    """MAX_SOURCE_STUDY_UIDS=50（防异常 case 把大量 UIDs 推到体检主检）。"""
    assert MAX_SOURCE_STUDY_UIDS == 50


def test_build_healthcheck_summary_ref_url_basic() -> None:
    """URL 拼装：base_url + 端点模板，visit_id 嵌入路径。"""
    url = build_healthcheck_summary_ref_url(
        "http://127.0.0.1:8082", "HC-20261013-00001"
    )
    assert url == (
        "http://127.0.0.1:8082/api/v1/healthcheck/"
        "HC-20261013-00001/summary-ref"
    )


def test_build_healthcheck_summary_ref_url_strip_trailing_slash() -> None:
    """base_url 末斜杠必须 trim，避免双斜杠。"""
    url = build_healthcheck_summary_ref_url(
        "http://127.0.0.1:8082/", "HC-VISIT-1"
    )
    assert url == "http://127.0.0.1:8082/api/v1/healthcheck/HC-VISIT-1/summary-ref"


def test_build_healthcheck_summary_ref_url_empty_base() -> None:
    with pytest.raises(HealthcheckPayloadError, match="Healthcheck base_url 为空"):
        build_healthcheck_summary_ref_url("", "HC-VISIT-1")


def test_build_healthcheck_summary_ref_url_empty_visit_id() -> None:
    with pytest.raises(HealthcheckPayloadError, match="visit_id 为空"):
        build_healthcheck_summary_ref_url("http://127.0.0.1:8082", "")


def test_build_healthcheck_target_endpoint() -> None:
    """outbox target_endpoint 含 hc:// 前缀 + visit_id（便于 retry_loop 路由）。"""
    ep = build_healthcheck_target_endpoint("HC-VISIT-XYZ")
    assert ep == "hc://summary-ref/HC-VISIT-XYZ"


def test_build_healthcheck_target_endpoint_diff_from_ris() -> None:
    """与 RISWriter 的 ris:// 前缀区分；同一 visitId vs 同一 study_uid
    可在 outbox 共存不冲突。"""
    hc_ep = build_healthcheck_target_endpoint("HC-1")
    # 与 RIS 不前缀冲突
    assert not hc_ep.startswith("ris://")


def test_build_healthcheck_payload_happy_path() -> None:
    """4 字段对齐 spec.md §4；summary_snippet 必含。"""
    kw = _payload_kwargs()
    payload = build_healthcheck_payload(**kw)
    assert payload == {
        "visit_id": kw["visit_id"],
        "summary_snippet": kw["summary_snippet"],
        "source_study_uids": kw["source_study_uids"],
        "model_version": kw["model_version"],
    }


def test_build_healthcheck_payload_default_model_version() -> None:
    """model_version 缺省 = DEFAULT_MODEL_VERSION（"0.1.0"）。"""
    kw = _payload_kwargs()
    kw.pop("model_version")
    payload = build_healthcheck_payload(**kw)
    assert payload["model_version"] == DEFAULT_MODEL_VERSION


def test_build_healthcheck_payload_multi_uid() -> None:
    """source_study_uids 可多条（同一 visitId 下多张胸片）。"""
    kw = _payload_kwargs()
    kw["source_study_uids"] = [
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_HC_1>",
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_HC_2>",
    ]
    payload = build_healthcheck_payload(**kw)
    assert len(payload["source_study_uids"]) == 2


def test_build_healthcheck_payload_empty_visit_id_rejected() -> None:
    kw = _payload_kwargs()
    kw["visit_id"] = ""
    with pytest.raises(HealthcheckPayloadError, match="visit_id 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_non_str_visit_id_rejected() -> None:
    kw = _payload_kwargs()
    kw["visit_id"] = 12345
    with pytest.raises(HealthcheckPayloadError, match="visit_id 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_empty_snippet_rejected() -> None:
    """体检主检拒绝空 snippet（信息科对接规范）。"""
    kw = _payload_kwargs()
    kw["summary_snippet"] = ""
    with pytest.raises(HealthcheckPayloadError, match="summary_snippet 为空"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_whitespace_only_snippet_rejected() -> None:
    """全空白 snippet 视为空（strip 后判定）。"""
    kw = _payload_kwargs()
    kw["summary_snippet"] = "   \t  \n  "
    with pytest.raises(HealthcheckPayloadError, match="summary_snippet 为空"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_non_str_snippet_rejected() -> None:
    kw = _payload_kwargs()
    kw["summary_snippet"] = 123
    with pytest.raises(HealthcheckPayloadError, match="summary_snippet 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_none_snippet_rejected() -> None:
    kw = _payload_kwargs()
    kw["summary_snippet"] = None
    with pytest.raises(HealthcheckPayloadError, match="summary_snippet 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_empty_uids_rejected() -> None:
    """source_study_uids 至少 1 条。"""
    kw = _payload_kwargs()
    kw["source_study_uids"] = []
    with pytest.raises(
        HealthcheckPayloadError, match="source_study_uids 为空"
    ):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_non_list_uids_rejected() -> None:
    kw = _payload_kwargs()
    kw["source_study_uids"] = "not-a-list"
    with pytest.raises(
        HealthcheckPayloadError, match="source_study_uids 必须为 list"
    ):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_uid_with_empty_str_rejected() -> None:
    kw = _payload_kwargs()
    kw["source_study_uids"] = ["valid-uid", ""]
    with pytest.raises(
        HealthcheckPayloadError, match="source_study_uids 含非法元素"
    ):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_uid_non_str_rejected() -> None:
    kw = _payload_kwargs()
    kw["source_study_uids"] = ["valid-uid", 12345]
    with pytest.raises(
        HealthcheckPayloadError, match="source_study_uids 含非法元素"
    ):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_too_many_uids_rejected() -> None:
    """超过 MAX_SOURCE_STUDY_UIDS 上限拒绝。"""
    kw = _payload_kwargs()
    kw["source_study_uids"] = [f"uid-{i}" for i in range(MAX_SOURCE_STUDY_UIDS + 1)]
    with pytest.raises(
        HealthcheckPayloadError, match="source_study_uids 超上限"
    ):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_max_uids_accepted() -> None:
    """恰好 MAX_SOURCE_STUDY_UIDS 接受。"""
    kw = _payload_kwargs()
    kw["source_study_uids"] = [f"uid-{i}" for i in range(MAX_SOURCE_STUDY_UIDS)]
    payload = build_healthcheck_payload(**kw)
    assert len(payload["source_study_uids"]) == MAX_SOURCE_STUDY_UIDS


def test_build_healthcheck_payload_empty_model_version_rejected() -> None:
    kw = _payload_kwargs()
    kw["model_version"] = ""
    with pytest.raises(HealthcheckPayloadError, match="model_version 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_non_str_model_version_rejected() -> None:
    kw = _payload_kwargs()
    kw["model_version"] = 0.1
    with pytest.raises(HealthcheckPayloadError, match="model_version 非法"):
        build_healthcheck_payload(**kw)


def test_build_healthcheck_payload_no_phi_leak() -> None:
    """payload 中不能出现明文 PatientID / AccessionNumber（仅 visit_id + UIDs）。"""
    kw = _payload_kwargs()
    payload = build_healthcheck_payload(**kw)
    payload_str = json.dumps(payload, ensure_ascii=False)
    # 没有"patient_id" / "accession" 键
    assert '"patient_id"' not in payload_str
    assert '"accession"' not in payload_str
    assert '"visit_id"' in payload_str


# =====================================================================
# 2. make_summary_snippet（caller 辅助函数）
# =====================================================================
def test_make_summary_snippet_positive_with_prob() -> None:
    """阳性：所见 + 概率 + 建议。"""
    snippet = make_summary_snippet(
        finding_text="右下肺野见少量气胸征象",
        max_prob=0.87,
        positive_flag=True,
        extra_suggestion="建议进一步 CT",
    )
    assert snippet.startswith(SUMMARY_SNIPPET_PREFIX)
    assert "右下肺野见少量气胸征象" in snippet
    assert "0.87" in snippet
    assert "建议进一步 CT" in snippet
    assert snippet.endswith("。")


def test_make_summary_snippet_negative_no_prob() -> None:
    """阴性：不带概率 + "未见明显异常"语气。"""
    snippet = make_summary_snippet(
        finding_text="双肺野清晰，心影不大",
        max_prob=0.10,
        positive_flag=False,
    )
    assert snippet.startswith(SUMMARY_SNIPPET_PREFIX)
    assert "双肺野清晰" in snippet
    # 阴性不应带概率括号
    assert "（0.10）" not in snippet


def test_make_summary_snippet_strips_draft_marker() -> None:
    """DRAFT_MARKER 自动剥（避免"草稿"两字进体检主检）。"""
    finding = "右下肺野见少量气胸征象（草稿 / 未经医师确认）"
    snippet = make_summary_snippet(
        finding_text=finding,
        max_prob=0.85,
        positive_flag=True,
    )
    assert "草稿" not in snippet
    assert "未经医师确认" not in snippet


def test_make_summary_snippet_empty_finding_text_falls_back() -> None:
    """空 finding_text 兜底（阴性 → "未见明显异常"；阳性 → "请人工复核"）。"""
    snippet_neg = make_summary_snippet(
        finding_text="", max_prob=0.10, positive_flag=False
    )
    assert "未见明显异常" in snippet_neg
    snippet_pos = make_summary_snippet(
        finding_text="", max_prob=0.85, positive_flag=True
    )
    assert "请人工复核" in snippet_pos


def test_make_summary_snippet_empty_extra_suggestion() -> None:
    """空 extra_suggestion 不写尾部。"""
    snippet = make_summary_snippet(
        finding_text="右下肺野见少量气胸征象",
        max_prob=0.87,
        positive_flag=True,
        extra_suggestion="",
    )
    # 没有 "，，"
    assert "，，。" not in snippet
    assert snippet.endswith("。")


def test_make_summary_snippet_no_diagnosis_or_treatment() -> None:
    """安全边界：snippet 不写诊断/治疗建议（CLAUDE.md 全局安全边界）。"""
    snippet = make_summary_snippet(
        finding_text="右下肺野见少量气胸征象",
        max_prob=0.87,
        positive_flag=True,
        extra_suggestion="建议进一步 CT",
    )
    # 不出现诊断/治疗敏感词
    assert "诊断为" not in snippet
    assert "确诊" not in snippet
    assert "处方" not in snippet
    assert "治疗" not in snippet


# =====================================================================
# 3. 工厂 build_healthcheck_writer 校验
# =====================================================================
def test_build_healthcheck_writer_requires_outbox(
    config: CxrTriageConfig,
) -> None:
    with pytest.raises(
        HealthcheckWriterError, match="必须显式传入 outbox"
    ):
        build_healthcheck_writer(
            config=config, outbox=None, audit_logger=MagicMock()
        )


def test_build_healthcheck_writer_requires_audit(
    config: CxrTriageConfig,
) -> None:
    """audit_logger 缺省抛错。"""
    with pytest.raises(
        HealthcheckWriterError, match="必须显式传入 audit_logger"
    ):
        build_healthcheck_writer(
            config=config, outbox=MagicMock(), audit_logger=None
        )


# =====================================================================
# 4. happy-path 200
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
async def test_write_summary_ref_200_success(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """体检主检 200 → True + audit_log 写 healthcheck.written + outbox 不入条。"""
    mock_client = _MockAsyncClient(_MockResponse(200, "OK"))
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw, study_uid="study-1")
    assert result is True

    # HTTP 调用必含 Bearer token + 正确 URL + payload 4 字段
    assert len(mock_client.calls) == 1
    call = mock_client.calls[0]
    assert (
        call["headers"]["Authorization"]
        == f"Bearer {config.healthcheck_api_token}"
    )
    assert call["headers"]["Content-Type"].startswith("application/json")
    assert kw["visit_id"] in call["url"]
    assert "summary-ref" in call["url"]
    body = call["json"]
    assert set(body.keys()) == {
        "visit_id",
        "summary_snippet",
        "source_study_uids",
        "model_version",
    }
    assert body["visit_id"] == kw["visit_id"]
    assert body["summary_snippet"] == kw["summary_snippet"]
    assert body["source_study_uids"] == kw["source_study_uids"]
    assert body["model_version"] == kw["model_version"]

    # audit_log 落 healthcheck.written
    rows = await audit.list_audit_for_study("study-1")
    events = [r["event_type"] for r in rows]
    assert EVENT_HEALTHCHECK_WRITTEN in events
    written = [
        r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_WRITTEN
    ][0]
    assert written["payload"]["status_code"] == 200
    assert written["payload"]["visit_id"] == kw["visit_id"]
    assert written["actor_role"] == "sr"

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


@pytest.mark.asyncio
async def test_write_summary_ref_200_without_study_uid(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """study_uid 可空 → audit 用 visit_id 兜底。"""
    mock_client = _MockAsyncClient(_MockResponse(200, "OK"))
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw)  # 不传 study_uid
    assert result is True
    rows = await audit.list_audit_for_study(kw["visit_id"])
    assert any(r["event_type"] == EVENT_HEALTHCHECK_WRITTEN for r in rows)
    await writer.aclose()


# =====================================================================
# 5. 失败路径：5xx
# =====================================================================
@pytest.mark.asyncio
async def test_write_summary_ref_500_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """体检主检 500 → False + outbox 入条 + audit_log 写 healthcheck.queued_for_retry。"""
    mock_client = _MockAsyncClient(_MockResponse(500, "Internal Server Error"))
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw, study_uid="study-500")
    assert result is False

    # outbox 必入条（hc:// 前缀）
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    row = pending[0]
    assert row["study_uid"] == "study-500"
    assert row["target_endpoint"] == build_healthcheck_target_endpoint(
        kw["visit_id"]
    )
    assert row["retry_count"] == 0
    # payload 含 4 字段
    assert "summary_snippet" in row["payload"]
    assert row["payload"]["visit_id"] == kw["visit_id"]

    # audit_log 落 healthcheck.queued_for_retry
    rows = await audit.list_audit_for_study("study-500")
    events = [r["event_type"] for r in rows]
    assert EVENT_HEALTHCHECK_QUEUED in events
    fail_audit = [
        r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_QUEUED
    ][0]
    assert "Healthcheck HTTP 500" in fail_audit["payload"]["error"]
    assert (
        fail_audit["payload"]["target_endpoint"]
        == build_healthcheck_target_endpoint(kw["visit_id"])
    )
    assert fail_audit["actor_role"] == "sr"

    # stats
    stats = writer.stats
    assert stats["write_attempts"] == 1
    assert stats["write_successes"] == 0
    assert stats["write_failures"] == 1
    assert stats["outbox_enqueues"] == 1

    await writer.aclose()


@pytest.mark.asyncio
async def test_write_summary_ref_503_retryable(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """体检主检 503 → 入 outbox。"""
    mock_client = _MockAsyncClient(_MockResponse(503, "Service Unavailable"))
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    kw["visit_id"] = "HC-VISIT-503"
    result = await writer.write_summary_ref(**kw, study_uid="study-503")
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    assert "503" in (
        await audit.list_audit_for_study("study-503")
    )[0]["payload"]["error"]
    await writer.aclose()


@pytest.mark.asyncio
async def test_write_summary_ref_4xx_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """体检主检 400/404 → 入 outbox（视作可重试）。"""
    for status in (400, 404, 422):
        mock_client = _MockAsyncClient(_MockResponse(status, f"err-{status}"))
        writer = HealthcheckWriter(
            config=config,
            outbox=outbox,
            audit_logger=audit,
            http_client=mock_client,  # type: ignore[arg-type]
        )
        kw = _payload_kwargs()
        kw["visit_id"] = f"HC-VISIT-{status}"
        result = await writer.write_summary_ref(**kw, study_uid=f"study-{status}")
        assert result is False
        # outbox 入条
        pending = await outbox.dequeue(limit=10)
        assert any(p["study_uid"] == f"study-{status}" for p in pending)
        await writer.aclose()


# =====================================================================
# 6. 失败路径：网络错 / 超时
# =====================================================================
class _TimeoutClient:
    async def post(self, *args, **kwargs):
        raise httpx.TimeoutException("Healthcheck timeout 5s")

    async def aclose(self) -> None:
        pass


class _ConnectErrorClient:
    async def post(self, *args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_write_summary_ref_timeout_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """httpx.TimeoutException → 入 outbox + audit。"""
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=_TimeoutClient(),  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw, study_uid="study-to")
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    rows = await audit.list_audit_for_study("study-to")
    assert "TimeoutException" in rows[0]["payload"]["error"]
    await writer.aclose()


@pytest.mark.asyncio
async def test_write_summary_ref_connect_error_failure(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """httpx.ConnectError → 入 outbox + audit。"""
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=_ConnectErrorClient(),  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw, study_uid="study-ce")
    assert result is False
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    await writer.aclose()


# =====================================================================
# 7. outbox 入条失败兜底
# =====================================================================
class _FailingOutbox:
    """模拟 outbox.enqueue 抛异常（DB 损坏场景）。"""

    async def enqueue(self, **kwargs) -> int:
        raise RuntimeError("DB locked")


@pytest.mark.asyncio
async def test_write_summary_ref_outbox_enqueue_fails(
    config: CxrTriageConfig, audit: AuditLogger
) -> None:
    """outbox 入条失败 → 写 healthcheck.outbox_enqueue_failed audit 行（不丢事件）。"""
    mock_client = _MockAsyncClient(_MockResponse(500, "Healthcheck down"))
    writer = HealthcheckWriter(
        config=config,
        outbox=_FailingOutbox(),  # type: ignore[arg-type]
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )
    kw = _payload_kwargs()
    result = await writer.write_summary_ref(**kw, study_uid="study-oef")
    assert result is False

    rows = await audit.list_audit_for_study("study-oef")
    events = [r["event_type"] for r in rows]
    assert EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED in events
    fail_audit = [
        r
        for r in rows
        if r["event_type"] == EVENT_HEALTHCHECK_OUTBOX_ENQUEUE_FAILED
    ][0]
    assert "outbox_error" in fail_audit["payload"]
    assert "primary_error" in fail_audit["payload"]
    assert fail_audit["actor_role"] == "sr"
    await writer.aclose()


# =====================================================================
# 8. 注入 client 默认懒加载
# =====================================================================
@pytest.mark.asyncio
async def test_writer_aclose_idempotent(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """aclose() 二次调用不抛（无外部 client 时也是 noop）。"""
    writer = HealthcheckWriter(config=config, outbox=outbox, audit_logger=audit)
    await writer.aclose()
    await writer.aclose()  # 二次幂等


@pytest.mark.asyncio
async def test_writer_lazy_client_creation(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """未注入 client 时，第一次 write_summary_ref 时懒创建；aclose 关闭自管 client。"""
    writer = HealthcheckWriter(config=config, outbox=outbox, audit_logger=audit)
    assert writer._http_client is None

    # 直接调用实际 httpx 会真发请求；这里只验证懒创建机制
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
# 9. 并发
# =====================================================================
@pytest.mark.asyncio
async def test_write_summary_ref_concurrent_5_success(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """并发 5 个 visitId 全部 200 → 5 条 audit + 0 outbox。"""

    class _CountingClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, json, headers):
            self.calls += 1
            return _MockResponse(200, "OK")

        async def aclose(self) -> None:
            pass

    client = _CountingClient()
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=client,  # type: ignore[arg-type]
    )

    async def write_one(idx: int) -> bool:
        kw = _payload_kwargs()
        kw["visit_id"] = f"HC-VISIT-CONC-{idx}"
        kw["source_study_uids"] = [f"1.2.826.0.1.3680043.8.498.<HC_{idx}>"]
        return await writer.write_summary_ref(**kw, study_uid=f"study-conc-{idx}")

    results = await asyncio.gather(*[write_one(i) for i in range(5)])
    assert all(r is True for r in results)

    # 5 条 audit
    for i in range(5):
        rows = await audit.list_audit_for_study(f"study-conc-{i}")
        assert any(
            r["event_type"] == EVENT_HEALTHCHECK_WRITTEN for r in rows
        )

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
async def test_write_summary_ref_concurrent_5_all_500(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """并发 5 个 visitId 全部 500 → 5 条 outbox + 5 条 audit。"""
    mock_client = _MockAsyncClient(_MockResponse(500, "down"))
    writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
    )

    async def write_one(idx: int) -> bool:
        kw = _payload_kwargs()
        kw["visit_id"] = f"HC-VISIT-FAIL-{idx}"
        kw["source_study_uids"] = [f"1.2.826.0.1.3680043.8.498.<HC_F_{idx}>"]
        return await writer.write_summary_ref(**kw, study_uid=f"study-fail-{idx}")

    results = await asyncio.gather(*[write_one(i) for i in range(5)])
    assert all(r is False for r in results)

    # 5 条 outbox
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 5
    # 5 条 audit
    for i in range(5):
        rows = await audit.list_audit_for_study(f"study-fail-{i}")
        assert any(
            r["event_type"] == EVENT_HEALTHCHECK_QUEUED for r in rows
        )
    stats = writer.stats
    assert stats["write_failures"] == 5
    assert stats["outbox_enqueues"] == 5
    await writer.aclose()


# =====================================================================
# 10. outbox 共存：RIS vs 主检前缀区分
# =====================================================================
@pytest.mark.asyncio
async def test_outbox_target_endpoint_prefix_diff_from_ris(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """Healthcheck 与 RIS 用同一 outbox 表，前缀区分；不互相污染。"""
    # RIS 写入（mock 500 → 入 outbox）
    from cxr_triage.writers.ris_writer import (
        RIS_TARGET_PREFIX,
        RISWriter,
        build_ris_target_endpoint,
    )

    ris_client = _MockAsyncClient(_MockResponse(500, "down"))
    ris_writer = RISWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=ris_client,  # type: ignore[arg-type]
    )
    study_uid = "1.2.826.0.1.3680043.8.498.<HC_RIS_COEXIST>"
    await ris_writer.write_finding_draft(
        study_uid=study_uid,
        patient_id_hash="hash123",
        finding_text="所见",
        finding_struct={"pathologies": [], "key_qa": [], "grounding_boxes": []},
        positive_flag=False,
        draft_ts="2026-10-13T07:23:11Z",
    )
    await ris_writer.aclose()

    # Healthcheck 也写入
    hc_client = _MockAsyncClient(_MockResponse(500, "down"))
    hc_writer = HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=hc_client,  # type: ignore[arg-type]
    )
    await hc_writer.write_summary_ref(
        visit_id="HC-VISIT-COEXIST",
        summary_snippet="胸片：未见明显异常。",
        source_study_uids=[study_uid],
        study_uid=study_uid,
    )
    await hc_writer.aclose()

    # 共存 2 条：一条 RIS 一条 HC
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 2
    ris_rows = [r for r in pending if r["target_endpoint"].startswith(RIS_TARGET_PREFIX)]
    hc_rows = [
        r
        for r in pending
        if r["target_endpoint"].startswith(HEALTHCHECK_TARGET_PREFIX)
    ]
    assert len(ris_rows) == 1
    assert len(hc_rows) == 1


# =====================================================================
# 11. 真实 respx 拦截（端到端协议兼容）
# =====================================================================
@pytest.mark.asyncio
async def test_real_httpx_round_trip_with_respx(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """真实 httpx.AsyncClient + respx 拦截 → 200 + 出站 body/header 正确。

    路径模板（spec.md §4）：``/api/v1/healthcheck/{visit_id}/summary-ref``。
    """
    respx = pytest.importorskip("respx")
    from httpx import Response as HttpxResponse

    base = config.healthcheck_api_url.rstrip("/")
    captured: dict[str, Any] = {}

    def _side_effect(request):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return HttpxResponse(200, json={"status": "ok"})

    with respx.mock(base_url=base, assert_all_called=False) as mock:
        mock.post(path__regex=r".*/summary-ref$").mock(side_effect=_side_effect)

        async with httpx.AsyncClient(timeout=5.0) as client:
            writer = HealthcheckWriter(
                config=config,
                outbox=outbox,
                audit_logger=audit,
                http_client=client,
            )
            kw = _payload_kwargs()
            result = await writer.write_summary_ref(
                **kw, study_uid="study-respx"
            )
            assert result is True

            headers_lower = {
                k.lower(): v for k, v in captured["headers"].items()
            }
            assert "authorization" in headers_lower
            assert (
                headers_lower["authorization"]
                == f"Bearer {config.healthcheck_api_token}"
            )
            assert kw["visit_id"] in captured["url"] or (
                kw["visit_id"]
                .replace("<", "%3C")
                .replace(">", "%3E")
                in captured["url"]
            )
            assert set(captured["body"].keys()) == {
                "visit_id",
                "summary_snippet",
                "source_study_uids",
                "model_version",
            }
            await writer.aclose()
