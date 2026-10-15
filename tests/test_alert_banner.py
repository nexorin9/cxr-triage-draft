"""失败告警 banner 测试（Task 27）。

覆盖：
1. Outbox 告警回调注册 / 注销 / 触发（persistence/outbox.py）
2. AlertEvent 数据契约校验
3. build_alert_banner_html 渲染（CSS class / 中文文案 / 隐藏态）
4. build_alert_event_payload 序列化（含 banner_html 字段）
5. AlertPublisher 构造 / 单条 raised 推送 / summary / dismiss
6. AlertPublisher 限重（dedupe_window）
7. AlertPublisher SSE publish 异常容错
8. AlertPublisher audit 写入容错
10. AlertPublisher 多回调同时触发（顺序调用 + 异常隔离）
11. CSS / JS 资源文件契约
13. 端到端：outbox enqueue → 4 次 mark_retry → status=alerting → callback fire → SSE event → HTML/CSS 全链路
"""
from __future__ import annotations

import asyncio
import json as json_mod
import os
from pathlib import Path

import pytest

from cxr_triage.config import CxrTriageConfig
from cxr_triage.persistence import Database, Outbox, init_db
from cxr_triage.persistence.outbox import MAX_RETRY_BEFORE_ALERT
from cxr_triage.writers.alert_banner import (
    ALERT_BANNER_CSS_CLASS,
    ALERT_BANNER_CSS_CLASS_HIDDEN,
    ALERT_BANNER_DATA_ATTR,
    ALERT_BANNER_DEFAULT_MESSAGE_ZH,
    ALERT_BANNER_ELEMENT_ID,
    ALERT_DEDUPE_WINDOW_SECONDS,
    ALERT_EVENT_NAME,
    ALERT_KIND_DISMISSED,
    ALERT_KIND_RAISED,
    ALERT_KIND_SUMMARY,
    ALERT_SEVERITY_ERROR,
    ALERT_SEVERITY_WARNING,
    EVENT_ALERT_DISMISSED,
    EVENT_ALERT_PUSHED,
    AlertBannerError,
    AlertEvent,
    AlertPublisher,
    build_alert_banner_html,
    build_alert_event_payload,
    build_alert_publisher,
    build_dismissed_banner_html,
    describe_target_endpoint,
    is_target_endpoint_known,
)
from cxr_triage.writers.audit import AuditLogger

WEB_STATIC_DIR = (
    Path(__file__).parent.parent
    / "src"
    / "cxr_triage"
    / "web"
    / "static"
)
CSS_FILE = WEB_STATIC_DIR / "style.css"
JS_FILE = WEB_STATIC_DIR / "alert_banner.js"


# =====================================================================
# fixtures
# =====================================================================
@pytest.fixture
async def db(tmp_path):
    """SQLite 内存数据库 + 4 表 schema。"""
    path = tmp_path / "test_alert.db"
    database = Database(path)
    await database.connect()
    await init_db(database)
    yield database
    await database.close()


@pytest.fixture
async def outbox(db):
    return Outbox(db)


@pytest.fixture
async def audit_logger(db, tmp_path):
    logger_ = AuditLogger(db=db, tool_calls_dir=tmp_path / "tool_calls")
    return logger_


@pytest.fixture
def captured_sse():
    """用 list 捕获所有推过来的 (study_uid, payload) 元组。"""
    captured: list[tuple[str, dict]] = []

    async def publish(study_uid: str, payload: dict) -> None:
        captured.append((study_uid, payload))

    return captured, publish


@pytest.fixture
async def alert_publisher(outbox, audit_logger, captured_sse):
    _, publish = captured_sse
    return AlertPublisher(outbox, audit_logger, publish)


# =====================================================================
# 1. AlertEvent 数据契约
# =====================================================================
class TestAlertEventContract:
    def test_minimal_construction_sets_default_ts(self):
        ev = AlertEvent(
            outbox_id=1,
            study_uid="1.2.3",
            target_endpoint="ris://finding-draft/1.2.3",
            retry_count=4,
            last_error="HTTP 500",
            message_zh="RIS 写回失败",
        )
        assert ev.ts  # 自动设默认 ISO8601
        assert ev.severity == ALERT_SEVERITY_ERROR
        assert ev.kind == ALERT_KIND_RAISED
        assert ev.total_alert_count == 0

    def test_invalid_severity_rejected(self):
        with pytest.raises(AlertBannerError, match="severity 非法"):
            AlertEvent(
                outbox_id=1,
                study_uid="x",
                target_endpoint="ris://x",
                retry_count=4,
                last_error="e",
                message_zh="m",
                severity="critical",
            )

    def test_invalid_kind_rejected(self):
        with pytest.raises(AlertBannerError, match="kind 非法"):
            AlertEvent(
                outbox_id=1,
                study_uid="x",
                target_endpoint="ris://x",
                retry_count=4,
                last_error="e",
                message_zh="m",
                kind="alert.custom",
            )

    def test_warning_severity_accepted(self):
        ev = AlertEvent(
            outbox_id=1,
            study_uid="x",
            target_endpoint="hc://x",
            retry_count=4,
            last_error="",
            message_zh="weak",
            severity=ALERT_SEVERITY_WARNING,
            kind=ALERT_KIND_SUMMARY,
        )
        assert ev.severity == ALERT_SEVERITY_WARNING


# =====================================================================
# 2. build_alert_banner_html 渲染
# =====================================================================
class TestBuildAlertBannerHtml:
    def test_default_severity_uses_error_class(self):
        html = build_alert_banner_html(alert_count=3)
        assert ALERT_BANNER_CSS_CLASS in html
        assert ALERT_BANNER_ELEMENT_ID in html
        assert 'data-cxr-alert-count="3"' in html
        assert ALERT_BANNER_DATA_ATTR in html

    def test_warning_severity_uses_warning_class(self):
        html = build_alert_banner_html(
            alert_count=1, severity=ALERT_SEVERITY_WARNING
        )
        assert "cxr-alert-banner--warning" in html
        assert "cxr-alert-banner--error" not in html

    def test_default_message_includes_count(self):
        html = build_alert_banner_html(alert_count=5)
        assert "5" in html
        assert "请联系信息科" in html

    def test_custom_message_used(self):
        html = build_alert_banner_html(
            alert_count=2, message="RIS 写回失败，请联系工程师。"
        )
        assert "RIS 写回失败" in html
        # 自定义 message 不带 {count} 占位符，原样保留
        assert "{count}" not in html

    def test_message_with_placeholder_gets_formatted(self):
        html = build_alert_banner_html(
            alert_count=4, message="已有 {count} 条失败，请联系信息科"
        )
        assert "4" in html
        assert "已有" in html

    def test_message_html_escaped(self):
        html = build_alert_banner_html(
            alert_count=1,
            message='<script>alert("xss")</script>',
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_negative_count_rejected(self):
        with pytest.raises(AlertBannerError, match="alert_count"):
            build_alert_banner_html(alert_count=-1)

    def test_invalid_severity_rejected(self):
        with pytest.raises(AlertBannerError, match="severity 非法"):
            build_alert_banner_html(alert_count=1, severity="info")

    def test_zero_count_still_renders(self):
        html = build_alert_banner_html(alert_count=0)
        assert 'data-cxr-alert-count="0"' in html

    def test_dismissed_banner_html(self):
        html = build_dismissed_banner_html()
        assert ALERT_BANNER_CSS_CLASS_HIDDEN in html
        assert 'data-cxr-alert-count="0"' in html
        assert 'data-cxr-alert-banner="0"' in html


# =====================================================================
# 3. build_alert_event_payload 序列化
# =====================================================================
class TestBuildAlertEventPayload:
    def test_required_fields_present(self):
        ev = AlertEvent(
            outbox_id=42,
            study_uid="1.2.3.4",
            target_endpoint="ris://finding-draft/1.2.3.4",
            retry_count=4,
            last_error="HTTP 500",
            message_zh="RIS 写回失败",
            total_alert_count=2,
        )
        payload = build_alert_event_payload(ev)
        assert payload["event"] == ALERT_EVENT_NAME
        assert payload["kind"] == ALERT_KIND_RAISED
        assert payload["severity"] == ALERT_SEVERITY_ERROR
        assert payload["outbox_id"] == 42
        assert payload["study_uid"] == "1.2.3.4"
        assert payload["target_endpoint"].startswith("ris://")
        assert payload["retry_count"] == 4
        assert payload["last_error"] == "HTTP 500"
        assert payload["total_alert_count"] == 2
        assert payload["ts"]
        assert payload["banner_html"]
        assert ALERT_BANNER_CSS_CLASS in payload["banner_html"]

    def test_summary_kind_in_payload(self):
        ev = AlertEvent(
            outbox_id=0,
            study_uid="",
            target_endpoint="summary",
            retry_count=5,
            last_error="",
            message_zh="",
            kind=ALERT_KIND_SUMMARY,
            total_alert_count=3,
        )
        payload = build_alert_event_payload(ev)
        assert payload["kind"] == ALERT_KIND_SUMMARY
        assert payload["total_alert_count"] == 3

    def test_payload_is_json_serializable(self):
        ev = AlertEvent(
            outbox_id=1,
            study_uid="x",
            target_endpoint="ris://x",
            retry_count=4,
            last_error="",
            message_zh="m",
        )
        payload = build_alert_event_payload(ev)
        # 双 round-trip 必须保形
        s = json_mod.dumps(payload, ensure_ascii=False)
        rt = json_mod.loads(s)
        assert rt["study_uid"] == "x"
        assert rt["message_zh"] == "m"


# =====================================================================
# 4. describe / is_target_endpoint_known
# =====================================================================
class TestTargetEndpointHelpers:
    def test_ris_prefix_known(self):
        assert is_target_endpoint_known("ris://finding-draft/1.2.3")
        assert describe_target_endpoint("ris://finding-draft/1.2.3") == "RIS"

    def test_hc_prefix_known(self):
        assert is_target_endpoint_known("hc://summary-ref/V1")
        assert describe_target_endpoint("hc://summary-ref/V1") == "体检主检"

    def test_unknown_prefix_returns_generic(self):
        assert not is_target_endpoint_known("custom://x")
        assert describe_target_endpoint("custom://x") == "外发目标"

    def test_empty_or_non_string_rejected(self):
        assert not is_target_endpoint_known("")
        assert not is_target_endpoint_known(None)  # type: ignore[arg-type]
        assert describe_target_endpoint("") == "外发目标"
        assert describe_target_endpoint(None) == "外发目标"  # type: ignore[arg-type]


# =====================================================================
# 5. Outbox 告警回调注册 + 触发（persistence 层）
# =====================================================================
class TestOutboxAlertCallback:
    async def test_register_rejects_none(self, outbox):
        with pytest.raises(TypeError):
            outbox.register_alert_callback(None)  # type: ignore[arg-type]

    async def test_register_rejects_non_callable(self, outbox):
        with pytest.raises(TypeError):
            outbox.register_alert_callback("not-a-callable")  # type: ignore[arg-type]

    async def test_unregister_returns_true_when_present(self, outbox):
        async def cb(p):
            pass
        outbox.register_alert_callback(cb)
        assert outbox.unregister_alert_callback(cb) is True

    async def test_unregister_returns_false_when_absent(self, outbox):
        async def cb(p):
            pass
        assert outbox.unregister_alert_callback(cb) is False

    async def test_clear_alert_callbacks(self, outbox):
        async def cb(p):
            pass
        outbox.register_alert_callback(cb)
        outbox.register_alert_callback(cb)
        outbox.clear_alert_callbacks()
        # 重复注销应 False（验证已清空）
        assert outbox.unregister_alert_callback(cb) is False

    async def test_mark_retry_under_threshold_does_not_fire(self, outbox):
        """retry_count=1（首次失败）→ status 仍 pending → 不 fire。"""
        fired: list[dict] = []

        async def cb(p):
            fired.append(p)

        outbox.register_alert_callback(cb)
        oid = await outbox.enqueue(
            study_uid="1.2.1",
            target_endpoint="ris://finding-draft/1.2.1",
            payload={"x": 1},
        )
        new_status = await outbox.mark_retry(oid, "HTTP 503")
        assert new_status == "pending"
        assert fired == []  # 没触发

    async def test_mark_retry_over_threshold_fires_callback(self, outbox):
        """连续失败 4 次（retry_count=4 > MAX_RETRY_BEFORE_ALERT=3）→ alerting → fire。"""
        fired: list[dict] = []

        async def cb(p):
            fired.append(p)

        outbox.register_alert_callback(cb)
        oid = await outbox.enqueue(
            study_uid="1.2.2",
            target_endpoint="ris://finding-draft/1.2.2",
            payload={"y": 1},
        )
        # 跑 4 次 mark_retry（retry_count=4 触发）
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            ns = await outbox.mark_retry(oid, f"err-{i}")
        assert ns == "alerting"
        # callback 应被 fire 一次
        assert len(fired) == 1
        payload = fired[0]
        assert payload["outbox_id"] == oid
        assert payload["study_uid"] == "1.2.2"
        assert payload["target_endpoint"] == "ris://finding-draft/1.2.2"
        assert payload["retry_count"] == MAX_RETRY_BEFORE_ALERT + 1
        assert payload["last_error"].startswith("err-")
        assert payload["ts"]

    async def test_callback_exception_does_not_break_mark_retry(self, outbox):
        """回调抛异常 → mark_retry 不抛；状态仍 alerting。"""
        async def bad_cb(p):
            raise RuntimeError("callback bad")

        outbox.register_alert_callback(bad_cb)
        oid = await outbox.enqueue(
            study_uid="1.2.3",
            target_endpoint="ris://finding-draft/1.2.3",
            payload={"z": 1},
        )
        # 跑 4 次
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            ns = await outbox.mark_retry(oid, f"err-{i}")
        assert ns == "alerting"  # 不抛 + 状态正确
        # alert 行仍写入 DB
        row = await outbox.get_by_id(oid)
        assert row["status"] == "alerting"

    async def test_multiple_callbacks_fire_in_order(self, outbox):
        """多个回调按注册顺序触发，互不污染。"""
        order: list[str] = []

        async def cb_a(p):
            order.append("a")

        async def cb_b(p):
            order.append("b")

        outbox.register_alert_callback(cb_a)
        outbox.register_alert_callback(cb_b)
        oid = await outbox.enqueue(
            study_uid="1.2.4",
            target_endpoint="ris://finding-draft/1.2.4",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")
        assert order == ["a", "b"]

    async def test_callback_exception_isolates_other_callbacks(self, outbox):
        """一个回调抛异常不阻断 b/c 后续回调。"""
        order: list[str] = []

        async def cb_bad(p):
            order.append("bad-start")
            raise RuntimeError("nope")

        async def cb_b(p):
            order.append("b")

        async def cb_c(p):
            order.append("c")

        outbox.register_alert_callback(cb_bad)
        outbox.register_alert_callback(cb_b)
        outbox.register_alert_callback(cb_c)
        oid = await outbox.enqueue(
            study_uid="1.2.5",
            target_endpoint="ris://finding-draft/1.2.5",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")
        # bad 起异常后 b + c 仍跑
        assert "b" in order
        assert "c" in order
        assert order.index("b") > order.index("bad-start")
        assert order.index("c") > order.index("b")

    async def test_get_by_id_after_alerting(self, outbox):
        """alerting 行 get_by_id 返回 status='alerting' + last_error 完整。"""
        oid = await outbox.enqueue(
            study_uid="1.2.6",
            target_endpoint="ris://finding-draft/1.2.6",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, "persistent failure")
        row = await outbox.get_by_id(oid)
        assert row["status"] == "alerting"
        assert row["retry_count"] == MAX_RETRY_BEFORE_ALERT + 1
        assert "persistent failure" in row["last_error"]

    async def test_count_alerting_increments(self, outbox):
        """count_alerting 反映所有 alerting 行（RIS + 主检前缀混合）。"""
        # RIS 行
        oid1 = await outbox.enqueue(
            study_uid="1.2.7",
            target_endpoint="ris://finding-draft/1.2.7",
            payload={},
        )
        # 主检行
        oid2 = await outbox.enqueue(
            study_uid="1.2.8",
            target_endpoint="hc://summary-ref/V1",
            payload={},
        )
        # 让两条都 alerting
        for oid in (oid1, oid2):
            for i in range(MAX_RETRY_BEFORE_ALERT + 1):
                await outbox.mark_retry(oid, f"err-{i}")
        n = await outbox.count_alerting()
        assert n == 2


# =====================================================================
# 6. AlertPublisher 构造 + 推送
# =====================================================================
class TestAlertPublisher:
    async def test_construct_rejects_none_outbox(self, audit_logger, captured_sse):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError, match="outbox"):
            AlertPublisher(None, audit_logger, publish)

    async def test_construct_rejects_none_audit(self, outbox, captured_sse):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError, match="audit"):
            AlertPublisher(outbox, None, publish)

    async def test_construct_rejects_none_sse(self, outbox, audit_logger):
        with pytest.raises(AlertBannerError, match="sse_publish"):
            AlertPublisher(outbox, audit_logger, None)  # type: ignore[arg-type]

    async def test_construct_rejects_non_callable_sse(self, outbox, audit_logger):
        with pytest.raises(AlertBannerError, match="sse_publish"):
            AlertPublisher(outbox, audit_logger, "not callable")  # type: ignore[arg-type]

    async def test_construct_rejects_invalid_dedupe_window(
        self, outbox, audit_logger, captured_sse
    ):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError, match="dedupe_window"):
            AlertPublisher(
                outbox, audit_logger, publish, dedupe_window_seconds=-1
            )

    async def test_construct_rejects_non_numeric_dedupe_window(
        self, outbox, audit_logger, captured_sse
    ):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError, match="dedupe_window"):
            AlertPublisher(
                outbox, audit_logger, publish, dedupe_window_seconds="bad"  # type: ignore[arg-type]
            )

    async def test_factory_function(self, outbox, audit_logger, captured_sse):
        _, publish = captured_sse
        p = build_alert_publisher(outbox, audit_logger, publish)
        assert isinstance(p, AlertPublisher)
        assert p.dedupe_window_seconds == ALERT_DEDUPE_WINDOW_SECONDS

    async def test_factory_rejects_none(self, audit_logger, captured_sse):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError):
            build_alert_publisher(None, audit_logger, publish)

    async def test_factory_rejects_none_audit(self, outbox, captured_sse):
        _, publish = captured_sse
        with pytest.raises(AlertBannerError):
            build_alert_publisher(outbox, None, publish)

    async def test_factory_rejects_none_sse(self, outbox, audit_logger):
        with pytest.raises(AlertBannerError):
            build_alert_publisher(outbox, audit_logger, None)

    async def test_initial_stats_zero(
        self, alert_publisher  # noqa: ARG002 — fixture
    ):
        s = alert_publisher.stats
        assert s["raised_pushed"] == 0
        assert s["raised_deduped"] == 0
        assert s["summary_pushed"] == 0
        assert s["dismissed_pushed"] == 0
        assert s["publish_errors"] == 0

    async def test_clear_dedupe_cache(self, alert_publisher):
        alert_publisher._recent_raises[("ris://x", "x")] = "2026-01-01T00:00:00.000Z"
        alert_publisher._last_dismissed_iso = "2026-01-01T00:00:00.000Z"
        alert_publisher.clear_dedupe_cache()
        assert alert_publisher._recent_raises == {}
        assert alert_publisher._last_dismissed_iso is None


# =====================================================================
# 7. AlertPublisher.publish_alert_summary
# =====================================================================
class TestAlertPublisherSummary:
    async def test_no_alerts_returns_none(
        self, alert_publisher, captured_sse, outbox
    ):
        captured, _ = captured_sse
        # 当前无 alerting → 不推
        result = await alert_publisher.publish_alert_summary()
        assert result is None
        assert captured == []

    async def test_publishes_when_alerts_exist(
        self, alert_publisher, captured_sse, outbox
    ):
        captured, _ = captured_sse
        # 制造 2 条 alerting 行
        for i in range(2):
            oid = await outbox.enqueue(
                study_uid=f"1.2.{i}",
                target_endpoint=f"ris://finding-draft/1.2.{i}",
                payload={},
            )
            for j in range(MAX_RETRY_BEFORE_ALERT + 1):
                await outbox.mark_retry(oid, f"err-{j}")
        captured.clear()  # 清掉 mark_retry 路径的 raises
        result = await alert_publisher.publish_alert_summary()
        assert result is not None
        assert result["kind"] == ALERT_KIND_SUMMARY
        assert result["total_alert_count"] == 2
        assert len(captured) == 1
        study_uid, payload = captured[0]
        assert study_uid == "alert"
        assert payload["event"] == ALERT_EVENT_NAME
        assert alert_publisher.stats["summary_pushed"] == 1


# =====================================================================
# 8. AlertPublisher.dismiss_alert
# =====================================================================
class TestAlertPublisherDismiss:
    async def test_dismiss_pushes_event(self, alert_publisher, captured_sse):
        captured, _ = captured_sse
        ok = await alert_publisher.dismiss_alert()
        assert ok is True
        assert len(captured) == 1
        study_uid, payload = captured[0]
        assert payload["kind"] == ALERT_KIND_DISMISSED
        assert alert_publisher.stats["dismissed_pushed"] == 1

    async def test_dismiss_records_last_dismissed_iso(self, alert_publisher):
        await alert_publisher.dismiss_alert()
        assert alert_publisher._last_dismissed_iso
        # 第二次 dismiss 不推（去重，但本类目前不严格去重）
        await alert_publisher.dismiss_alert()
        # 第二条仍推（API 设计上信息科可重复 dismiss 同步前端）
        assert alert_publisher.stats["dismissed_pushed"] == 2

    async def test_dismiss_with_reason(self, alert_publisher, captured_sse):
        captured, _ = captured_sse
        await alert_publisher.dismiss_alert(reason="信息科处理完成")
        _, payload = captured[0]
        assert payload["last_error"] == "信息科处理完成"


# =====================================================================
# 9. AlertPublisher.on_outbox_alert_transition（回调入口）
# =====================================================================
class TestAlertPublisherCallback:
    async def test_raises_alert_on_outbox_callback(
        self, alert_publisher, captured_sse, outbox
    ):
        captured, _ = captured_sse
        # 把 publisher 装到 outbox 上
        outbox.register_alert_callback(alert_publisher.on_outbox_alert_transition)
        oid = await outbox.enqueue(
            study_uid="1.2.10",
            target_endpoint="ris://finding-draft/1.2.10",
            payload={},
        )
        # 跑 4 次让状态变 alerting（每次 mark_retry 都会 fire 回调，
        # 但只有「pending→alerting 转移」一次；内部 publisher 去重）
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")
        # 应有 1 条 raised（dedupe 防止 4 次 mark_retry 推 4 次）
        assert len(captured) == 1
        study_uid, payload = captured[0]
        assert payload["kind"] == ALERT_KIND_RAISED
        assert payload["outbox_id"] == oid
        assert payload["study_uid"] == "1.2.10"
        assert payload["target_endpoint"] == "ris://finding-draft/1.2.10"
        assert payload["retry_count"] == MAX_RETRY_BEFORE_ALERT + 1
        assert alert_publisher.stats["raised_pushed"] == 1

    async def test_no_alert_when_status_under_threshold(
        self, alert_publisher, captured_sse, outbox
    ):
        captured, _ = captured_sse
        outbox.register_alert_callback(alert_publisher.on_outbox_alert_transition)
        oid = await outbox.enqueue(
            study_uid="1.2.11",
            target_endpoint="ris://finding-draft/1.2.11",
            payload={},
        )
        # 仅 1 次 mark_retry → 仍 pending → 不 fire
        await outbox.mark_retry(oid, "HTTP 503")
        assert captured == []
        assert alert_publisher.stats["raised_pushed"] == 0

    async def test_dedupe_window_prevents_repeat(
        self, alert_publisher, captured_sse, outbox
    ):
        """两个不同的 outbox 行都 alerting → 都被推（dedupe_key 不同）。"""
        captured, _ = captured_sse
        outbox.register_alert_callback(alert_publisher.on_outbox_alert_transition)
        oid1 = await outbox.enqueue(
            study_uid="1.2.12",
            target_endpoint="ris://finding-draft/1.2.12",
            payload={},
        )
        oid2 = await outbox.enqueue(
            study_uid="1.2.13",
            target_endpoint="ris://finding-draft/1.2.13",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid1, f"err-{i}")
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid2, f"err-{i}")
        # 两条 study_uid 不同 → 都被推
        assert len(captured) == 2
        assert alert_publisher.stats["raised_pushed"] == 2

    async def test_dedupe_window_short(self, outbox, audit_logger):
        """dedupe_window=0 → 仍推（边界：窗口不阻挡）。"""
        captured = []

        async def publish(s, p):
            captured.append((s, p))

        publisher = AlertPublisher(
            outbox, audit_logger, publish, dedupe_window_seconds=0
        )
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)
        oid = await outbox.enqueue(
            study_uid="1.2.14",
            target_endpoint="ris://finding-draft/1.2.14",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")
        assert len(captured) >= 1

    async def test_dedupe_window_dedupes_within_window(
        self, outbox, audit_logger
    ):
        """dedupe_window 内同 key 第二次 fire 不推。"""
        captured = []

        async def publish(s, p):
            captured.append((s, p))

        publisher = AlertPublisher(
            outbox,
            audit_logger,
            publish,
            dedupe_window_seconds=600.0,  # 10 分钟窗口
        )
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)
        # 手动 fire 两次（同 key）
        payload = {
            "outbox_id": 1,
            "study_uid": "1.2.15",
            "target_endpoint": "ris://finding-draft/1.2.15",
            "retry_count": 4,
            "last_error": "err",
            "ts": "2026-10-13T00:00:00.000Z",
        }
        await publisher.on_outbox_alert_transition(payload)
        # 第二次同 key 在 10 分钟窗口内 → 去重
        payload["ts"] = "2026-10-13T00:05:00.000Z"
        await publisher.on_outbox_alert_transition(payload)
        assert len(captured) == 1
        assert publisher.stats["raised_deduped"] == 1

    async def test_dedupe_window_passes_after_window_expiry(
        self, outbox, audit_logger
    ):
        """dedupe_window 外（时间已过）→ 仍推。"""
        captured = []

        async def publish(s, p):
            captured.append((s, p))

        publisher = AlertPublisher(
            outbox,
            audit_logger,
            publish,
            dedupe_window_seconds=60.0,  # 60 秒窗口
        )
        payload = {
            "outbox_id": 1,
            "study_uid": "1.2.16",
            "target_endpoint": "ris://finding-draft/1.2.16",
            "retry_count": 4,
            "last_error": "",
            "ts": "2026-10-13T00:00:00.000Z",
        }
        await publisher.on_outbox_alert_transition(payload)
        # 间隔 > 60s
        payload["ts"] = "2026-10-13T00:02:00.000Z"
        await publisher.on_outbox_alert_transition(payload)
        assert len(captured) == 2

    async def test_invalid_payload_ignored(self, alert_publisher, captured_sse):
        captured, _ = captured_sse
        # 非 dict → warn 不抛 + 不推
        await alert_publisher.on_outbox_alert_transition("not a dict")  # type: ignore[arg-type]
        assert captured == []

    async def test_missing_field_ignored(self, alert_publisher, captured_sse):
        captured, _ = captured_sse
        # 缺 outbox_id → warn + 不推
        await alert_publisher.on_outbox_alert_transition(
            {"study_uid": "x", "target_endpoint": "ris://x"}
        )
        assert captured == []

    async def test_invalid_int_field_ignored(
        self, alert_publisher, captured_sse
    ):
        captured, _ = captured_sse
        await alert_publisher.on_outbox_alert_transition(
            {
                "outbox_id": "not-an-int",
                "study_uid": "x",
                "target_endpoint": "ris://x",
                "retry_count": 4,
                "last_error": "",
                "ts": "2026-10-13T00:00:00.000Z",
            }
        )
        assert captured == []


# =====================================================================
# 10. AlertPublisher SSE publish 异常容错
# =====================================================================
class TestAlertPublisherFaultTolerance:
    async def test_sse_failure_does_not_raise(
        self, outbox, audit_logger, captured_sse
    ):
        async def bad_publish(s, p):
            raise RuntimeError("SSE broken")

        publisher = AlertPublisher(outbox, audit_logger, bad_publish)
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)
        oid = await outbox.enqueue(
            study_uid="1.2.20",
            target_endpoint="ris://finding-draft/1.2.20",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            # 不抛
            await outbox.mark_retry(oid, f"err-{i}")
        assert publisher.stats["publish_errors"] >= 1

    async def test_audit_failure_does_not_block_sse(
        self, outbox, audit_logger
    ):
        """audit 写失败时 SSE publish 仍成功，stats 计数。"""
        captured = []

        async def publish(s, p):
            captured.append((s, p))

        # 把 audit_logger 替换成抛异常的
        class BadAudit:
            async def audit_log_insert(self, **kw):
                raise RuntimeError("audit broken")

        publisher = AlertPublisher(outbox, BadAudit(), publish)  # type: ignore[arg-type]
        # summary 必须有 count_alerting > 0
        oid = await outbox.enqueue(
            study_uid="1.2.21",
            target_endpoint="ris://finding-draft/1.2.21",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")
        # 直接推 summary
        result = await publisher.publish_alert_summary()
        assert result is not None
        assert len(captured) == 1


# =====================================================================
# 11. CSS / JS 资源文件契约
# =====================================================================
class TestStaticAssets:
    def test_css_file_exists(self):
        assert CSS_FILE.exists(), f"CSS 文件不存在: {CSS_FILE}"

    def test_css_file_has_error_class(self):
        content = CSS_FILE.read_text(encoding="utf-8")
        assert ".cxr-alert-banner--error" in content
        # 红底白字
        assert "background-color: #d32f2f" in content or "#d32f2f" in content
        assert "color: #ffffff" in content or "color:#ffffff" in content

    def test_css_file_has_warning_class(self):
        content = CSS_FILE.read_text(encoding="utf-8")
        assert ".cxr-alert-banner--warning" in content

    def test_css_file_has_hidden_class(self):
        content = CSS_FILE.read_text(encoding="utf-8")
        assert ".cxr-alert-banner--hidden" in content
        assert "display: none" in content

    def test_css_file_has_responsive(self):
        content = CSS_FILE.read_text(encoding="utf-8")
        assert "@media" in content

    def test_js_file_exists(self):
        assert JS_FILE.exists(), f"JS 文件不存在: {JS_FILE}"

    def test_js_file_has_handler_class(self):
        content = JS_FILE.read_text(encoding="utf-8")
        assert "AlertBannerHandler" in content
        assert "EventSource" in content

    def test_js_file_has_event_kinds(self):
        content = JS_FILE.read_text(encoding="utf-8")
        assert "alert.raised" in content
        assert "alert.dismissed" in content
        assert "alert.summary" in content

    def test_js_file_has_banner_root_id(self):
        content = JS_FILE.read_text(encoding="utf-8")
        assert ALERT_BANNER_ELEMENT_ID in content

    def test_js_file_has_health_probe(self):
        content = JS_FILE.read_text(encoding="utf-8")
        # /api/health 兜底
        assert "/api/health" in content
        assert "outbox_alerts" in content

    def test_js_file_handles_event_addEventListener(self):
        content = JS_FILE.read_text(encoding="utf-8")
        assert "addEventListener" in content

    def test_js_file_html_escapes(self):
        content = JS_FILE.read_text(encoding="utf-8")
        assert "escapeHtml" in content

    def test_css_file_minimum_size(self):
        """CSS 文件不能太短（避免空壳）。"""
        size = CSS_FILE.stat().st_size
        assert size > 500, f"CSS 文件 {size} bytes 太小"

    def test_js_file_minimum_size(self):
        """JS 文件不能太短。"""
        size = JS_FILE.stat().st_size
        assert size > 1500, f"JS 文件 {size} bytes 太小"


# =====================================================================
# 12. 端到端：outbox mark_retry → SSE → HTML 全链路
# =====================================================================
class TestEndToEnd:
    async def test_outbox_to_html_full_flow(
        self, outbox, audit_logger, captured_sse
    ):
        """完整 happy-path：enqueue → 4 次 mark_retry → SSE push → HTML 包含全字段。"""
        captured, _ = captured_sse
        publisher = AlertPublisher(outbox, audit_logger, captured_sse[1])
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)

        # 入条
        oid = await outbox.enqueue(
            study_uid="1.2.30",
            target_endpoint="ris://finding-draft/1.2.30",
            payload={"patient_id_hash": "abc123"},
        )
        # 跑 4 次失败
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"HTTP 503 retry-{i}")

        # SSE 应收到一条 alert 事件
        assert len(captured) == 1
        study_uid, payload = captured[0]
        assert payload["event"] == ALERT_EVENT_NAME
        assert payload["kind"] == ALERT_KIND_RAISED
        assert payload["study_uid"] == "1.2.30"
        assert payload["target_endpoint"] == "ris://finding-draft/1.2.30"
        assert payload["outbox_id"] == oid
        assert payload["retry_count"] == MAX_RETRY_BEFORE_ALERT + 1
        assert payload["total_alert_count"] >= 1
        # banner_html 必须含 CSS class 与中文文案
        html = payload["banner_html"]
        assert ALERT_BANNER_CSS_CLASS in html
        assert ALERT_BANNER_ELEMENT_ID in html
        assert 'data-cxr-alert-count="1"' in html
        assert "请联系信息科" in html
        # audit 应有 alert.pushed 行
        rows = await audit_logger.list_audit_for_study("1.2.30")
        alert_pushed = [
            r for r in rows if r["event_type"] == EVENT_ALERT_PUSHED
        ]
        assert len(alert_pushed) == 1

    async def test_two_endpoints_different_keys(
        self, outbox, audit_logger, captured_sse
    ):
        """RIS + 主检同时 alerting → SSE 收到 2 条（不同 dedupe key）。"""
        captured, _ = captured_sse
        publisher = AlertPublisher(outbox, audit_logger, captured_sse[1])
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)

        for endpoint, study in [
            ("ris://finding-draft/1.2.40", "1.2.40"),
            ("hc://summary-ref/V40", "1.2.40-hc"),  # 注意 hc 用不同 study
        ]:
            oid = await outbox.enqueue(
                study_uid=study, target_endpoint=endpoint, payload={}
            )
            for i in range(MAX_RETRY_BEFORE_ALERT + 1):
                await outbox.mark_retry(oid, f"err-{i}")

        # 2 条 alerts（key 不同）
        assert len(captured) == 2
        targets = {p["target_endpoint"] for _, p in captured}
        assert "ris://finding-draft/1.2.40" in targets
        assert "hc://summary-ref/V40" in targets

    async def test_publish_alert_summary_after_recovery(
        self, outbox, audit_logger, captured_sse
    ):
        """修复后 dismiss → summary 不再推（count_alerting == 0）。"""
        captured, _ = captured_sse
        publisher = AlertPublisher(outbox, audit_logger, captured_sse[1])
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)

        # 触发一条 alert
        oid = await outbox.enqueue(
            study_uid="1.2.50",
            target_endpoint="ris://finding-draft/1.2.50",
            payload={},
        )
        for i in range(MAX_RETRY_BEFORE_ALERT + 1):
            await outbox.mark_retry(oid, f"err-{i}")

        captured.clear()
        # count_alerting > 0 → summary 推
        s1 = await publisher.publish_alert_summary()
        assert s1 is not None
        assert len(captured) == 1

        # 模拟修复：手动把 outbox 行状态切回 sent
        await outbox.mark_sent(oid)
        captured.clear()
        # count_alerting == 0 → summary 不推
        s2 = await publisher.publish_alert_summary()
        assert s2 is None
        assert captured == []


# =====================================================================
# 13. writers/__init__ 符号导出
# =====================================================================
class TestWritersInitExports:
    def test_alert_publisher_symbols_exported(self):
        from cxr_triage import writers

        for name in [
            "AlertBannerError",
            "AlertEvent",
            "AlertPublisher",
            "build_alert_banner_html",
            "build_alert_event_payload",
            "build_alert_publisher",
        ]:
            assert hasattr(writers, name), f"writers.{name} 未导出"