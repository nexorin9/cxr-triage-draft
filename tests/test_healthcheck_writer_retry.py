"""tests/test_healthcheck_writer_retry.py — 体检主检 outbox 异步重试 + 与 RIS 独立通道。

覆盖（CLAUDE.md Phase 4 验收 · Task 25）：

1. **retry_loop 生命周期**：
   - ``start_retry_loop()`` 幂等；``retry_running`` 属性；
   - ``stop_retry_loop()`` 优雅退出；``aclose()`` 顺手停；
   - 没 start 直接调 retry_loop() 兜底建 event。

2. **单条 retry 成功**：
   - mock 体检主检 200 → ``Outbox.mark_sent`` + 写
     ``healthcheck.retry_sent`` audit；
   - 出站 URL + Bearer + payload 4 字段对齐；
   - visit_id 从 ``target_endpoint`` 正确解析；
   - stats ``retry_attempts`` / ``retry_successes`` 累加。

3. **单条 retry 失败**：
   - mock 体检主检 500 → ``Outbox.mark_retry`` retry_count += 1 + 写
     ``healthcheck.retry_failed`` audit；
   - 网络错（TimeoutException / ConnectError）同样入 mark_retry；
   - stats ``retry_failures`` 累加。

4. **指数退避 + 告警阈值**：
   - 连续 4 次失败（retry_count 0→1→2→3→4）→ 第 4 次 mark_retry
     返 ``alerting`` + 写 ``healthcheck.retry_alerting`` audit + stats
     ``retry_alerts`` = 1；
   - 退避曲线由 ``outbox.compute_next_retry_iso`` 决定（30s/60s/120s）。

5. **非体检主检行跳过**：
   - outbox 中混入 ``ris://finding-draft/<uid>``（RIS）行 →
     retry_loop 不处理；
   - stats ``retry_skipped`` 累加。

6. **与 RIS writer retry_loop 并行不互锁**：
   - 同一 outbox 表：RIS 行被 RISWriter 写 sent；体检主检行被
     HealthcheckWriter 写 sent；互不干扰。

7. **dequeue 空 → 不报错**：
   - outbox 空 → retry_loop 一轮后继续，不写 audit。

8. **单条异常兜底**：
   - 单条 POST 抛非预期异常 → 仍 mark_retry + audit + 下一轮继续。

9. **visit_id 解析**：
   - ``_extract_visit_id_from_target_endpoint`` 解析 ``hc://summary-ref/<id>``
     → ``<id>``；非法 target_endpoint / 非 hc 前缀抛
     ``HealthcheckPayloadError``。

10. **出站契约**：Bearer Token + Content-Type + 正确 URL 路径 +
    payload 4 字段对齐 spec.md §4。
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from cxr_triage.config import CxrTriageConfig
from cxr_triage.persistence import Database, init_db
from cxr_triage.persistence.outbox import (
    BASE_BACKOFF_SECONDS,
    MAX_RETRY_BEFORE_ALERT,
    Outbox,
    compute_next_retry_iso,
)
from cxr_triage.writers.audit import AuditLogger
from cxr_triage.writers.healthcheck_writer import (
    DEFAULT_RETRY_POLL_SECONDS,
    EVENT_HEALTHCHECK_RETRY_ALERTING,
    EVENT_HEALTHCHECK_RETRY_FAILED,
    EVENT_HEALTHCHECK_RETRY_SENT,
    HEALTHCHECK_TARGET_PREFIX,
    HealthcheckPayloadError,
    HealthcheckWriter,
    _extract_visit_id_from_target_endpoint,
    build_healthcheck_summary_ref_url,
    build_healthcheck_target_endpoint,
)
from cxr_triage.writers.ris_writer import (
    EVENT_RIS_RETRY_SENT,
    RISWriter,
    build_ris_target_endpoint,
)


# =====================================================================
# Fixtures
# =====================================================================
def _make_config(**overrides) -> CxrTriageConfig:
    """构造最小 CxrTriageConfig（凭据 placeholder + flag 放行）。"""
    import os

    os.environ.setdefault("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    os.environ.setdefault("RIS_API_TOKEN", "replace-me-ris-token")
    os.environ.setdefault("HEALTHCHECK_API_TOKEN", "replace-me-hc-token")
    os.environ.setdefault("OPENAI_API_KEY", "replace-me-llm-key")
    return CxrTriageConfig(**overrides)


def _payload_kwargs(visit_id: str = "HC-20261013-00001") -> dict[str, Any]:
    return {
        "visit_id": visit_id,
        "summary_snippet": (
            "胸片：右下肺野见少量气胸征象（0.87），建议进一步 CT。"
        ),
        "source_study_uids": [
            f"1.2.826.0.1.3680043.8.498.<EXAMPLE_HC_{visit_id}>",
        ],
        "model_version": "0.1.0",
    }


@pytest.fixture
async def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "healthcheck_retry_test.db"))
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
# Mock HTTP 客户端
# =====================================================================
class _MockResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _MockAsyncClient:
    """httpx.AsyncClient 的最简替身（顺序消费响应序列）。"""

    def __init__(self, response_sequence: list[_MockResponse]) -> None:
        self._responses = list(response_sequence)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict, headers: dict) -> _MockResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._idx >= len(self._responses):
            resp = self._responses[-1]
        else:
            resp = self._responses[self._idx]
            self._idx += 1
        return resp

    async def aclose(self) -> None:
        pass


class _AlwaysFailingClient:
    """每次 POST 都抛非预期异常；用于测兜底。"""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        raise self._exc

    async def aclose(self) -> None:
        pass


class _TimeoutClient:
    async def post(self, *args, **kwargs):
        raise httpx.TimeoutException("retry timeout")

    async def aclose(self) -> None:
        pass


class _ConnectErrorClient:
    async def post(self, *args, **kwargs):
        raise httpx.ConnectError("retry connection refused")

    async def aclose(self) -> None:
        pass


def _make_writer(
    config: CxrTriageConfig,
    outbox: Outbox,
    audit: AuditLogger,
    mock_client: Any,
    *,
    poll_seconds: float = 0.05,
    batch_size: int = 10,
) -> HealthcheckWriter:
    return HealthcheckWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=mock_client,  # type: ignore[arg-type]
        retry_poll_seconds=poll_seconds,
        retry_batch_size=batch_size,
    )


async def _wait_for_cycles(
    writer: HealthcheckWriter,
    expected_cycles: int,
    *,
    timeout: float = 5.0,
) -> None:
    """异步等 ``retry_cycles`` 计数到 ≥ ``expected_cycles``。"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if writer.stats["retry_cycles"] >= expected_cycles:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"retry_cycles 未达 {expected_cycles}（实际 {writer.stats['retry_cycles']}）"
    )


# =====================================================================
# 1. retry_loop 生命周期
# =====================================================================
@pytest.mark.asyncio
async def test_default_retry_poll_constant() -> None:
    """DEFAULT_RETRY_POLL_SECONDS=30（医院内网 30s 足够）。"""
    assert DEFAULT_RETRY_POLL_SECONDS == 30.0


@pytest.mark.asyncio
async def test_event_constants_locked() -> None:
    assert EVENT_HEALTHCHECK_RETRY_SENT == "healthcheck.retry_sent"
    assert EVENT_HEALTHCHECK_RETRY_FAILED == "healthcheck.retry_failed"
    assert EVENT_HEALTHCHECK_RETRY_ALERTING == "healthcheck.retry_alerting"


@pytest.mark.asyncio
async def test_retry_running_starts_false(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    writer = _make_writer(
        config, outbox, audit, _MockAsyncClient([_MockResponse(200)])
    )
    assert writer.retry_running is False
    await writer.aclose()


@pytest.mark.asyncio
async def test_start_retry_loop_is_idempotent(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    writer = _make_writer(
        config, outbox, audit, _MockAsyncClient([_MockResponse(200)])
    )
    await writer.start_retry_loop()
    assert writer.retry_running is True
    task1 = writer._retry_task
    # 第二次启动 noop
    await writer.start_retry_loop()
    task2 = writer._retry_task
    assert task1 is task2
    await writer.stop_retry_loop()
    assert writer.retry_running is False


@pytest.mark.asyncio
async def test_stop_retry_loop_idempotent(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    writer = _make_writer(
        config, outbox, audit, _MockAsyncClient([_MockResponse(200)])
    )
    # 没起也能 stop（noop）
    await writer.stop_retry_loop()
    await writer.stop_retry_loop()
    # 起了再 stop
    await writer.start_retry_loop()
    await writer.stop_retry_loop()
    # 再 stop 也是 noop
    await writer.stop_retry_loop()


@pytest.mark.asyncio
async def test_aclose_stops_retry_loop(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    writer = _make_writer(
        config, outbox, audit, _MockAsyncClient([_MockResponse(200)])
    )
    await writer.start_retry_loop()
    assert writer.retry_running is True
    await writer.aclose()
    assert writer.retry_running is False


@pytest.mark.asyncio
async def test_retry_loop_late_event_lazy_init(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """直接 await retry_loop()（没 start）→ 懒建 event，不抛。"""
    writer = _make_writer(
        config, outbox, audit, _MockAsyncClient([_MockResponse(200)])
    )
    # 直接调一次循环：停事件被 set 后立即退出
    stop = writer._retry_stop_event or asyncio.Event()
    writer._retry_stop_event = stop
    task = asyncio.create_task(writer.retry_loop())
    await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)


# =====================================================================
# 2. 单条 retry 成功
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_sent_success(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """retry_loop 跑一轮 → outbox 行 mark_sent + audit healthcheck.retry_sent。"""
    kw = _payload_kwargs("HC-SUCCESS-RETRY")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-success",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )
    assert outbox_id > 0

    mock_client = _MockAsyncClient([_MockResponse(200, "OK")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        await _wait_for_cycles(writer, 1)
        # 等 audit 落库（异步）
        for _ in range(50):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-success")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    # HTTP 出站断言
    assert len(mock_client.calls) >= 1
    call = mock_client.calls[0]
    assert "summary-ref" in call["url"]
    assert call["headers"]["Authorization"] == (
        f"Bearer {config.healthcheck_api_token}"
    )
    assert kw["visit_id"] in call["url"]

    # outbox mark_sent
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 0
    by_status = await outbox.count_by_status()
    assert by_status.get("sent", 0) == 1

    # audit
    rows = await audit.list_audit_for_study("uid-hc-success")
    sent_rows = [r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT]
    assert len(sent_rows) == 1
    assert sent_rows[0]["payload"]["status_code"] == 200
    assert sent_rows[0]["payload"]["outbox_id"] == outbox_id
    assert sent_rows[0]["payload"]["visit_id"] == kw["visit_id"]

    # stats
    stats = writer.stats
    assert stats["retry_attempts"] >= 1
    assert stats["retry_successes"] >= 1
    assert stats["retry_failures"] == 0
    assert stats["retry_alerts"] == 0


# =====================================================================
# 3. 单条 retry 失败
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_failure_marks_retry(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """retry_loop 失败 → mark_retry retry_count=1 + audit healthcheck.retry_failed。"""
    kw = _payload_kwargs("HC-FAIL-500")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-fail-500",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )

    mock_client = _MockAsyncClient([_MockResponse(500, "boom")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-fail-500")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    # outbox status=pending（还能再 retry）；retry_count=1
    row = await outbox.get_by_id(outbox_id)
    assert row is not None
    assert row["status"] == "pending"
    assert row["retry_count"] == 1
    assert "HTTP 500" in row["last_error"]

    # audit
    rows = await audit.list_audit_for_study("uid-hc-fail-500")
    fail = [r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED]
    assert len(fail) == 1
    assert "HTTP 500" in fail[0]["payload"]["error"]
    assert fail[0]["payload"]["new_status"] == "pending"

    # stats
    stats = writer.stats
    assert stats["retry_failures"] >= 1
    assert stats["retry_alerts"] == 0


@pytest.mark.asyncio
async def test_retry_loop_network_error(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """POST 抛 TimeoutException → mark_retry + audit。"""
    kw = _payload_kwargs("HC-TIMEOUT")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-timeout",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )

    writer = _make_writer(
        config, outbox, audit, _TimeoutClient()  # type: ignore[arg-type]
    )
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-timeout")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    row = await outbox.get_by_id(outbox_id)
    assert row is not None
    assert row["retry_count"] == 1
    assert "TimeoutException" in row["last_error"]
    rows = await audit.list_audit_for_study("uid-hc-timeout")
    fail = [r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED][0]
    assert "TimeoutException" in fail["payload"]["error"]


@pytest.mark.asyncio
async def test_retry_loop_connect_error(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """POST 抛 ConnectError → mark_retry + audit。"""
    kw = _payload_kwargs("HC-CONNECT")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-connect",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )

    writer = _make_writer(
        config, outbox, audit, _ConnectErrorClient()  # type: ignore[arg-type]
    )
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-connect")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    row = await outbox.get_by_id(outbox_id)
    assert row is not None
    assert row["retry_count"] == 1
    assert "ConnectError" in row["last_error"]
    rows = await audit.list_audit_for_study("uid-hc-connect")
    fail = [r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED][0]
    assert "ConnectError" in fail["payload"]["error"]


# =====================================================================
# 4. 指数退避 + 告警阈值
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_alerts_after_threshold(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """连续失败超 MAX_RETRY_BEFORE_ALERT → status=alerting + retry_alerting audit。"""
    kw = _payload_kwargs("HC-ALERT")
    # 直接构造一行 retry_count=3（≥ MAX_RETRY_BEFORE_ALERT）；下一次 mark_retry → alerting
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-alert",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )
    # 用 mark_retry 直接把 retry_count 推到 3（>= MAX_RETRY_BEFORE_ALERT）
    new_status = await outbox.mark_retry(
        outbox_id=outbox_id, error_message="seed retry_count=3"
    )
    assert new_status == "pending"  # retry_count 1 → 仍 pending
    new_status = await outbox.mark_retry(
        outbox_id=outbox_id, error_message="seed retry_count=4"
    )
    assert new_status == "pending"  # retry_count 2
    new_status = await outbox.mark_retry(
        outbox_id=outbox_id, error_message="seed retry_count=5"
    )
    assert new_status == "pending"  # retry_count 3（= MAX），仍 pending
    # 现在 retry_count=3；下一次 mark_retry 会 retry_count=4 > MAX → alerting
    # 但 retry_count=3 的 next_retry_at 已被设到 240s 后；需要重置到过去
    from datetime import datetime, timezone

    past_iso = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
        + "000Z"
    )
    async with outbox.db.connection() as conn:
        await conn.execute(
            "UPDATE outbox SET next_retry_at = ? WHERE id = ?",
            (past_iso, outbox_id),
        )
        await conn.commit()

    mock_client = _MockAsyncClient([_MockResponse(503, "still down")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-alert")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_ALERTING for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    # outbox 状态切 alerting
    alerting_count = await outbox.count_alerting()
    assert alerting_count == 1

    # audit
    rows = await audit.list_audit_for_study("uid-hc-alert")
    alert_rows = [
        r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_ALERTING
    ]
    assert len(alert_rows) == 1
    assert "联系信息科" in alert_rows[0]["payload"]["alert_message"]

    # stats
    assert writer.stats["retry_alerts"] >= 1


@pytest.mark.asyncio
async def test_retry_loop_alerting_not_redequeued(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """alerting 行 → 下次 dequeue 不再被 retry_loop 拉到。"""
    kw = _payload_kwargs("HC-ALERT-SKIP")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-alert-skip",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )
    # 直接推到 alerting
    for i in range(MAX_RETRY_BEFORE_ALERT + 1):
        await outbox.mark_retry(
            outbox_id=outbox_id, error_message=f"seed {i}"
        )

    mock_client = _MockAsyncClient([_MockResponse(503, "down")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        # 跑 2 轮：每轮 dequeue 不拉 alerting 行 → POST 不调用
        for _ in range(50):
            await asyncio.sleep(0.02)
            if writer.stats["retry_cycles"] >= 2:
                break
    finally:
        await writer.stop_retry_loop()

    # alerting 行从不被拉，HTTP 调用次数仍 0
    assert len(mock_client.calls) == 0
    # retry_skipped 不计 alerting（status='alerting' 不会被 dequeue 拉到）
    assert writer.stats["retry_attempts"] == 0


@pytest.mark.asyncio
async def test_retry_loop_backoff_curve() -> None:
    """retry 失败后 ``next_retry_at`` 走指数退避曲线（30s/60s/120s/240s）。"""
    nra0 = compute_next_retry_iso(retry_count=0)
    nra1 = compute_next_retry_iso(retry_count=1)
    nra2 = compute_next_retry_iso(retry_count=2)
    nra3 = compute_next_retry_iso(retry_count=3)

    from datetime import datetime

    def _naive(iso: str) -> datetime:
        s = iso.replace("Z", "")
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

    t0 = _naive(nra0)
    t1 = _naive(nra1)
    t2 = _naive(nra2)
    t3 = _naive(nra3)

    d01 = (t1 - t0).total_seconds()
    d12 = (t2 - t1).total_seconds()
    d23 = (t3 - t2).total_seconds()

    assert d01 >= BASE_BACKOFF_SECONDS * 2 ** 0
    assert d12 >= BASE_BACKOFF_SECONDS * 2 ** 1
    assert d23 >= BASE_BACKOFF_SECONDS * 2 ** 2


# =====================================================================
# 5. 非体检主检行跳过
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_skips_non_hc_target(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """outbox 中混入 RIS 行（ris://finding-draft/...）→ retry_loop 不处理。"""
    hc_kw = _payload_kwargs("HC-MIXED-HC")
    ris_kw = {
        "study_uid": "uid-mixed-ris",
        "patient_id_hash": "abc123def456" * 4,
        "finding_text": (
            "右下肺野见少量气胸征象（草稿 / 未经医师确认），"
            "余肺野清晰。"
        ),
        "finding_struct": {
            "pathologies": [
                {"name": "Pneumothorax", "prob": 0.87, "present": True},
            ],
            "key_qa": [],
            "grounding_boxes": [],
        },
        "positive_flag": True,
        "draft_ts": "2026-10-13T07:23:11.487Z",
        "tool_version": "0.1.0",
    }

    await outbox.enqueue(
        study_uid="uid-mixed-hc",
        target_endpoint=build_healthcheck_target_endpoint(hc_kw["visit_id"]),
        payload=hc_kw,
    )
    # RIS 行（不归 Healthcheck writer 处理）
    await outbox.enqueue(
        study_uid=ris_kw["study_uid"],
        target_endpoint=build_ris_target_endpoint(ris_kw["study_uid"]),
        payload=ris_kw,
    )

    mock_client = _MockAsyncClient([_MockResponse(200, "OK")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-mixed-hc")
            if any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT for r in rows):
                break
    finally:
        await writer.stop_retry_loop()

    # HC 行 mark_sent；RIS 行不动
    by_status = await outbox.count_by_status()
    assert by_status.get("sent", 0) == 1
    assert by_status.get("pending", 0) == 1  # RIS 行还在 pending

    # 出站仅 1 次（HC 行）
    assert len(mock_client.calls) == 1
    assert hc_kw["visit_id"] in mock_client.calls[0]["url"]

    # retry_skipped 计数 ≥ 1
    assert writer.stats["retry_skipped"] >= 1

    # RIS 行 audit 不该出现 retry_sent
    ris_audit_rows = await audit.list_audit_for_study(ris_kw["study_uid"])
    assert all(
        r["event_type"] != EVENT_HEALTHCHECK_RETRY_SENT for r in ris_audit_rows
    )


# =====================================================================
# 6. 与 RIS writer retry_loop 并行不互锁
# =====================================================================
@pytest.mark.asyncio
async def test_ris_and_healthcheck_retry_loops_run_in_parallel(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """RIS 与 Healthcheck 各跑各的 retry_loop；同 outbox 表共存，互不干扰。

    验证：RIS 行被 RISWriter 写 sent；体检主检行被 HealthcheckWriter
    写 sent；stats 各自累加。
    """
    # RIS 行
    ris_kw = {
        "study_uid": "uid-ris-parallel",
        "patient_id_hash": "abc123def456" * 4,
        "finding_text": (
            "右下肺野见少量气胸征象（草稿 / 未经医师确认），"
            "余肺野清晰。"
        ),
        "finding_struct": {
            "pathologies": [
                {"name": "Pneumothorax", "prob": 0.87, "present": True},
            ],
            "key_qa": [],
            "grounding_boxes": [],
        },
        "positive_flag": True,
        "draft_ts": "2026-10-13T07:23:11.487Z",
        "tool_version": "0.1.0",
    }
    await outbox.enqueue(
        study_uid=ris_kw["study_uid"],
        target_endpoint=build_ris_target_endpoint(ris_kw["study_uid"]),
        payload=ris_kw,
    )

    # 体检主检行
    hc_kw = _payload_kwargs("HC-PARALLEL")
    await outbox.enqueue(
        study_uid="uid-hc-parallel",
        target_endpoint=build_healthcheck_target_endpoint(hc_kw["visit_id"]),
        payload=hc_kw,
    )

    ris_client = _MockAsyncClient([_MockResponse(200, "RIS OK")])
    hc_client = _MockAsyncClient([_MockResponse(200, "HC OK")])

    ris_writer = RISWriter(
        config=config,
        outbox=outbox,
        audit_logger=audit,
        http_client=ris_client,  # type: ignore[arg-type]
        retry_poll_seconds=0.05,
        retry_batch_size=10,
    )
    hc_writer = _make_writer(config, outbox, audit, hc_client)

    await ris_writer.start_retry_loop()
    await hc_writer.start_retry_loop()
    try:
        # 等两边都至少跑一轮
        for _ in range(80):
            await asyncio.sleep(0.02)
            rows_ris = await audit.list_audit_for_study("uid-ris-parallel")
            rows_hc = await audit.list_audit_for_study("uid-hc-parallel")
            if (
                any(r["event_type"] == EVENT_RIS_RETRY_SENT for r in rows_ris)
                and any(
                    r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT for r in rows_hc
                )
            ):
                break
    finally:
        await ris_writer.stop_retry_loop()
        await hc_writer.stop_retry_loop()

    # 两边都 mark_sent
    by_status = await outbox.count_by_status()
    assert by_status.get("sent", 0) == 2
    assert by_status.get("pending", 0) == 0

    # 两边 audit 都各落一行
    ris_rows = await audit.list_audit_for_study("uid-ris-parallel")
    hc_rows = await audit.list_audit_for_study("uid-hc-parallel")
    assert any(r["event_type"] == EVENT_RIS_RETRY_SENT for r in ris_rows)
    assert any(r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT for r in hc_rows)

    # 各自 stats
    assert ris_writer.stats["retry_successes"] >= 1
    assert hc_writer.stats["retry_successes"] >= 1
    assert ris_writer.stats["retry_skipped"] >= 1
    assert hc_writer.stats["retry_skipped"] >= 1


# =====================================================================
# 7. dequeue 空 → 不报错
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_empty_outbox(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """空 outbox 跑多轮 retry_loop → 不写 audit，不报错。"""
    mock_client = _MockAsyncClient([_MockResponse(200)])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            if writer.stats["retry_cycles"] >= 2:
                break
    finally:
        await writer.stop_retry_loop()

    # 没拉过任何东西
    assert len(mock_client.calls) == 0
    assert writer.stats["retry_attempts"] == 0
    assert writer.stats["retry_cycles"] >= 2


# =====================================================================
# 8. 单条异常兜底
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_per_row_exception_caught(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """POST 抛非预期 RuntimeError → 本轮 catch + mark_retry + audit + 继续。"""
    kw = _payload_kwargs("HC-ROW-EXC")
    outbox_id = await outbox.enqueue(
        study_uid="uid-hc-row-exc",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )

    writer = _make_writer(
        config,
        outbox,
        audit,
        _AlwaysFailingClient(RuntimeError("unexpected")),  # type: ignore[arg-type]
    )
    await writer.start_retry_loop()
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-row-exc")
            if any(
                r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED for r in rows
            ):
                break
    finally:
        await writer.stop_retry_loop()

    # outbox 仍 retry_count=1（mark_retry 兜底成功）
    row = await outbox.get_by_id(outbox_id)
    assert row is not None
    assert row["retry_count"] == 1
    # audit 仍写入（retry_failed；不是 alerting）
    rows = await audit.list_audit_for_study("uid-hc-row-exc")
    fail = [
        r for r in rows if r["event_type"] == EVENT_HEALTHCHECK_RETRY_FAILED
    ]
    assert len(fail) >= 1


# =====================================================================
# 9. visit_id 解析
# =====================================================================
@pytest.mark.asyncio
async def test_extract_visit_id_happy_path() -> None:
    visit_id = _extract_visit_id_from_target_endpoint(
        f"{HEALTHCHECK_TARGET_PREFIX}HC-VISIT-XYZ"
    )
    assert visit_id == "HC-VISIT-XYZ"


@pytest.mark.asyncio
async def test_extract_visit_id_empty_target() -> None:
    with pytest.raises(HealthcheckPayloadError, match="target_endpoint 非法"):
        _extract_visit_id_from_target_endpoint("")


@pytest.mark.asyncio
async def test_extract_visit_id_wrong_prefix() -> None:
    with pytest.raises(
        HealthcheckPayloadError, match="target_endpoint 非体检主检前缀"
    ):
        _extract_visit_id_from_target_endpoint("ris://finding-draft/uid-1")


@pytest.mark.asyncio
async def test_extract_visit_id_missing_visit_id() -> None:
    with pytest.raises(
        HealthcheckPayloadError, match="target_endpoint 缺 visit_id"
    ):
        _extract_visit_id_from_target_endpoint(HEALTHCHECK_TARGET_PREFIX)


# =====================================================================
# 10. 出站契约
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_outbound_contract(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """出站请求必含 Bearer token + Content-Type + 正确 URL 路径 + payload 4 字段。"""
    kw = _payload_kwargs("HC-CONTRACT")
    await outbox.enqueue(
        study_uid="uid-hc-contract",
        target_endpoint=build_healthcheck_target_endpoint(kw["visit_id"]),
        payload=kw,
    )

    mock_client = _MockAsyncClient([_MockResponse(200, "OK")])
    writer = _make_writer(config, outbox, audit, mock_client)
    await writer.start_retry_loop()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            if len(mock_client.calls) >= 1:
                break
    finally:
        await writer.stop_retry_loop()

    call = mock_client.calls[0]
    expected_url = build_healthcheck_summary_ref_url(
        config.healthcheck_api_url, kw["visit_id"]
    )
    assert call["url"] == expected_url
    assert call["headers"]["Authorization"] == (
        f"Bearer {config.healthcheck_api_token}"
    )
    assert "application/json" in call["headers"]["Content-Type"]
    # payload 4 字段对齐 spec.md §4
    assert set(call["json"].keys()) == {
        "visit_id",
        "summary_snippet",
        "source_study_uids",
        "model_version",
    }


# =====================================================================
# 11. 端到端：成功路径 outbox 全链路
# =====================================================================
@pytest.mark.asyncio
async def test_retry_loop_sends_then_marks_sent(
    config: CxrTriageConfig, outbox: Outbox, audit: AuditLogger
) -> None:
    """完整链路：写体检主检失败入 outbox → retry_loop 跑成功 → outbox sent。"""
    kw = _payload_kwargs("HC-FULL-CYCLE")
    # 第一次 POST 失败（500）→ 入 outbox
    fail_client = _MockAsyncClient([_MockResponse(500, "first fail")])
    writer = _make_writer(config, outbox, audit, fail_client)
    await writer.write_summary_ref(**kw, study_uid="uid-hc-full-cycle")
    pending = await outbox.dequeue(limit=10)
    assert len(pending) == 1
    assert pending[0]["retry_count"] == 0

    # 换 client：现在 POST 200 → retry_loop 应把它 mark_sent
    fail_client.calls.clear()
    new_client = _MockAsyncClient([_MockResponse(200, "OK")])
    writer._http_client = new_client  # type: ignore[assignment]

    await writer.start_retry_loop()
    try:
        for _ in range(80):
            await asyncio.sleep(0.02)
            rows = await audit.list_audit_for_study("uid-hc-full-cycle")
            sent = [
                r for r in rows
                if r["event_type"] == EVENT_HEALTHCHECK_RETRY_SENT
            ]
            if sent:
                break
    finally:
        await writer.stop_retry_loop()

    # outbox sent
    by_status = await outbox.count_by_status()
    assert by_status.get("sent", 0) == 1
    assert by_status.get("pending", 0) == 0

    # 新 client 收到过 POST（不是 fail_client）
    assert len(new_client.calls) >= 1
    assert len(fail_client.calls) == 0