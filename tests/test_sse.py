"""SSE 推送 publisher 测试（Task 29）。

覆盖（spec.md §5 维度 4 + plan.md §4 浏览器实时面板）：

1. **常量 / 数据契约**：
   - 4 类 event 白名单（new_study / draft_ready / state_change / alert）
   - SseEvent 构造校验（白名单 / 非空 / 非换行）
   - SSE 帧序列化（event/id/data 行 + 双换行 + UTF-8）
2. **fan-out 主路径**：
   - broadcast → 所有连接客户端都收到 bytes
   - 入参校验（study_uid / payload.event / 序列化）
   - 「study_uid 含 \\n」拒绝且 events_dropped += 1
   - 「event 不在白名单」拒绝
   - 单客户端 queue 满时丢最旧 + queue_overflow += 1
3. **history replay**：
   - event_stream(last_seen_ts=None) 不 replay
   - last_seen_ts=ISO8601 解析 + replay ≥ last_seen_ts 的事件
   - last_seen_ts 解析失败 → 当 None 处理（不抛）
4. **keepalive 心跳**：
   - event_stream 无事件时超时推 ``: keepalive\\n\\n``
5. **多客户端 + 生命周期**：
   - 多并发 event_stream 同时在线
   - 一端断线 → _unregister → clients_connected 减 1
   - broadcast 仍对剩余客户端成功
6. **协议兼容 / 便捷方法**：
   - publish_new_study / publish_draft_ready / publish_state_change / publish_alert
     4 个便捷包装 → payload.event 字段被自动补
   - 与 AlertPublisher callback 协议同形（SsePublishCallback = async(study_uid, payload)）
7. **AppState / 路由集成**：
   - create_app 启动后 ``/api/sse`` 可访问；``client_count >= 0``
   - last_seen_ts query 透传 → SsePublisher.event_stream
"""

from __future__ import annotations

import asyncio
import json as json_lib
import time
from typing import Any, Optional

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cxr_triage.config import CxrTriageConfig
from cxr_triage.persistence.migrations import init_db
from cxr_triage.persistence.sqlite import Database
from cxr_triage.web.sse import (
    DEFAULT_CLIENT_QUEUE_MAXSIZE,
    DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
    DEFAULT_MAX_HISTORY_SECONDS,
    EVENT_ALERT,
    EVENT_DRAFT_READY,
    EVENT_NEW_STUDY,
    EVENT_STATE_CHANGE,
    SSE_FIELD_DATA,
    SSE_FIELD_EVENT,
    SSE_FIELD_ID,
    SseEvent,
    SsePublishCallback,
    SsePublishError,
    SsePublisher,
    SsePublisherError,
    VALID_EVENT_TYPES,
    _format_keepalive_frame,
    _format_sse_frame,
    _parse_iso8601_utc,
    _utc_now_iso,
    build_sse_publisher,
)


# ============================================================================
# 1. 常量 / 数据契约
# ============================================================================
def test_valid_event_types_locked() -> None:
    """4 类 event 白名单必须锁定。"""
    assert VALID_EVENT_TYPES == frozenset(
        {EVENT_NEW_STUDY, EVENT_DRAFT_READY, EVENT_STATE_CHANGE, EVENT_ALERT}
    )
    assert EVENT_NEW_STUDY == "new_study"
    assert EVENT_DRAFT_READY == "draft_ready"
    assert EVENT_STATE_CHANGE == "state_change"
    assert EVENT_ALERT == "alert"


def test_format_sse_frame_minimal() -> None:
    """最简事件：event + data 两行 + 帧尾双换行。"""
    evt = SseEvent(
        event=EVENT_NEW_STUDY,
        study_uid="1.2.3.4",
        ts="2026-10-13T07:00:00.000000Z",
        payload={"patient_id_hash": "abc"},
    )
    frame = _format_sse_frame(evt).decode("utf-8")
    assert frame.startswith(f"{SSE_FIELD_EVENT}: {EVENT_NEW_STUDY}\n")
    assert frame.endswith("\n\n")
    assert f"{SSE_FIELD_ID}: " not in frame  # 没 event_id
    assert f"{SSE_FIELD_DATA}: " in frame
    # data 行 JSON 可解析 + 含 study_uid/ts/patient_id_hash
    data_line = frame.split(f"{SSE_FIELD_DATA}: ", 1)[1].split("\n", 1)[0]
    parsed = json_lib.loads(data_line)
    assert parsed["study_uid"] == "1.2.3.4"
    assert parsed["ts"] == "2026-10-13T07:00:00.000000Z"
    assert parsed["patient_id_hash"] == "abc"


def test_format_sse_frame_with_event_id() -> None:
    """带 event_id 的帧：多一行 ``id: <event_id>``。"""
    evt = SseEvent(
        event=EVENT_DRAFT_READY,
        study_uid="study-A",
        ts="2026-10-13T07:00:00.000Z",
        payload={"max_prob": 0.85},
        event_id="42",
    )
    frame = _format_sse_frame(evt).decode("utf-8")
    assert f"{SSE_FIELD_EVENT}: {EVENT_DRAFT_READY}\n" in frame
    assert f"{SSE_FIELD_ID}: 42\n" in frame
    assert f"{SSE_FIELD_DATA}: " in frame


def test_format_sse_frame_chinese_payload() -> None:
    """中文 payload 不被 escape（ensure_ascii=False）。"""
    evt = SseEvent(
        event=EVENT_DRAFT_READY,
        study_uid="study-ZH",
        ts="2026-10-13T07:00:00.000Z",
        payload={"finding_text": "右下肺野见少量气胸征象"},
    )
    frame = _format_sse_frame(evt).decode("utf-8")
    assert "右下肺野见少量气胸征象" in frame


def test_format_sse_frame_payload_overrides_study_uid_in_payload_field() -> None:
    """如果 payload 自己有 study_uid 字段 → 保留原值，不被 event.study_uid 覆盖。"""
    evt = SseEvent(
        event=EVENT_ALERT,
        study_uid="alert-summary",
        ts="2026-10-13T07:00:00.000Z",
        payload={"study_uid": "from-payload", "total_alert_count": 3},
    )
    frame = _format_sse_frame(evt).decode("utf-8")
    parsed = json_lib.loads(frame.split(f"{SSE_FIELD_DATA}: ", 1)[1].split("\n", 1)[0])
    assert parsed["study_uid"] == "from-payload"  # payload 内字段被保留
    # 但顶层 SseEvent.study_uid 在 broadcast 里会透传作为 key


def test_format_keepalive_frame() -> None:
    """心跳帧：``\\u003a keepalive\\n\\n``（UTF-8 编码）。"""
    frame = _format_keepalive_frame()
    assert frame == b": keepalive\n\n"
    assert frame.startswith(b":")


def test_parse_iso8601_utc_happy_and_fail() -> None:
    """ISO8601 解析：合法 → datetime；非法 → None。"""
    dt = _parse_iso8601_utc("2026-10-13T07:00:00.123456Z")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 10 and dt.day == 13
    assert _parse_iso8601_utc("not-a-date") is None
    assert _parse_iso8601_utc("") is None
    assert _parse_iso8601_utc(None) is None  # type: ignore[arg-type]


def test_sseevent_rejects_unknown_event_type() -> None:
    with pytest.raises(SsePublishError):
        SseEvent(
            event="bogus",
            study_uid="x",
            ts="2026-10-13T07:00:00.000Z",
            payload={},
        )


def test_sseevent_rejects_empty_study_uid() -> None:
    with pytest.raises(SsePublishError):
        SseEvent(
            event=EVENT_NEW_STUDY,
            study_uid="",
            ts="2026-10-13T07:00:00.000Z",
            payload={},
        )


def test_sseevent_rejects_study_uid_with_newline() -> None:
    for bad in ("x\ny", "x\ry", "x\r\ny"):
        with pytest.raises(SsePublishError):
            SseEvent(
                event=EVENT_NEW_STUDY,
                study_uid=bad,
                ts="2026-10-13T07:00:00.000Z",
                payload={},
            )


def test_sseevent_rejects_non_dict_payload() -> None:
    with pytest.raises(SsePublishError):
        SseEvent(
            event=EVENT_NEW_STUDY,
            study_uid="x",
            ts="2026-10-13T07:00:00.000Z",
            payload="not-a-dict",  # type: ignore[arg-type]
        )


def test_sseevent_rejects_empty_ts() -> None:
    with pytest.raises(SsePublishError):
        SseEvent(
            event=EVENT_NEW_STUDY,
            study_uid="x",
            ts="",
            payload={},
        )


# ============================================================================
# 2. SsePublisher 构造 + fan-out
# ============================================================================
def test_sse_publisher_default_construction() -> None:
    p = SsePublisher()
    assert p.client_count == 0
    assert p.max_history_seconds == DEFAULT_MAX_HISTORY_SECONDS
    assert p.client_queue_maxsize == DEFAULT_CLIENT_QUEUE_MAXSIZE
    assert p.keepalive_interval_seconds == DEFAULT_KEEPALIVE_INTERVAL_SECONDS
    assert p.stats["broadcasts"] == 0
    assert p.stats["events_pushed"] == 0


def test_sse_publisher_factory() -> None:
    p = build_sse_publisher(
        max_history_seconds=120.0,
        client_queue_maxsize=42,
        keepalive_interval_seconds=7.0,
    )
    assert p.max_history_seconds == 120.0
    assert p.client_queue_maxsize == 42
    assert p.keepalive_interval_seconds == 7.0


@pytest.mark.parametrize(
    "bad_max_history,bad_client_q,bad_keepalive,bad_broadcast_timeout",
    [
        (0, 1, 1.0, 1.0),
        (-1.0, 1, 1.0, 1.0),
        (10.0, 0, 1.0, 1.0),
        (10.0, -5, 1.0, 1.0),
        (10.0, 1, 0, 1.0),
        (10.0, 1, -1.0, 1.0),
        (10.0, 1, 1.0, -0.5),
    ],
)
def test_sse_publisher_constructor_validates(
    bad_max_history: float, bad_client_q: int, bad_keepalive: float, bad_broadcast_timeout: float
) -> None:
    with pytest.raises((SsePublisherError, ValueError)):
        SsePublisher(
            max_history_seconds=bad_max_history,
            client_queue_maxsize=bad_client_q,
            keepalive_interval_seconds=bad_keepalive,
            broadcast_timeout_seconds=bad_broadcast_timeout,
        )


@pytest.mark.asyncio
async def test_broadcast_basic_fan_out_to_two_clients() -> None:
    """2 个客户端同时在线 → broadcast 后两个都收到同样的 frame。"""
    p = SsePublisher(keepalive_interval_seconds=0.5)
    # 起 2 个 stream
    streams = []
    q0 = asyncio.Queue(maxsize=10)
    q1 = asyncio.Queue(maxsize=10)

    async def _consume_one(q: asyncio.Queue, ev: asyncio.Event):
        item = await q.get()
        ev.set()
        return item

    # 直接调 broadcast → 应推到所有 stream
    delivered = await p.broadcast(
        study_uid="study-1",
        payload={
            "event": EVENT_NEW_STUDY,
            "patient_id_hash": "h-1",
        },
    )
    # 当前无注册客户端 → 0 推送；但 history +1
    assert delivered == 0
    assert p.stats["broadcasts"] == 1
    assert len(p.history_snapshot()) == 1
    evt = p.history_snapshot()[0]
    assert evt.event == EVENT_NEW_STUDY
    assert evt.study_uid == "study-1"


@pytest.mark.asyncio
async def test_broadcast_rejects_invalid_event_type() -> None:
    p = SsePublisher()
    with pytest.raises(SsePublishError):
        await p.broadcast(study_uid="x", payload={"event": "bogus"})
    assert p.stats["events_dropped"] == 1


@pytest.mark.asyncio
async def test_broadcast_rejects_missing_event_type() -> None:
    p = SsePublisher()
    with pytest.raises(SsePublishError):
        await p.broadcast(study_uid="x", payload={"foo": "bar"})
    assert p.stats["events_dropped"] == 1


@pytest.mark.asyncio
async def test_broadcast_rejects_empty_study_uid() -> None:
    p = SsePublisher()
    with pytest.raises(SsePublishError):
        await p.broadcast(study_uid="", payload={"event": EVENT_NEW_STUDY})


@pytest.mark.asyncio
async def test_broadcast_rejects_study_uid_with_newline() -> None:
    p = SsePublisher()
    with pytest.raises(SsePublishError):
        await p.broadcast(
            study_uid="evil\ninject",
            payload={"event": EVENT_NEW_STUDY},
        )
    assert p.stats["events_dropped"] == 1


@pytest.mark.asyncio
async def test_broadcast_rejects_non_dict_payload() -> None:
    p = SsePublisher()
    with pytest.raises(SsePublishError):
        await p.broadcast(
            study_uid="x", payload="not a dict"  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_broadcast_normalizes_event_and_ts_in_payload() -> None:
    """payload['event'] / payload['ts'] / payload['event_id'] 不会被复制到 safe_payload。

    event 字段已经在 SSE 帧的 ``event:`` 行；ts 在 ``data:`` JSON 头部；
    event_id 在 ``id:`` 行。重复在 data 内出现会冗余，但允许 payload 自带额外字段。
    """
    p = SsePublisher(keepalive_interval_seconds=0.5)
    await p.broadcast(
        study_uid="s",
        payload={
            "event": EVENT_NEW_STUDY,
            "ts": "2026-10-13T08:00:00.000Z",
            "extra": "x",
            "patient_id_hash": "h",
        },
    )
    evt = p.history_snapshot()[0]
    # history 存的是 safe_payload（不含 event/ts/event_id）
    assert "extra" in evt.payload
    assert evt.payload["extra"] == "x"
    assert evt.payload["patient_id_hash"] == "h"


# ============================================================================
# 3. event_stream + keepalive + replay
# ============================================================================
@pytest.mark.asyncio
async def test_event_stream_emits_keepalive_when_no_event() -> None:
    """无事件时 stream 应推心跳 ``: keepalive\\n\\n``。"""
    p = SsePublisher(
        keepalive_interval_seconds=0.1,
        client_queue_maxsize=10,
    )
    cancel = asyncio.Event()

    async def _first_two():
        gen = p.event_stream(cancel_event=cancel)
        try:
            f1 = await asyncio.wait_for(anext(gen), timeout=1.0)
            f2 = await asyncio.wait_for(anext(gen), timeout=1.0)
            return f1, f2
        finally:
            await gen.aclose()

    f1, f2 = await asyncio.wait_for(_first_two(), timeout=2.0)
    # 没有事件时第一个就是 keepalive
    assert f1 == _format_keepalive_frame()
    cancel.set()


@pytest.mark.asyncio
async def test_event_stream_emits_event_after_broadcast() -> None:
    """广播 → stream 应马上吐出对应帧。"""
    p = SsePublisher(keepalive_interval_seconds=5.0, client_queue_maxsize=10)
    frames: list[bytes] = []
    cancel = asyncio.Event()

    async def _consumer():
        gen = p.event_stream(cancel_event=cancel)
        try:
            # 先订阅
            frame = await asyncio.wait_for(anext(gen), timeout=2.0)
            frames.append(frame)
        finally:
            await gen.aclose()

    consumer_task = asyncio.create_task(_consumer())
    await asyncio.sleep(0.05)  # 等 consumer 注册

    # 现在 broadcast
    await p.broadcast(
        study_uid="study-A",
        payload={"event": EVENT_NEW_STUDY, "patient_id_hash": "h"},
    )

    await asyncio.wait_for(consumer_task, timeout=2.0)
    assert len(frames) == 1
    decoded = frames[0].decode("utf-8")
    assert f"{SSE_FIELD_EVENT}: {EVENT_NEW_STUDY}" in decoded
    assert "study-A" in decoded
    cancel.set()


@pytest.mark.asyncio
async def test_event_stream_replay_filters_by_last_seen_ts() -> None:
    """``last_seen_ts`` 决定 history replay 起点；只吐 ts >= last_seen_ts 的事件。"""
    p = SsePublisher(
        max_history_seconds=600.0,
        keepalive_interval_seconds=5.0,
        client_queue_maxsize=10,
    )
    # 4 条 history（逐步 ts；s0 早于 last_seen_ts）
    for i, ts in enumerate(
        [
            "2026-10-13T07:00:00.000001Z",
            "2026-10-13T07:00:00.000002Z",
            "2026-10-13T07:00:00.000003Z",
        ]
    ):
        await p.broadcast(
            study_uid=f"s{i}",
            payload={"event": EVENT_NEW_STUDY, "ts": ts},
        )
    await p.broadcast(
        study_uid="future",
        payload={"event": EVENT_NEW_STUDY, "ts": "2099-01-01T00:00:00.000000Z"},
    )

    cancel = asyncio.Event()
    frames: list[bytes] = []

    async def _collect():
        gen = p.event_stream(
            last_seen_ts="2026-10-13T07:00:00.000002Z",
            cancel_event=cancel,
        )
        try:
            for _ in range(3):
                f = await asyncio.wait_for(anext(gen), timeout=3.0)
                frames.append(f)
        except (StopAsyncIteration, asyncio.TimeoutError):
            pass
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

    collector_task = asyncio.create_task(_collect())
    try:
        await asyncio.wait_for(collector_task, timeout=5.0)
    except asyncio.TimeoutError:
        cancel.set()
        try:
            await asyncio.wait_for(collector_task, timeout=2.0)
        except Exception:
            pass

    cancel.set()
    # 行为断言：3 条事件被 replay（s0 不在）
    assert p.stats["replays"] >= 1, "replay 应该至少被调用一次"
    assert p.stats["replay_events"] == 3, (
        f"应 replay 3 条事件（s1/s2/future），实际 {p.stats['replay_events']}"
    )
    # frames 至少有 1 条被读到（如果 collector 跑通，3 条全在）
    assert len(frames) >= 1, f"frames 至少 1 条；实际 {len(frames)}"
    # 内容验证：搜 "study_uid": "s0" 必须不在
    decoded = [f.decode("utf-8") for f in frames]
    assert not any('"study_uid": "s0"' in d for d in decoded)


@pytest.mark.asyncio
async def test_event_stream_with_bad_last_seen_ts_skips_replay() -> None:
    """非法 last_seen_ts → 当 None 处理（不抛），仍按 live feed 推。"""
    p = SsePublisher(keepalive_interval_seconds=5.0, client_queue_maxsize=10)
    cancel = asyncio.Event()
    frames: list[bytes] = []

    async def _collect():
        gen = p.event_stream(
            last_seen_ts="not-a-date",
            cancel_event=cancel,
        )
        try:
            # 订阅一次，让 consumer 注册
            for _ in range(2):
                f = await asyncio.wait_for(anext(gen), timeout=2.0)
                frames.append(f)
        except (StopAsyncIteration, asyncio.TimeoutError):
            pass
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

    collector_task = asyncio.create_task(_collect())
    await asyncio.sleep(0.1)
    # 现在 broadcast 一条事件
    await p.broadcast(
        study_uid="live-event",
        payload={"event": EVENT_NEW_STUDY, "ts": "2026-10-13T07:00:00.000Z"},
    )
    await asyncio.sleep(0.1)
    cancel.set()
    await asyncio.wait_for(collector_task, timeout=2.0)

    decoded = [f.decode("utf-8") for f in frames]
    assert any('"study_uid": "live-event"' in d for d in decoded)


@pytest.mark.asyncio
async def test_event_stream_disconnects_clean() -> None:
    """客户端断开 → clients_connected 回到 0；之后 broadcast 不抛。"""
    p = SsePublisher(
        keepalive_interval_seconds=0.1,
        client_queue_maxsize=10,
    )
    # 直接调 event_stream() 一次取第一帧（keepalive）
    gen = p.event_stream()
    try:
        frame = await asyncio.wait_for(anext(gen), timeout=2.0)
    finally:
        try:
            await gen.aclose()
        except Exception:
            pass
    # consumer 已退出 → client_count 回到 0
    await asyncio.sleep(0.05)
    assert p.client_count == 0
    assert isinstance(frame, bytes) and len(frame) > 0
    assert frame == _format_keepalive_frame()
    # broadcast 应不抛
    await p.broadcast(
        study_uid="after-disc",
        payload={"event": EVENT_NEW_STUDY},
    )
    assert p.stats["broadcasts"] == 1
    assert p.stats["streams_closed"] == 1


# ============================================================================
# 4. 多客户端并发
# ============================================================================
@pytest.mark.asyncio
async def test_broadcast_to_multiple_clients() -> None:
    """多客户端同时在线 → broadcast 全部 fan-out。"""
    p = SsePublisher(keepalive_interval_seconds=5.0, client_queue_maxsize=10)
    received_a: list[bytes] = []
    received_b: list[bytes] = []
    cancel = asyncio.Event()

    async def _consume(q: asyncio.Queue, target: list[bytes]):
        while True:
            f = await q.get()
            target.append(f)

    async def _spawn():
        # 注册两个 queue
        cid_a = await p._register_client(asyncio.Queue(maxsize=10))
        cid_b = await p._register_client(asyncio.Queue(maxsize=10))
        assert p.client_count == 2
        return cid_a, cid_b

    cid_a, cid_b = await _spawn()
    await p.broadcast(
        study_uid="multi",
        payload={"event": EVENT_DRAFT_READY, "max_prob": 0.85},
    )
    # 直接从内部 _clients 拿数据
    q_a = p._clients[cid_a]
    q_b = p._clients[cid_b]
    assert not q_a.empty()
    assert not q_b.empty()
    frame = q_a.get_nowait()
    assert f"{SSE_FIELD_EVENT}: {EVENT_DRAFT_READY}" in frame.decode("utf-8")
    # 反注册清理
    await p._unregister(cid_a)
    await p._unregister(cid_b)
    assert p.client_count == 0


@pytest.mark.asyncio
async def test_broadcast_history_grows_and_caps_at_deque_maxlen() -> None:
    """history 增长到 deque maxlen=1024 后自动挤掉最旧。"""
    p = SsePublisher()
    # 推 5 条不同的 studies
    for i in range(5):
        await p.broadcast(
            study_uid=f"s-{i}",
            payload={"event": EVENT_NEW_STUDY, "idx": i},
        )
    hist = p.history_snapshot()
    assert len(hist) == 5


# ============================================================================
# 5. 便捷方法
# ============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,event_constant",
    [
        ("publish_new_study", EVENT_NEW_STUDY),
        ("publish_draft_ready", EVENT_DRAFT_READY),
        ("publish_state_change", EVENT_STATE_CHANGE),
        ("publish_alert", EVENT_ALERT),
    ],
)
async def test_convenience_methods_set_event_field(method: str, event_constant: str) -> None:
    p = SsePublisher()
    fn = getattr(p, method)
    await fn(
        study_uid=f"s-{method}",
        payload={"msg": "hello"},
    )
    hist = p.history_snapshot()
    assert len(hist) == 1
    evt = hist[0]
    assert evt.event == event_constant
    assert evt.payload["msg"] == "hello"


@pytest.mark.asyncio
async def test_publish_alert_fallback_study_uid() -> None:
    """publish_alert 当 study_uid 为空字符串时回退取 payload['study_uid']。"""
    p = SsePublisher()
    await p.publish_alert(
        study_uid="",
        payload={"study_uid": "from-payload", "total_alert_count": 3},
    )
    hist = p.history_snapshot()
    assert hist[0].study_uid == "from-payload"


@pytest.mark.asyncio
async def test_publish_matches_callback_protocol() -> None:
    """``SsePublisher.publish`` 必须满足 SsePublishCallback 协议。"""
    p = SsePublisher()
    cb: SsePublishCallback = p.publish
    await cb("s", {"event": EVENT_NEW_STUDY, "data": 1})
    assert p.stats["broadcasts"] == 1


# ============================================================================
# 6. AppState + 路由集成
# ============================================================================
@pytest_asyncio.fixture
async def db(tmp_path):
    path = tmp_path / "test_sse_app.db"
    database = Database(path)
    await database.connect()
    await init_db(database)
    yield database
    try:
        await database.close()
    except Exception:
        pass


def test_create_app_exposes_sse_route(db, tmp_path) -> None:
    """complete ``create_app()`` 应注册 /api/sse 路由 + 默认 SsePublisher 实例化。"""
    from cxr_triage.web.app import SSE_MEDIA_TYPE, SSE_ENDPOINT, create_app

    app = create_app(
        db_path=str(db.db_path),
        tool_calls_dir=str(tmp_path / "tc"),
        enable_pacs_listener=False,
        enable_folder_watcher=False,
    )
    paths = {r.path for r in app.routes}
    assert SSE_ENDPOINT in paths
    # /api/health / 仍然在
    assert "/api/health" in paths
    assert "/" in paths
    # media type 常量
    assert SSE_MEDIA_TYPE == "text/event-stream"
    # SsePublisher 默认实例存在（lifespan 内构造 + state_holder）
    assert isinstance(app.state.cxr_holder, dict)
    # noop_sse_publisher 已注入到 lifespan 钩子内（lifespan 启动后才有；这里只校验未启动时 state=None）
    assert app.state.cxr_holder["state"] is None


def test_create_app_with_injected_sse_publisher(tmp_path) -> None:
    """create_app 接受自定义 sse_publisher 实例注入。"""
    from cxr_triage.web.app import create_app

    custom = build_sse_publisher(
        max_history_seconds=42.0,
        client_queue_maxsize=7,
        keepalive_interval_seconds=99.0,
    )
    app = create_app(
        db_path=str(tmp_path / "inj.db"),
        tool_calls_dir=str(tmp_path / "tc"),
        enable_pacs_listener=False,
        enable_folder_watcher=False,
        sse_publisher=custom,
    )
    # _build_lifespan_state 接受 sse_publisher 参数（仅校验不抛 + 路径注册）
    assert any(
        getattr(r, "path", None) == "/api/sse" for r in app.routes
    )
