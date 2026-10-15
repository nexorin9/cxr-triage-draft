"""SSE 推送 publisher — 浏览器实时面板的核心通道。

设计原则（spec.md §5 维度 4 + §6 工作闭环 + plan.md §4 浏览器实时面板）：

1. **4 类事件**：
   - ``new_study``       入队（PACS 落片 → incoming_studies 完成基础元数据解析）
   - ``draft_ready``     草稿就绪（StateGraph 5 件工具跑完 → triage_queue 已入列）
   - ``state_change``    状态变更（jr 确认 / reviewer 复核 / sr 签发 / 驳回）
   - ``alert``           失败告警（Task 27 outbox 连续失败 → 顶部红 banner）

2. **fan-out 模型**：单进程内 ``asyncio.Queue`` 一对一推送；每客户端
   独享一个 ``asyncio.Queue`` 避免互相阻塞；disconnect 时自动出队。

3. **history 重连**：客户端 ``?last_seen_ts=2026-10-13T07:00:00Z`` 时
   先 replay ``max_history_seconds`` 内的事件，再切到 live feed（断线
   重连期间出的事件不丢，飞检复盘也对得上事件顺序）。

4. **不静默丢数据**：bottleneck 时丢头不丢尾的「新事件覆盖」策略，
   每客户端最多 ``client_queue_maxsize`` 缓冲；超出覆盖最早的待发
   事件并记 ``stats['queue_overflow'] += 1``，保证 latest 一致。

5. **不耦合业务层**：业务模块（trigger / state_machine / outbox /
   alert_publisher）只调用 ``broadcast`` 协议（``async def publish``），
   不直接 import ``SsePublisher`` 内私有结构；测试可注入 FakePublisher。

6. **可观测性**：stats / 客户端数 / queue size 全部可查；Falcon 风格
   状态接口 + 单元测试友好。

7. **安全**：单 study_uid 不允许含 ``\n`` / ``\r``，避免 SSE 帧破坏；
   event type 仅允许白名单 4 个；payload 必须 JSON 可序列化；异常
   兜底走 logger.warning，不阻断主路径。

参考：本模块**新建**（源仓库 ``medrax`` 用 Gradio ``.stream()``，不
适用医院 HIS 浏览器面板的 SSE 协议）；与 ``writers/alert_banner.py``
``SsePublishCallback`` 协议同形，便于 alert_publisher 直接接入。
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    Optional,
    Protocol,
    Union,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 常量
# =====================================================================

# 4 类事件白名单（前端 EventSource.addEventListener(name, handler) 用）
EVENT_NEW_STUDY: str = "new_study"
EVENT_DRAFT_READY: str = "draft_ready"
EVENT_STATE_CHANGE: str = "state_change"
EVENT_ALERT: str = "alert"

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {EVENT_NEW_STUDY, EVENT_DRAFT_READY, EVENT_STATE_CHANGE, EVENT_ALERT}
)

# 默认参数（生产可经 create_app 里注入；测试可覆盖）
DEFAULT_MAX_HISTORY_SECONDS: float = 300.0          # 5 分钟重连窗口
DEFAULT_CLIENT_QUEUE_MAXSIZE: int = 100             # 单客户端 queue 上限
DEFAULT_KEEPALIVE_INTERVAL_SECONDS: float = 15.0    # 心跳间隔，避免 proxy 切断
DEFAULT_BROADCAST_TIMEOUT_SECONDS: float = 0.5      # 单客户端 push 阻塞上限

# SSE 协议字段
SSE_FIELD_EVENT: str = "event"
SSE_FIELD_DATA: str = "data"
SSE_FIELD_ID: str = "id"
SSE_KEEPALIVE_COMMENT: str = ": keepalive\n\n"

# ISO8601 UTC 时间戳格式（与 audit/writer/trigger 模块同源）
_ISO8601_UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# =====================================================================
# 异常
# =====================================================================
class SsePublisherError(RuntimeError):
    """SsePublisher 协议 / 序列化 / 客户端异常基类。"""


class SsePublishError(SsePublisherError):
    """``broadcast`` 序列化失败 / event type 不合法时抛。"""


# =====================================================================
# 数据契约
# =====================================================================
@dataclass(frozen=True)
class SseEvent:
    """单条 SSE 事件（与 Task 29 SsePublisher.broadcast 入参对齐）。

    字段说明：
    - ``event``: 4 类白名单之一（new_study / draft_ready / state_change / alert）
    - ``study_uid``: 业务主键；用作 last_seen_ts 幂等去重的 key
    - ``ts``: UTC ISO8601 毫秒
    - ``payload``: dict 字段；序列化时调 ``json.dumps(payload, ensure_ascii=False)``
    - ``event_id``: SSE ``id:`` 字段，可选；用于浏览器 EventSource.lastEventId
    """

    event: str
    study_uid: str
    ts: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.event not in VALID_EVENT_TYPES:
            raise SsePublishError(
                f"SseEvent.event 不在白名单 {sorted(VALID_EVENT_TYPES)}; 收到 {self.event!r}"
            )
        if not isinstance(self.study_uid, str) or not self.study_uid:
            raise SsePublishError(
                f"SseEvent.study_uid 必须是非空字符串；收到 {self.study_uid!r}"
            )
        if "\n" in self.study_uid or "\r" in self.study_uid:
            raise SsePublishError(
                "SseEvent.study_uid 不允许换行符（避免 SSE 帧破坏）"
            )
        if not isinstance(self.ts, str) or not self.ts:
            raise SsePublishError(
                f"SseEvent.ts 必须是非空字符串；收到 {self.ts!r}"
            )
        if not isinstance(self.payload, dict):
            raise SsePublishError(
                f"SseEvent.payload 必须是 dict；收到 {type(self.payload).__name__}"
            )


# 与 ``engine/trigger.py`` 与 ``writers/alert_banner.py`` 同形协议
SsePublishCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


# =====================================================================
# 工具
# =====================================================================
def _utc_now_iso() -> str:
    """UTC ISO8601（毫秒）— 与 audit / writer / trigger 模块同源。"""
    return datetime.now(timezone.utc).strftime(_ISO8601_UTC_FORMAT)


def _parse_iso8601_utc(ts: str) -> Optional[datetime]:
    """解析 ISO8601 字符串 → ``datetime``；解析失败返回 ``None``。"""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.strptime(ts, _ISO8601_UTC_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_sse_frame(event: SseEvent) -> bytes:
    """把 ``SseEvent`` 序列化为 SSE 帧（UTF-8 bytes）。

    帧格式（标准 SSE）::

        event: <event>
        id: <event_id or empty>
        data: <json payload + study_uid + ts>

    行尾 ``\\n`` 终止；帧尾双 ``\\n``；UTF-8 编码。
    """
    full_payload: dict[str, Any] = dict(event.payload)
    full_payload.setdefault("study_uid", event.study_uid)
    full_payload.setdefault("ts", event.ts)
    try:
        data_str = json.dumps(full_payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise SsePublishError(
            f"SSE payload JSON 序列化失败: {exc}; payload={full_payload!r}"
        ) from exc

    lines: list[str] = [f"{SSE_FIELD_EVENT}: {event.event}"]
    if event.event_id:
        lines.append(f"{SSE_FIELD_ID}: {event.event_id}")
    # SSE 协议：data 字段单行；如果 data 含换行则需拆分为多 data: 行
    # 这里我们把 json 字符串内的换行透传，前端 parse 后语义一致。
    lines.append(f"{SSE_FIELD_DATA}: {data_str}")
    # 双换行结束帧
    frame = "\n".join(lines) + "\n\n"
    return frame.encode("utf-8")


def _format_keepalive_frame() -> bytes:
    """SSE comment 心跳（``keep-alive``）— 阻止中间 proxy 切断。"""
    return SSE_KEEPALIVE_COMMENT.encode("utf-8")


# =====================================================================
# 客户端订阅契约
# =====================================================================
class _ClientSink(Protocol):
    """单客户端订阅句柄（asyncio.Queue 兼容协议）。

    实现可以是 ``asyncio.Queue`` 或测试用的 ``deque``，但必须支持：

    - ``put_nowait(item)``  — 非阻塞入队（满则抛 ``asyncio.QueueFull``）
    - ``get()``              — 阻塞出队（协程）
    - ``task_done()``        — 标记一次 get 已处理
    """

    def put_nowait(self, item: bytes) -> None: ...  # type: ignore[override]
    async def get(self) -> bytes: ...
    def task_done(self) -> None: ...


# =====================================================================
# 主类：SsePublisher
# =====================================================================
class SsePublisher:
    """浏览器实时面板 SSE 推送 publisher（fan-out + history replay）。

    典型用法（FastAPI 启动 lifespan + 路由）::

        # lifespan：
        publisher = SsePublisher()
        app.state.sse_publisher = publisher

        # 业务模块接 SsePublishCallback：
        trigger = build_triage_trigger(..., sse_publish=publisher.publish)
        alert_publisher = build_alert_publisher(..., sse_publish=publisher.publish)

        # 路由：
        @app.get("/api/sse")
        async def api_sse(last_seen_ts: str | None = None):
            return StreamingResponse(
                publisher.event_stream(last_seen_ts=last_seen_ts),
                media_type="text/event-stream",
            )

    线程模型：所有方法都是 async；``broadcast`` 非阻塞入队，满了丢最旧
    并 ``stats['queue_overflow'] += 1``；``event_stream`` 是异步生成
    器，循环 yield ``bytes`` 直到客户端断开。
    """

    def __init__(
        self,
        *,
        max_history_seconds: float = DEFAULT_MAX_HISTORY_SECONDS,
        client_queue_maxsize: int = DEFAULT_CLIENT_QUEUE_MAXSIZE,
        keepalive_interval_seconds: float = DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
        broadcast_timeout_seconds: float = DEFAULT_BROADCAST_TIMEOUT_SECONDS,
        clock: Optional[Callable[[], str]] = None,
    ) -> None:
        # ---- 参数校验 ----
        if not isinstance(max_history_seconds, (int, float)) or max_history_seconds <= 0:
            raise SsePublisherError(
                f"max_history_seconds 必须为正数；收到 {max_history_seconds!r}"
            )
        if not isinstance(client_queue_maxsize, int) or client_queue_maxsize < 1:
            raise SsePublisherError(
                f"client_queue_maxsize 必须为正整数；收到 {client_queue_maxsize!r}"
            )
        if not isinstance(keepalive_interval_seconds, (int, float)) or keepalive_interval_seconds <= 0:
            raise SsePublisherError(
                f"keepalive_interval_seconds 必须为正数；收到 {keepalive_interval_seconds!r}"
            )
        if not isinstance(broadcast_timeout_seconds, (int, float)) or broadcast_timeout_seconds < 0:
            raise SsePublisherError(
                f"broadcast_timeout_seconds 必须为非负数；收到 {broadcast_timeout_seconds!r}"
            )
        if clock is not None and not callable(clock):
            raise SsePublisherError("clock 必须是可调用对象或 None")

        self._max_history_seconds = float(max_history_seconds)
        self._client_queue_maxsize = int(client_queue_maxsize)
        self._keepalive_interval = float(keepalive_interval_seconds)
        self._broadcast_timeout = float(broadcast_timeout_seconds)
        self._clock: Callable[[], str] = clock or _utc_now_iso

        # 客户端注册表（弱连接：disconnect 自动 unregister）
        self._clients: dict[int, asyncio.Queue] = {}
        self._client_meta: dict[int, dict[str, Any]] = {}
        self._next_client_id = 0

        # history 环形缓冲（按 ts 排序；超出 max_history_seconds 自动挤掉最旧）
        # 单 deque 即可（append / pop from left 都 O(1)；广播 O(N) scan）
        self._history: Deque[SseEvent] = deque(maxlen=1024)

        # 协程锁（注册 / 反注册串行；broadcast 取快照后并行 put）
        self._lock = asyncio.Lock()

        # 统计
        self._stats: dict[str, int] = {
            "broadcasts": 0,           # broadcast() 总调用次数
            "events_pushed": 0,        # 序列化成功的事件数
            "events_dropped": 0,       # 序列化失败被丢弃的事件数
            "queue_overflow": 0,       # 单客户端 queue 满时丢头计数
            "clients_connected": 0,    # 当前活跃客户端数（峰值）—— 实时见 property
            "clients_disconnected": 0, # 累计 disconnect 次数
            "streams_opened": 0,       # event_stream() 累计打开次数
            "streams_closed": 0,       # 累计正常关闭次数
            "replays": 0,              # history replay 累计次数
            "replay_events": 0,        # replay 出去的事件总数
            "keepalives_sent": 0,      # 心跳次数
        }

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:
        """只读 stats 快照（便于单测断言）。"""
        snap = dict(self._stats)
        snap["clients_connected"] = len(self._clients)
        return snap

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def max_history_seconds(self) -> float:
        return self._max_history_seconds

    @property
    def client_queue_maxsize(self) -> int:
        return self._client_queue_maxsize

    @property
    def keepalive_interval_seconds(self) -> float:
        return self._keepalive_interval

    def history_snapshot(self) -> list[SseEvent]:
        """返回当前 history 列表的拷贝（仅测试 / 排查用）。"""
        return list(self._history)

    # ------------------------------------------------------------------
    # 业务回调：SsePublishCallback 协议
    # ------------------------------------------------------------------
    async def publish(self, study_uid: str, payload: dict[str, Any]) -> None:
        """业务层用：异步 publish 一次事件（兼容 SsePublishCallback 协议）。

        Parameters
        ----------
        study_uid:
            主键；面板根据它做列表增量更新与 last_seen_ts 幂等。
        payload:
            必须含 ``event``（4 类之一）+ 其它字段；可含 ``ts`` 否则自动补。

        Raises
        ------
        SsePublishError
            ``event`` 不在白名单 / ``payload`` 不可序列化时抛。
        """
        await self.broadcast(study_uid=study_uid, payload=payload)

    # ------------------------------------------------------------------
    # 业务便捷方法
    # ------------------------------------------------------------------
    async def publish_new_study(
        self, study_uid: str, payload: dict[str, Any]
    ) -> None:
        """便捷包装：``event="new_study"``。"""
        p = dict(payload)
        p["event"] = EVENT_NEW_STUDY
        await self.broadcast(study_uid=study_uid, payload=p)

    async def publish_draft_ready(
        self, study_uid: str, payload: dict[str, Any]
    ) -> None:
        """便捷包装：``event="draft_ready"``（Task 18 trigger 调用）。"""
        p = dict(payload)
        p["event"] = EVENT_DRAFT_READY
        await self.broadcast(study_uid=study_uid, payload=p)

    async def publish_state_change(
        self, study_uid: str, payload: dict[str, Any]
    ) -> None:
        """便捷包装：``event="state_change"``。"""
        p = dict(payload)
        p["event"] = EVENT_STATE_CHANGE
        await self.broadcast(study_uid=study_uid, payload=p)

    async def publish_alert(
        self, study_uid: str, payload: dict[str, Any]
    ) -> None:
        """便捷包装：``event="alert"``（Task 27 AlertPublisher 调用）。

        ``study_uid`` 在 alert 类可传空字符串，回退由 ``payload`` 内
        ``study_uid`` 字段提供。
        """
        p = dict(payload)
        p["event"] = EVENT_ALERT
        await self.broadcast(
            study_uid=study_uid or str(p.get("study_uid") or "alert"),
            payload=p,
        )

    # ------------------------------------------------------------------
    # 主路径：broadcast
    # ------------------------------------------------------------------
    async def broadcast(
        self, study_uid: str, payload: dict[str, Any]
    ) -> int:
        """广播一次事件到所有连接的客户端，并写入 history。

        Returns
        -------
        int
            实际推到的客户端数（断线客户端不计）。
        """
        self._stats["broadcasts"] += 1

        # ---- 参数校验 ----
        if not isinstance(payload, dict):
            self._stats["events_dropped"] += 1
            raise SsePublishError(
                f"payload 必须是 dict；收到 {type(payload).__name__}"
            )
        raw_event = payload.get("event")
        if not isinstance(raw_event, str) or raw_event not in VALID_EVENT_TYPES:
            self._stats["events_dropped"] += 1
            raise SsePublishError(
                f"payload.event 必须在白名单 {sorted(VALID_EVENT_TYPES)}; "
                f"收到 {raw_event!r}"
            )
        if not isinstance(study_uid, str) or not study_uid:
            self._stats["events_dropped"] += 1
            raise SsePublishError(
                f"study_uid 必须是非空字符串；收到 {study_uid!r}"
            )
        if "\n" in study_uid or "\r" in study_uid:
            self._stats["events_dropped"] += 1
            raise SsePublishError(
                "study_uid 不允许换行符（避免 SSE 帧破坏）"
            )

        # ---- 构造 SseEvent ----
        ts = str(payload.get("ts") or self._clock())
        event_id = str(payload.get("event_id")) if payload.get("event_id") else None
        # 复制 payload（不污染上游业务 dict）+ event/ts 字段保留（前端可读）
        safe_payload: dict[str, Any] = {}
        for k, v in payload.items():
            if k in {"event", "ts", "event_id"}:
                continue
            safe_payload[k] = v
        try:
            event = SseEvent(
                event=raw_event,
                study_uid=study_uid,
                ts=ts,
                payload=safe_payload,
                event_id=event_id,
            )
        except SsePublishError:
            self._stats["events_dropped"] += 1
            raise

        # ---- 序列化 ----
        try:
            frame = _format_sse_frame(event)
        except SsePublishError:
            self._stats["events_dropped"] += 1
            raise

        # ---- history 写入（环形缓冲；deque maxlen=1024 自动收尾） ----
        # 注意：不再按"事件 ts - max_history"主动 trim；这是因为事件 ts
        # 可能来自业务（未来时间戳）而不是"现在"，裁剪基线不稳；
        # 真实按 max_history_seconds 过滤的逻辑在 event_stream() 的
        # replay 里（基于 self._clock() 算 cutoff），这里只做 buffer 上限。
        self._history.append(event)

        # ---- fan-out（snapshot clients + put_nowait） ----
        async with self._lock:
            snapshot = list(self._clients.items())
        delivered = 0
        for client_id, queue in snapshot:
            try:
                # 满了丢最旧（保证 latest 可达）
                while queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                        self._stats["queue_overflow"] += 1
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(frame)
                delivered += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SSE push client_id=%s 失败: %s", client_id, exc
                )
                # 客户端不可达 → 反注册（避免泄漏）
                await self._unregister(client_id)
                self._stats["clients_disconnected"] += 1

        self._stats["events_pushed"] += 1
        return delivered

    # ------------------------------------------------------------------
    # event_stream：异步 SSE 生成器
    # ------------------------------------------------------------------
    async def event_stream(
        self,
        last_seen_ts: Optional[str] = None,
        *,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[bytes]:
        """异步 SSE 生成器；用于 FastAPI ``StreamingResponse``。

        Parameters
        ----------
        last_seen_ts:
            客户端断线重连时传的 ``lastEventId`` / ``?last_seen_ts=`` 。
            server 会先 replay ``max_history_seconds`` 内 ``ts >=
            last_seen_ts`` 的事件；再切到 live feed。
        cancel_event:
            测试用：注入 ``asyncio.Event`` 让 stream 在 set 时优雅关闭
            （FastAPI 客户端断开时，框架会取消生成器）。

        Yields
        ------
        bytes
            SSE 帧（UTF-8 编码）；每个连接对应一个独立 asyncio.Queue。

        Notes
        -----
        - heartbeat 周期：每 ``keepalive_interval_seconds`` 推一行注释帧
          （``: keepalive\\n\\n``），防止 nginx/中间层切断长连接。
        - ``max_queue_wait_seconds`` 由 ``keepalive_interval`` 决定
          （避免事件稀疏时连接挂死）。
        """
        self._stats["streams_opened"] += 1
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._client_queue_maxsize)
        client_id = await self._register_client(queue)

        # ---- 重连 replay：先吐 max_history_seconds 内 ts >= last_seen_ts 的事件 ----
        if last_seen_ts is not None:
            last_dt = _parse_iso8601_utc(last_seen_ts)
            if last_dt is not None:
                cutoff_dt_iso = (
                    last_dt
                    - __import__("datetime").timedelta(
                        seconds=self._max_history_seconds
                    )
                ).strftime(_ISO8601_UTC_FORMAT)
                replayed = 0
                for evt in list(self._history):
                    if evt.ts >= cutoff_dt_iso and evt.ts >= last_seen_ts:
                        try:
                            frame = _format_sse_frame(evt)
                            queue.put_nowait(frame)
                            replayed += 1
                        except asyncio.QueueFull:
                            self._stats["queue_overflow"] += 1
                            break
                        except SsePublishError:
                            continue
                self._stats["replays"] += 1
                self._stats["replay_events"] += replayed

        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    # 阻塞等下一个事件；keepalive 间隔内无事件则推心跳
                    frame = await asyncio.wait_for(
                        queue.get(), timeout=self._keepalive_interval
                    )
                    yield frame
                except asyncio.TimeoutError:
                    yield _format_keepalive_frame()
                    self._stats["keepalives_sent"] += 1
                except asyncio.CancelledError:
                    # 客户端断开 → FastAPI 取消 task
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SSE stream 异常 client_id=%s: %s", client_id, exc)
                    break
        finally:
            await self._unregister(client_id)
            self._stats["streams_closed"] += 1
            try:
                queue.task_done()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 内部：客户端注册 / 反注册
    # ------------------------------------------------------------------
    async def _register_client(self, queue: asyncio.Queue) -> int:
        async with self._lock:
            cid = self._next_client_id
            self._next_client_id += 1
            self._clients[cid] = queue
            self._client_meta[cid] = {
                "registered_at": self._clock(),
                "queue_size": 0,
            }
            return cid

    async def _unregister(self, client_id: int) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)
            self._client_meta.pop(client_id, None)


# =====================================================================
# 工厂
# =====================================================================
def build_sse_publisher(
    *,
    max_history_seconds: float = DEFAULT_MAX_HISTORY_SECONDS,
    client_queue_maxsize: int = DEFAULT_CLIENT_QUEUE_MAXSIZE,
    keepalive_interval_seconds: float = DEFAULT_KEEPALIVE_INTERVAL_SECONDS,
    broadcast_timeout_seconds: float = DEFAULT_BROADCAST_TIMEOUT_SECONDS,
) -> SsePublisher:
    """工厂：构造 ``SsePublisher``（与 create_app 注入点对齐）。"""
    return SsePublisher(
        max_history_seconds=max_history_seconds,
        client_queue_maxsize=client_queue_maxsize,
        keepalive_interval_seconds=keepalive_interval_seconds,
        broadcast_timeout_seconds=broadcast_timeout_seconds,
    )


__all__ = [
    # 常量
    "EVENT_NEW_STUDY",
    "EVENT_DRAFT_READY",
    "EVENT_STATE_CHANGE",
    "EVENT_ALERT",
    "VALID_EVENT_TYPES",
    "DEFAULT_MAX_HISTORY_SECONDS",
    "DEFAULT_CLIENT_QUEUE_MAXSIZE",
    "DEFAULT_KEEPALIVE_INTERVAL_SECONDS",
    "DEFAULT_BROADCAST_TIMEOUT_SECONDS",
    "SSE_FIELD_EVENT",
    "SSE_FIELD_DATA",
    "SSE_FIELD_ID",
    # 异常
    "SsePublisherError",
    "SsePublishError",
    # 数据契约
    "SseEvent",
    "SsePublishCallback",
    # 工具
    "_format_sse_frame",
    "_format_keepalive_frame",
    "_parse_iso8601_utc",
    # 主类
    "SsePublisher",
    "build_sse_publisher",
]
