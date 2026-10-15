"""失败告警 banner — outbox 连续失败 → SSE 推 alert → 浏览器顶部红色 banner。

设计原则（spec.md §5 维度 4 + 6 + plan.md §6 风险缓解）：

1. **可见性**：RIS / 体检主检写回连续失败 ``MAX_RETRY_BEFORE_ALERT=3``
   次后，浏览器面板顶部必须出现**红色高对比度** banner，避免「RIS 写
   不出去但医师看不到，报告已签发却未到达目标系统」的临床风险。
2. **不静默丢数据**：banner 触发条件 ``count_alerting() >= 1`` 由
   ``/api/health.outbox_alerts`` + SSE alert 事件双通道推送；任一通道
   抖动不影响另一通道。
3. **降级文案**：banner 默认文案「RIS / 体检主检写回连续失败 × 例，请联系
   信息科」—— 直接说"谁负责（信息科）/ 接住后果（写回没到）/ 现在该做
   什么"，避免空话。
4. **可观测性**：每条 alert 事件写 ``audit_log.alert_pushed``（actor_role=
   ``system``，event_type=``alert.pushed`` 或 ``alert.dismissed``）便于
   飞检 / 信息科追溯 banner 触发与恢复时间。
5. **不耦合业务层**：本模块只 import ``config`` / ``persistence`` /
   ``audit``；与 ``RISWriter`` / ``HealthcheckWriter`` 通过 ``Outbox``
   回调解耦（``Outbox.register_alert_callback``）。

参考：源仓库 ``medrax`` 不涉及外发写回重试与告警；本模块是医院对接层
的工程支撑（spec.md §7 开源对照：``**新建**：失败告警 banner``）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from ..persistence.outbox import MAX_RETRY_BEFORE_ALERT, Outbox
from .audit import AuditLogger

logger = logging.getLogger(__name__)


# =====================================================================
# 常量
# =====================================================================

# SSE event name（与 Task 29 SsePublisher 的 event 类型字段对齐）
ALERT_EVENT_NAME: str = "alert"

# SSE event 类型字段
ALERT_KIND_RAISED: str = "alert.raised"   # 新增告警（status=alerting）
ALERT_KIND_DISMISSED: str = "alert.dismissed"  # 告警消失（如手动重置后）
ALERT_KIND_SUMMARY: str = "alert.summary"  # 汇总（count_alerting >= 1）

# 告警事件严重度（前端 CSS 区分）
ALERT_SEVERITY_ERROR: str = "error"
ALERT_SEVERITY_WARNING: str = "warning"

# banner HTML 默认中文文案（科务会直白能听懂）
ALERT_BANNER_DEFAULT_MESSAGE_ZH: str = (
    "RIS / 体检主检写回连续失败 {count} 例，请联系信息科。"
)

# banner CSS class（与 src/cxr_triage/web/static/style.css 同源）
ALERT_BANNER_CSS_CLASS: str = "cxr-alert-banner cxr-alert-banner--error"
ALERT_BANNER_CSS_CLASS_HIDDEN: str = "cxr-alert-banner cxr-alert-banner--hidden"
ALERT_BANNER_DATA_ATTR: str = "data-cxr-alert-banner"

# banner HTML element id（前端 JS 定位）
ALERT_BANNER_ELEMENT_ID: str = "cxr-alert-banner-root"

# audit_log 事件命名（与 RISWriter / HealthcheckWriter 的 alert_* 区分；
# 本类是「banner 推送」事件，writer 类是「writer 自身 retry 告警」
# 事件；两端审计可交叉对照）
EVENT_ALERT_PUSHED: str = "alert.pushed"
EVENT_ALERT_DISMISSED: str = "alert.dismissed"

# 限重操作（防抖）：相同 study_uid + target_endpoint 在
# ALERT_DEDUPE_WINDOW_SECONDS 秒内仅推一条 alert（避免反复 retry 推
# 同一 outbox 行时 banner 闪烁）
ALERT_DEDUPE_WINDOW_SECONDS: float = 60.0


# =====================================================================
# 异常
# =====================================================================
class AlertBannerError(Exception):
    """alert banner 模块异常基类。"""


class AlertPublishError(AlertBannerError):
    """SSE 推 alert 事件失败（publisher 不可达 / 序列化失败）。"""


# =====================================================================
# 数据契约
# =====================================================================
@dataclass
class AlertEvent:
    """单条告警事件（与 SSE payload 字段对齐）。

    字段说明：
    - ``outbox_id``: 触达 alerting 的 outbox 行主键。
    - ``study_uid``: 失败 study_uid（飞检可回查）。
    - ``target_endpoint``: RIS / 体检主检前缀区分（"ris://" vs "hc://"）。
    - ``retry_count``: 失败次数（>= MAX_RETRY_BEFORE_ALERT + 1）。
    - ``last_error``: 最近的失败原因（截前 500 字符）。
    - ``message_zh``: 渲染到 banner 上的中文文案。
    - ``severity``: "error" / "warning"（前端 CSS class 区分）。
    - ``kind``: ALERT_KIND_RAISED / DISMISSED / SUMMARY。
    - ``ts``: 事件时间（UTC ISO8601）。
    - ``total_alert_count``: 当前 outbox 中 status=alerting 的总数（banner
      文案需要）。
    """

    outbox_id: int
    study_uid: str
    target_endpoint: str
    retry_count: int
    last_error: str
    message_zh: str
    severity: str = ALERT_SEVERITY_ERROR
    kind: str = ALERT_KIND_RAISED
    ts: str = ""
    total_alert_count: int = 0

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = _utc_now_iso()
        # 校验 severity / kind 在白名单内
        if self.severity not in {ALERT_SEVERITY_ERROR, ALERT_SEVERITY_WARNING}:
            raise AlertBannerError(
                f"AlertEvent.severity 非法: {self.severity!r}"
            )
        if self.kind not in {
            ALERT_KIND_RAISED,
            ALERT_KIND_DISMISSED,
            ALERT_KIND_SUMMARY,
        }:
            raise AlertBannerError(
                f"AlertEvent.kind 非法: {self.kind!r}"
            )


# =====================================================================
# 工具
# =====================================================================
def _utc_now_iso() -> str:
    """UTC ISO8601；与 ``audit.utc_now_iso`` 同形。"""
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "Z"


def build_alert_event_payload(event: AlertEvent) -> dict[str, Any]:
    """把 ``AlertEvent`` 序列化为 SSE payload dict。

    字段约定（与 Task 29 SSE publisher 对齐）：

    - ``event``: ALERT_EVENT_NAME（"alert"）
    - ``kind``: ALERT_KIND_*（"alert.raised" / "alert.dismissed" / ...）
    - ``severity``: ALERT_SEVERITY_*（CSS class 区分）
    - ``outbox_id``: int
    - ``study_uid``: str
    - ``target_endpoint``: str
    - ``retry_count``: int
    - ``last_error``: str（截前 500）
    - ``message_zh``: str（banner 默认文案）
    - ``total_alert_count``: int
    - ``ts``: str（UTC ISO8601）
    - ``banner_html``: str（前端可直接 innerHTML；与 CSS class 对齐）
    """
    return {
        "event": ALERT_EVENT_NAME,
        "kind": event.kind,
        "severity": event.severity,
        "outbox_id": int(event.outbox_id),
        "study_uid": event.study_uid,
        "target_endpoint": event.target_endpoint,
        "retry_count": int(event.retry_count),
        "last_error": event.last_error,
        "message_zh": event.message_zh,
        "total_alert_count": int(event.total_alert_count),
        "ts": event.ts,
        "banner_html": build_alert_banner_html(
            alert_count=int(event.total_alert_count),
            message=event.message_zh,
            severity=event.severity,
        ),
    }


def build_alert_banner_html(
    alert_count: int,
    *,
    message: Optional[str] = None,
    severity: str = ALERT_SEVERITY_ERROR,
) -> str:
    """构造 banner HTML 字符串（前端可直接 innerHTML 渲染）。

    CSS class 与 ``src/cxr_triage/web/static/style.css`` 的
    ``.cxr-alert-banner--error`` / ``.cxr-alert-banner--warning`` 对齐。

    severity = ALERT_SEVERITY_ERROR → 红底白字（默认）。
    severity = ALERT_SEVERITY_WARNING → 黄底黑字（弱一级）。

    banner 默认 ``hidden=False``；调用方传 ``severity="warning"`` 时不
    走 ``hidden`` class，便于前端 CSS 选择。
    """
    if alert_count < 0:
        raise AlertBannerError("alert_count 不能为负")
    if severity not in {ALERT_SEVERITY_ERROR, ALERT_SEVERITY_WARNING}:
        raise AlertBannerError(f"severity 非法: {severity!r}")
    if severity == ALERT_SEVERITY_ERROR:
        css_class = ALERT_BANNER_CSS_CLASS
    else:
        css_class = ALERT_BANNER_CSS_CLASS.replace("--error", "--warning")

    msg = message or ALERT_BANNER_DEFAULT_MESSAGE_ZH.format(count=alert_count)
    if "{count}" in msg:
        msg = msg.format(count=alert_count)
    # 转义双引号 / 反斜杠（防止 message 内有引号破坏 HTML）
    safe_msg = (
        msg.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    safe_count = str(int(alert_count))
    return (
        f'<div id="{ALERT_BANNER_ELEMENT_ID}" '
        f'class="{css_class}" '
        f'{ALERT_BANNER_DATA_ATTR}="1" '
        f'data-cxr-alert-count="{safe_count}">'
        f'<span class="cxr-alert-banner__icon" aria-hidden="true">⚠</span>'
        f'<span class="cxr-alert-banner__message">{safe_msg}</span>'
        f'</div>'
    )


def build_dismissed_banner_html() -> str:
    """banner 消失时的占位 HTML（前端用 ``hidden`` 槽位恢复默认）。"""
    return (
        f'<div id="{ALERT_BANNER_ELEMENT_ID}" '
        f'class="{ALERT_BANNER_CSS_CLASS_HIDDEN}" '
        f'{ALERT_BANNER_DATA_ATTR}="0" '
        f'data-cxr-alert-count="0"></div>'
    )


def is_target_endpoint_known(endpoint: str) -> bool:
    """检查 ``target_endpoint`` 是否是 RIS / 体检主检已知前缀。

    已知前缀：
    - ``ris://`` — RIS finding-draft 写回
    - ``hc://`` — 体检主检 summary-ref 写回

    未知前缀（含自定义外发）也接受（不抛），但 banner 文案会用通用
    "外发目标"。
    """
    if not isinstance(endpoint, str) or not endpoint:
        return False
    return endpoint.startswith("ris://") or endpoint.startswith("hc://")


def describe_target_endpoint(endpoint: str) -> str:
    """把 ``target_endpoint`` 前缀翻译成中文显示名（banner 文案可读性）。"""
    if not isinstance(endpoint, str) or not endpoint:
        return "外发目标"
    if endpoint.startswith("ris://"):
        return "RIS"
    if endpoint.startswith("hc://"):
        return "体检主检"
    return "外发目标"


# =====================================================================
# SsePublishCallback 协议
# =====================================================================
# 与 engine/trigger.py 的 SsePublishCallback 同形；Task 29 的 SsePublisher
# 实例化时直接满足本协议（async def broadcast(study_uid, payload)）。
SsePublishCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


# =====================================================================
# AlertPublisher
# =====================================================================
class AlertPublisher:
    """把 outbox 告警转换成 SSE 推 alert 事件 → 浏览器顶部红色 banner。

    典型用法（FastAPI 启动 lifespan）::

        outbox = Outbox(db)
        publisher = AlertPublisher(
            outbox=outbox,
            audit=audit,
            sse_publish=sse_publisher.publish,  # Task 29 实现
        )
        outbox.register_alert_callback(publisher.on_outbox_alert_transition)

    线程模型：所有方法都是 async；标记回调（on_outbox_alert_transition）
    必须 await，因为 mark_retry 是 async 函数内部 await 触发。
    """

    def __init__(
        self,
        outbox: Outbox,
        audit: AuditLogger,
        sse_publish: SsePublishCallback,
        *,
        dedupe_window_seconds: float = ALERT_DEDUPE_WINDOW_SECONDS,
    ) -> None:
        if outbox is None:
            raise AlertBannerError("AlertPublisher 拒绝 outbox=None")
        if audit is None:
            raise AlertBannerError("AlertPublisher 拒绝 audit=None")
        if sse_publish is None or not callable(sse_publish):
            raise AlertBannerError("AlertPublisher 拒绝 sse_publish=None")
        if not isinstance(dedupe_window_seconds, (int, float)) or dedupe_window_seconds < 0:
            raise AlertBannerError(
                f"dedupe_window_seconds 必须非负数: {dedupe_window_seconds!r}"
            )

        self._outbox = outbox
        self._audit = audit
        self._sse_publish = sse_publish
        self._dedupe_window = float(dedupe_window_seconds)

        # 限重：同一 (target_endpoint, study_uid) 在 dedupe_window 内
        # 最多推一条 raised（dict 记 last_pushed_iso 时间戳）。
        self._recent_raises: dict[tuple[str, str], str] = {}

        # 限重：最近一次 dismissed 的 ts；dedupe_window 内不重复推
        # dismissed（避免反复推空）。
        self._last_dismissed_iso: Optional[str] = None

        # 统计：发出的 alert 数量（成功 / 被去重 / 失败）
        self._stats: dict[str, int] = {
            "raised_pushed": 0,
            "raised_deduped": 0,
            "summary_pushed": 0,
            "dismissed_pushed": 0,
            "publish_errors": 0,
        }

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:
        """只读副本，便于测试断言。"""
        return dict(self._stats)

    @property
    def dedupe_window_seconds(self) -> float:
        return self._dedupe_window

    # ------------------------------------------------------------------
    # 注册到 outbox 的回调入口
    # ------------------------------------------------------------------
    async def on_outbox_alert_transition(self, payload: dict[str, Any]) -> None:
        """``Outbox.register_alert_callback`` 注册的回调入口。

        ``payload`` 由 ``Outbox.mark_retry`` 写入，字段：
        ``outbox_id / study_uid / target_endpoint / retry_count /
        last_error / ts``。本方法只关心「触达 alerting」一种语义——
        Outbox 仅在 ``new_status == 'alerting'`` 时触发本回调。
        """
        if not isinstance(payload, dict):
            logger.warning(
                "AlertPublisher 收到非 dict 回调 payload: %r", payload
            )
            return

        try:
            outbox_id = int(payload["outbox_id"])
            study_uid = str(payload["study_uid"])
            target_endpoint = str(payload["target_endpoint"])
            retry_count = int(payload["retry_count"])
            last_error = str(payload.get("last_error", ""))
            ts = str(payload.get("ts", _utc_now_iso()))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "AlertPublisher 回调 payload 缺字段/类型错: %s payload=%r",
                exc,
                payload,
            )
            return

        # 取当前 outbox alerting 总数（写 banner message）
        try:
            total_alert_count = await self._outbox.count_alerting()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AlertPublisher count_alerting 失败: %s", exc
            )
            total_alert_count = 0

        # 限重：dedupe_window 内同 (target_endpoint, study_uid) 已推过则跳过
        key = (target_endpoint, study_uid)
        last_ts = self._recent_raises.get(key)
        if last_ts is not None:
            try:
                last_dt = datetime.strptime(
                    last_ts, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=timezone.utc)
                now_dt = datetime.strptime(
                    ts, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=timezone.utc)
                if (now_dt - last_dt).total_seconds() < self._dedupe_window:
                    self._stats["raised_deduped"] += 1
                    logger.info(
                        "alert raised deduped key=%s window=%ss",
                        key,
                        self._dedupe_window,
                    )
                    return
            except ValueError:
                pass  # ts 解析失败 → 不去重
        self._recent_raises[key] = ts

        # 构造 alert event
        message_zh = ALERT_BANNER_DEFAULT_MESSAGE_ZH.format(
            count=total_alert_count
        )
        event = AlertEvent(
            outbox_id=outbox_id,
            study_uid=study_uid,
            target_endpoint=target_endpoint,
            retry_count=retry_count,
            last_error=last_error,
            message_zh=message_zh,
            severity=ALERT_SEVERITY_ERROR,
            kind=ALERT_KIND_RAISED,
            ts=ts,
            total_alert_count=total_alert_count,
        )

        await self._publish_event(event, event_type=EVENT_ALERT_PUSHED)

    # ------------------------------------------------------------------
    # 公开方法：手动推汇总
    # ------------------------------------------------------------------
    async def publish_alert_summary(self) -> Optional[dict[str, Any]]:
        """主动查询 ``count_alerting()`` 并推一条 summary alert 事件。

        Returns
        -------
        Optional[dict]
            推成功的 payload；count_alerting == 0 时**不**推送（无
            alert → 无 banner），返回 ``None``。
        """
        try:
            count = await self._outbox.count_alerting()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AlertPublisher count_alerting 失败: %s", exc)
            return None
        if count <= 0:
            return None

        ts = _utc_now_iso()
        event = AlertEvent(
            outbox_id=0,
            study_uid="",
            target_endpoint="summary",
            retry_count=MAX_RETRY_BEFORE_ALERT + 1,
            last_error="",
            message_zh=ALERT_BANNER_DEFAULT_MESSAGE_ZH.format(count=count),
            severity=ALERT_SEVERITY_ERROR,
            kind=ALERT_KIND_SUMMARY,
            ts=ts,
            total_alert_count=count,
        )
        return await self._publish_event(event, event_type=EVENT_ALERT_PUSHED)

    # ------------------------------------------------------------------
    # 公开方法：手动 dismiss（信息科处理告警后调用）
    # ------------------------------------------------------------------
    async def dismiss_alert(
        self, *, reason: Optional[str] = None
    ) -> bool:
        """推一条 dismissed 事件让前端 banner 消失。

        Returns
        -------
        bool
            是否推送成功。
        """
        ts = _utc_now_iso()
        self._last_dismissed_iso = ts
        event = AlertEvent(
            outbox_id=0,
            study_uid="",
            target_endpoint="dismissed",
            retry_count=0,
            last_error=reason or "",
            message_zh="",
            severity=ALERT_SEVERITY_ERROR,
            kind=ALERT_KIND_DISMISSED,
            ts=ts,
            total_alert_count=0,
        )
        result = await self._publish_event(
            event, event_type=EVENT_ALERT_DISMISSED
        )
        return result is not None

    # ------------------------------------------------------------------
    # 公开方法：清除限重缓存（测试 / 信息科手动重置用）
    # ------------------------------------------------------------------
    def clear_dedupe_cache(self) -> None:
        self._recent_raises.clear()
        self._last_dismissed_iso = None

    # ------------------------------------------------------------------
    # 内部：推 SSE 事件 + 写 audit
    # ------------------------------------------------------------------
    async def _publish_event(
        self, event: AlertEvent, *, event_type: str
    ) -> Optional[dict[str, Any]]:
        """序列化 → SSE publish → 写 audit；任一失败仅 warn 不抛。"""
        try:
            payload = build_alert_event_payload(event)
        except Exception as exc:  # noqa: BLE001
            self._stats["publish_errors"] += 1
            logger.warning("alert payload 序列化失败: %s event=%r", exc, event)
            return None

        try:
            await self._sse_publish(event.study_uid or "alert", payload)
        except Exception as exc:  # noqa: BLE001
            self._stats["publish_errors"] += 1
            logger.warning(
                "alert SSE publish 失败: %s payload=%r", exc, payload
            )
            return None

        # 写 audit（不抛，避免下游 SSE 抖动污染审计）
        try:
            await self._audit.audit_log_insert(
                actor_role="system",
                study_uid=event.study_uid or "alert",
                event_type=event_type,
                payload={
                    "kind": event.kind,
                    "severity": event.severity,
                    "outbox_id": int(event.outbox_id),
                    "target_endpoint": event.target_endpoint,
                    "retry_count": int(event.retry_count),
                    "total_alert_count": int(event.total_alert_count),
                    "ts": event.ts,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert audit 写入失败: %s", exc)

        # stats 累加
        if event.kind == ALERT_KIND_RAISED:
            self._stats["raised_pushed"] += 1
        elif event.kind == ALERT_KIND_SUMMARY:
            self._stats["summary_pushed"] += 1
        elif event.kind == ALERT_KIND_DISMISSED:
            self._stats["dismissed_pushed"] += 1
        return payload


# =====================================================================
# 工厂
# =====================================================================
def build_alert_publisher(
    outbox: Outbox,
    audit: AuditLogger,
    sse_publish: SsePublishCallback,
    *,
    dedupe_window_seconds: float = ALERT_DEDUPE_WINDOW_SECONDS,
) -> AlertPublisher:
    """工厂：构造 ``AlertPublisher``。

    显式参数优先；``outbox`` / ``audit`` / ``sse_publish`` 都必填
    （避免隐式依赖导致循环引用）。
    """
    if outbox is None:
        raise AlertBannerError("build_alert_publisher 必须显式传入 outbox")
    if audit is None:
        raise AlertBannerError("build_alert_publisher 必须显式传入 audit")
    if sse_publish is None or not callable(sse_publish):
        raise AlertBannerError(
            "build_alert_publisher 必须显式传入 sse_publish callback"
        )
    return AlertPublisher(
        outbox=outbox,
        audit=audit,
        sse_publish=sse_publish,
        dedupe_window_seconds=dedupe_window_seconds,
    )


__all__ = [
    # 常量
    "ALERT_EVENT_NAME",
    "ALERT_KIND_RAISED",
    "ALERT_KIND_DISMISSED",
    "ALERT_KIND_SUMMARY",
    "ALERT_SEVERITY_ERROR",
    "ALERT_SEVERITY_WARNING",
    "ALERT_BANNER_DEFAULT_MESSAGE_ZH",
    "ALERT_BANNER_CSS_CLASS",
    "ALERT_BANNER_CSS_CLASS_HIDDEN",
    "ALERT_BANNER_DATA_ATTR",
    "ALERT_BANNER_ELEMENT_ID",
    "ALERT_DEDUPE_WINDOW_SECONDS",
    "EVENT_ALERT_PUSHED",
    "EVENT_ALERT_DISMISSED",
    # 异常
    "AlertBannerError",
    "AlertPublishError",
    # 数据契约
    "AlertEvent",
    "SsePublishCallback",
    # 工具
    "build_alert_event_payload",
    "build_alert_banner_html",
    "build_dismissed_banner_html",
    "describe_target_endpoint",
    "is_target_endpoint_known",
    # 主类
    "AlertPublisher",
    "build_alert_publisher",
]