"""outbox — 写回异步重试表（RIS / 体检主检 / 其他外发目标统一通道）。

设计原则（spec.md §5 维度 4 + 6 + plan.md §5 深度 (a) + §6 风险缓解）：

1. **统一通道**：RIS finding-draft、体检主检 summary-ref、未来其它
   写回都走 ``Outbox.enqueue(target_endpoint=...)``；不退化为两个独立
   重试类。
2. **指数退避**：``next_retry_at = now + 30s * (2 ** retry_count)``
   - retry_count=0 → 30s 后
   - retry_count=1 → 60s 后
   - retry_count=2 → 120s 后
   - retry_count=3 → 240s 后（连续失败 3 次告警 + 状态 alerting）
3. **状态机**：
   ``pending``（待发） → ``sent``（成功） 一去不返
   ``pending`` → ``failed_retry``（失败但可重试） → ``alerting``（超阈值）
4. **不静默丢数据**：失败必写 ``last_error``；状态变更写 ``audit_log``
   由调用方负责（Outbox 本身只管表内字段，避免循环依赖）。
5. **dequeue 拉取条件**：``status='pending' AND next_retry_at <= now``，
   按 ``id ASC`` 顺序（FIFO）保证同一 study 不被饿死。
6. **告警回调（Task 27）**：``mark_retry`` 触达 ``alerting`` 时**同步**
   触发可选 ``on_alert_transition`` 回调（payload 含 outbox_id /
   study_uid / target_endpoint / retry_count / last_error / ts），便于
   writers/alert_banner.AlertPublisher 接 SSE 推浏览器顶部红色 banner
   「RIS / 体检主检写回连续失败 × 例，请联系信息科」。回调抛异常仅
   warn 不阻断主路径（避免下游 SSE 抖动污染 outbox 表）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from .sqlite import Database

# 告警回调签名：async def cb(payload: dict) -> None
AlertTransitionCallback = Callable[[dict[str, Any]], Awaitable[None]]

logger = logging.getLogger(__name__)


# 退避配置：30s × 2^retry_count；第 3 次后状态切到 alerting 触发 banner。
BASE_BACKOFF_SECONDS = 30
MAX_RETRY_BEFORE_ALERT = 3


def _utc_now_iso() -> str:
    """UTC ISO8601；与 SQLite CURRENT_TIMESTAMP 同形，便于跨栈排序。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _iso_to_dt(iso: str) -> datetime:
    """字符串 → ``datetime``；缺微秒兜底。"""
    try:
        return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%f").replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def compute_next_retry_iso(retry_count: int, now: Optional[datetime] = None) -> str:
    """纯函数：``next_retry_at = now + BASE * 2 ** retry_count``。

    用于测试 + 业务侧预览。
    """
    from datetime import timedelta

    base = datetime.now(timezone.utc) if now is None else now
    delay = timedelta(seconds=BASE_BACKOFF_SECONDS * (2 ** retry_count))
    return (base + delay).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class Outbox:
    """写回异步重试表。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        # 告警状态切换回调列表（Task 27）：mark_retry 触达 alerting
        # 时同步 fire；多个回调依次调用（不并发），单回调抛异常仅记
        # warn 不阻断主路径。
        self._alert_callbacks: list[AlertTransitionCallback] = []

    # ------------------------------------------------------------------
    # 告警回调注册（Task 27）
    # ------------------------------------------------------------------
    def register_alert_callback(
        self, callback: AlertTransitionCallback
    ) -> None:
        """注册一个 ``on_alert_transition`` 回调。

        同一实例可注册多个回调（顺序调用）。传入 ``None`` 抛
        ``TypeError``（fail-fast）。
        """
        if callback is None:
            raise TypeError("register_alert_callback 拒绝 None")
        if not callable(callback):
            raise TypeError("register_alert_callback 拒绝不可调用对象")
        self._alert_callbacks.append(callback)

    def unregister_alert_callback(
        self, callback: AlertTransitionCallback
    ) -> bool:
        """取消一个告警回调；返回是否成功移除（用于测试清理）。"""
        try:
            self._alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def clear_alert_callbacks(self) -> None:
        """清空所有告警回调（主要用于测试隔离）。"""
        self._alert_callbacks.clear()

    async def _fire_alert_callbacks(self, payload: dict[str, Any]) -> None:
        """触发所有调用的回调；任一抛异常仅 warn 不阻断主路径。"""
        for cb in list(self._alert_callbacks):
            try:
                await cb(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "outbox alert_callback 抛异常 err=%s payload=%s",
                    exc,
                    payload,
                )

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    async def enqueue(
        self,
        study_uid: str,
        target_endpoint: str,
        payload: dict[str, Any],
        *,
        next_retry_at: Optional[str] = None,
    ) -> int:
        """把一次写回入 outbox；返回 ``id``。

        ``next_retry_at`` 缺省 = 当前时间（即"立刻可被 retry_loop 拉取"）。
        """
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        nra = next_retry_at or _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO outbox
                    (study_uid, target_endpoint, payload_json,
                     status, retry_count, next_retry_at)
                VALUES (?, ?, ?, 'pending', 0, ?)
                """,
                (study_uid, target_endpoint, payload_json, nra),
            )
            row_id = cur.lastrowid
            await conn.commit()
        logger.info(
            "outbox enqueue id=%s study=%s endpoint=%s",
            row_id,
            study_uid,
            target_endpoint,
        )
        return int(row_id or 0)

    # ------------------------------------------------------------------
    # 拉取（retry_loop 主路径）
    # ------------------------------------------------------------------
    async def dequeue(self, limit: int = 10) -> list[dict[str, Any]]:
        """拉取 status=pending 且 next_retry_at <= now 的行（按 id ASC FIFO）。

        返回每行 ``{"id": int, "study_uid": str, "target_endpoint": str,
        "payload": dict, "retry_count": int}``。

        本方法**不**修改任何字段；调用方（retry_loop）拿到行后自行
        POST，成功调 ``mark_sent``、失败调 ``mark_retry``。
        """
        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, study_uid, target_endpoint, payload_json, retry_count
                FROM outbox
                WHERE status = 'pending' AND next_retry_at <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (now_iso, limit),
            )
            rows = await cur.fetchall()
            await cur.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": int(row[0]),
                    "study_uid": row[1],
                    "target_endpoint": row[2],
                    "payload": json.loads(row[3]),
                    "retry_count": int(row[4]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # 标记成功
    # ------------------------------------------------------------------
    async def mark_sent(self, outbox_id: int) -> None:
        """标记为 sent；不可逆。"""
        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            await conn.execute(
                """
                UPDATE outbox
                SET status = 'sent',
                    updated_at = ?,
                    last_error = ''
                WHERE id = ?
                """,
                (now_iso, outbox_id),
            )
            await conn.commit()
        logger.info("outbox mark_sent id=%s", outbox_id)

    # ------------------------------------------------------------------
    # 标记失败重试（指数退避 + 超阈值 alerting）
    # ------------------------------------------------------------------
    async def mark_retry(
        self,
        outbox_id: int,
        error_message: str,
    ) -> str:
        """失败重试：retry_count += 1，重算 ``next_retry_at``，写 ``last_error``。

        超 ``MAX_RETRY_BEFORE_ALERT`` 次 → 状态切到 ``alerting``（不再
        重试，等人工介入）。返回**新状态**（``pending`` / ``alerting``）。

        状态由 ``pending`` 切到 ``alerting`` 时同步触发
        ``_alert_callbacks``（Task 27），让 writers/alert_banner 推送
        SSE 告警事件并触发浏览器顶部红色 banner。
        """
        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT retry_count, study_uid, target_endpoint "
                "FROM outbox WHERE id = ?",
                (outbox_id,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is None:
                raise KeyError(f"outbox id={outbox_id} 不存在")
            new_retry = int(row[0]) + 1
            row_study_uid: str = row[1]
            row_target_endpoint: str = row[2]

            if new_retry > MAX_RETRY_BEFORE_ALERT:
                # 连续失败 3 次（retry_count 4）→ alerting
                new_status = "alerting"
                next_retry = now_iso  # 不再被 dequeue 拉取
            else:
                new_status = "pending"
                next_retry = compute_next_retry_iso(new_retry)

            await conn.execute(
                """
                UPDATE outbox
                SET status = ?,
                    retry_count = ?,
                    next_retry_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    new_status,
                    new_retry,
                    next_retry,
                    error_message[:1000],  # 防止极端长 stack 撑爆 TEXT
                    now_iso,
                    outbox_id,
                ),
            )
            await conn.commit()

        if new_status == "alerting":
            logger.error(
                "outbox alerting id=%s retry_count=%s last_error=%s",
                outbox_id,
                new_retry,
                error_message[:200],
            )
            # Task 27：触发告警回调让 SSE publisher 推送 alert 事件
            # → 浏览器顶部红色 banner 「RIS / 体检主检写回连续
            # 失败 × 例，请联系信息科」
            await self._fire_alert_callbacks(
                {
                    "outbox_id": int(outbox_id),
                    "study_uid": row_study_uid,
                    "target_endpoint": row_target_endpoint,
                    "retry_count": int(new_retry),
                    "last_error": error_message[:500],
                    "ts": now_iso,
                }
            )
        else:
            logger.warning(
                "outbox mark_retry id=%s retry_count=%s next_retry=%s err=%s",
                outbox_id,
                new_retry,
                next_retry,
                error_message[:200],
            )
        return new_status

    # ------------------------------------------------------------------
    # 监控 / 调试
    # ------------------------------------------------------------------
    async def get_by_id(self, outbox_id: int) -> Optional[dict[str, Any]]:
        """按 ``id`` 拉取单条 outbox 行（用于 retry_loop 测试 + 调试）。

        不修改任何字段；返回 ``None`` 表示不存在。返回 ``dict`` 含
        ``id / study_uid / target_endpoint / payload / status /
        retry_count / next_retry_at / last_error / created_at / updated_at``。
        """
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, study_uid, target_endpoint, payload_json,
                       status, retry_count, next_retry_at, last_error,
                       created_at, updated_at
                FROM outbox
                WHERE id = ?
                """,
                (outbox_id,),
            )
            row = await cur.fetchone()
            await cur.close()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "study_uid": row[1],
            "target_endpoint": row[2],
            "payload": json.loads(row[3]),
            "status": row[4],
            "retry_count": int(row[5]),
            "next_retry_at": row[6],
            "last_error": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }

    async def reset_next_retry_at(self, outbox_id: int) -> None:
        """把指定 outbox 行的 ``next_retry_at`` 置为当前时间（便于测试重试命中）。

        仅供测试用；正常生产链路不应调用（下次重试时间由退避曲线决定）。
        """
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        ) + "Z"
        async with self.db.connection() as conn:
            await conn.execute(
                "UPDATE outbox SET next_retry_at = ? WHERE id = ?",
                (now_iso, outbox_id),
            )
            await conn.commit()

    async def count_by_status(self) -> dict[str, int]:
        """按 status 汇总（用于 /api/health.outbox_alerts）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT status, COUNT(*) FROM outbox GROUP BY status"
            )
            rows = await cur.fetchall()
            await cur.close()
            return {r[0]: int(r[1]) for r in rows}

    async def count_alerting(self) -> int:
        """连续失败 ≥ 3 次的告警条数；浏览器面板顶部 banner 触发条件。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = 'alerting'"
            )
            row = await cur.fetchone()
            await cur.close()
            return int(row[0]) if row else 0


__all__ = [
    "Outbox",
    "BASE_BACKOFF_SECONDS",
    "MAX_RETRY_BEFORE_ALERT",
    "compute_next_retry_iso",
    "AlertTransitionCallback",
]