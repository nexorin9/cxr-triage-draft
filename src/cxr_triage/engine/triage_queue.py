"""triage_queue — 阳性优先级队列（按 max_prob DESC），浏览器第一栏数据源。

设计原则（spec.md §5 维度 1 + 3 + plan.md §5 深度 (a) 真实算法）：

1. **真实算法**：max_prob 排序用 ``idx_triage_queue_max_prob`` 索引；
   ``peek_next`` / ``list_pending`` 走 ``ORDER BY max_prob DESC``，由
   SQLite 倒序扫描索引取前 N 条（O(log n + k)）。
2. **同 study 二次入站幂等**：(``study_uid`` 主键) ``INSERT OR REPLACE``
   — 同一张胸片第二次 StateGraph 出新结果时刷新 ``max_prob / finding_text
   / positive_flag / inserted_at``；``dispatched_at`` 重置为 NULL 允许
   重新被取走（驳回 + 重跑场景）。
3. **find_text 仅缓存摘要**：队列表只存 ``finding_text``；完整 ``draft``
   （含 ``classifier_probs / key_qa / grounding_boxes`` 等）由
   ``state.StateIdempotencyStore`` 缓存，**不**进队列表（避免胖行 +
   让队列 SELECT 走更窄索引）。
4. **dispatched_at 软标记**：取走并**不**删行；浏览器面板需要历史
   「已处理」栏时仍可查。``list_pending`` 默认 ``WHERE dispatched_at
   IS NULL``，``list_dispatched`` 反之。
5. **审计留痕**：入队 / 取走两条事件都写 ``audit_log``，含
   ``actor_role=system`` + ``study_uid`` + ``event_type=triage_queued
   / triage_dispatched`` + payload 摘要（max_prob / positive_flag /
   finding_text 截断）。
6. **并发安全**：SQLite 串行化写；同 study 二次 enqueue 走 ``INSERT OR
   REPLACE`` 自然合并；并发 ``mark_dispatched`` 由 ``UPDATE ... WHERE
   dispatched_at IS NULL`` 守——两条同时执行，一条 rowcount=1、一条
   rowcount=0（与 state_machine 形为「成功 / 已派发」语义一致）。
7. **不静默改数据**：``max_prob < 0`` / ``> 1`` / 空 ``study_uid`` /
   ``finding_text`` 含 NUL 等都拒绝入队；抛 ``TriageQueueError``。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..persistence.sqlite import Database

logger = logging.getLogger(__name__)


# 默认拉取上限（浏览器面板一次渲染 50 条足够；超过则分页）
DEFAULT_DISPATCH_LIMIT = 50

# audit_log payload 里 finding_text 摘要截断长度（避免 audit 表被胖
# 行撑大；完整 draft 仍在 idempotency 缓存）。
_AUDIT_FINDING_SNIPPET_CHARS = 200


class TriageQueueError(ValueError):
    """入参 / 状态非法时抛。"""


def _utc_now_iso() -> str:
    """UTC ISO8601；与 SQLite CURRENT_TIMESTAMP 同形，便于跨栈排序。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _validate_prob(max_prob: float) -> float:
    """校验 max_prob ∈ [0.0, 1.0]；通过则返回规范化后的 float。"""
    if not isinstance(max_prob, (int, float)):
        raise TriageQueueError(
            f"max_prob 必须是数字；收到 {type(max_prob).__name__}: {max_prob!r}"
        )
    if max_prob < 0.0 or max_prob > 1.0:
        raise TriageQueueError(
            f"max_prob 必须在 [0.0, 1.0] 区间；收到 {max_prob}"
        )
    return float(max_prob)


def _validate_study_uid(study_uid: str) -> str:
    """校验 study_uid：非空 + 不含 NUL 字符；通过则返回 strip 后的字符串。"""
    if not isinstance(study_uid, str):
        raise TriageQueueError(
            f"study_uid 必须是字符串；收到 {type(study_uid).__name__}"
        )
    s = study_uid.strip()
    if not s:
        raise TriageQueueError("study_uid 不能为空")
    if "\x00" in s:
        raise TriageQueueError("study_uid 含 NUL 字符")
    return s


class TriageQueue:
    """阳性优先级队列 — 按 ``max_prob`` 降序维护浏览器第一栏。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    async def enqueue(
        self,
        study_uid: str,
        max_prob: float,
        draft: dict[str, Any],
        *,
        positive_flag: Optional[bool] = None,
    ) -> int:
        """把一条阳性 / 阴性草稿入阳性优先级队列。

        Parameters
        ----------
        study_uid:
            DICOM ``StudyInstanceUID``；**主键**；同 study 二次入站
            走 ``INSERT OR REPLACE`` 刷新 max_prob / finding_text /
            positive_flag / inserted_at，并重置 dispatched_at 为 NULL。
        max_prob:
            18 病理概率最大值；``[0.0, 1.0]`` 区间。
        draft:
            完整 StateGraph 输出 dict。**本方法只缓存 ``finding_text``
            字段**到队列表；其余键（如 ``classifier_probs / key_qa /
            grounding_boxes`` 等）由调用方另行写入
            ``state.StateIdempotencyStore``。
        positive_flag:
            显式传入时按传入值落表；``None`` 时从 ``max_prob >= 0.5``
            启发式推断（与 config 默认阈值 0.5 对齐）。

        Returns
        -------
        int
            ``rowcount``；新插入返回 1，覆盖更新返回 2（SQLite OR REPLACE
            语义）。便于上层区分「真新进」与「重跑覆盖」。
        """
        uid = _validate_study_uid(study_uid)
        prob = _validate_prob(max_prob)
        if not isinstance(draft, dict):
            raise TriageQueueError(
                f"draft 必须是 dict；收到 {type(draft).__name__}"
            )
        finding_text = draft.get("finding_text", "")
        if finding_text is None:
            finding_text = ""
        if not isinstance(finding_text, str):
            raise TriageQueueError(
                f"draft.finding_text 必须是字符串；收到 {type(finding_text).__name__}"
            )
        if "\x00" in finding_text:
            raise TriageQueueError("draft.finding_text 含 NUL 字符")
        if positive_flag is None:
            positive_flag = bool(prob >= 0.5)
        elif not isinstance(positive_flag, bool):
            raise TriageQueueError(
                f"positive_flag 必须是 bool 或 None；收到 {type(positive_flag).__name__}"
            )

        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO triage_queue
                    (study_uid, max_prob, positive_flag, finding_text,
                     inserted_at, dispatched_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(study_uid) DO UPDATE SET
                    max_prob       = excluded.max_prob,
                    positive_flag  = excluded.positive_flag,
                    finding_text   = excluded.finding_text,
                    inserted_at    = excluded.inserted_at,
                    dispatched_at  = NULL
                """,
                (
                    uid,
                    prob,
                    1 if positive_flag else 0,
                    finding_text,
                    now_iso,
                ),
            )
            rowcount = cur.rowcount or 0
            await conn.commit()

            # 同步写 audit_log 行；actor_role=system（StateGraph 入队）
            payload = {
                "max_prob": prob,
                "positive_flag": positive_flag,
                "finding_snippet": finding_text[:_AUDIT_FINDING_SNIPPET_CHARS],
            }
            await conn.execute(
                """
                INSERT INTO audit_log
                    (actor_role, ts, study_uid, event_type, payload_json)
                VALUES ('system', ?, ?, 'triage_queued', ?)
                """,
                (now_iso, uid, json.dumps(payload, ensure_ascii=False)),
            )
            await conn.commit()
            await cur.close()

        logger.info(
            "triage_queue enqueue study=%s max_prob=%.4f positive=%s rowcount=%s",
            uid,
            prob,
            positive_flag,
            rowcount,
        )
        return int(rowcount)

    # ------------------------------------------------------------------
    # 拉取（不修改；仅返回）
    # ------------------------------------------------------------------
    async def peek_next(self) -> Optional[dict[str, Any]]:
        """取下一条最高 ``max_prob`` 且 ``dispatched_at IS NULL`` 的草稿。

        **不**修改 dispatched_at；调用方拿到草稿后自行决定是否调用
        ``mark_dispatched``（允许「先看 → 再决定」的人机协同场景）。

        返回 ``None`` 表示队列空。
        """
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, max_prob, positive_flag, finding_text,
                       inserted_at, dispatched_at
                FROM triage_queue
                WHERE dispatched_at IS NULL
                ORDER BY max_prob DESC, inserted_at ASC
                LIMIT 1
                """,
            )
            row = await cur.fetchone()
            await cur.close()
        if row is None:
            return None
        return _row_to_dict(row)

    async def list_pending(
        self,
        limit: int = DEFAULT_DISPATCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """按 ``max_prob`` 降序拉取待处理列表（浏览器面板一次渲染用）。

        tie-break：``inserted_at`` 升序（同分时先入队优先）。
        """
        if not isinstance(limit, int) or limit <= 0:
            raise TriageQueueError(
                f"limit 必须是正整数；收到 {limit!r}"
            )
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, max_prob, positive_flag, finding_text,
                       inserted_at, dispatched_at
                FROM triage_queue
                WHERE dispatched_at IS NULL
                ORDER BY max_prob DESC, inserted_at ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            await cur.close()
        return [_row_to_dict(r) for r in rows]

    async def list_dispatched(
        self,
        limit: int = DEFAULT_DISPATCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """拉取已派发历史（与 list_pending 互为对照）。"""
        if not isinstance(limit, int) or limit <= 0:
            raise TriageQueueError(
                f"limit 必须是正整数；收到 {limit!r}"
            )
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, max_prob, positive_flag, finding_text,
                       inserted_at, dispatched_at
                FROM triage_queue
                WHERE dispatched_at IS NOT NULL
                ORDER BY dispatched_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            await cur.close()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 取走（标记 dispatched_at）
    # ------------------------------------------------------------------
    async def mark_dispatched(
        self,
        study_uid: str,
        actor_role: str = "system",
    ) -> bool:
        """标记 study 为「已派发」（dispatched_at = now）。

        ``actor_role`` 用于审计：浏览器面板由 jr 点确认时传 ``"jr"``，
        触发器自动派发传 ``"system"``。

        Returns
        -------
        bool
            True = 本次成功标记；False = 已被别人先标记（CAS 失配）或
            不存在该 study_uid。**不抛**。
        """
        uid = _validate_study_uid(study_uid)
        if not isinstance(actor_role, str) or not actor_role.strip():
            raise TriageQueueError(
                f"actor_role 必须是非空字符串；收到 {actor_role!r}"
            )

        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE triage_queue
                SET dispatched_at = ?
                WHERE study_uid = ? AND dispatched_at IS NULL
                """,
                (now_iso, uid),
            )
            rowcount = cur.rowcount or 0
            await conn.commit()
            await cur.close()

            if rowcount == 0:
                # CAS 失配：要么 study 不存在，要么已被别人标记
                # 复查存在性，给审计 / 飞检一个明确信号
                cur2 = await conn.execute(
                    "SELECT max_prob FROM triage_queue WHERE study_uid = ?",
                    (uid,),
                )
                row2 = await cur2.fetchone()
                await cur2.close()
                if row2 is None:
                    logger.warning(
                        "triage_queue mark_dispatched study=%s not_in_queue",
                        uid,
                    )
                    return False
                # 已 dispatched 的合法情况（前端刷新导致重复点确认）
                logger.info(
                    "triage_queue mark_dispatched study=%s already_dispatched",
                    uid,
                )
                return False

            # 写 audit_log 行
            payload = {"actor_role": actor_role}
            await conn.execute(
                """
                INSERT INTO audit_log
                    (actor_role, ts, study_uid, event_type, payload_json)
                VALUES (?, ?, ?, 'triage_dispatched', ?)
                """,
                (actor_role, now_iso, uid, json.dumps(payload, ensure_ascii=False)),
            )
            await conn.commit()

        logger.info(
            "triage_queue mark_dispatched study=%s actor=%s",
            uid,
            actor_role,
        )
        return True

    # ------------------------------------------------------------------
    # 监控 / 调试
    # ------------------------------------------------------------------
    async def count_pending(self) -> int:
        """待派发条数（/api/health 与面板状态指示）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM triage_queue WHERE dispatched_at IS NULL"
            )
            row = await cur.fetchone()
            await cur.close()
            return int(row[0]) if row else 0

    async def count_dispatched(self) -> int:
        """已派发条数。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM triage_queue WHERE dispatched_at IS NOT NULL"
            )
            row = await cur.fetchone()
            await cur.close()
            return int(row[0]) if row else 0

    async def count_by_positive(self) -> dict[str, int]:
        """按 positive_flag 分组条数（用于 /api/health 阳性比例统计）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT positive_flag, COUNT(*) FROM triage_queue GROUP BY positive_flag"
            )
            rows = await cur.fetchall()
            await cur.close()
            out = {"positive": 0, "negative": 0}
            for r in rows:
                key = "positive" if int(r[0]) == 1 else "negative"
                out[key] = int(r[1])
            return out

    async def get_by_study(self, study_uid: str) -> Optional[dict[str, Any]]:
        """按 study_uid 查单条（含 dispatched_at），便于驳回后回查。"""
        uid = _validate_study_uid(study_uid)
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, max_prob, positive_flag, finding_text,
                       inserted_at, dispatched_at
                FROM triage_queue
                WHERE study_uid = ?
                """,
                (uid,),
            )
            row = await cur.fetchone()
            await cur.close()
        if row is None:
            return None
        return _row_to_dict(row)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """sqlite3.Row → 标准 dict；positive_flag 转回 bool。"""
    return {
        "study_uid": row[0],
        "max_prob": float(row[1]),
        "positive_flag": bool(int(row[2])),
        "finding_text": row[3] or "",
        "inserted_at": row[4],
        "dispatched_at": row[5],
    }


__all__ = [
    "TriageQueue",
    "TriageQueueError",
    "DEFAULT_DISPATCH_LIMIT",
]
