"""5 态状态机 + CAS 并发防护 + 驳回回退 — spec.md §5 维度 3。

设计原则：

1. **5 态主线**::

       draft_pending → jr_confirmed → sr_reviewed → sr_signed
       rejected → draft_pending （驳回后自动回退的循环）

   ``rejected`` 是过程态，最终态仍是 ``draft_pending``；把它实现成一
   次性的"中转站"，避免遗留的 ``rejected`` 行堆积而需要单独清扫。

2. **actor_role 枚举**（spec.md §3）：``jr`` / ``reviewer`` / ``sr`` /
   ``system``。每条跃迁只允许一个或一组特定角色推进：

   - ``draft_pending → jr_confirmed``    only ``jr``
   - ``jr_confirmed → sr_reviewed``      only ``reviewer``
   - ``sr_reviewed  → sr_signed``        only ``sr``
   - ``*            → rejected``         ``jr`` / ``reviewer`` / ``sr``
   - ``rejected     → draft_pending``     only ``system``（驳回回退）

3. **CAS（compare-and-set）并并发**（spec.md §5 维度 3）：
   ``UPDATE triage_states SET current_state = ? WHERE study_uid = ?
   AND current_state = ?``；SQLite 单连接串行化写，并发两条 UPDATE
   命中同一行 → 只有先到的 ``affected_rows = 1``，后到的 ``= 0``。
   业务层据此判断是否需要刷新界面与提示"另一名医师已抢先签发"。

4. **审计同步**：每次状态变更同步写一条 ``audit_log``（actor_role +
   ts + study_uid + event_type + payload_json）；不在 transition 主体
   之外做 audit，规避"状态已变更、审计忘写"的隐患。

5. **不自动签名 / 不回写临床状态**：状态机只负责面板流转；RIS 写回、
   体检主检 summary-ref、签字等由 writer 层负责（拒绝越界）。

参考：源仓库 ``medrax/agent/agent.py:91-100`` 状态机思想（has_tool_calls
条件边判断状态）；本任务是把它的"消息级"抽象落到"医师流转级"。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional

from ..persistence.sqlite import Database

logger = logging.getLogger(__name__)


# =====================================================================
# 枚举与常量
# =====================================================================


class CxrState(str, Enum):
    """5 态枚举。"""

    DRAFT_PENDING = "draft_pending"  # 初始态：StateGraph 跑完入队
    JR_CONFIRMED = "jr_confirmed"   # jr 医师在面板点确认
    SR_REVIEWED = "sr_reviewed"    # reviewer 复核
    SR_SIGNED = "sr_signed"         # sr 终审签发（终态）
    REJECTED = "rejected"           # 中转：驳回后立即回 DRAFT_PENDING


class ActorRole(str, Enum):
    """操作者角色枚举。"""

    JR = "jr"                 # 放射初诊医师
    REVIEWER = "reviewer"     # 主检复核医师
    SR = "sr"                 # 终审签发医师
    SYSTEM = "system"         # 系统自动（如驳回后回退）


# 跃迁白名单：(from_state, to_state) → 允许的 actor_role 集合
# 写死为不可变 dict，便于测试与外部 import。
VALID_TRANSITIONS: dict[tuple[CxrState, CxrState], frozenset[ActorRole]] = {
    # 正常流转
    (CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED): frozenset({ActorRole.JR}),
    (CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED): frozenset({ActorRole.REVIEWER}),
    (CxrState.SR_REVIEWED, CxrState.SR_SIGNED): frozenset({ActorRole.SR}),
    # 驳回：任意医师都可发起
    (CxrState.DRAFT_PENDING, CxrState.REJECTED): frozenset(
        {ActorRole.JR, ActorRole.REVIEWER, ActorRole.SR}
    ),
    (CxrState.JR_CONFIRMED, CxrState.REJECTED): frozenset(
        {ActorRole.REVIEWER, ActorRole.SR}
    ),
    (CxrState.SR_REVIEWED, CxrState.REJECTED): frozenset({ActorRole.SR}),
    # 驳回自动回退
    (CxrState.REJECTED, CxrState.DRAFT_PENDING): frozenset({ActorRole.SYSTEM}),
}


# =====================================================================
# 数据类与异常
# =====================================================================


class StateTransitionError(ValueError):
    """非法跃迁 / 角色不匹配 / 状态不匹配异常。"""


@dataclass(frozen=True)
class TransitionResult:
    """``transition()`` 返回值。

    Attributes
    ----------
    success:
        ``True`` 表示 CAS 命中、状态已写入；``False`` 表示当前状态不是
        ``from_state``（并发冲突），状态未变更。
    study_uid:
        目标研究实例 UID。
    from_state:
        本次 transition **期望**的源状态。
    to_state:
        本次 transition 期望的目标状态；驳回时这是 ``rejected``，
        **第二次** ``system`` 推进会让 ``to_state`` = ``draft_pending``。
    new_state:
        ``success=True`` 时的最终状态（驳回流程结束后是 ``draft_pending``）；
        ``success=False`` 时返回 ``query_state()`` 的实际状态。
    actor_role:
        本次 transition 的操作者角色。
    audit_log_id:
        本次 transition 写入 ``audit_log`` 的行 ID（失败时为 ``None``）。
    from_state_actual:
        ``success=False`` 时数据库中实际状态（用于诊断并发冲突）。
    """

    success: bool
    study_uid: str
    from_state: CxrState
    to_state: CxrState
    new_state: CxrState
    actor_role: ActorRole
    audit_log_id: Optional[int]
    from_state_actual: Optional[CxrState] = None


# =====================================================================
# 时间戳工具
# =====================================================================


def _utc_now_iso() -> str:
    """UTC ISO8601（与 persistence 层其它时间戳对齐）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _coerce_state(value: Any) -> CxrState:
    """``str`` / ``CxrState`` → ``CxrState``；未知值抛 ``StateTransitionError``。"""
    if isinstance(value, CxrState):
        return value
    s = str(value or "").strip().lower()
    for st in CxrState:
        if st.value == s:
            return st
    raise StateTransitionError(f"未知状态: {value!r}")


def _coerce_role(value: Any) -> ActorRole:
    """``str`` / ``ActorRole`` → ``ActorRole``。"""
    if isinstance(value, ActorRole):
        return value
    s = str(value or "").strip().lower()
    for r in ActorRole:
        if r.value == s:
            return r
    raise StateTransitionError(f"未知角色: {value!r}")


# =====================================================================
# 主类
# =====================================================================


class CxrStateMachine:
    """5 态状态机主类。

    Parameters
    ----------
    db:
        已 ``await db.connect()`` 的 ``Database`` 实例（业务层 lifespan
        启动时拿到，传进来）。
    audit_hook:
        可选回调 ``async def audit_hook(actor_role, study_uid,
        event_type, payload)``；用于：飞检审计集成 / 多通道落 audit_log
        / 实时 SSE 推 audit 事件。**注意**：本类在 transition / reject
        主体内仍会同步写 ``audit_log`` 表一行（确保审计不漏），hook
        失败不阻断主流程但会打 warning。
    """

    def __init__(
        self,
        db: Database,
        *,
        audit_hook: Optional[Any] = None,
    ) -> None:
        self.db = db
        self._audit_hook = audit_hook

    # ------------------------------------------------------------------
    # 内部：写 audit_log
    # ------------------------------------------------------------------
    async def _write_audit(
        self,
        *,
        actor_role: ActorRole,
        study_uid: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> Optional[int]:
        """写一行 ``audit_log``；失败抛 ``StateTransitionError``。

        主体内已包好 try/except 保证 hook 失败不阻断主流程；本方法
        自身失败时**会**阻断（避免"状态已写、审计漏写"）。
        """
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        ts = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "INSERT INTO audit_log "
                "(actor_role, ts, study_uid, event_type, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (actor_role.value, ts, study_uid, event_type, payload_json),
            )
            row_id = cur.lastrowid
            await conn.commit()
            await cur.close()

        # 业务 hook（可选；失败仅 warn）
        if self._audit_hook is not None:
            try:
                res = self._audit_hook(
                    actor_role=actor_role.value,
                    study_uid=study_uid,
                    event_type=event_type,
                    payload=payload,
                    ts=ts,
                )
                # 允许 hook 是 async
                if hasattr(res, "__await__"):
                    await res
            except Exception:  # noqa: BLE001
                logger.warning(
                    "audit_hook 失败（已写入 audit_log）：study=%s event=%s",
                    study_uid,
                    event_type,
                )
        return int(row_id or 0)

    # ------------------------------------------------------------------
    # 公共：状态查询 / 初始化
    # ------------------------------------------------------------------
    async def get_or_init_state(self, study_uid: str) -> CxrState:
        """查状态；不存在则 INSERT 一行 ``draft_pending`` 后返回。

        业务语义：每次 transition 前先调它，确保研究实例已经进入状态
        机主表（默认 draft_pending）。幂等。
        """
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT current_state FROM triage_states WHERE study_uid = ?",
                (study_uid,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is not None:
                return _coerce_state(row[0])

        # 不存在 → INSERT 默认 draft_pending
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO triage_states
                    (study_uid, current_state, updated_at, updated_by)
                VALUES (?, ?, ?, ?)
                """,
                (study_uid, CxrState.DRAFT_PENDING.value, _utc_now_iso(), ActorRole.SYSTEM.value),
            )
            await conn.commit()
        await self._write_audit(
            actor_role=ActorRole.SYSTEM,
            study_uid=study_uid,
            event_type="state_initialized",
            payload={
                "initial_state": CxrState.DRAFT_PENDING.value,
            },
        )
        return CxrState.DRAFT_PENDING

    async def query_state(self, study_uid: str) -> Optional[CxrState]:
        """查状态；不存在返回 ``None``（不初始化）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT current_state FROM triage_states WHERE study_uid = ?",
                (study_uid,),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is None:
                return None
            return _coerce_state(row[0])

    # ------------------------------------------------------------------
    # 公共：列表 / 计数（浏览器三栏面板用）
    # ------------------------------------------------------------------
    async def list_by_state(
        self,
        state: CxrState,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按状态列举研究；返回 ``[{"study_uid", "current_state",
        "updated_at", "updated_by"}]``。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT study_uid, current_state, updated_at, updated_by
                FROM triage_states
                WHERE current_state = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (state.value, limit),
            )
            rows = await cur.fetchall()
            await cur.close()
        return [
            {
                "study_uid": r[0],
                "current_state": r[1],
                "updated_at": r[2],
                "updated_by": r[3],
            }
            for r in rows
        ]

    async def count_by_state(self) -> dict[str, int]:
        """按状态计数；给 /api/health 与面板头部 chip 统计用。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                "SELECT current_state, COUNT(*) FROM triage_states GROUP BY current_state"
            )
            rows = await cur.fetchall()
            await cur.close()
        return {r[0]: int(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # 公共：状态推进（核心）
    # ------------------------------------------------------------------
    async def transition(
        self,
        study_uid: str,
        from_state: Any,
        to_state: Any,
        actor_role: Any,
        *,
        payload: Optional[dict[str, Any]] = None,
    ) -> TransitionResult:
        """核心方法：一次状态推进。

        Parameters
        ----------
        study_uid:
            DICOM StudyInstanceUID。
        from_state, to_state:
            期望的源状态与目标状态（``str`` 或 ``CxrState``）。
        actor_role:
            操作者角色（``str`` 或 ``ActorRole``）。
        payload:
            额外审计载荷（如驳回原因）；会被序列化进 ``audit_log``。

        Returns
        -------
        ``TransitionResult(success, from_state, to_state, new_state, ...)``

        Notes
        -----
        - **驳回**：请用 ``reject()``；本方法也会接受 ``to_state=rejected``
          但不会自动回 ``draft_pending``，仅作"先停在 rejected"的特殊
          场景（一般不在浏览器面板用）。
        - **并发冲突**：本方法在 ``success=False`` 时**不抛异常**，调用
          方根据 ``result.from_state_actual`` 判断下一步（刷新界面 /
          弹"已被另一名医师抢先签发"提示）。
        - **非法跃迁 / 角色不匹配**：抛 ``StateTransitionError``。
        """
        from_st = _coerce_state(from_state)
        to_st = _coerce_state(to_state)
        role = _coerce_role(actor_role)
        if to_st == CxrState.REJECTED:
            raise StateTransitionError(
                "请用 reject() 发起驳回；transition() 不会自动回 draft_pending"
            )
        allowed_roles = VALID_TRANSITIONS.get((from_st, to_st))
        if allowed_roles is None:
            raise StateTransitionError(
                f"非法跃迁：{from_st.value} → {to_st.value} 不在白名单"
            )
        if role not in allowed_roles:
            raise StateTransitionError(
                f"角色 {role.value} 不被允许跃迁 {from_st.value} → {to_st.value}；"
                f"仅允许 {sorted(r.value for r in allowed_roles)}"
            )

        # 确保状态行存在
        await self.get_or_init_state(study_uid)

        return await self._cas_transition(
            study_uid=study_uid,
            from_state=from_st,
            to_state=to_st,
            actor_role=role,
            payload=payload or {},
            event_type="state_change",
        )

    async def reject(
        self,
        study_uid: str,
        from_state: Any,
        actor_role: Any,
        *,
        reason: str = "",
    ) -> TransitionResult:
        """驳回：``* → rejected → draft_pending``（一次调用完成两步）。

        Parameters
        ----------
        study_uid:
            研究实例 UID。
        from_state:
            期望的源状态（jr_confirmed / sr_reviewed / draft_pending）；
            不同源状态允许驳回的角色不同（见 ``VALID_TRANSITIONS``）。
        actor_role:
            驳回者角色（jr / reviewer / sr）。
        reason:
            驳回原因（会写入 audit_log）。

        Returns
        -------
        ``TransitionResult``，``new_state`` 为 ``draft_pending``（驳回
        回退后的最终态）。

        Raises
        ------
        StateTransitionError:
            角色无驳回权限 / from_state 不允许驳回（sr_signed 不可驳回）。
        """
        from_st = _coerce_state(from_state)
        role = _coerce_role(actor_role)
        if from_st == CxrState.SR_SIGNED:
            raise StateTransitionError(
                "sr_signed 已是终态；不允许驳回（如需整改请走其它流程）"
            )
        if from_st == CxrState.REJECTED:
            raise StateTransitionError(
                "已是 rejected 中转态；本方法拒绝重复驳回"
            )

        allowed_roles = VALID_TRANSITIONS.get((from_st, CxrState.REJECTED))
        if allowed_roles is None or role not in allowed_roles:
            raise StateTransitionError(
                f"角色 {role.value} 无权驳回 {from_st.value}"
            )

        # 确保状态行存在
        await self.get_or_init_state(study_uid)

        # 第一步：from_state → rejected（CAS）
        payload_step1 = {
            "from_state": from_st.value,
            "to_state": CxrState.REJECTED.value,
            "reason": reason[:500],
        }
        result_step1 = await self._cas_transition(
            study_uid=study_uid,
            from_state=from_st,
            to_state=CxrState.REJECTED,
            actor_role=role,
            payload=payload_step1,
            event_type="state_reject_step1",
        )
        if not result_step1.success:
            return TransitionResult(
                success=False,
                study_uid=study_uid,
                from_state=from_st,
                to_state=CxrState.REJECTED,
                new_state=result_step1.new_state,
                actor_role=role,
                audit_log_id=result_step1.audit_log_id,
                from_state_actual=result_step1.from_state_actual,
            )

        # 第二步：rejected → draft_pending（system 推进，CAS）
        payload_step2 = {
            "from_state": CxrState.REJECTED.value,
            "to_state": CxrState.DRAFT_PENDING.value,
            "reason": reason[:500],
            "reject_audit_log_id": result_step1.audit_log_id,
        }
        result_step2 = await self._cas_transition(
            study_uid=study_uid,
            from_state=CxrState.REJECTED,
            to_state=CxrState.DRAFT_PENDING,
            actor_role=ActorRole.SYSTEM,
            payload=payload_step2,
            event_type="state_reject_step2",
        )
        return TransitionResult(
            success=result_step2.success,
            study_uid=study_uid,
            from_state=from_st,
            to_state=CxrState.DRAFT_PENDING,
            new_state=result_step2.new_state,
            actor_role=role,
            audit_log_id=result_step2.audit_log_id,
            from_state_actual=result_step2.from_state_actual,
        )

    # ------------------------------------------------------------------
    # 私有：CAS 推进实现
    # ------------------------------------------------------------------
    async def _cas_transition(
        self,
        *,
        study_uid: str,
        from_state: CxrState,
        to_state: CxrState,
        actor_role: ActorRole,
        payload: dict[str, Any],
        event_type: str,
    ) -> TransitionResult:
        """核心 CAS：``UPDATE triage_states SET current_state=? WHERE
        study_uid=? AND current_state=?``；``affected_rows = 0`` 即视为
        并发冲突（CAS 失配），**不抛异常**返回 ``success=False``。
        """
        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE triage_states
                SET current_state = ?, updated_at = ?, updated_by = ?
                WHERE study_uid = ? AND current_state = ?
                """,
                (
                    to_state.value,
                    now_iso,
                    actor_role.value,
                    study_uid,
                    from_state.value,
                ),
            )
            affected = cur.rowcount or 0
            await conn.commit()
            await cur.close()

        audit_payload = dict(payload)
        audit_payload["from_state"] = from_state.value
        audit_payload["to_state"] = to_state.value
        audit_payload["cas_success"] = bool(affected)

        # CAS 失配仍写 audit（标 cas_success=False；给飞检追溯"未生效的
        # 尝试"留底，避免"看 audit 不知道曾有医师试图签发而被抢签"）
        audit_id = await self._write_audit(
            actor_role=actor_role,
            study_uid=study_uid,
            event_type=event_type,
            payload=audit_payload,
        )

        if affected > 0:
            logger.info(
                "state transition ok: %s %s → %s by %s (audit=%s)",
                study_uid,
                from_state.value,
                to_state.value,
                actor_role.value,
                audit_id,
            )
            return TransitionResult(
                success=True,
                study_uid=study_uid,
                from_state=from_state,
                to_state=to_state,
                new_state=to_state,
                actor_role=actor_role,
                audit_log_id=audit_id,
            )

        # 并发冲突：查实际状态
        actual = await self.query_state(study_uid)
        logger.warning(
            "state CAS conflict: %s expected %s, actual=%s (audit=%s)",
            study_uid,
            from_state.value,
            (actual.value if actual else "<none>"),
            audit_id,
        )
        return TransitionResult(
            success=False,
            study_uid=study_uid,
            from_state=from_state,
            to_state=to_state,
            new_state=(actual if actual is not None else from_state),
            actor_role=actor_role,
            audit_log_id=audit_id,
            from_state_actual=actual,
        )

    # ------------------------------------------------------------------
    # 调试 / 监控
    # ------------------------------------------------------------------
    async def list_audit_for_study(
        self,
        study_uid: str,
        *,
        event_type_prefix: str = "state_",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """拉取某 study 的全部状态变更审计行（调试与飞检追溯）。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT id, actor_role, ts, event_type, payload_json
                FROM audit_log
                WHERE study_uid = ? AND event_type LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (study_uid, f"{event_type_prefix}%", limit),
            )
            rows = await cur.fetchall()
            await cur.close()
        return [
            {
                "id": int(r[0]),
                "actor_role": r[1],
                "ts": r[2],
                "event_type": r[3],
                "payload": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]


__all__ = [
    "CxrState",
    "ActorRole",
    "CxrStateMachine",
    "StateTransitionError",
    "TransitionResult",
    "VALID_TRANSITIONS",
]
