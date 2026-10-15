"""tests/test_state_machine.py — 5 态状态机 + CAS 并发防护 + 驳回回退。

覆盖（CLAUDE.md Phase 4 验收 · Task 7）：

1. 5 态正常流转（draft_pending → jr_confirmed → sr_reviewed → sr_signed）；
2. 角色白名单（jr / reviewer / sr 各司其职，越权拒绝）；
3. 非法跃迁（直接跳 stage / sr_signed 再次推进）拒绝抛 StateTransitionError；
4. **CAS 并发冲突**：同 study_uid 两路同时 sr_sign → 只有一条成功，
   另一条返回 ``TransitionResult.success=False`` + ``from_state_actual``；
5. **驳回回退**：jr / reviewer / sr 任一发起驳回 → 一次调用完成两步
   （``from_state → rejected`` → ``draft_pending``），并分别落 audit；
6. 驳回权限边界：sr_signed 不可驳回；reviewer 不能驳回 draft_pending；
7. 审计留痕：每次 transition + 每次 reject 步骤都写 audit_log 行；
8. 状态查询 / 计数 / 按状态列举（list_by_state / count_by_state）。

测试约定：每条测试用独立 tmp_db 路径，避免跨测试污染；并发测试用
``asyncio.gather`` 跑两条 transition 模拟两个医师同时签发。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from cxr_triage.persistence import (
    Database,
    init_db,
    list_tables,
    table_exists,
)
from cxr_triage.state import (
    ActorRole,
    CxrState,
    CxrStateMachine,
    StateTransitionError,
    TransitionResult,
    VALID_TRANSITIONS,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test_state.db")
    await d.connect()
    await init_db(d)
    yield d
    await d.close()


@pytest.fixture
async def sm(db: Database) -> CxrStateMachine:
    return CxrStateMachine(db)


# 已知占位 UID；避免"看起来像真数据"
EX_UID_DRAFT = "1.2.826.0.1.3680043.8.498.<EXAMPLE_DRAFT>"
EX_UID_JR = "1.2.826.0.1.3680043.8.498.<EXAMPLE_JR>"
EX_UID_SR = "1.2.826.0.1.3680043.8.498.<EXAMPLE_SR>"
EX_UID_RJ = "1.2.826.0.1.3680043.8.498.<EXAMPLE_RJ>"


# ----------------------------------------------------------------------
# 工具：把 audit_log 行拉成 list[(id, actor, ts, event, payload_dict)]
# ----------------------------------------------------------------------
async def _fetch_audit(db: Database, study_uid: str) -> list[dict[str, Any]]:
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT id, actor_role, event_type, payload_json "
            "FROM audit_log WHERE study_uid = ? ORDER BY id ASC",
            (study_uid,),
        )
        rows = await cur.fetchall()
        await cur.close()
    return [
        {
            "id": int(r[0]),
            "actor_role": r[1],
            "event_type": r[2],
            "payload": json.loads(r[3]) if r[3] else {},
        }
        for r in rows
    ]


# ======================================================================
# 1. Schema & 初始化
# ======================================================================
async def test_state_machine_table_created(db: Database):
    """init_db 后 triage_states 表存在。"""
    assert await table_exists(db, "triage_states") is True


async def test_init_db_can_skip_state_machine(tmp_path: Path):
    """``include_state_machine=False`` 不建状态机表（场景测试用）。"""
    d = Database(tmp_path / "no_state.db")
    await d.connect()
    try:
        await init_db(d, include_state_machine=False)
        tables = set(await list_tables(d))
        assert "triage_states" not in tables
        assert "incoming_studies" in tables
        assert "outbox" in tables
    finally:
        await d.close()


async def test_get_or_init_state_creates_draft_pending(sm: CxrStateMachine):
    """首次查询 → 自动建 draft_pending 行 + audit state_initialized。"""
    state = await sm.get_or_init_state(EX_UID_DRAFT)
    assert state == CxrState.DRAFT_PENDING

    audit = await _fetch_audit(sm.db, EX_UID_DRAFT)
    assert len(audit) == 1
    assert audit[0]["event_type"] == "state_initialized"
    assert audit[0]["actor_role"] == ActorRole.SYSTEM.value
    assert audit[0]["payload"]["initial_state"] == "draft_pending"


async def test_get_or_init_state_idempotent(sm: CxrStateMachine):
    """二次查询相同 UID → 不重建行、不多写 audit。"""
    s1 = await sm.get_or_init_state(EX_UID_DRAFT)
    s2 = await sm.get_or_init_state(EX_UID_DRAFT)
    assert s1 == s2 == CxrState.DRAFT_PENDING
    audit = await _fetch_audit(sm.db, EX_UID_DRAFT)
    assert len(audit) == 1  # 仅 state_initialized 一次


async def test_query_state_returns_none_for_unknown(sm: CxrStateMachine):
    """未初始化且无业务上下文 → query_state 返回 None（不自动建）。"""
    res = await sm.query_state("1.2.826.0.1.3680043.8.498.<EXAMPLE_UNKNOWN>")
    assert res is None


# ======================================================================
# 2. 5 态正常流转 + 角色白名单
# ======================================================================
async def test_happy_path_three_roles(sm: CxrStateMachine):
    """draft_pending → jr_confirmed → sr_reviewed → sr_signed 全流程。"""
    await sm.get_or_init_state(EX_UID_DRAFT)

    r1 = await sm.transition(EX_UID_DRAFT, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    assert r1.success is True
    assert r1.new_state == CxrState.JR_CONFIRMED

    r2 = await sm.transition(EX_UID_DRAFT, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    assert r2.success is True
    assert r2.new_state == CxrState.SR_REVIEWED

    r3 = await sm.transition(EX_UID_DRAFT, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.SR)
    assert r3.success is True
    assert r3.new_state == CxrState.SR_SIGNED

    assert await sm.query_state(EX_UID_DRAFT) == CxrState.SR_SIGNED


async def test_role_mismatch_rejected_by_transition(sm: CxrStateMachine):
    """draft_pending → jr_confirmed 只允许 jr；reviewer / sr / system 一律拒绝。"""
    await sm.get_or_init_state(EX_UID_DRAFT)

    with pytest.raises(StateTransitionError, match="角色"):
        await sm.transition(
            EX_UID_DRAFT,
            CxrState.DRAFT_PENDING,
            CxrState.JR_CONFIRMED,
            ActorRole.REVIEWER,
        )
    with pytest.raises(StateTransitionError, match="角色"):
        await sm.transition(
            EX_UID_DRAFT,
            CxrState.DRAFT_PENDING,
            CxrState.JR_CONFIRMED,
            ActorRole.SR,
        )
    with pytest.raises(StateTransitionError, match="角色"):
        await sm.transition(
            EX_UID_DRAFT,
            CxrState.DRAFT_PENDING,
            CxrState.JR_CONFIRMED,
            ActorRole.SYSTEM,
        )
    # 状态未变
    assert await sm.query_state(EX_UID_DRAFT) == CxrState.DRAFT_PENDING


async def test_role_mismatch_review_cannot_sign(sm: CxrStateMachine):
    """reviewer 无权 sr_sign。"""
    await sm.get_or_init_state(EX_UID_JR)
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_JR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    with pytest.raises(StateTransitionError, match="角色"):
        await sm.transition(EX_UID_JR, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.REVIEWER)
    assert await sm.query_state(EX_UID_JR) == CxrState.SR_REVIEWED


async def test_role_mismatch_jr_cannot_review(sm: CxrStateMachine):
    """jr 无权 sr_review。"""
    await sm.get_or_init_state(EX_UID_JR)
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    with pytest.raises(StateTransitionError, match="角色"):
        await sm.transition(EX_UID_JR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.JR)
    assert await sm.query_state(EX_UID_JR) == CxrState.JR_CONFIRMED


async def test_illegal_skip_jr_pending_to_sr_signed(sm: CxrStateMachine):
    """draft_pending → sr_signed（非白名单跃迁）拒绝。"""
    await sm.get_or_init_state(EX_UID_SR)
    with pytest.raises(StateTransitionError, match="非法跃迁"):
        await sm.transition(EX_UID_SR, CxrState.DRAFT_PENDING, CxrState.SR_SIGNED, ActorRole.SR)
    assert await sm.query_state(EX_UID_SR) == CxrState.DRAFT_PENDING


async def test_sr_signed_terminal_cannot_be_advanced(sm: CxrStateMachine):
    """sr_signed 是终态；再走任何 transition 都拒绝。"""
    await sm.get_or_init_state(EX_UID_SR)
    await sm.transition(EX_UID_SR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_SR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    await sm.transition(EX_UID_SR, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.SR)
    assert await sm.query_state(EX_UID_SR) == CxrState.SR_SIGNED

    with pytest.raises(StateTransitionError, match="非法跃迁"):
        # sr_signed 没有任何白名单出边
        await sm.transition(EX_UID_SR, CxrState.SR_SIGNED, CxrState.JR_CONFIRMED, ActorRole.JR)


# ======================================================================
# 3. 字符串/枚举互转
# ======================================================================
async def test_string_role_and_state_accepted(sm: CxrStateMachine):
    """``str`` 入参也能被正确接受（FastAPI JSON payload 用）。"""
    await sm.get_or_init_state(EX_UID_JR)
    r = await sm.transition(EX_UID_JR, "draft_pending", "jr_confirmed", "jr")
    assert r.success is True
    assert r.new_state == CxrState.JR_CONFIRMED


async def test_unknown_state_raises(sm: CxrStateMachine):
    """未知 from_state 字符串 → 抛 StateTransitionError。"""
    await sm.get_or_init_state(EX_UID_JR)
    with pytest.raises(StateTransitionError, match="未知状态"):
        await sm.transition(EX_UID_JR, "mystery_state", "jr_confirmed", "jr")


async def test_unknown_role_raises(sm: CxrStateMachine):
    """未知 actor_role 字符串 → 抛 StateTransitionError。"""
    await sm.get_or_init_state(EX_UID_JR)
    with pytest.raises(StateTransitionError, match="未知角色"):
        await sm.transition(EX_UID_JR, "draft_pending", "jr_confirmed", "ghost_role")


# ======================================================================
# 4. CAS 并发防护
# ======================================================================
async def test_cas_conflict_returns_false_no_raise(sm: CxrStateMachine):
    """期望 from_state 不匹配 → 返回 success=False，不抛。"""
    await sm.get_or_init_state(EX_UID_JR)
    # 实际状态 = draft_pending；尝试 draft_pending → jr_confirmed 但
    # 用错误的 from_state 假装已 jr_confirmed，CAS 必失败
    r = await sm.transition(
        EX_UID_JR,
        CxrState.JR_CONFIRMED,  # 错误的 from_state
        CxrState.SR_REVIEWED,
        ActorRole.REVIEWER,
    )
    assert r.success is False
    assert r.from_state_actual == CxrState.DRAFT_PENDING
    # CAS 失败也写 audit（标 cas_success=False），便于飞检追溯"未生效尝试"
    assert r.audit_log_id is not None and r.audit_log_id > 0
    # 实际状态未变
    assert await sm.query_state(EX_UID_JR) == CxrState.DRAFT_PENDING


async def test_cas_concurrent_two_sr_signs_only_one_wins(sm: CxrStateMachine):
    """两路并发 sr_sign → 只有一条 UPDATE 影响行 = 1，另一条 0。"""
    await sm.get_or_init_state(EX_UID_SR)
    await sm.transition(EX_UID_SR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_SR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    assert await sm.query_state(EX_UID_SR) == CxrState.SR_REVIEWED

    # 收集并发开始前的 audit id，作为分界线
    audit_before = await _fetch_audit(sm.db, EX_UID_SR)
    last_audit_id_before = max((r["id"] for r in audit_before), default=0)

    # 两路并发 sr_sign
    r1, r2 = await asyncio.gather(
        sm.transition(EX_UID_SR, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.SR),
        sm.transition(EX_UID_SR, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.SR),
    )
    successes = sorted([r1.success, r2.success])
    assert successes == [False, True], "恰好一路成功一路失败"
    # 状态最终是 sr_signed（赢家写的）
    assert await sm.query_state(EX_UID_SR) == CxrState.SR_SIGNED
    # 仅看并发期间新增的 audit 行
    audit = await _fetch_audit(sm.db, EX_UID_SR)
    cas_records = [r for r in audit if r["id"] > last_audit_id_before and r["event_type"] == "state_change"]
    assert len(cas_records) == 2
    success_count = sum(1 for r in cas_records if r["payload"]["cas_success"] is True)
    assert success_count == 1
    fail_count = sum(1 for r in cas_records if r["payload"]["cas_success"] is False)
    assert fail_count == 1


async def test_cas_conflict_audit_records_cas_success_false(sm: CxrStateMachine):
    """CAS 失败的 audit 行 payload.cas_success = False（飞检可还原）。"""
    await sm.get_or_init_state(EX_UID_JR)
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    # 用错的 from_state 再推一次
    r = await sm.transition(
        EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR
    )
    assert r.success is False
    audit = await _fetch_audit(sm.db, EX_UID_JR)
    last = audit[-1]
    assert last["event_type"] == "state_change"
    assert last["payload"]["cas_success"] is False
    assert last["payload"]["from_state"] == "draft_pending"
    assert last["payload"]["to_state"] == "jr_confirmed"


# ======================================================================
# 5. 驳回回退
# ======================================================================
async def test_reject_from_jr_confirmed_returns_to_draft(sm: CxrStateMachine):
    """jr_confirmed 被 reviewer 驳回 → 自动回到 draft_pending。"""
    await sm.get_or_init_state(EX_UID_RJ)
    await sm.transition(EX_UID_RJ, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    assert await sm.query_state(EX_UID_RJ) == CxrState.JR_CONFIRMED

    r = await sm.reject(
        EX_UID_RJ, from_state=CxrState.JR_CONFIRMED, actor_role=ActorRole.REVIEWER, reason="心影异常"
    )
    assert r.success is True
    assert r.new_state == CxrState.DRAFT_PENDING
    # 最终态是 draft_pending（不是 rejected）
    assert await sm.query_state(EX_UID_RJ) == CxrState.DRAFT_PENDING

    # audit 写了 4 条：state_initialized + 1 state_change + 2 state_reject_step*
    audit = await _fetch_audit(sm.db, EX_UID_RJ)
    types = [a["event_type"] for a in audit]
    assert "state_reject_step1" in types
    assert "state_reject_step2" in types


async def test_reject_from_sr_reviewed_returns_to_draft(sm: CxrStateMachine):
    """sr_reviewed 被 sr 驳回 → draft_pending。"""
    await sm.get_or_init_state(EX_UID_RJ)
    await sm.transition(EX_UID_RJ, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_RJ, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    r = await sm.reject(EX_UID_RJ, from_state=CxrState.SR_REVIEWED, actor_role=ActorRole.SR, reason="size 不对")
    assert r.success is True
    assert r.new_state == CxrState.DRAFT_PENDING
    assert await sm.query_state(EX_UID_RJ) == CxrState.DRAFT_PENDING


async def test_reject_from_draft_pending_allowed(sm: CxrStateMachine):
    """draft_pending 被 jr 驳回 → draft_pending（其实就保持，但写两条 audit）。"""
    await sm.get_or_init_state(EX_UID_RJ)
    r = await sm.reject(EX_UID_RJ, from_state=CxrState.DRAFT_PENDING, actor_role=ActorRole.JR, reason="完全错误")
    assert r.success is True
    assert r.new_state == CxrState.DRAFT_PENDING
    # audit 两条：step1 (DP→rejected) + step2 (rejected→DP)
    audit = await _fetch_audit(sm.db, EX_UID_RJ)
    step1 = [a for a in audit if a["event_type"] == "state_reject_step1"]
    step2 = [a for a in audit if a["event_type"] == "state_reject_step2"]
    assert len(step1) == 1
    assert len(step2) == 1
    assert step1[0]["payload"]["reason"] == "完全错误"


async def test_reject_terminal_sr_signed_forbidden(sm: CxrStateMachine):
    """sr_signed 是终态；不允许驳回（如需整改走其它流程）。"""
    await sm.get_or_init_state(EX_UID_SR)
    await sm.transition(EX_UID_SR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_SR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    await sm.transition(EX_UID_SR, CxrState.SR_REVIEWED, CxrState.SR_SIGNED, ActorRole.SR)

    with pytest.raises(StateTransitionError, match="sr_signed"):
        await sm.reject(EX_UID_SR, from_state=CxrState.SR_SIGNED, actor_role=ActorRole.SR)
    assert await sm.query_state(EX_UID_SR) == CxrState.SR_SIGNED


async def test_reject_role_authority_at_draft_pending(sm: CxrStateMachine):
    """draft_pending：jr / reviewer / sr 三个角色都能驳回（未确认阶段谁来驳都行）。"""
    await sm.get_or_init_state(EX_UID_RJ)

    # jr 驳回 draft_pending → OK
    r = await sm.reject(EX_UID_RJ, from_state=CxrState.DRAFT_PENDING, actor_role=ActorRole.JR, reason="jr 驳回")
    assert r.success is True
    assert r.new_state == CxrState.DRAFT_PENDING

    # reviewer 驳回 draft_pending（另一个 study）→ 也 OK
    await sm.get_or_init_state("1.2.826.0.1.3680043.8.498.<EXAMPLE_RJ2>")
    r = await sm.reject(
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_RJ2>",
        from_state=CxrState.DRAFT_PENDING,
        actor_role=ActorRole.REVIEWER,
        reason="reviewer 驳回",
    )
    assert r.success is True

    # sr 驳回 draft_pending（再一个 study）→ 也 OK
    await sm.get_or_init_state("1.2.826.0.1.3680043.8.498.<EXAMPLE_RJ3>")
    r = await sm.reject(
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_RJ3>",
        from_state=CxrState.DRAFT_PENDING,
        actor_role=ActorRole.SR,
        reason="sr 驳回",
    )
    assert r.success is True


async def test_reject_double_reject_after_already_rejected_rejected_input(sm: CxrStateMachine):
    """调用 reject(from_state=REJECTED) 直接拒绝（已经被驳回就别再驳）。"""
    await sm.get_or_init_state(EX_UID_RJ)
    with pytest.raises(StateTransitionError, match="已是 rejected"):
        await sm.reject(EX_UID_RJ, from_state=CxrState.REJECTED, actor_role=ActorRole.JR)


async def test_reject_cas_conflict_at_step1(sm: CxrStateMachine):
    """驳回 step1 CAS 失配（并发）→ 返回 success=False，不走 step2。"""
    await sm.get_or_init_state(EX_UID_RJ)
    await sm.transition(EX_UID_RJ, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    # 路径上另一路先把 jr_confirmed 推进到 sr_reviewed
    await sm.transition(EX_UID_RJ, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)
    # 尝试从 jr_confirmed 驳回：实际状态已变 sr_reviewed，CAS 失配
    r = await sm.reject(EX_UID_RJ, from_state=CxrState.JR_CONFIRMED, actor_role=ActorRole.REVIEWER)
    assert r.success is False
    assert r.from_state_actual == CxrState.SR_REVIEWED
    # 数据库状态没回到 draft_pending
    assert await sm.query_state(EX_UID_RJ) == CxrState.SR_REVIEWED
    # audit 只有 step1（标 cas_success=False），无 step2
    audit = await _fetch_audit(sm.db, EX_UID_RJ)
    reject_audits = [a for a in audit if a["event_type"].startswith("state_reject")]
    assert len(reject_audits) == 1
    assert reject_audits[0]["payload"]["cas_success"] is False


# ======================================================================
# 6. transition() 拒绝 to_state=rejected（统一走 reject()）
# ======================================================================
async def test_transition_to_rejected_directly_rejected(sm: CxrStateMachine):
    """to_state=rejected 必须走 reject()，transition() 不允许单独接受。"""
    await sm.get_or_init_state(EX_UID_RJ)
    with pytest.raises(StateTransitionError, match="请用 reject"):
        await sm.transition(
            EX_UID_RJ,
            CxrState.DRAFT_PENDING,
            CxrState.REJECTED,
            ActorRole.JR,
        )


# ======================================================================
# 7. 列表 / 计数
# ======================================================================
async def test_list_by_state(sm: CxrStateMachine):
    """按状态列 study_uid；多条同态按 updated_at DESC。"""
    await sm.get_or_init_state(EX_UID_DRAFT)
    await sm.get_or_init_state(EX_UID_JR)
    await sm.get_or_init_state(EX_UID_SR)
    await sm.get_or_init_state(EX_UID_RJ)
    # 让 EX_UID_JR 推进到 jr_confirmed
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    # 让 EX_UID_SR 推进到 sr_reviewed
    await sm.transition(EX_UID_SR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_SR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)

    draft_rows = await sm.list_by_state(CxrState.DRAFT_PENDING)
    jr_rows = await sm.list_by_state(CxrState.JR_CONFIRMED)
    sr_rows = await sm.list_by_state(CxrState.SR_REVIEWED)

    draft_uids = {r["study_uid"] for r in draft_rows}
    assert {EX_UID_DRAFT, EX_UID_RJ}.issubset(draft_uids)
    jr_uids = {r["study_uid"] for r in jr_rows}
    assert EX_UID_JR in jr_uids
    sr_uids = {r["study_uid"] for r in sr_rows}
    assert EX_UID_SR in sr_uids


async def test_count_by_state(sm: CxrStateMachine):
    """按状态计数。"""
    await sm.get_or_init_state(EX_UID_DRAFT)
    await sm.get_or_init_state(EX_UID_JR)
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)

    counts = await sm.count_by_state()
    assert counts.get("draft_pending", 0) >= 1
    assert counts.get("jr_confirmed", 0) == 1


# ======================================================================
# 8. 审计顺序与拒绝写 audit
# ======================================================================
async def test_every_transition_writes_audit(sm: CxrStateMachine):
    """每次 transition 主体在 audit_log 落一行（含 cas_success 字段）。"""
    await sm.get_or_init_state(EX_UID_DRAFT)
    r1 = await sm.transition(EX_UID_DRAFT, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    assert r1.audit_log_id is not None and r1.audit_log_id > 0
    audit = await _fetch_audit(sm.db, EX_UID_DRAFT)
    changes = [a for a in audit if a["event_type"] == "state_change"]
    assert len(changes) == 1
    assert changes[0]["actor_role"] == ActorRole.JR.value
    assert changes[0]["payload"]["cas_success"] is True


async def test_reject_audit_records_reason(sm: CxrStateMachine):
    """驳回 step1 audit 记录 reason 与 from/to_state。"""
    await sm.get_or_init_state(EX_UID_RJ)
    await sm.transition(EX_UID_RJ, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.reject(EX_UID_RJ, from_state=CxrState.JR_CONFIRMED, actor_role=ActorRole.REVIEWER, reason="心影异常需重拍")
    audit = await _fetch_audit(sm.db, EX_UID_RJ)
    step1 = next(a for a in audit if a["event_type"] == "state_reject_step1")
    assert step1["actor_role"] == ActorRole.REVIEWER.value
    assert step1["payload"]["reason"] == "心影异常需重拍"
    assert step1["payload"]["from_state"] == "jr_confirmed"
    assert step1["payload"]["to_state"] == "rejected"
    assert step1["payload"]["cas_success"] is True


async def test_illegal_transition_does_not_write_audit_for_state_change(sm: CxrStateMachine):
    """非法跃迁/角色越权在 transition() 入口直接抛；audit_log 不写
    state_change 行（state_initialized 行的存在与否取决于是否先调过
    get_or_init_state）。这里只验证：仅 1 行 audit 是 state_initialized。
    """
    await sm.get_or_init_state(EX_UID_JR)
    with pytest.raises(StateTransitionError, match="非法跃迁"):
        await sm.transition(
            EX_UID_JR,
            CxrState.DRAFT_PENDING,
            CxrState.SR_SIGNED,
            ActorRole.SR,
        )
    audit = await _fetch_audit(sm.db, EX_UID_JR)
    changes = [a for a in audit if a["event_type"] == "state_change"]
    assert changes == []  # 非法跃迁根本没写 state_change


# ======================================================================
# 9. 注入 audit_hook（可选 hook 不阻断主流程）
# ======================================================================
async def test_audit_hook_receives_payload(db: Database):
    """audit_hook（可选）接到 actor_role / event_type / payload / ts 字段。"""
    seen: list[dict[str, Any]] = []

    async def hook(**kw):
        seen.append(kw)

    sm = CxrStateMachine(db, audit_hook=hook)
    await sm.transition(
        EX_UID_JR,
        CxrState.DRAFT_PENDING,
        CxrState.JR_CONFIRMED,
        ActorRole.JR,
        payload={"remark": "首检确认"},
    )
    state_changes = [s for s in seen if s["event_type"] == "state_change"]
    assert len(state_changes) == 1
    assert state_changes[0]["actor_role"] == "jr"
    assert state_changes[0]["payload"]["remark"] == "首检确认"
    assert "ts" in state_changes[0]


async def test_audit_hook_failure_does_not_break_transition(db: Database):
    """hook 抛异常 → warning 记日志但不阻断 transition 与 audit_log 写入。"""

    async def bad_hook(**kw):
        raise RuntimeError("hook boom")

    sm = CxrStateMachine(db, audit_hook=bad_hook)
    r = await sm.transition(
        EX_UID_JR,
        CxrState.DRAFT_PENDING,
        CxrState.JR_CONFIRMED,
        ActorRole.JR,
    )
    assert r.success is True
    audit = await _fetch_audit(db, EX_UID_JR)
    assert any(a["event_type"] == "state_change" for a in audit)


# ======================================================================
# 10. list_audit_for_study 调试接口
# ======================================================================
async def test_list_audit_for_study_filters_state_events(sm: CxrStateMachine):
    """list_audit_for_study 只返回 event_type LIKE 'state_%' 的行。"""
    await sm.get_or_init_state(EX_UID_JR)
    await sm.transition(EX_UID_JR, CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED, ActorRole.JR)
    await sm.transition(EX_UID_JR, CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED, ActorRole.REVIEWER)

    rows = await sm.list_audit_for_study(EX_UID_JR)
    types = [r["event_type"] for r in rows]
    # 应包含 state_initialized + 2 state_change
    assert "state_initialized" in types
    assert types.count("state_change") == 2


# ======================================================================
# 11. VALID_TRANSITIONS 白名单表格完整性（防止配置漂移）
# ======================================================================
def test_valid_transitions_covers_required_edges():
    """白名单必须覆盖 4 条 happy-path + 3 驳回 + 1 驳回回退 = 8 条。"""
    expected = {
        (CxrState.DRAFT_PENDING, CxrState.JR_CONFIRMED),
        (CxrState.JR_CONFIRMED, CxrState.SR_REVIEWED),
        (CxrState.SR_REVIEWED, CxrState.SR_SIGNED),
        (CxrState.DRAFT_PENDING, CxrState.REJECTED),
        (CxrState.JR_CONFIRMED, CxrState.REJECTED),
        (CxrState.SR_REVIEWED, CxrState.REJECTED),
        (CxrState.REJECTED, CxrState.DRAFT_PENDING),
    }
    actual = set(VALID_TRANSITIONS.keys())
    assert expected.issubset(actual)
