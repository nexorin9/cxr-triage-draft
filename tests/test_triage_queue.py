"""tests/test_triage_queue.py — 阳性优先级队列（按 max_prob 降序）。

覆盖（CLAUDE.md Phase 4 验收 · Task 10）：

1. **enqueue**：单条入队 + INSERT OR REPLACE 覆盖更新（同 study 二次入站
   刷新 max_prob / finding_text / positive_flag / inserted_at，并把
   dispatched_at 重置为 NULL 允许重派发）；
2. **peek_next**：3 条不同 max_prob 入队 → 拉最高；mark_dispatched
   后下一条降到次高；空队列返回 None；
3. **list_pending**：ORDER BY max_prob DESC + inserted_at ASC tie-break；
4. **mark_dispatched**：CAS 防护（两条同 study 并发 → 一成功一失配），
   失配时返 False 不抛；不存在 study 返 False；
5. **count_pending / count_dispatched / count_by_positive** 监控接口；
6. **get_by_study**：驳回场景下回查；
7. **audit_log**：enqueue 写 ``triage_queued`` + mark_dispatched 写
   ``triage_dispatched`` 各一行（actor_role=system 默认）；
8. **入参校验**：空 / 非字符串 study_uid、max_prob 越界、非 dict draft、
   finding_text 含 NUL、limit ≤ 0、positive_flag 非 bool/None 等都拒绝；
9. **list_dispatched** 历史已派发栏。

测试约定：每条测试用独立 tmp_db 路径，避免跨测试污染；并发用
``asyncio.gather`` 两条 mark_dispatched 模拟两个医师同时确认。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from cxr_triage.engine import (
    DEFAULT_DISPATCH_LIMIT,
    TriageQueue,
    TriageQueueError,
)
from cxr_triage.persistence import Database, init_db


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test_tq.db")
    await d.connect()
    await init_db(d)
    yield d
    await d.close()


@pytest.fixture
async def tq(db: Database) -> TriageQueue:
    return TriageQueue(db)


# 占位 UID（与既有测试一致，避免「看起来像真数据」）
EX_UID_A = "1.2.826.0.1.3680043.8.498.<EXAMPLE_TQ_A>"
EX_UID_B = "1.2.826.0.1.3680043.8.498.<EXAMPLE_TQ_B>"
EX_UID_C = "1.2.826.0.1.3680043.8.498.<EXAMPLE_TQ_C>"


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


def _draft(text: str) -> dict[str, Any]:
    """构造最小 draft dict（仅 finding_text 字段；其余字段由调用方另行
    写入 idempotency）。"""
    return {
        "finding_text": text,
        "classifier_probs": {"Pneumothorax": 0.85},  # 不入队列表，仅校验 dict 接收
        "positive_flag": True,
    }


# ======================================================================
# 1. enqueue
# ======================================================================
async def test_enqueue_single_study(tq: TriageQueue, db: Database):
    """单条入队 + 行可读出 + audit_log 落 1 条 triage_queued。"""
    rowcount = await tq.enqueue(
        EX_UID_A, 0.85, _draft("右下肺野见气胸征象（草稿）"),
        positive_flag=True,
    )
    assert rowcount == 1

    row = await tq.get_by_study(EX_UID_A)
    assert row is not None
    assert row["study_uid"] == EX_UID_A
    assert row["max_prob"] == pytest.approx(0.85)
    assert row["positive_flag"] is True
    assert "气胸" in row["finding_text"]
    assert row["dispatched_at"] is None
    assert row["inserted_at"]

    audits = await _fetch_audit(db, EX_UID_A)
    assert len(audits) == 1
    assert audits[0]["event_type"] == "triage_queued"
    assert audits[0]["actor_role"] == "system"
    assert audits[0]["payload"]["max_prob"] == pytest.approx(0.85)
    assert audits[0]["payload"]["positive_flag"] is True


async def test_enqueue_same_study_replaces(tq: TriageQueue, db: Database):
    """同 study 二次入站 → INSERT OR REPLACE 刷新 max_prob / finding_text /
    positive_flag / inserted_at，并把 dispatched_at 重置为 NULL。"""
    await tq.enqueue(EX_UID_A, 0.60, _draft("首版所见"), positive_flag=True)
    first = await tq.get_by_study(EX_UID_A)
    first_inserted = first["inserted_at"]
    # 派发后再二次入站
    await tq.mark_dispatched(EX_UID_A)
    assert (await tq.get_by_study(EX_UID_A))["dispatched_at"] is not None

    # 二次入站（同 study，更高 prob，新所见）
    rowcount = await tq.enqueue(EX_UID_A, 0.95, _draft("重跑所见"), positive_flag=True)
    # SQLite OR REPLACE 二次更新通常返回 rowcount=2（删除 + 插入）
    assert rowcount >= 1

    row = await tq.get_by_study(EX_UID_A)
    assert row["max_prob"] == pytest.approx(0.95)
    assert row["finding_text"] == "重跑所见"
    assert row["dispatched_at"] is None  # 重置回 NULL 允许重新派发
    # inserted_at 应当被刷新（不一定单调推进，但**字符串字序**应 ≥ 首版）
    assert row["inserted_at"] >= first_inserted

    # audit_log 应当有 2 行 triage_queued + 1 行 triage_dispatched
    audits = await _fetch_audit(db, EX_UID_A)
    assert len(audits) == 3
    assert [a["event_type"] for a in audits] == [
        "triage_queued",
        "triage_dispatched",
        "triage_queued",
    ]


async def test_enqueue_negative_default_flag(tq: TriageQueue):
    """positive_flag=None 且 max_prob < 0.5 → positive_flag 自动 False。"""
    await tq.enqueue(EX_UID_A, 0.30, _draft("正常胸片"))
    row = await tq.get_by_study(EX_UID_A)
    assert row["positive_flag"] is False

    await tq.enqueue(EX_UID_A, 0.49, _draft("边界阴性"))
    row = await tq.get_by_study(EX_UID_A)
    assert row["positive_flag"] is False

    await tq.enqueue(EX_UID_A, 0.51, _draft("刚过线"))
    row = await tq.get_by_study(EX_UID_A)
    assert row["positive_flag"] is True


async def test_enqueue_rejects_invalid_prob(tq: TriageQueue):
    """max_prob 越界 / 非数字拒绝。"""
    for bad in (-0.01, 1.01, 2.0, -1.0):
        with pytest.raises(TriageQueueError, match="max_prob"):
            await tq.enqueue(EX_UID_A, bad, _draft("bad"))
    for bad in ("0.5", None, [0.5]):
        with pytest.raises(TriageQueueError, match="max_prob"):
            await tq.enqueue(EX_UID_A, bad, _draft("bad"))


async def test_enqueue_rejects_invalid_study_uid(tq: TriageQueue):
    """空 / 非字符串 / 含 NUL 的 study_uid 拒绝。"""
    with pytest.raises(TriageQueueError, match="study_uid"):
        await tq.enqueue("", 0.5, _draft("empty"))
    with pytest.raises(TriageQueueError, match="study_uid"):
        await tq.enqueue("   ", 0.5, _draft("whitespace"))
    with pytest.raises(TriageQueueError, match="study_uid"):
        await tq.enqueue(123, 0.5, _draft("int"))
    with pytest.raises(TriageQueueError, match="study_uid"):
        await tq.enqueue(None, 0.5, _draft("none"))
    with pytest.raises(TriageQueueError, match="NUL"):
        await tq.enqueue("1.2.3\x00bad", 0.5, _draft("nul"))


async def test_enqueue_rejects_invalid_draft(tq: TriageQueue):
    """draft 非 dict / finding_text 含 NUL 拒绝。"""
    with pytest.raises(TriageQueueError, match="draft"):
        await tq.enqueue(EX_UID_A, 0.5, "not a dict")
    with pytest.raises(TriageQueueError, match="finding_text"):
        await tq.enqueue(EX_UID_A, 0.5, {"finding_text": 12345})
    with pytest.raises(TriageQueueError, match="NUL"):
        await tq.enqueue(EX_UID_A, 0.5, {"finding_text": "bad\x00char"})


async def test_enqueue_rejects_invalid_positive_flag(tq: TriageQueue):
    """positive_flag 非 bool/None 拒绝。"""
    for bad in (1, 0, "yes", [True]):
        with pytest.raises(TriageQueueError, match="positive_flag"):
            await tq.enqueue(EX_UID_A, 0.5, _draft("d"), positive_flag=bad)


async def test_enqueue_missing_finding_text_defaults_empty(tq: TriageQueue):
    """draft 缺 finding_text 字段 → 默认空串；仍可入队。"""
    await tq.enqueue(EX_UID_A, 0.5, {"foo": "bar"})
    row = await tq.get_by_study(EX_UID_A)
    assert row["finding_text"] == ""


# ======================================================================
# 2. peek_next / list_pending
# ======================================================================
async def test_peek_next_empty_queue(tq: TriageQueue):
    """空队列 → peek_next 返回 None；count_pending=0。"""
    assert await tq.peek_next() is None
    assert await tq.count_pending() == 0


async def test_peek_next_returns_highest_max_prob(tq: TriageQueue):
    """3 条不同 max_prob 入队 → peek_next 拉最高。"""
    await tq.enqueue(EX_UID_A, 0.40, _draft("低"), positive_flag=False)
    await tq.enqueue(EX_UID_B, 0.85, _draft("高"), positive_flag=True)
    await tq.enqueue(EX_UID_C, 0.60, _draft("中"), positive_flag=True)

    top = await tq.peek_next()
    assert top is not None
    assert top["study_uid"] == EX_UID_B
    assert top["max_prob"] == pytest.approx(0.85)
    # peek_next 不修改；count_pending 仍为 3
    assert await tq.count_pending() == 3
    assert await tq.count_dispatched() == 0


async def test_peek_next_skips_dispatched(tq: TriageQueue):
    """peek_next 跳过已 dispatched；连续调用降序拉完。"""
    await tq.enqueue(EX_UID_A, 0.40, _draft("低"), positive_flag=False)
    await tq.enqueue(EX_UID_B, 0.85, _draft("高"), positive_flag=True)
    await tq.enqueue(EX_UID_C, 0.60, _draft("中"), positive_flag=True)

    # 把最高的 B 派发掉
    assert await tq.mark_dispatched(EX_UID_B) is True

    # peek_next 现在应返回次高 C
    top = await tq.peek_next()
    assert top["study_uid"] == EX_UID_C

    # 把 C 也派发
    assert await tq.mark_dispatched(EX_UID_C) is True

    # 现在只剩 A
    top = await tq.peek_next()
    assert top["study_uid"] == EX_UID_A
    assert await tq.count_pending() == 1


async def test_list_pending_ordering_by_max_prob(tq: TriageQueue):
    """list_pending 按 max_prob DESC + inserted_at ASC tie-break。"""
    await tq.enqueue(EX_UID_A, 0.40, _draft("低"), positive_flag=False)
    await tq.enqueue(EX_UID_B, 0.85, _draft("高"), positive_flag=True)
    await tq.enqueue(EX_UID_C, 0.60, _draft("中"), positive_flag=True)

    rows = await tq.list_pending(limit=10)
    uids = [r["study_uid"] for r in rows]
    assert uids == [EX_UID_B, EX_UID_C, EX_UID_A]
    assert all(r["dispatched_at"] is None for r in rows)


async def test_list_pending_tie_break_by_inserted_at(tq: TriageQueue):
    """同 max_prob 时按 inserted_at ASC（先入队优先）。"""
    # 三条同 max_prob=0.50；按写入顺序
    await tq.enqueue(EX_UID_A, 0.50, _draft("first"))
    await tq.enqueue(EX_UID_B, 0.50, _draft("second"))
    await tq.enqueue(EX_UID_C, 0.50, _draft("third"))

    rows = await tq.list_pending(limit=10)
    uids = [r["study_uid"] for r in rows]
    assert uids == [EX_UID_A, EX_UID_B, EX_UID_C]


async def test_list_pending_respects_limit(tq: TriageQueue):
    """list_pending(limit=N) 只拉 N 条。"""
    for i in range(5):
        await tq.enqueue(
            f"1.2.826.0.1.3680043.8.498.<EXAMPLE_LIM_{i}>",
            0.50 + i * 0.05,
            _draft(f"draft {i}"),
        )

    rows = await tq.list_pending(limit=3)
    assert len(rows) == 3
    # 应是 max_prob 最高的 3 条
    assert rows[0]["max_prob"] == pytest.approx(0.70)
    assert rows[1]["max_prob"] == pytest.approx(0.65)
    assert rows[2]["max_prob"] == pytest.approx(0.60)


async def test_list_pending_rejects_invalid_limit(tq: TriageQueue):
    """limit ≤ 0 / 非 int 拒绝。"""
    for bad in (0, -1, "5", None, 1.5):
        with pytest.raises(TriageQueueError, match="limit"):
            await tq.list_pending(limit=bad)


async def test_default_dispatch_limit_constant():
    """DEFAULT_DISPATCH_LIMIT 是正整数（约定 50）。"""
    assert isinstance(DEFAULT_DISPATCH_LIMIT, int)
    assert DEFAULT_DISPATCH_LIMIT > 0


# ======================================================================
# 3. mark_dispatched + CAS
# ======================================================================
async def test_mark_dispatched_sets_timestamp(tq: TriageQueue, db: Database):
    """mark_dispatched 把 dispatched_at 置为非 NULL；audit_log 落 1 行。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("d"))
    assert (await tq.get_by_study(EX_UID_A))["dispatched_at"] is None

    ok = await tq.mark_dispatched(EX_UID_A, actor_role="jr")
    assert ok is True

    row = await tq.get_by_study(EX_UID_A)
    assert row["dispatched_at"] is not None

    audits = await _fetch_audit(db, EX_UID_A)
    dispatched = [a for a in audits if a["event_type"] == "triage_dispatched"]
    assert len(dispatched) == 1
    assert dispatched[0]["actor_role"] == "jr"
    assert dispatched[0]["payload"] == {"actor_role": "jr"}


async def test_mark_dispatched_nonexistent_returns_false(tq: TriageQueue):
    """不存在 study → 返回 False 不抛。"""
    ok = await tq.mark_dispatched(
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_NEVER>",
        actor_role="jr",
    )
    assert ok is False


async def test_mark_dispatched_idempotent_second_call_returns_false(tq: TriageQueue):
    """重复 mark_dispatched → 第二次 False（CAS 失配，不抛）。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("d"))
    assert await tq.mark_dispatched(EX_UID_A) is True
    # 第二次派发：应当 False，且 dispatched_at 不变
    before = (await tq.get_by_study(EX_UID_A))["dispatched_at"]
    assert await tq.mark_dispatched(EX_UID_A) is False
    after = (await tq.get_by_study(EX_UID_A))["dispatched_at"]
    assert after == before


async def test_mark_dispatched_cas_concurrent_only_one_wins(tq: TriageQueue):
    """并发两条 mark_dispatched 同 study → 一条 True 一条 False。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("d"))
    results = await asyncio.gather(
        tq.mark_dispatched(EX_UID_A, actor_role="jr"),
        tq.mark_dispatched(EX_UID_A, actor_role="reviewer"),
    )
    successes = [r for r in results if r is True]
    failures = [r for r in results if r is False]
    assert len(successes) == 1
    assert len(failures) == 1


async def test_mark_dispatched_rejects_empty_actor_role(tq: TriageQueue):
    """actor_role 非字符串 / 空字符串拒绝。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("d"))
    for bad in ("", "   ", None, 123, ["jr"]):
        with pytest.raises(TriageQueueError, match="actor_role"):
            await tq.mark_dispatched(EX_UID_A, actor_role=bad)


async def test_mark_dispatched_after_re_enqueue_resets(tq: TriageQueue):
    """派发后再次 enqueue → dispatched_at 重置为 NULL → 可再次派发。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("first"))
    assert await tq.mark_dispatched(EX_UID_A) is True
    # 驳回 + 重跑场景
    await tq.enqueue(EX_UID_A, 0.95, _draft("rerun"))
    row = await tq.get_by_study(EX_UID_A)
    assert row["dispatched_at"] is None
    assert row["max_prob"] == pytest.approx(0.95)
    # 再次派发应当成功
    assert await tq.mark_dispatched(EX_UID_A, actor_role="reviewer") is True


# ======================================================================
# 4. count_* 监控接口
# ======================================================================
async def test_count_pending_and_dispatched(tq: TriageQueue):
    """count_pending / count_dispatched 互相独立计数。"""
    assert await tq.count_pending() == 0
    assert await tq.count_dispatched() == 0

    await tq.enqueue(EX_UID_A, 0.50, _draft("a"))
    await tq.enqueue(EX_UID_B, 0.50, _draft("b"))
    await tq.enqueue(EX_UID_C, 0.50, _draft("c"))
    assert await tq.count_pending() == 3
    assert await tq.count_dispatched() == 0

    await tq.mark_dispatched(EX_UID_A)
    await tq.mark_dispatched(EX_UID_B)
    assert await tq.count_pending() == 1
    assert await tq.count_dispatched() == 2


async def test_count_by_positive(tq: TriageQueue):
    """count_by_positive 返回 {positive: N, negative: M}。"""
    await tq.enqueue(EX_UID_A, 0.40, _draft("neg"), positive_flag=False)
    await tq.enqueue(EX_UID_B, 0.85, _draft("pos"), positive_flag=True)
    await tq.enqueue(EX_UID_C, 0.90, _draft("pos"), positive_flag=True)

    counts = await tq.count_by_positive()
    assert counts == {"positive": 2, "negative": 1}


async def test_count_by_positive_empty_queue(tq: TriageQueue):
    """空队列 → 全 0（不抛 KeyError）。"""
    counts = await tq.count_by_positive()
    assert counts == {"positive": 0, "negative": 0}


# ======================================================================
# 5. list_dispatched + get_by_study
# ======================================================================
async def test_list_dispatched_ordering(tq: TriageQueue):
    """list_dispatched 按 dispatched_at DESC（最新派发在前）。"""
    await tq.enqueue(EX_UID_A, 0.50, _draft("a"))
    await tq.enqueue(EX_UID_B, 0.50, _draft("b"))
    await tq.enqueue(EX_UID_C, 0.50, _draft("c"))
    await tq.mark_dispatched(EX_UID_A)
    await asyncio.sleep(0.05)
    await tq.mark_dispatched(EX_UID_B)
    await asyncio.sleep(0.05)
    await tq.mark_dispatched(EX_UID_C)

    rows = await tq.list_dispatched(limit=10)
    uids = [r["study_uid"] for r in rows]
    assert uids == [EX_UID_C, EX_UID_B, EX_UID_A]
    assert all(r["dispatched_at"] is not None for r in rows)


async def test_get_by_study_returns_full_dict(tq: TriageQueue):
    """get_by_study 返回完整 6 字段 dict。"""
    await tq.enqueue(EX_UID_A, 0.85, _draft("draft text"), positive_flag=True)
    row = await tq.get_by_study(EX_UID_A)
    assert set(row.keys()) == {
        "study_uid",
        "max_prob",
        "positive_flag",
        "finding_text",
        "inserted_at",
        "dispatched_at",
    }


async def test_get_by_study_nonexistent_returns_none(tq: TriageQueue):
    """不存在的 study → 返回 None 不抛。"""
    assert await tq.get_by_study(
        "1.2.826.0.1.3680043.8.498.<EXAMPLE_NEVER>"
    ) is None


async def test_get_by_study_rejects_invalid_study_uid(tq: TriageQueue):
    """空 / NUL study_uid 拒绝（与 enqueue 一致）。"""
    with pytest.raises(TriageQueueError, match="study_uid"):
        await tq.get_by_study("")
    with pytest.raises(TriageQueueError, match="NUL"):
        await tq.get_by_study("1.2.3\x00bad")


# ======================================================================
# 6. 并发 + 端到端
# ======================================================================
async def test_concurrent_enqueue_distinct_studies(tq: TriageQueue):
    """并发 5 条不同 study 入队 → count_pending=5；peek_next 拉最高。"""
    uids = [
        f"1.2.826.0.1.3680043.8.498.<EXAMPLE_CONC_{i}>"
        for i in range(5)
    ]
    probs = [0.30, 0.50, 0.70, 0.90, 0.95]
    await asyncio.gather(*[
        tq.enqueue(uid, p, _draft(f"d{idx}"))
        for idx, (uid, p) in enumerate(zip(uids, probs))
    ])
    assert await tq.count_pending() == 5
    top = await tq.peek_next()
    assert top["study_uid"] == uids[-1]  # max_prob=0.95


async def test_concurrent_enqueue_same_study_idempotent(tq: TriageQueue):
    """并发 5 条同 study 入队 → 只剩 1 行；最后 max_prob 是赢家。"""
    winners = await asyncio.gather(*[
        tq.enqueue(EX_UID_A, 0.10 + i * 0.10, _draft(f"v{i}"))
        for i in range(5)
    ])
    assert all(w >= 1 for w in winners)  # 至少 1 行受影响
    row = await tq.get_by_study(EX_UID_A)
    # 最后一次写入应当是 max_prob=0.50（i=4），但 SQLite 串行化下顺序不确定；
    # 这里只断言 max_prob ∈ 入参集合
    assert row["max_prob"] in {0.10, 0.20, 0.30, 0.40, 0.50}
    assert await tq.count_pending() == 1
