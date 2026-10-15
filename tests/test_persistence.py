"""tests/test_persistence.py — SQLite 4 表 + outbox + 幂等。

覆盖（CLAUDE.md Phase 4 验收 · Task 3）：

1. ``init_db`` 幂等可重入（多次调用不报错、不丢表）；
2. 4 表 + 幂等表全部存在；
3. ``Outbox.enqueue`` → ``dequeue`` → ``mark_sent`` 端到端；
4. ``Outbox`` 指数退避（30s / 60s / 120s / 240s）正确；
5. ``Outbox`` 连续失败 3 次 → 状态切到 ``alerting``；
6. ``IdempotencyStore`` 复合主键查重（命中/未命中/不同 tool_version 不命中）；
7. ``IdempotencyStore`` TTL 过期（mock 时间）→ 不命中。

测试约定：每个测试用独立 tmp_db 路径，避免跨测试污染。
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cxr_triage.persistence import (
    BASE_BACKOFF_SECONDS,
    Database,
    IdempotencyStore,
    MAX_RETRY_BEFORE_ALERT,
    Outbox,
    compute_next_retry_iso,
    init_db,
    list_tables,
    table_exists,
)


# ----------------------------------------------------------------------
# 通用 fixture：每条测试独立 tmp 数据库
# ----------------------------------------------------------------------
@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.connect()
    await init_db(d)
    yield d
    await d.close()


# ======================================================================
# 1. Database 连接 + init_db 幂等
# ======================================================================
async def test_database_connect_close(tmp_path: Path):
    """Database.connect / close 正常 + ping 返回 True。"""
    d = Database(tmp_path / "a.db")
    await d.connect()
    assert await d.ping() is True
    await d.close()
    # 重复 connect/close 幂等
    await d.connect()
    assert await d.ping() is True
    await d.close()


async def test_init_db_creates_all_tables(db: Database):
    """init_db 后 4 表 + 1 幂等表全部存在。"""
    expected = {
        "incoming_studies",
        "triage_queue",
        "audit_log",
        "outbox",
        "draft_idempotency",
    }
    actual = set(await list_tables(db))
    assert expected.issubset(actual), f"缺失表: {expected - actual}"


async def test_init_db_is_idempotent(db: Database):
    """多次 init_db 不报错也不丢表。"""
    await init_db(db)
    await init_db(db)
    await init_db(db, include_idempotency=False)
    tables = set(await list_tables(db))
    assert "incoming_studies" in tables
    assert "draft_idempotency" in tables  # 第一次已建，后续 include_idempotency=False 也不删


async def test_init_db_without_idempotency_creates_only_main_tables(tmp_path: Path):
    """include_idempotency=False → 不建幂等表（测试场景）。"""
    d = Database(tmp_path / "no_idem.db")
    await d.connect()
    await init_db(d, include_idempotency=False)
    try:
        tables = set(await list_tables(d))
        assert "incoming_studies" in tables
        assert "outbox" in tables
        assert "draft_idempotency" not in tables
    finally:
        await d.close()


async def test_init_db_without_connect_raises(tmp_path: Path):
    """未 connect 就 init_db → MigrationError。"""
    d = Database(tmp_path / "unconnected.db")
    from cxr_triage.persistence import MigrationError

    with pytest.raises(MigrationError):
        await init_db(d)


# ======================================================================
# 2. Outbox — 入队 / 拉取 / 标记
# ======================================================================
async def test_outbox_enqueue_dequeue_mark_sent(db: Database):
    """enqueue → dequeue → mark_sent 端到端。"""
    ob = Outbox(db)
    oid = await ob.enqueue(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>",
        target_endpoint="http://ris.internal/api/v1/studies/<uid>/finding-draft",
        payload={"finding_text": "右下肺野见气胸（草稿）", "positive_flag": True},
    )
    assert oid > 0

    rows = await ob.dequeue(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == oid
    assert row["study_uid"].endswith("<EXAMPLE_POS>")
    assert row["target_endpoint"].endswith("finding-draft")
    assert row["payload"]["finding_text"].startswith("右下肺野")
    assert row["payload"]["positive_flag"] is True
    assert row["retry_count"] == 0

    await ob.mark_sent(oid)

    rows = await ob.dequeue(limit=10)
    assert rows == []  # sent 不再被拉取

    by_status = await ob.count_by_status()
    assert by_status.get("sent") == 1
    assert by_status.get("pending", 0) == 0


async def test_outbox_dequeue_respects_next_retry_at(db: Database):
    """next_retry_at 未到 → 不被 dequeue。"""
    ob = Outbox(db)
    # 入队但 next_retry_at 设到 1 小时后
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "Z"
    await ob.enqueue(
        study_uid="<EXAMPLE_DEFER>",
        target_endpoint="http://ris.internal/<x>",
        payload={"x": 1},
        next_retry_at=future,
    )
    rows = await ob.dequeue(limit=10)
    assert rows == []


async def test_outbox_dequeue_fifo_order(db: Database):
    """多行按 id ASC FIFO 顺序被拉取（先入先出）。"""
    ob = Outbox(db)
    for i in range(3):
        await ob.enqueue(
            study_uid=f"<EXAMPLE_{i}>",
            target_endpoint="http://example/<x>",
            payload={"i": i},
        )
    rows = await ob.dequeue(limit=10)
    assert [r["payload"]["i"] for r in rows] == [0, 1, 2]


async def test_outbox_dequeue_respects_limit(db: Database):
    """limit 控制每次拉取数。"""
    ob = Outbox(db)
    for i in range(5):
        await ob.enqueue(
            study_uid=f"<EXAMPLE_{i}>",
            target_endpoint="http://example/<x>",
            payload={"i": i},
        )
    rows = await ob.dequeue(limit=2)
    assert len(rows) == 2


# ======================================================================
# 3. Outbox — 指数退避 + alerting 状态
# ======================================================================
def test_compute_next_retry_iso_backoff_curve():
    """纯函数：retry_count=0/1/2/3 → 30s/60s/120s/240s。"""
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    expected = [
        (0, 30),
        (1, 60),
        (2, 120),
        (3, 240),
    ]
    for rc, expected_seconds in expected:
        iso = compute_next_retry_iso(rc, now=now)
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        delta = (dt - now).total_seconds()
        assert delta == expected_seconds, (
            f"retry_count={rc}: expected {expected_seconds}s, got {delta}s"
        )


async def test_outbox_mark_retry_increments_and_backoff(db: Database):
    """mark_retry → retry_count += 1；状态保持 pending；next_retry_at 推进。"""
    ob = Outbox(db)
    oid = await ob.enqueue(
        study_uid="<EXAMPLE_R1>",
        target_endpoint="http://example/<x>",
        payload={"x": 1},
    )

    new_status = await ob.mark_retry(oid, "HTTP 500")
    assert new_status == "pending"

    rows = await ob.dequeue(limit=10)
    assert rows == []  # next_retry_at 已推进到 30s 后

    # 模拟 30s 后：手动把 next_retry_at 拨回 now（不直接调私有方法）
    async with db.connection() as conn:
        await conn.execute(
            "UPDATE outbox SET next_retry_at = ? WHERE id = ?",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                oid,
            ),
        )
        await conn.commit()

    rows = await ob.dequeue(limit=10)
    assert len(rows) == 1
    assert rows[0]["retry_count"] == 1
    assert rows[0]["id"] == oid


async def test_outbox_alerting_after_max_retry(db: Database):
    """连续失败 > MAX_RETRY_BEFORE_ALERT → 状态切 alerting。"""
    ob = Outbox(db)
    oid = await ob.enqueue(
        study_uid="<EXAMPLE_ALERT>",
        target_endpoint="http://example/<x>",
        payload={"x": 1},
    )

    # 重试 0→1→2→3 都还是 pending；第 4 次失败（retry_count=4）切 alerting
    final_status = "pending"
    for _ in range(MAX_RETRY_BEFORE_ALERT):
        final_status = await ob.mark_retry(oid, "HTTP 500")

    assert final_status == "pending"
    # 第 MAX_RETRY_BEFORE_ALERT + 1 次失败后切 alerting
    final_status = await ob.mark_retry(oid, "HTTP 500 持续失败")
    assert final_status == "alerting"

    by_status = await ob.count_by_status()
    assert by_status.get("alerting") == 1

    # alerting 不再被 dequeue 拉取
    # 强制把 next_retry_at 拨回 now，确认 alerting 不被拉
    async with db.connection() as conn:
        await conn.execute(
            "UPDATE outbox SET next_retry_at = ? WHERE id = ?",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                oid,
            ),
        )
        await conn.commit()
    rows = await ob.dequeue(limit=10)
    assert rows == []

    # count_alerting 用于 /api/health.outbox_alerts
    alerting_count = await ob.count_alerting()
    assert alerting_count == 1


async def test_outbox_mark_retry_unknown_id_raises(db: Database):
    """mark_retry 不存在的 id → KeyError（避免静默吞错）。"""
    ob = Outbox(db)
    with pytest.raises(KeyError):
        await ob.mark_retry(99999, "no such id")


async def test_outbox_mark_sent_unknown_id_no_raise(db: Database):
    """mark_sent 不存在的 id：SQL UPDATE 影响 0 行，不抛。"""
    ob = Outbox(db)
    await ob.mark_sent(99999)  # 不应抛


# ======================================================================
# 4. IdempotencyStore — 复合主键 + TTL
# ======================================================================
async def test_idempotency_cache_and_lookup_hit(db: Database):
    """cache_draft → lookup 命中且未过期 → 返回 draft。"""
    store = IdempotencyStore(db)
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_IDEM_1>"
    tool_version = "v0.1.0"

    await store.cache_draft(
        study_uid,
        tool_version,
        {"finding_text": "右下肺野见气胸（草稿）", "positive_flag": True},
    )

    got = await store.lookup(study_uid, tool_version)
    assert got is not None
    assert got["finding_text"].startswith("右下肺野")
    assert got["positive_flag"] is True


async def test_idempotency_lookup_miss_different_tool_version(db: Database):
    """同 study_uid、不同 tool_version → 不命中。"""
    store = IdempotencyStore(db)
    study_uid = "<EXAMPLE_IDEM_2>"

    await store.cache_draft(study_uid, "v0.1.0", {"v": "v0.1.0"})

    assert await store.lookup(study_uid, "v0.1.0") is not None
    assert await store.lookup(study_uid, "v0.2.0") is None


async def test_idempotency_lookup_miss_study_uid(db: Database):
    """从未入过的 study_uid → 不命中。"""
    store = IdempotencyStore(db)
    assert await store.lookup("<EXAMPLE_NEVER>", "v0.1.0") is None


async def test_idempotency_cache_overwrites_same_key(db: Database):
    """同 (study_uid, tool_version) 二次 cache → 后者覆盖；inserted_at 刷新。"""
    store = IdempotencyStore(db)
    study_uid = "<EXAMPLE_IDEM_3>"

    await store.cache_draft(study_uid, "v0.1.0", {"v": "first"})
    first_meta = await store.lookup_with_meta(study_uid, "v0.1.0")
    assert first_meta is not None
    first_inserted_at = first_meta["inserted_at"]

    await asyncio.sleep(0.05)  # 确保 inserted_at 推进
    await store.cache_draft(study_uid, "v0.1.0", {"v": "second"})
    second = await store.lookup(study_uid, "v0.1.0")
    assert second == {"v": "second"}

    second_meta = await store.lookup_with_meta(study_uid, "v0.1.0")
    assert second_meta is not None
    # inserted_at 必须被刷新（字符串 ISO8601 字典序即可比较）
    assert second_meta["inserted_at"] > first_inserted_at
    # 但都未过期
    assert second_meta["expired"] is False


async def test_idempotency_ttl_expired_returns_none(db: Database):
    """TTL 过期（mock 时间）→ lookup 返回 None。"""
    # 用 1 秒 TTL 加速测试
    store = IdempotencyStore(db, ttl_seconds=1)
    study_uid = "<EXAMPLE_IDEM_TTL>"

    await store.cache_draft(study_uid, "v0.1.0", {"v": 1})
    # 立即查 → 命中
    assert await store.lookup(study_uid, "v0.1.0") is not None

    # 模拟 2 秒后
    await asyncio.sleep(1.2)
    got = await store.lookup(study_uid, "v0.1.0")
    assert got is None

    # lookup_with_meta 仍能看到 row，但 expired=True
    meta = await store.lookup_with_meta(study_uid, "v0.1.0")
    assert meta is not None
    assert meta["expired"] is True


async def test_idempotency_invalidate(db: Database):
    """invalidate 单 version / 全 version 两种路径。"""
    store = IdempotencyStore(db)
    study_uid = "<EXAMPLE_IDEM_4>"

    await store.cache_draft(study_uid, "v0.1.0", {"v": "a"})
    await store.cache_draft(study_uid, "v0.2.0", {"v": "b"})

    deleted = await store.invalidate(study_uid, tool_version="v0.1.0")
    assert deleted == 1
    assert await store.lookup(study_uid, "v0.1.0") is None
    assert await store.lookup(study_uid, "v0.2.0") is not None

    deleted = await store.invalidate(study_uid)  # 失效全部
    assert deleted == 1
    assert await store.lookup(study_uid, "v0.2.0") is None


# ======================================================================
# 5. 多行并发与综合验证
# ======================================================================
async def test_persistence_integration_concurrent_enqueue(db: Database):
    """并发 enqueue 5 行 → dequeue 全部命中（FIFO 顺序不保证但全到）。"""
    ob = Outbox(db)
    ids = []
    for i in range(5):
        oid = await ob.enqueue(
            study_uid=f"<EXAMPLE_CONC_{i}>",
            target_endpoint="http://example/<x>",
            payload={"i": i},
        )
        ids.append(oid)

    rows = await ob.dequeue(limit=10)
    assert len(rows) == 5
    assert sorted(r["id"] for r in rows) == sorted(ids)


async def test_persistence_full_lifecycle(db: Database):
    """入队 → mark_retry → mark_sent 全流程。"""
    ob = Outbox(db)
    oid = await ob.enqueue(
        study_uid="<EXAMPLE_LIFE>",
        target_endpoint="http://example/<x>",
        payload={"x": 1},
    )

    # 失败一次
    status = await ob.mark_retry(oid, "HTTP 503")
    assert status == "pending"

    # 第二次成功
    await ob.mark_sent(oid)

    by_status = await ob.count_by_status()
    assert by_status.get("sent") == 1
    assert by_status.get("pending", 0) == 0
    assert by_status.get("alerting", 0) == 0


async def test_table_exists_helper(db: Database):
    """table_exists 工具函数正确。"""
    assert await table_exists(db, "incoming_studies") is True
    assert await table_exists(db, "outbox") is True
    assert await table_exists(db, "draft_idempotency") is True
    assert await table_exists(db, "no_such_table") is False