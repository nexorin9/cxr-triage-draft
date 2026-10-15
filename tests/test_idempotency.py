"""tests/test_idempotency.py — state 层幂等缓存（StateIdempotencyStore）。

覆盖（CLAUDE.md Phase 4 验收 · Task 8）：

1. **同 study_uid + tool_version 二次查 → 命中复用**：cache → lookup
   拿到原 draft；二次 cache 同 key → 覆盖。
2. **不同 tool_version 二次查 → 不命中**：同 study_uid 不同版本
   视为不同草稿，命中不到。
3. **TTL 过期（mock 时间）→ 不命中**：用 1 秒 TTL + asyncio.sleep
   验证过期语义；``lookup_with_meta`` 仍能返回 ``expired=True``。
4. **state 层 default tool_version**：不传 tool_version 时默认走
   当前产品 ``__version__``；持久化层不传会报 KeyError。
5. **invalidate_study**：一次失效该 Study 全部 tool_version；驳回
   场景下用于清空旧草稿。
6. **驳回业务场景端到端**：cache → invalidate_study → lookup
   返回 None；再 cache 新草稿 → lookup 命中。
7. **JSON 序列化兼容**：draft 含中文所见 + 嵌套结构（finding_text /
   finding_struct / grounding_boxes）原样存原样取。
8. **state 层包装 ≠ 持久化层**：TTL 与 default tool_version 语义
   与持久化层保持一致（同一份 SQLite 表）。

测试约定：每条测试用独立 tmp_db 路径，避免跨测试污染；TTL 测试用
1 秒 TTL + ``asyncio.sleep(1.2)`` 加速验证。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from cxr_triage import __version__
from cxr_triage.persistence import Database, init_db
from cxr_triage.state import (
    DEFAULT_TOOL_VERSION,
    STATE_IDEMPOTENCY_TTL_SECONDS,
    StateIdempotencyStore,
)


# ----------------------------------------------------------------------
# 通用 fixture：每条测试独立 tmp 数据库（已建幂等表）
# ----------------------------------------------------------------------
@pytest.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "state_idem.db")
    await d.connect()
    await init_db(d)  # 默认 include_idempotency=True（建 draft_idempotency）
    yield d
    await d.close()


# ----------------------------------------------------------------------
# fixture：典型 CXR draft（含中文 + 嵌套结构）
# ----------------------------------------------------------------------
@pytest.fixture
def sample_draft() -> dict[str, Any]:
    return {
        "finding_text": "右下肺野见少量气胸征象（草稿 / 未经医师确认），余肺野清晰。",
        "finding_struct": {
            "pathologies": [
                {"name": "Pneumothorax", "prob": 0.78},
                {"name": "Infiltration", "prob": 0.12},
            ],
            "key_qa": [
                {"q": "心影大小", "a": "心影未见明显增大"},
                {"q": "纵隔居中否", "a": "纵隔居中"},
            ],
        },
        "grounding_boxes": [
            {"x": 0.62, "y": 0.31, "w": 0.18, "h": 0.22, "label": "Pneumothorax", "prob": 0.78},
        ],
        "positive_flag": True,
        "draft_ts": "2026-10-13T05:30:00.000000Z",
    }


# ======================================================================
# 1. state 层常量与默认值
# ======================================================================
def test_default_tool_version_matches_package_version():
    """DEFAULT_TOOL_VERSION == __version__（避免误用别版本草稿）。"""
    assert DEFAULT_TOOL_VERSION == __version__


def test_state_ttl_is_5_minutes():
    """STATE_IDEMPOTENCY_TTL_SECONDS = 300（spec.md §5 维度 5 硬约束）。"""
    assert STATE_IDEMPOTENCY_TTL_SECONDS == 5 * 60


def test_store_default_ttl_is_5_minutes(db: Database):
    """StateIdempotencyStore 默认 TTL = 5 分钟。"""
    store = StateIdempotencyStore(db)
    assert store.ttl_seconds == STATE_IDEMPOTENCY_TTL_SECONDS == 300


def test_store_uses_custom_ttl(db: Database):
    """测试可注入小 TTL（不破坏默认）。"""
    store = StateIdempotencyStore(db, ttl_seconds=1)
    assert store.ttl_seconds == 1


# ======================================================================
# 2. 命中复用：同一 study_uid + tool_version 二次查
# ======================================================================
async def test_cache_then_lookup_returns_same_draft(
    db: Database, sample_draft: dict[str, Any]
):
    """cache_draft → lookup 拿到原 draft。"""
    store = StateIdempotencyStore(db)
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_STATE_IDEM_1>"

    await store.cache_draft(study_uid, sample_draft)
    got = await store.lookup(study_uid)
    assert got is not None
    assert got["finding_text"].startswith("右下肺野")
    assert got["positive_flag"] is True
    # 中文所见与嵌套结构原样存原样取
    assert got["finding_struct"]["pathologies"][0]["name"] == "Pneumothorax"
    assert got["grounding_boxes"][0]["label"] == "Pneumothorax"


async def test_cache_overwrites_same_key_with_later_draft(db: Database):
    """同 (study_uid, tool_version) 二次 cache → 后者覆盖。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_STATE_IDEM_OVERWRITE>"

    await store.cache_draft(study_uid, {"v": "first", "positive_flag": False})
    await store.cache_draft(study_uid, {"v": "second", "positive_flag": True})

    got = await store.lookup(study_uid)
    assert got == {"v": "second", "positive_flag": True}


async def test_cache_overwrites_refreshes_inserted_at(db: Database):
    """同 key 二次 cache → inserted_at 刷新（避免 TTL 错误判定过期）。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_STATE_IDEM_REFRESH>"

    await store.cache_draft(study_uid, {"v": 1})
    first_meta = await store.lookup_with_meta(study_uid)
    assert first_meta is not None
    first_inserted_at = first_meta["inserted_at"]

    await asyncio.sleep(0.05)  # 确保 inserted_at 推进
    await store.cache_draft(study_uid, {"v": 2})
    second_meta = await store.lookup_with_meta(study_uid)
    assert second_meta is not None
    assert second_meta["inserted_at"] > first_inserted_at
    assert second_meta["expired"] is False


# ======================================================================
# 3. 不同 tool_version 不命中
# ======================================================================
async def test_lookup_miss_different_tool_version(db: Database):
    """同 study_uid、不同 tool_version → 不命中。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_STATE_IDEM_VERSION>"

    await store.cache_draft(study_uid, {"v": "a"}, tool_version="v0.1.0")

    # 同版本命中
    assert await store.lookup(study_uid, tool_version="v0.1.0") is not None
    # 不同版本不命中
    assert await store.lookup(study_uid, tool_version="v0.2.0") is None


async def test_lookup_miss_never_inserted_study_uid(db: Database):
    """从未入过的 study_uid → 不命中。"""
    store = StateIdempotencyStore(db)
    assert await store.lookup("<EXAMPLE_NEVER_INSERTED>") is None


async def test_default_tool_version_is_used_when_omitted(db: Database):
    """不传 tool_version → 用 DEFAULT_TOOL_VERSION（== __version__）。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_DEFAULT_VERSION>"

    await store.cache_draft(study_uid, {"v": "default-version"})
    # 不传 tool_version 仍能命中（说明默认版本号生效）
    assert await store.lookup(study_uid) is not None
    # 显式用 DEFAULT_TOOL_VERSION 也能命中
    assert await store.lookup(study_uid, tool_version=DEFAULT_TOOL_VERSION) is not None


# ======================================================================
# 4. TTL 过期（mock 时间 → asyncio.sleep）
# ======================================================================
async def test_ttl_expired_returns_none(db: Database):
    """TTL 过期（mock 时间：asyncio.sleep(1.2) + 1 秒 TTL）→ lookup 返回 None。"""
    store = StateIdempotencyStore(db, ttl_seconds=1)
    study_uid = "<EXAMPLE_STATE_IDEM_TTL>"

    await store.cache_draft(study_uid, {"v": 1})
    # 立即查 → 命中
    assert await store.lookup(study_uid) is not None

    # 等过期
    await asyncio.sleep(1.2)
    got = await store.lookup(study_uid)
    assert got is None


async def test_lookup_with_meta_returns_expired_true_after_ttl(db: Database):
    """过期后 lookup_with_meta 仍能返回，但 expired=True。"""
    store = StateIdempotencyStore(db, ttl_seconds=1)
    study_uid = "<EXAMPLE_STATE_IDEM_TTL_META>"

    await store.cache_draft(study_uid, {"v": 1})
    await asyncio.sleep(1.2)
    meta = await store.lookup_with_meta(study_uid)
    assert meta is not None
    assert meta["expired"] is True
    assert meta["age_seconds"] >= 1.2
    assert meta["draft"] == {"v": 1}


# ======================================================================
# 5. invalidate_study — 驳回场景
# ======================================================================
async def test_invalidate_study_deletes_all_versions(db: Database):
    """invalidate_study → 一次失效该 Study 全部 tool_version。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_STATE_IDEM_INVALIDATE_ALL>"

    await store.cache_draft(study_uid, {"v": "a"}, tool_version="v0.1.0")
    await store.cache_draft(study_uid, {"v": "b"}, tool_version="v0.2.0")
    await store.cache_draft(study_uid, {"v": "c"}, tool_version="v0.3.0")

    deleted = await store.invalidate_study(study_uid)
    assert deleted == 3
    assert await store.lookup(study_uid, tool_version="v0.1.0") is None
    assert await store.lookup(study_uid, tool_version="v0.2.0") is None
    assert await store.lookup(study_uid, tool_version="v0.3.0") is None


async def test_invalidate_study_only_targets_one_study(db: Database):
    """invalidate_study 不影响其它 Study。"""
    store = StateIdempotencyStore(db)
    study_a = "<EXAMPLE_STUDY_A>"
    study_b = "<EXAMPLE_STUDY_B>"

    await store.cache_draft(study_a, {"v": "a"})
    await store.cache_draft(study_b, {"v": "b"})

    deleted = await store.invalidate_study(study_a)
    assert deleted == 1
    assert await store.lookup(study_a) is None
    assert await store.lookup(study_b) is not None  # 未受影响


async def test_invalidate_single_version_keeps_others(db: Database):
    """invalidate(study_uid, tool_version) 只失效单 version。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_SINGLE_INVALIDATE>"

    await store.cache_draft(study_uid, {"v": "old"}, tool_version="v0.1.0")
    await store.cache_draft(study_uid, {"v": "new"}, tool_version="v0.2.0")

    deleted = await store.invalidate(study_uid, tool_version="v0.1.0")
    assert deleted == 1
    assert await store.lookup(study_uid, tool_version="v0.1.0") is None
    assert await store.lookup(study_uid, tool_version="v0.2.0") is not None


async def test_invalidate_default_tool_version(db: Database):
    """invalidate 不传 tool_version → 用 DEFAULT_TOOL_VERSION。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_DEFAULT_INVALIDATE>"

    await store.cache_draft(study_uid, {"v": 1})  # 默认版本
    await store.cache_draft(study_uid, {"v": 2}, tool_version="other")

    deleted = await store.invalidate(study_uid)  # 默认版本失效
    assert deleted == 1
    assert await store.lookup(study_uid) is None
    assert await store.lookup(study_uid, tool_version="other") is not None


# ======================================================================
# 6. 驳回业务场景端到端
# ======================================================================
async def test_reject_workflow_end_to_end(
    db: Database, sample_draft: dict[str, Any]
):
    """驳回到 draft_pending → 清旧草稿 → 重新跑 → 新草稿命中。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_REJECT_SCENARIO>"

    # 1) jr 看到阳性 → cache 草稿
    await store.cache_draft(study_uid, sample_draft)
    assert await store.lookup(study_uid) is not None

    # 2) reviewer 驳回 → 走 CxrStateMachine.reject()（实际不在本测试）
    #    同步清空旧草稿
    await store.invalidate_study(study_uid)
    assert await store.lookup(study_uid) is None

    # 3) TriageTrigger 重新跑 5 件工具 → 写新草稿
    new_draft = {
        **sample_draft,
        "finding_text": "驳回后重审：右下肺野气胸征象改善（草稿）。",
        "positive_flag": False,
    }
    await store.cache_draft(study_uid, new_draft)

    # 4) 新草稿命中且是「驳回后版本」
    got = await store.lookup(study_uid)
    assert got is not None
    assert "驳回后重审" in got["finding_text"]
    assert got["positive_flag"] is False


# ======================================================================
# 7. JSON 序列化兼容（含中文 + 嵌套结构）
# ======================================================================
async def test_complex_nested_draft_round_trip(
    db: Database, sample_draft: dict[str, Any]
):
    """含中文所见 + 嵌套结构的 draft 序列化往返一致。"""
    store = StateIdempotencyStore(db)
    study_uid = "<EXAMPLE_COMPLEX_DRAFT>"

    await store.cache_draft(study_uid, sample_draft)
    got = await store.lookup(study_uid)

    # 中文所见原样
    assert got["finding_text"] == sample_draft["finding_text"]
    # 嵌套 pathologies
    assert got["finding_struct"]["pathologies"] == sample_draft["finding_struct"]["pathologies"]
    # 嵌套 key_qa（含两个问答）
    assert got["finding_struct"]["key_qa"] == sample_draft["finding_struct"]["key_qa"]
    # grounding_boxes 坐标 [0,1] 区间
    for box in got["grounding_boxes"]:
        assert 0.0 <= box["x"] <= 1.0
        assert 0.0 <= box["y"] <= 1.0
        assert 0.0 <= box["w"] <= 1.0
        assert 0.0 <= box["h"] <= 1.0


# ======================================================================
# 8. state 层与持久化层写入同一表（共享 SQLite）
# ======================================================================
async def test_state_and_persistence_share_underlying_table(
    db: Database, sample_draft: dict[str, Any]
):
    """StateIdempotencyStore 写入 → persistence.IdempotencyStore.lookup
    能命中（说明两者共享同一 draft_idempotency 表）。"""
    from cxr_triage.persistence import IdempotencyStore

    state_store = StateIdempotencyStore(db)
    persist_store = IdempotencyStore(db)

    study_uid = "<EXAMPLE_SHARED_TABLE>"
    await state_store.cache_draft(study_uid, sample_draft)

    # 用持久化层 lookup → 仍能命中
    got = await persist_store.lookup(study_uid, DEFAULT_TOOL_VERSION)
    assert got is not None
    assert got["finding_text"] == sample_draft["finding_text"]
