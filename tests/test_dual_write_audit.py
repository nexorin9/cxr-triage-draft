"""tests/test_dual_write_audit.py — AuditLogger 双写对账（RIS + 体检主检）。

覆盖（CLAUDE.md Phase 4 验收 · Task 26 spec.md §5 维度 6 飞检双写对账）：

1. happy path — RIS 200 + 主检 200 → audit_log 两条 row（ris.dual_write +
   healthcheck.dual_write）；ts 统一；payload.status 均为 "ok"；
2. 部分失败 — RIS 500 + 主检 200 → audit_log 两条（一 ris.dual_write_failed
   + 一 healthcheck.dual_write）；payload.status 分别 fail / ok；
3. 双失败 — RIS 500 + 主检 500 → 两条均 *_failed；
4. ts 统一 — 同一 ``write_dual_write`` 调用两行 ts 字面相同；
5. ts 自定义 — 入参 ts 同时进入两条 audit_log；
6. 默认 actor_role = "sr"（终审签发）；
7. 自定义 actor_role（reviewer / jr / system）合法；非法值拒绝；
8. ris_status / healthcheck_status 白名单校验（"ok" / "fail"），其他拒绝；
9. study_uid 必填，空串 / None / 非字符串拒绝；
10. ris_extra / healthcheck_extra 可选 dict 合并到对应行 payload；
11. payload 必含 kind = "dual_write" + pair_ts == 自身 ts；
12. 并发 5 组双写 → audit_log 累计 10 行；每对 ts 一致；
13. 入参校验 — 无 audit_log 行被错误地写出（先校验失败时不写）；
14. event_type 命名 ris.dual_write / ris.dual_write_failed /
    healthcheck.dual_write / healthcheck.dual_write_failed 锁定。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cxr_triage.persistence import Database, init_db
from cxr_triage.writers.audit import (
    DUAL_WRITE_DEFAULT_ACTOR_ROLE,
    DUAL_WRITE_EVENT_HEALTHCHECK_FAILED,
    DUAL_WRITE_EVENT_HEALTHCHECK_OK,
    DUAL_WRITE_EVENT_RIS_FAILED,
    DUAL_WRITE_EVENT_RIS_OK,
    DUAL_WRITE_VALID_STATUSES,
    VALID_AUDIT_ACTOR_ROLES,
    AuditLogger,
    AuditLoggerError,
)


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "dual_write_test.db"


@pytest.fixture
def tmp_tool_calls_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tool_calls"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
async def db(tmp_db_path: Path) -> Database:
    database = Database(str(tmp_db_path))
    await database.connect()
    await init_db(database)
    yield database
    await database.close()


@pytest.fixture
async def logger(db: Database, tmp_tool_calls_dir: Path) -> AuditLogger:
    return AuditLogger(db=db, tool_calls_dir=tmp_tool_calls_dir)


async def _fetch_audit_rows(database: Database, study_uid: str) -> list[dict]:
    """从 audit_log 拉某 study_uid 全行（异步入口）。"""
    async with database.connection() as conn:
        cur = await conn.execute(
            "SELECT id, actor_role, ts, study_uid, event_type, payload_json "
            "FROM audit_log WHERE study_uid = ? ORDER BY id ASC",
            (study_uid,),
        )
        rows = await cur.fetchall()
        await cur.close()
    out: list[dict] = []
    for row in rows:
        try:
            payload_obj = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            payload_obj = row["payload_json"]
        out.append(
            {
                "id": row["id"],
                "actor_role": row["actor_role"],
                "ts": row["ts"],
                "study_uid": row["study_uid"],
                "event_type": row["event_type"],
                "payload": payload_obj,
            }
        )
    return out


# =====================================================================
# 常量层 / 模块契约
# =====================================================================
def test_dual_write_statuses_are_exactly_ok_and_fail() -> None:
    assert DUAL_WRITE_VALID_STATUSES == frozenset({"ok", "fail"})


def test_dual_write_event_types_are_locked() -> None:
    assert DUAL_WRITE_EVENT_RIS_OK == "ris.dual_write"
    assert DUAL_WRITE_EVENT_RIS_FAILED == "ris.dual_write_failed"
    assert DUAL_WRITE_EVENT_HEALTHCHECK_OK == "healthcheck.dual_write"
    assert DUAL_WRITE_EVENT_HEALTHCHECK_FAILED == "healthcheck.dual_write_failed"


def test_dual_write_default_actor_role_is_sr() -> None:
    assert DUAL_WRITE_DEFAULT_ACTOR_ROLE == "sr"
    assert "sr" in VALID_AUDIT_ACTOR_ROLES


# =====================================================================
# happy path — RIS 200 + 主检 200
# =====================================================================
@pytest.mark.asyncio
async def test_happy_path_both_ok_writes_two_rows_with_unified_ts(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
    ris_id, hc_id = await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="ok",
    )

    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2, "RIS + 体检主检 双写应落 2 行 audit_log"

    # row1 = RIS
    ris_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_RIS_OK)
    assert ris_row["actor_role"] == "sr"
    assert ris_row["payload"]["kind"] == "dual_write"
    assert ris_row["payload"]["status"] == "ok"
    assert ris_row["payload"]["target"] == "ris"
    assert ris_row["payload"]["pair_ts"] == ris_row["ts"]

    # row2 = 体检主检
    hc_row = next(
        r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_HEALTHCHECK_OK
    )
    assert hc_row["actor_role"] == "sr"
    assert hc_row["payload"]["kind"] == "dual_write"
    assert hc_row["payload"]["status"] == "ok"
    assert hc_row["payload"]["target"] == "healthcheck"
    assert hc_row["payload"]["pair_ts"] == hc_row["ts"]

    # 双写必须 ts 字面相同（统一 UTC ISO8601）
    assert ris_row["ts"] == hc_row["ts"], "RIS 与体检主检两行 ts 必须统一"

    # 返回的 id 对应两条 row
    assert ris_id == ris_row["id"]
    assert hc_id == hc_row["id"]


# =====================================================================
# 部分失败 — RIS 500 + 主检 200
# =====================================================================
@pytest.mark.asyncio
async def test_partial_failure_ris_fail_healthcheck_ok_writes_two_rows(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>"
    ris_id, hc_id = await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="fail",
        healthcheck_status="ok",
        ris_extra={"http_status_code": 500},
    )

    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2, "部分失败也应落 2 行 audit_log（飞检对账需求）"

    ris_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_RIS_FAILED)
    hc_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_HEALTHCHECK_OK)

    # RIS 失败
    assert ris_row["payload"]["status"] == "fail"
    assert ris_row["payload"]["target"] == "ris"
    assert ris_row["payload"]["pair_ts"] == ris_row["ts"]
    assert ris_row["payload"]["http_status_code"] == 500  # extra 子键合并

    # 体检主检成功
    assert hc_row["payload"]["status"] == "ok"
    assert hc_row["payload"]["target"] == "healthcheck"
    assert hc_row["payload"]["pair_ts"] == hc_row["ts"]

    assert ris_row["ts"] == hc_row["ts"]
    assert ris_id == ris_row["id"]
    assert hc_id == hc_row["id"]


# =====================================================================
# 双失败
# =====================================================================
@pytest.mark.asyncio
async def test_dual_failure_both_fail_writes_two_failed_rows(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_BORDER>"
    ris_id, hc_id = await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="fail",
        healthcheck_status="fail",
        ris_extra={"http_status_code": 503},
        healthcheck_extra={"http_status_code": 500, "outbox_id": 99},
    )

    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    ris_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_RIS_FAILED)
    hc_row = next(
        r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_HEALTHCHECK_FAILED
    )
    assert ris_row["payload"]["status"] == "fail"
    assert ris_row["payload"]["http_status_code"] == 503
    assert hc_row["payload"]["status"] == "fail"
    assert hc_row["payload"]["http_status_code"] == 500
    assert hc_row["payload"]["outbox_id"] == 99
    assert ris_row["ts"] == hc_row["ts"]


# =====================================================================
# ts 自定义
# =====================================================================
@pytest.mark.asyncio
async def test_custom_ts_propagates_to_both_rows(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_CUSTOM_TS>"
    fixed_ts = "2026-10-13T07:23:11.123Z"
    ris_id, hc_id = await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="ok",
        ts=fixed_ts,
    )

    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    for row in rows:
        assert row["ts"] == fixed_ts, (
            f"自定义 ts 必须原样落 audit_log；got {row['ts']!r}"
        )
        assert row["payload"]["pair_ts"] == fixed_ts


# =====================================================================
# 默认 actor_role + 自定义 actor_role 合法性
# =====================================================================
@pytest.mark.asyncio
async def test_default_actor_role_is_sr(logger: AuditLogger, db: Database) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_DEFAULT_ACTOR>"
    await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="ok",
    )
    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    for row in rows:
        assert row["actor_role"] == "sr"


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_role", ["reviewer", "jr", "system"])
async def test_custom_actor_roles_are_accepted(
    logger: AuditLogger, db: Database, actor_role: str
) -> None:
    study_uid = f"1.2.826.0.1.3680043.8.498.<EXAMPLE_{actor_role.upper()}>"
    ris_id, hc_id = await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="fail",
        actor_role=actor_role,
    )
    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    for row in rows:
        assert row["actor_role"] == actor_role


@pytest.mark.asyncio
async def test_invalid_actor_role_rejected_no_rows_written(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_BAD_ACTOR>"
    with pytest.raises(AuditLoggerError, match="无效 actor_role"):
        await logger.write_dual_write(
            study_uid=study_uid,
            ris_status="ok",
            healthcheck_status="ok",
            actor_role="admin",
        )
    assert await _fetch_audit_rows(db, study_uid) == [], (
        "参数校验失败时不得写入 audit_log"
    )


# =====================================================================
# status 白名单
# =====================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ris_status,hc_status",
    [
        ("timeout", "ok"),  # 非法 ris_status
        ("ok", "timeout"),  # 非法 healthcheck_status
        ("200", "ok"),  # 数字字符串不规范
        ("", "ok"),  # 空串
        ("ok", ""),  # 空串
        (None, "ok"),  # None 拒绝
    ],
)
async def test_invalid_status_rejected_no_rows(
    logger: AuditLogger, db: Database, ris_status: object, hc_status: object
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_BAD_STATUS>"
    with pytest.raises(AuditLoggerError):
        await logger.write_dual_write(
            study_uid=study_uid,
            ris_status=ris_status,  # type: ignore[arg-type]
            healthcheck_status=hc_status,  # type: ignore[arg-type]
        )
    assert await _fetch_audit_rows(db, study_uid) == []


# =====================================================================
# study_uid 校验
# =====================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_uid", ["", None, 123, " ", "\x00uid-with-nul"]
)
async def test_study_uid_validation_rejects_bad_inputs(
    logger: AuditLogger, db: Database, bad_uid: object
) -> None:
    with pytest.raises(AuditLoggerError):
        await logger.write_dual_write(
            study_uid=bad_uid,  # type: ignore[arg-type]
            ris_status="ok",
            healthcheck_status="ok",
        )


# =====================================================================
# extra 字段合并
# =====================================================================
@pytest.mark.asyncio
async def test_ris_extra_merges_into_ris_only_payload(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_RIS_EXTRA>"
    await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="ok",
        ris_extra={"http_status_code": 200, "payload_size_bytes": 1542},
    )
    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    ris_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_RIS_OK)
    hc_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_HEALTHCHECK_OK)
    assert ris_row["payload"]["http_status_code"] == 200
    assert ris_row["payload"]["payload_size_bytes"] == 1542
    # 主检行不应被 RIS 的 extra 污染
    assert "http_status_code" not in hc_row["payload"]
    assert "payload_size_bytes" not in hc_row["payload"]


@pytest.mark.asyncio
async def test_healthcheck_extra_merges_into_healthcheck_only_payload(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_HC_EXTRA>"
    await logger.write_dual_write(
        study_uid=study_uid,
        ris_status="fail",
        healthcheck_status="ok",
        healthcheck_extra={"outbox_id": 77, "http_status_code": 201},
    )
    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    ris_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_RIS_FAILED)
    hc_row = next(r for r in rows if r["event_type"] == DUAL_WRITE_EVENT_HEALTHCHECK_OK)
    assert hc_row["payload"]["outbox_id"] == 77
    assert hc_row["payload"]["http_status_code"] == 201
    assert "outbox_id" not in ris_row["payload"]


# =====================================================================
# 并发 — N 组双写全部落库
# =====================================================================
@pytest.mark.asyncio
async def test_concurrent_dual_writes_all_land_in_audit_log(
    logger: AuditLogger, db: Database
) -> None:
    n_pairs = 5
    study_uids = [
        f"1.2.826.0.1.3680043.8.498.<EXAMPLE_CONCURRENT_{i}>"
        for i in range(n_pairs)
    ]
    statuses_cycle = [
        ("ok", "ok"),
        ("fail", "ok"),
        ("ok", "fail"),
        ("fail", "fail"),
        ("ok", "ok"),
    ]

    results = await asyncio.gather(
        *(
            logger.write_dual_write(
                study_uid=study_uids[i],
                ris_status=statuses_cycle[i][0],
                healthcheck_status=statuses_cycle[i][1],
            )
            for i in range(n_pairs)
        )
    )

    # 检查每个 study 都落了 2 行 audit_log
    for idx, uid in enumerate(study_uids):
        rows = await _fetch_audit_rows(db, uid)
        assert len(rows) == 2, f"study_idx={idx} 应落 2 行 audit_log"
        ris_row = next(
            r
            for r in rows
            if r["event_type"] in (DUAL_WRITE_EVENT_RIS_OK, DUAL_WRITE_EVENT_RIS_FAILED)
        )
        hc_row = next(
            r
            for r in rows
            if r["event_type"]
            in (DUAL_WRITE_EVENT_HEALTHCHECK_OK, DUAL_WRITE_EVENT_HEALTHCHECK_FAILED)
        )
        assert ris_row["payload"]["status"] == statuses_cycle[idx][0]
        assert hc_row["payload"]["status"] == statuses_cycle[idx][1]
        assert ris_row["ts"] == hc_row["ts"]
        ris_id, hc_id = results[idx]
        assert ris_id == ris_row["id"]
        assert hc_id == hc_row["id"]


# =====================================================================
# 多 study_uid 隔离
# =====================================================================
@pytest.mark.asyncio
async def test_multiple_studys_dont_cross_audit_rows(
    logger: AuditLogger, db: Database
) -> None:
    uid_a = "1.2.826.0.1.3680043.8.498.<EXAMPLE_A>"
    uid_b = "1.2.826.0.1.3680043.8.498.<EXAMPLE_B>"
    await logger.write_dual_write(uid_a, "ok", "ok")
    await logger.write_dual_write(uid_b, "fail", "ok")

    rows_a = await _fetch_audit_rows(db, uid_a)
    rows_b = await _fetch_audit_rows(db, uid_b)

    assert len(rows_a) == 2 and len(rows_b) == 2
    for r in rows_a:
        assert r["study_uid"] == uid_a
    for r in rows_b:
        assert r["study_uid"] == uid_b


# =====================================================================
# ts 单调推进 + 不同时刻双写不会被合并
# =====================================================================
@pytest.mark.asyncio
async def test_consecutive_dual_writes_produce_distinct_ts(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_SEQUENCE>"
    await logger.write_dual_write(study_uid, "ok", "ok")
    await asyncio.sleep(0.05)  # 确保毫秒推进
    await logger.write_dual_write(study_uid, "ok", "fail")

    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 4

    pair1_ris = rows[0]
    pair1_hc = rows[1]
    pair2_ris = rows[2]
    pair2_hc = rows[3]

    # 同一对的两条 ts 完全一致
    assert pair1_ris["ts"] == pair1_hc["ts"]
    assert pair2_ris["ts"] == pair2_hc["ts"]
    # 不同对 ts 严格递增
    assert pair1_ris["ts"] < pair2_ris["ts"]


# =====================================================================
# 字段契约
# =====================================================================
@pytest.mark.asyncio
async def test_payload_kind_and_pair_ts_required(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_KIND>"
    await logger.write_dual_write(study_uid, "ok", "ok")
    rows = await _fetch_audit_rows(db, study_uid)
    assert len(rows) == 2
    for row in rows:
        payload = row["payload"]
        assert payload["kind"] == "dual_write", (
            "payload 必含 kind='dual_write'"
        )
        assert payload["pair_ts"] == row["ts"], "pair_ts 必须 == 自身 ts"
        assert payload["target"] in ("ris", "healthcheck")
        assert payload["status"] in {"ok", "fail"}


@pytest.mark.asyncio
async def test_study_uid_index_audit_lookup_returns_both_rows(
    logger: AuditLogger, db: Database
) -> None:
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_LOOKUP>"
    await logger.write_dual_write(study_uid, "ok", "ok")

    # list_audit_for_study 已由 test_audit.py 验证；这里只要验证双写
    # 行也能用既有的 list_audit_for_study 拿出来
    audit_rows = await logger.list_audit_for_study(study_uid)
    assert len(audit_rows) == 2
    event_types = {r["event_type"] for r in audit_rows}
    assert event_types == {DUAL_WRITE_EVENT_RIS_OK, DUAL_WRITE_EVENT_HEALTHCHECK_OK}


# =====================================================================
# 不耦合业务层 / 工厂注入兼容性
# =====================================================================
@pytest.mark.asyncio
async def test_write_dual_write_accepts_audit_logger_from_factory(
    tmp_db_path: Path, tmp_tool_calls_dir: Path
) -> None:
    """build_audit_logger 工厂构造的实例应支持双写。"""
    from cxr_triage.writers.audit import build_audit_logger

    instance = build_audit_logger(
        db_path=tmp_db_path,
        tool_calls_dir=tmp_tool_calls_dir,
    )
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_FACTORY>"
    ris_id, hc_id = await instance.write_dual_write(
        study_uid=study_uid,
        ris_status="ok",
        healthcheck_status="ok",
    )
    assert ris_id > 0 and hc_id > 0
    assert ris_id != hc_id, "两条 audit_log 行的 id 必须不同（双行非一条）"
    rows = await instance.list_audit_for_study(study_uid)
    assert len(rows) == 2
