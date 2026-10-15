"""tests/test_audit.py — AuditLogger 双层审计（SQLite audit_log + JSONL）。

覆盖（CLAUDE.md Phase 4 验收 · Task 9）：

1. ``audit_log_insert`` 写入 → SQLite audit_log 行存在 + 字段完整；
2. payload 接受 dict / str / None 三种形态；
3. actor_role 白名单（jr / reviewer / sr / system）；拒绝其他值；
4. event_type 必填，空串拒绝；
5. ts 覆盖：自定义 ts 写入与读出对齐；
6. ``tool_call_record`` 写入 → JSONL 文件一行合法 JSON；
7. JSONL 字段完整（ts / study_uid / tool / model / prompt_tokens /
   completion_tokens / latency_ms / status 全部 8 个 + 可选 extra）；
8. status 白名单（ok / fail / degraded）；拒绝其他值；
9. token / latency 必须非负整数；负值拒绝；
10. 空 tool 拒绝；
11. ts 自动按日期切片（YYYY-MM-DD 文件名）；
12. 并发写 audit_log + JSONL：``asyncio.gather`` 30 次混合写，全部
    字段不丢、不交错；
13. ``list_audit_for_study`` / ``count_audit_for_study`` / 读 JSONL
    回放字段完整；
14. 损坏 JSONL 行（手工插入坏 JSON）由 ``read_tool_call_records``
    跳过而非崩溃；
15. ensure_db 懒加载：构造时不连，第一次写入时建立；close() 后
    持有的 db 可重连；
16. ``build_audit_logger`` 工厂函数可用。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cxr_triage.persistence import Database, init_db
from cxr_triage.writers.audit import (
    DEFAULT_TOOL_CALLS_DIR,
    VALID_AUDIT_ACTOR_ROLES,
    VALID_TOOL_CALL_STATUSES,
    AuditLogger,
    AuditLoggerError,
    build_audit_logger,
    utc_now_iso,
    utc_today_iso,
)


# =====================================================================
# Fixtures（async，pytest-asyncio auto 模式）
# =====================================================================
@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "audit_test.db"


@pytest.fixture
def tmp_tool_calls_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tool_calls"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
async def db(tmp_db_path: Path) -> Database:
    """预建表结构的 Database fixture（含 schema + idempotency + state）。"""
    database = Database(str(tmp_db_path))
    await database.connect()
    await init_db(database)
    yield database
    await database.close()


@pytest.fixture
async def logger(
    db: Database, tmp_tool_calls_dir: Path
) -> AuditLogger:
    return AuditLogger(db=db, tool_calls_dir=tmp_tool_calls_dir)


# =====================================================================
# 1. 构造 / 工厂 / 懒加载
# =====================================================================
def test_default_tool_calls_dir_constant() -> None:
    """DEFAULT_TOOL_CALLS_DIR 字符串常量默认 /tmp/cxr-triage/tool_calls。"""
    assert DEFAULT_TOOL_CALLS_DIR == "/tmp/cxr-triage/tool_calls"


def test_valid_statuses_constant() -> None:
    """status 白名单：ok / fail / degraded 三选一。"""
    assert VALID_TOOL_CALL_STATUSES == frozenset({"ok", "fail", "degraded"})


def test_valid_actor_roles_constant() -> None:
    """actor_role 枚举与 state.ActorRole 一致：jr / reviewer / sr / system。"""
    assert VALID_AUDIT_ACTOR_ROLES == frozenset(
        {"jr", "reviewer", "sr", "system"}
    )


def test_build_audit_logger_factory(tmp_db_path: Path, tmp_tool_calls_dir: Path) -> None:
    """工厂函数返回 AuditLogger 实例。"""
    audit = build_audit_logger(
        db_path=str(tmp_db_path),
        tool_calls_dir=str(tmp_tool_calls_dir),
    )
    assert isinstance(audit, AuditLogger)
    assert audit.tool_calls_dir == tmp_tool_calls_dir


async def test_lazy_db_creation(tmp_path: Path) -> None:
    """未传 db 时构造不连；首次 ensure_db 才连。"""
    audit = AuditLogger(
        db_path=str(tmp_path / "lazy.db"),
        tool_calls_dir=str(tmp_path / "tc"),
    )
    # 构造时不连（私有 _db 仍是 None）
    assert audit._db is None

    # 首次 ensure_db 触发连接 + 写一行
    await audit.ensure_db()
    row_id = await audit.audit_log_insert(
        actor_role="system",
        study_uid="1.2.3.lazy",
        event_type="lazy.test",
        payload={"k": "v"},
    )
    assert row_id >= 1
    await audit.close()


# =====================================================================
# 2. utc_now_iso / utc_today_iso 时间工具
# =====================================================================
def test_utc_now_iso_format() -> None:
    """utc_now_iso 输出 ``YYYY-MM-DDTHH:MM:SS.mmmZ``。"""
    s = utc_now_iso()
    assert len(s) == 24
    assert s[4] == "-" and s[7] == "-"
    assert s[10] == "T" and s[-1] == "Z"
    assert s[13] == ":" and s[16] == ":"
    assert "." in s  # 毫秒段


def test_utc_today_iso_format() -> None:
    """utc_today_iso 输出 ``YYYY-MM-DD``。"""
    s = utc_today_iso()
    assert len(s) == 10
    assert s[4] == "-" and s[7] == "-"


# =====================================================================
# 3. audit_log_insert 写 SQLite 飞检业务事件流
# =====================================================================
async def test_audit_log_insert_dict_payload(logger: AuditLogger) -> None:
    """dict payload 自动 JSON 序列化；中文所见保留。"""
    row_id = await logger.audit_log_insert(
        actor_role="jr",
        study_uid="1.2.826.0.1.EXAMPLE.POS",
        event_type="state.transition",
        payload={
            "from_state": "draft_pending",
            "to_state": "jr_confirmed",
            "finding_text": "右下肺野见少量气胸征象（草稿）。",
        },
    )
    assert row_id >= 1

    rows = await logger.list_audit_for_study("1.2.826.0.1.EXAMPLE.POS")
    assert len(rows) == 1
    r = rows[0]
    assert r["actor_role"] == "jr"
    assert r["event_type"] == "state.transition"
    assert r["payload"]["from_state"] == "draft_pending"
    assert r["payload"]["to_state"] == "jr_confirmed"
    assert "气胸" in r["payload"]["finding_text"]  # 中文保留


async def test_audit_log_insert_str_payload(logger: AuditLogger) -> None:
    """str payload 原样写（已序列化好的 JSON）。"""
    await logger.audit_log_insert(
        actor_role="system",
        study_uid="study-X",
        event_type="pacs.cstore.received",
        payload='{"raw_bytes": 1234}',
    )
    rows = await logger.list_audit_for_study("study-X")
    assert len(rows) == 1
    assert rows[0]["payload"] == {"raw_bytes": 1234}


async def test_audit_log_insert_empty_str_payload(logger: AuditLogger) -> None:
    """空 str payload → ``{}``。"""
    await logger.audit_log_insert(
        actor_role="system",
        study_uid="study-Y",
        event_type="test.empty",
        payload="",
    )
    rows = await logger.list_audit_for_study("study-Y")
    assert rows[0]["payload"] == {}


async def test_audit_log_insert_none_payload(logger: AuditLogger) -> None:
    """None payload → ``{}``。"""
    await logger.audit_log_insert(
        actor_role="system",
        study_uid="study-Z",
        event_type="test.none",
        payload=None,
    )
    rows = await logger.list_audit_for_study("study-Z")
    assert rows[0]["payload"] == {}


async def test_audit_log_insert_custom_ts(logger: AuditLogger) -> None:
    """自定义 ts 写入与读出对齐（ISO8601 字符串原样）。"""
    custom_ts = "2026-10-13T07:23:11.487Z"
    await logger.audit_log_insert(
        actor_role="sr",
        study_uid="study-ts",
        event_type="ris.written",
        ts=custom_ts,
    )
    rows = await logger.list_audit_for_study("study-ts")
    assert rows[0]["ts"] == custom_ts


# =====================================================================
# 4. audit_log_insert 校验：actor_role / event_type 拒绝非法值
# =====================================================================
async def test_audit_log_insert_invalid_actor_role(logger: AuditLogger) -> None:
    """非法 actor_role 抛 AuditLoggerError；不写入。"""
    with pytest.raises(AuditLoggerError, match="无效 actor_role"):
        await logger.audit_log_insert(
            actor_role="admin",  # 不在白名单
            study_uid="study-bad-role",
            event_type="x.test",
            payload={},
        )
    n = await logger.count_audit_for_study("study-bad-role")
    assert n == 0


async def test_audit_log_insert_empty_event_type(logger: AuditLogger) -> None:
    """空 event_type 抛 AuditLoggerError。"""
    with pytest.raises(AuditLoggerError, match="event_type"):
        await logger.audit_log_insert(
            actor_role="system",
            study_uid="study-empty-evt",
            event_type="",
            payload={},
        )


async def test_audit_log_insert_unsupported_payload_type(
    logger: AuditLogger,
) -> None:
    """非 dict/str/None 类型 payload 拒绝。"""
    with pytest.raises(AuditLoggerError, match="payload 类型不支持"):
        await logger.audit_log_insert(
            actor_role="system",
            study_uid="study-bad-payload",
            event_type="x.test",
            payload=12345,
        )


# =====================================================================
# 5. tool_call_record 写 JSONL（模型调用元数据）
# =====================================================================
async def test_tool_call_record_happy_path(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """happy path：1 行 JSON 写入 tool_calls_<YYYY-MM-DD>.jsonl。"""
    path = await logger.tool_call_record(
        tool="cxr_keyqa",
        model="gpt-4o-2024-08",
        prompt_tokens=384,
        completion_tokens=128,
        latency_ms=1820,
        status="ok",
        study_uid="study-tc-1",
    )
    assert path.exists()
    assert path.parent == tmp_tool_calls_dir
    assert path.name.startswith("tool_calls_")
    assert path.suffix == ".jsonl"

    records = AuditLogger.read_tool_call_records(path)
    assert len(records) == 1
    r = records[0]
    assert r["tool"] == "cxr_keyqa"
    assert r["model"] == "gpt-4o-2024-08"
    assert r["prompt_tokens"] == 384
    assert r["completion_tokens"] == 128
    assert r["latency_ms"] == 1820
    assert r["status"] == "ok"
    assert r["study_uid"] == "study-tc-1"
    assert "ts" in r
    assert r["ts"].endswith("Z")  # ISO8601 + Z


async def test_tool_call_record_extra_field(
    logger: AuditLogger,
) -> None:
    """extra 字段落到 JSONL ``extra`` 子键。"""
    path = await logger.tool_call_record(
        tool="classifier",
        model="torchxrayvision-densenet121",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=85,
        status="ok",
        study_uid="study-extra",
        extra={"source": "local", "device": "cuda:0"},
    )
    records = AuditLogger.read_tool_call_records(path)
    assert records[0]["extra"] == {"source": "local", "device": "cuda:0"}


async def test_tool_call_record_daily_rotation(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """不同 UTC 日期落在不同 JSONL 文件。"""
    await logger.tool_call_record(
        tool="t1", model="m1", prompt_tokens=1, completion_tokens=1,
        latency_ms=10, status="ok", study_uid="s1",
        ts="2026-10-12T08:00:00.000Z",
    )
    await logger.tool_call_record(
        tool="t2", model="m1", prompt_tokens=2, completion_tokens=2,
        latency_ms=20, status="ok", study_uid="s2",
        ts="2026-10-13T08:00:00.000Z",
    )
    files = sorted(tmp_tool_calls_dir.glob("tool_calls_*.jsonl"))
    assert [p.name for p in files] == [
        "tool_calls_2026-10-12.jsonl",
        "tool_calls_2026-10-13.jsonl",
    ]
    r1 = AuditLogger.read_tool_call_records(files[0])
    r2 = AuditLogger.read_tool_call_records(files[1])
    assert r1[0]["tool"] == "t1"
    assert r2[0]["tool"] == "t2"


async def test_tool_call_record_empty_tool_rejected(logger: AuditLogger) -> None:
    """空 tool 拒绝。"""
    with pytest.raises(AuditLoggerError, match="tool"):
        await logger.tool_call_record(
            tool="",
            model="m",
            prompt_tokens=0, completion_tokens=0,
            latency_ms=0,
            status="ok",
        )


async def test_tool_call_record_invalid_status_rejected(logger: AuditLogger) -> None:
    """非法 status 拒绝。"""
    with pytest.raises(AuditLoggerError, match="无效 status"):
        await logger.tool_call_record(
            tool="t", model="m",
            prompt_tokens=0, completion_tokens=0,
            latency_ms=0,
            status="timeout",  # 不在 ok/fail/degraded
        )


async def test_tool_call_record_negative_tokens_rejected(
    logger: AuditLogger,
) -> None:
    """负数 token 拒绝。"""
    with pytest.raises(AuditLoggerError, match="prompt_tokens"):
        await logger.tool_call_record(
            tool="t", model="m",
            prompt_tokens=-1, completion_tokens=0,
            latency_ms=0,
            status="ok",
        )


async def test_tool_call_record_negative_latency_rejected(
    logger: AuditLogger,
) -> None:
    """负数 latency_ms 拒绝。"""
    with pytest.raises(AuditLoggerError, match="latency_ms"):
        await logger.tool_call_record(
            tool="t", model="m",
            prompt_tokens=0, completion_tokens=0,
            latency_ms=-1,
            status="ok",
        )


# =====================================================================
# 6. 并发写：audit_log + JSONL 同步落盘不丢不交错
# =====================================================================
async def test_concurrent_audit_log_writes_no_loss(logger: AuditLogger) -> None:
    """并发 30 条 audit_log_insert 全部落库，无丢失。"""
    study_uid = "study-concurrent-audit"

    async def _insert(i: int) -> None:
        await logger.audit_log_insert(
            actor_role="jr",
            study_uid=study_uid,
            event_type="concurrent.test",
            payload={"i": i},
        )

    await asyncio.gather(*[_insert(i) for i in range(30)])
    n = await logger.count_audit_for_study(study_uid)
    assert n == 30

    rows = await logger.list_audit_for_study(study_uid)
    # payload i 全部命中（去重后应有 0..29）
    idxs = sorted(r["payload"]["i"] for r in rows)
    assert idxs == list(range(30))


async def test_concurrent_tool_call_writes_no_loss(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """并发 30 行 tool_call_record 全部落单 JSONL 文件，每行 JSON 合法。"""
    file_paths: list[Path] = []

    async def _write(i: int) -> None:
        path = await logger.tool_call_record(
            tool="concurrent",
            model="gpt-4o",
            prompt_tokens=i,
            completion_tokens=i + 1,
            latency_ms=i * 10,
            status="ok",
            study_uid=f"study-{i}",
        )
        file_paths.append(path)

    await asyncio.gather(*[_write(i) for i in range(30)])

    # 全部落到同一 UTC 日期文件（同一日内调用）
    unique_paths = set(file_paths)
    assert len(unique_paths) == 1
    same_path = file_paths[0]
    assert same_path.exists()

    records = AuditLogger.read_tool_call_records(same_path)
    assert len(records) == 30

    # 每行 JSON 必合法，字段齐全；i 范围完整、无重复
    indices = sorted(r["prompt_tokens"] for r in records)
    assert indices == list(range(30))
    required_keys = {
        "ts", "study_uid", "tool", "model",
        "prompt_tokens", "completion_tokens",
        "latency_ms", "status",
    }
    for r in records:
        assert required_keys.issubset(r.keys())


async def test_concurrent_mixed_writes(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """混合 audit_log + tool_call_record 并发（asyncio.gather）皆完整。"""

    async def _audit(i: int) -> None:
        await logger.audit_log_insert(
            actor_role="sr", study_uid=f"s-{i}", event_type="dual.test",
            payload={"i": i},
        )

    async def _tool(i: int) -> None:
        await logger.tool_call_record(
            tool="t", model="m",
            prompt_tokens=i, completion_tokens=i,
            latency_ms=i, status="ok",
            study_uid=f"s-{i}",
        )

    coros: list = []
    for i in range(20):
        coros.append(_audit(i))
        coros.append(_tool(i))
    await asyncio.gather(*coros)

    # audit_log 验证：每个 study_uid 恰好 1 条，event_type=dual.test 共 20 条
    db = await logger.ensure_db()
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event_type = 'dual.test'"
        )
        row = await cur.fetchone()
        await cur.close()
    assert int(row["n"]) == 20

    # JSONL 验证：共 20 条
    jsonl_files = list(tmp_tool_calls_dir.glob("tool_calls_*.jsonl"))
    total_records: list[dict] = []
    for fp in jsonl_files:
        total_records.extend(AuditLogger.read_tool_call_records(fp))
    assert len(total_records) == 20


# =====================================================================
# 7. 损坏 JSONL 行：read_tool_call_records 跳过 + warn 不崩
# =====================================================================
async def test_read_tool_call_records_skips_bad_lines(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """手工写坏 JSON 行夹在好行之间；reader 跳过坏行、好行保留。"""
    good_path = await logger.tool_call_record(
        tool="good1", model="m",
        prompt_tokens=1, completion_tokens=1,
        latency_ms=10, status="ok",
        study_uid="g1",
    )
    # 追加一行损坏 JSON + 一行合法 JSON + 一行空行
    with open(good_path, "a", encoding="utf-8") as fh:
        fh.write("not-json\n")
        record2 = {
            "ts": "2026-10-13T08:00:00.001Z",
            "study_uid": "g2",
            "tool": "good2",
            "model": "m",
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "latency_ms": 20,
            "status": "ok",
        }
        fh.write(json.dumps(record2, ensure_ascii=False) + "\n")
        fh.write("\n")  # 空行

    records = AuditLogger.read_tool_call_records(good_path)
    # 跳过坏行与空行，剩 2 条
    assert len(records) == 2
    assert records[0]["study_uid"] == "g1"
    assert records[1]["study_uid"] == "g2"


# =====================================================================
# 8. list_audit_for_study / count_audit_for_study 空查询
# =====================================================================
async def test_list_audit_for_study_empty(logger: AuditLogger) -> None:
    """未写过该 study 的 list 返回空列表。"""
    rows = await logger.list_audit_for_study("never-written")
    assert rows == []


async def test_count_audit_for_study_empty(logger: AuditLogger) -> None:
    """未写过该 study 的 count 返回 0。"""
    n = await logger.count_audit_for_study("never-counted")
    assert n == 0


# =====================================================================
# 9. list_tool_call_files 列举
# =====================================================================
async def test_list_tool_call_files(
    logger: AuditLogger, tmp_tool_calls_dir: Path
) -> None:
    """list_tool_call_files 按文件名升序返回。"""
    # 还没写
    assert logger.list_tool_call_files() == []

    await logger.tool_call_record(
        tool="x", model="m", prompt_tokens=1, completion_tokens=1,
        latency_ms=1, status="ok", ts="2026-10-10T01:00:00.000Z",
    )
    await logger.tool_call_record(
        tool="x", model="m", prompt_tokens=1, completion_tokens=1,
        latency_ms=1, status="ok", ts="2026-10-11T01:00:00.000Z",
    )

    files = logger.list_tool_call_files()
    assert len(files) == 2
    assert [p.name for p in files] == [
        "tool_calls_2026-10-10.jsonl",
        "tool_calls_2026-10-11.jsonl",
    ]


# =====================================================================
# 10. 多 study_uid 隔离 + ts 单调性
# =====================================================================
async def test_audit_log_isolation_across_study_uids(logger: AuditLogger) -> None:
    """不同 study_uid 的 audit 行互不干扰；list_for_study 只返回自己的。"""
    for uid in ("study-A", "study-B", "study-C"):
        await logger.audit_log_insert(
            actor_role="system", study_uid=uid,
            event_type="isolation.test",
            payload={"uid": uid},
        )
    assert len(await logger.list_audit_for_study("study-A")) == 1
    assert len(await logger.list_audit_for_study("study-B")) == 1
    assert len(await logger.list_audit_for_study("study-C")) == 1


async def test_audit_log_ts_monotonic_within_study(logger: AuditLogger) -> None:
    """同 study 多次写入 list 顺序按 ts ASC。"""
    uid = "study-ts-order"
    ts_seq = [
        "2026-10-13T07:23:11.487Z",
        "2026-10-13T07:23:11.488Z",
        "2026-10-13T07:23:11.489Z",
    ]
    for ts in ts_seq:
        await logger.audit_log_insert(
            actor_role="system", study_uid=uid,
            event_type="ts.order", ts=ts,
        )
    rows = await logger.list_audit_for_study(uid)
    assert [r["ts"] for r in rows] == ts_seq
