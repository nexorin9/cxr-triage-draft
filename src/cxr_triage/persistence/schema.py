"""SQLite 4 表 DDL — incoming_studies / triage_queue / audit_log / outbox。

设计原则（spec.md §5 维度 1 + 3 + 6 + plan.md §5 深度 (a)）：

1. **统一时间戳**：所有 ``ts`` 字段用 ``TEXT`` 存 ISO8601 UTC（便于跨
   时区排序与人工阅读）；SQLite ``CURRENT_TIMESTAMP`` 仅在初始化时用。
2. **无外键约束**：4 表刻意不引入 FK，避免 outbox/audit_log 被级联
   删除；幂等由 ``(study_uid, tool_version)`` 复合主键保证。
3. **payload_json 用 TEXT**：不用 SQLite JSON1 extension（部分打包版本
   未启用）；序列化由 Python 端 ``json.dumps`` 负责。
4. **outbox 失败重试**：``retry_count`` + ``next_retry_at`` + ``last_error``
   + ``status`` (pending / sent / failed_retry / alerting) 完整。
5. **triage_queue 排序索引**：(``max_prob`` DESC) 用于浏览器第一栏阳性
   队列拉取。
6. **幂等可重入**：所有 DDL 都用 ``CREATE TABLE IF NOT EXISTS``，
   ``CREATE INDEX IF NOT EXISTS``；重复执行不会报错。

字段命名约定：
- 主键：单字段用 ``study_uid`` / ``id``；复合主键在表头明示。
- ``study_uid`` 统一用 ``TEXT``（DICOM UID 不超过 64 字节，无性能问题）。
- ``actor_role`` 枚举字符串：``jr`` / ``reviewer`` / ``sr`` / ``system``。
"""
from __future__ import annotations

# 4 表 + 2 索引；顺序敏感：先表后索引。
SCHEMA_DDL: list[str] = [
    # ------------------------------------------------------------------
    # 1. incoming_studies — PACS 入站队列
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS incoming_studies (
        study_uid        TEXT PRIMARY KEY,
        patient_id       TEXT NOT NULL DEFAULT '',
        patient_id_hash  TEXT NOT NULL DEFAULT '',
        accession_number TEXT NOT NULL DEFAULT '',
        modality         TEXT NOT NULL,
        study_date       TEXT NOT NULL DEFAULT '',
        patient_sex      TEXT NOT NULL DEFAULT '',
        patient_age      TEXT NOT NULL DEFAULT '',
        source           TEXT NOT NULL DEFAULT 'pacs_cstore',
        status           TEXT NOT NULL DEFAULT 'pending',
        max_prob         REAL NOT NULL DEFAULT 0.0,
        received_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    # ------------------------------------------------------------------
    # 2. triage_queue — 阳性优先级队列（按 max_prob DESC 索引）
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS triage_queue (
        study_uid     TEXT PRIMARY KEY,
        max_prob      REAL NOT NULL,
        positive_flag INTEGER NOT NULL DEFAULT 0,
        finding_text  TEXT NOT NULL DEFAULT '',
        inserted_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        dispatched_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_triage_queue_max_prob "
    "ON triage_queue (max_prob DESC)",
    # ------------------------------------------------------------------
    # 3. audit_log — 状态与外发流水（飞检审计）
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_role   TEXT NOT NULL,
        ts           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        study_uid    TEXT NOT NULL DEFAULT '',
        event_type   TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_log_study_uid "
    "ON audit_log (study_uid)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log (ts DESC)",
    # ------------------------------------------------------------------
    # 4. outbox — 写回异步重试表
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS outbox (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        study_uid       TEXT NOT NULL,
        target_endpoint  TEXT NOT NULL,
        payload_json    TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'pending',
        retry_count     INTEGER NOT NULL DEFAULT 0,
        next_retry_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        last_error      TEXT NOT NULL DEFAULT '',
        created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outbox_status_next_retry "
    "ON outbox (status, next_retry_at)",
    "CREATE INDEX IF NOT EXISTS idx_outbox_study_uid "
    "ON outbox (study_uid)",
]


# 5 表外的复合主键幂等表（plan.md §5 深度 (a)）：
# 由 ``idempotency.py`` 单独管理；不在此枚举，便于单元测试可独立创建。
IDEMPOTENCY_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS draft_idempotency (
        study_uid    TEXT NOT NULL,
        tool_version TEXT NOT NULL,
        draft_json   TEXT NOT NULL,
        inserted_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        PRIMARY KEY (study_uid, tool_version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_draft_idempotency_inserted_at "
    "ON draft_idempotency (inserted_at)",
]


# ----------------------------------------------------------------------
# 5. triage_states — 5 态状态机主表（spec.md §5 维度 3 业务状态生命周期）
# ----------------------------------------------------------------------
# 与 incoming_studies.status 完全独立：状态机记录医师流转（draft_pending
# → jr_confirmed → sr_reviewed → sr_signed，被驳回 → rejected 后回
# draft_pending），incoming_studies.status 仍然是 PACS 队列状态。
# 表里 study_uid 主键 + current_state 由 CAS 守；每次 transition 同步
# 写 audit_log 行（actor_role + ts + from_state + to_state + payload 摘要）。
STATE_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS triage_states (
        study_uid     TEXT PRIMARY KEY,
        current_state TEXT NOT NULL DEFAULT 'draft_pending',
        updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        updated_by    TEXT NOT NULL DEFAULT 'system'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_triage_states_current_state "
    "ON triage_states (current_state)",
    "CREATE INDEX IF NOT EXISTS idx_triage_states_updated_at "
    "ON triage_states (updated_at DESC)",
]


__all__ = ["SCHEMA_DDL", "IDEMPOTENCY_DDL", "STATE_DDL"]