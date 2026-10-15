"""persistence — SQLite 持久化层（aiosqlite + 4 表 + outbox + 幂等 + 状态机）。

公开入口：
- ``Database``           aiosqlite 单连接 + WAL 异步上下文管理器
- ``init_db``            4 表 + 幂等表 + 状态机表的幂等 DDL 初始化
- ``IdempotencyStore``   (study_uid, tool_version) 幂等查询
- ``Outbox``             写回异步重试表（指数退避 30s/60s/120s/240s）
- ``compute_next_retry_iso``  纯函数：retry_count → next_retry_at
- ``list_tables``        调试 / 健康检查
"""
from __future__ import annotations

from .idempotency import IDEMPOTENCY_TTL_SECONDS, IdempotencyStore
from .migrations import MigrationError, init_db, list_tables, table_exists
from .outbox import (
    BASE_BACKOFF_SECONDS,
    MAX_RETRY_BEFORE_ALERT,
    Outbox,
    compute_next_retry_iso,
)
from .schema import IDEMPOTENCY_DDL, SCHEMA_DDL, STATE_DDL
from .sqlite import DEFAULT_DB_PATH, Database

__all__ = [
    # connection
    "Database",
    "DEFAULT_DB_PATH",
    # migrations
    "init_db",
    "list_tables",
    "table_exists",
    "MigrationError",
    # schema
    "SCHEMA_DDL",
    "IDEMPOTENCY_DDL",
    "STATE_DDL",
    # idempotency
    "IdempotencyStore",
    "IDEMPOTENCY_TTL_SECONDS",
    # outbox
    "Outbox",
    "compute_next_retry_iso",
    "BASE_BACKOFF_SECONDS",
    "MAX_RETRY_BEFORE_ALERT",
]