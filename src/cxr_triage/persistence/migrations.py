"""migrations — 4 表 + 幂等表初始化。

设计原则（spec.md §5 维度 1 + plan.md §5 深度 (a)）：

1. **幂等可重入**：所有 DDL 走 ``CREATE ... IF NOT EXISTS``；同一 db 调
   多次 ``init_db`` 不会报错，也不会重复建索引。
2. **统一入口**：业务层（FastAPI lifespan、测试 setup）只调
   ``await init_db(db)``；不直接碰 schema.py 的 DDL 列表。
3. **WAL + 外键**：连接由 ``Database.connect()`` 完成；本模块只补 DDL。
4. **失败语义**：任何 DDL 失败 → 抛 ``MigrationError``，调用方决定是
   重试还是放弃。

不引入 alembic 等重型迁移框架（4 表 + 1 幂等表不值得）；如未来要加列，
按 ``ALTER TABLE ... ADD COLUMN`` 增量补丁的方式扩 init_db。
"""
from __future__ import annotations

import logging
from typing import Optional

from .schema import IDEMPOTENCY_DDL, SCHEMA_DDL, STATE_DDL
from .sqlite import Database

logger = logging.getLogger(__name__)


class MigrationError(RuntimeError):
    """DDL 执行失败时抛。"""


async def init_db(
    db: Database,
    *,
    include_idempotency: bool = True,
    include_state_machine: bool = True,
) -> None:
    """初始化 4 表 + 幂等表（可选）+ 5 态状态机表（可选）。

    Parameters
    ----------
    db:
        已通过 ``await db.connect()`` 打开连接的 Database 实例。
    include_idempotency:
        是否一并创建幂等表；测试场景可关。
    include_state_machine:
        是否一并创建 5 态状态机表；测试场景可关。
    """
    if db._conn is None:  # type: ignore[attr-defined]  # noqa: SLF001
        raise MigrationError("Database 未连接；请先 await db.connect()")

    statements = list(SCHEMA_DDL)
    if include_idempotency:
        statements.extend(IDEMPOTENCY_DDL)
    if include_state_machine:
        statements.extend(STATE_DDL)

    async with db.connection() as conn:
        for stmt in statements:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                logger.exception("DDL failed: %s", stmt.strip().split("\n", 1)[0])
                raise MigrationError(f"DDL failed: {exc}") from exc

    logger.info(
        "init_db OK: %d statements (idempotency=%s, state_machine=%s)",
        len(statements),
        include_idempotency,
        include_state_machine,
    )


async def table_exists(db: Database, table_name: str) -> bool:
    """查询 sqlite_master 是否存在指定表（测试用）。"""
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = await cur.fetchone()
        await cur.close()
        return row is not None


async def list_tables(db: Database) -> list[str]:
    """列出当前 db 全部表名（测试 + /api/health 用）。"""
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [r[0] for r in rows]


__all__ = ["MigrationError", "init_db", "list_tables", "table_exists"]