"""SQLite 连接池与 Database 类 — 体检胸片所见草稿系统的持久化基座。

设计原则（spec.md §5 维度 1 + 4 + 6 + plan.md §5 深度 (a)）：

1. **aiosqlite 单连接异步上下文管理器**：SQLite 写并发有限，单连接 + WAL
   已经足够（体检日均 200+ 张胸片，旺季 300+），业务层用 ``async with
   db.connection() as conn:`` 即取即用，不暴露连接本身。
2. **WAL 日志模式**：允许并发读 + 单写；浏览器面板的 SELECT 与触发器的
   INSERT 不会互锁。
3. **PRAGMA foreign_keys=ON**：4 表之间不引入外键约束（避免级联删除把
   审计行一起删了），但显式打开以便将来加 FK。
4. **路径注入**：db_path 必传，缺省 ``/tmp/cxr-triage.db`` 仅供测试；生产
   须走 config.data_dir 或显式参数。
5. **零业务字段**：本文件**不**导入业务模块（state_machine、triage_queue
   等），只暴露 ``Database`` 与 ``init_db``，避免循环依赖。

约定：
- **单例**：每个进程只创建一个 ``Database`` 实例；FastAPI lifespan 中
  通过 ``app.state.db`` 持有。
- **连接管理**：aiosqlite 单连接同写并发由 SQLite 串行化；本封装不对外
  提供 ``execute`` 直通入口（避免业务侧乱开事务），只暴露 ``connection``
  上下文管理器给同级 ``Outbox`` / ``IdempotencyStore`` / ``StateMachine``
  等专用类使用。
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/tmp/cxr-triage.db"


class Database:
    """aiosqlite 单连接 + WAL；``connection()`` 异步上下文管理器即取即用。

    Parameters
    ----------
    db_path:
        SQLite 文件路径；缺省 ``/tmp/cxr-triage.db``（仅供本地调试）。
        生产环境建议显式传入 ``config.data_dir/cxr-triage.db``。
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock_count = 0  # 简化并发追踪；调试期可见

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """打开连接 + 启用 WAL；幂等（重复调用只返回一次）。"""
        if self._conn is not None:
            return
        # ``isolation_level=None`` 让 aiosqlite 进入 autocommit 模式，调用方
        # 用 ``await conn.execute("BEGIN")`` 显式开事务；适合 outbox 这类
        # 需要显式回滚的场景。
        self._conn = await aiosqlite.connect(
            str(self.db_path), isolation_level=None
        )
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        logger.info("SQLite connected: %s (WAL enabled)", self.db_path)

    async def close(self) -> None:
        """关闭连接；幂等。"""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite closed: %s", self.db_path)

    # ------------------------------------------------------------------
    # 上下文管理器入口
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """异步上下文管理器：返回当前连接。

        业务层调用方式::

            async with db.connection() as conn:
                await conn.execute("INSERT INTO ...", (...))
                await conn.commit()

        重复进入同一连接是合法的（SQLite 串行化）；不要并发 ``await
        conn.execute()``，请显式 ``asyncio.Lock`` 守住临界区或改走
        ``aiosqlite.connect`` 多连接（本期不引入）。
        """
        if self._conn is None:
            await self.connect()
        assert self._conn is not None  # for type-checker
        try:
            self._lock_count += 1
            yield self._conn
        finally:
            self._lock_count -= 1

    # ------------------------------------------------------------------
    # 健康检查（FastAPI /api/health 用）
    # ------------------------------------------------------------------
    async def ping(self) -> bool:
        """SELECT 1 探活。"""
        try:
            async with self.connection() as conn:
                cur = await conn.execute("SELECT 1")
                row = await cur.fetchone()
                await cur.close()
                return row is not None and row[0] == 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQLite ping failed: %s", exc)
            return False


__all__ = ["Database", "DEFAULT_DB_PATH"]