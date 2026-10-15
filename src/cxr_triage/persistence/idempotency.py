"""study_uid + tool_version 幂等表 — 同 study + 同 tool_version 二次入站复用草稿。

设计原则（spec.md §5 维度 5 + plan.md §5 深度 (a)）：

1. **复合主键 (study_uid, tool_version)**：避免重复调 LLM。同一张胸片
   5 分钟内二次入站直接复用 draft，不重跑 5 件工具。
2. **TTL 5 分钟**：从 ``inserted_at`` 起算；过期则视为失效（业务方
   可重新生成）。本常量由 ``_TTL_SECONDS`` 控制。
3. **JSON 载荷**：draft 全量落 ``draft_json TEXT``；调用方负责序列化。
4. **不入 audit_log**：幂等命中是性能优化，不属于业务事件；但可在调
   用方写一行 audit ``idempotency_hit`` 备查。

不引入 TTL 后台清扫（5 分钟量级不大；过期行累积后下次 ``init_db`` 走
VACUUM 即可；生产可用 ``DELETE WHERE inserted_at < ?`` 周期清理）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .sqlite import Database

logger = logging.getLogger(__name__)

# 5 分钟 TTL；过长会浪费空间，过短会重复跑 LLM。
_TTL_SECONDS = 5 * 60

# 包级公开别名（idempotency.__init__ 也用得到）
IDEMPOTENCY_TTL_SECONDS = _TTL_SECONDS


def _utc_now_iso() -> str:
    """返回 UTC ISO8601 字符串（与 SQLite CURRENT_TIMESTAMP 同形）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_iso(ts: str) -> "datetime":
    """解析 ``YYYY-MM-DDTHH:MM:SS.<microseconds>Z``；缺微秒兜底。

    始终返回 tz-aware datetime（UTC）。
    """
    from datetime import datetime, timezone

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%f",
        "%Y-%m-%dT%H:%M:%fZ",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
        except ValueError:
            continue
        # 强制加 UTC（输入已带 Z，按 UTC 处理）
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    raise ValueError(f"无法解析 ISO8601 字符串: {ts!r}")


class IdempotencyStore:
    """study_uid + tool_version 幂等表。"""

    def __init__(self, db: Database, ttl_seconds: int = _TTL_SECONDS) -> None:
        self.db = db
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    async def cache_draft(
        self,
        study_uid: str,
        tool_version: str,
        draft: dict[str, Any],
    ) -> None:
        """写入 / 覆盖 (study_uid, tool_version) 草稿。

        ``ON CONFLICT(study_uid, tool_version) DO UPDATE`` 保证：
        - 首次入站：INSERT
        - 同 key 二次入站：刷新 inserted_at 与 draft_json
        """
        payload = json.dumps(draft, ensure_ascii=False, default=str)
        now_iso = _utc_now_iso()
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO draft_idempotency (study_uid, tool_version, draft_json, inserted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(study_uid, tool_version) DO UPDATE SET
                    draft_json  = excluded.draft_json,
                    inserted_at = excluded.inserted_at
                """,
                (study_uid, tool_version, payload, now_iso),
            )
            await conn.commit()
        logger.debug("idempotency cache_draft: %s v=%s", study_uid, tool_version)

    # ------------------------------------------------------------------
    # 读取（命中且未过期才返回）
    # ------------------------------------------------------------------
    async def lookup(
        self,
        study_uid: str,
        tool_version: str,
    ) -> Optional[dict[str, Any]]:
        """命中且未过期 → 返回 draft dict；否则 → None。

        命中但过期：仍返回 None（视为失效）；若调用方希望区分
        "过期"与"未命中"，可用 ``lookup_with_meta``。
        """
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT draft_json, inserted_at FROM draft_idempotency
                WHERE study_uid = ? AND tool_version = ?
                """,
                (study_uid, tool_version),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is None:
                return None
            draft_json, inserted_at = row[0], row[1]

        # TTL 检查
        from datetime import datetime, timezone

        inserted_dt = _parse_iso(inserted_at)
        now_dt = datetime.now(timezone.utc)
        age = (now_dt - inserted_dt).total_seconds()
        if age > self.ttl_seconds:
            logger.debug(
                "idempotency expired: %s v=%s age=%.1fs ttl=%ds",
                study_uid,
                tool_version,
                age,
                self.ttl_seconds,
            )
            return None

        return json.loads(draft_json)

    async def lookup_with_meta(
        self,
        study_uid: str,
        tool_version: str,
    ) -> Optional[dict[str, Any]]:
        """调试 / 监控用：返回 ``{"draft": dict, "inserted_at": str,
        "age_seconds": float, "expired": bool}``。"""
        async with self.db.connection() as conn:
            cur = await conn.execute(
                """
                SELECT draft_json, inserted_at FROM draft_idempotency
                WHERE study_uid = ? AND tool_version = ?
                """,
                (study_uid, tool_version),
            )
            row = await cur.fetchone()
            await cur.close()
            if row is None:
                return None
            draft_json, inserted_at = row[0], row[1]

        from datetime import datetime, timezone

        inserted_dt = _parse_iso(inserted_at)
        now_dt = datetime.now(timezone.utc)
        age = (now_dt - inserted_dt).total_seconds()
        return {
            "draft": json.loads(draft_json),
            "inserted_at": inserted_at,
            "age_seconds": age,
            "expired": age > self.ttl_seconds,
        }

    # ------------------------------------------------------------------
    # 失效（仅供测试 / 运维使用）
    # ------------------------------------------------------------------
    async def invalidate(self, study_uid: str, tool_version: Optional[str] = None) -> int:
        """显式失效；返回删除的行数。``tool_version=None`` 失效该 study 全部版本。"""
        async with self.db.connection() as conn:
            if tool_version is None:
                cur = await conn.execute(
                    "DELETE FROM draft_idempotency WHERE study_uid = ?",
                    (study_uid,),
                )
            else:
                cur = await conn.execute(
                    "DELETE FROM draft_idempotency "
                    "WHERE study_uid = ? AND tool_version = ?",
                    (study_uid, tool_version),
                )
            await conn.commit()
            return cur.rowcount or 0


__all__ = ["IdempotencyStore", "IDEMPOTENCY_TTL_SECONDS"]