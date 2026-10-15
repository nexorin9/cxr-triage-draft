"""state 层幂等缓存 — study_uid + tool_version 草稿复用。

业务定位（spec.md §5 维度 5 + plan.md §5 深度 (a) 真实算法）：

- **同 Study 同 tool_version 二次入站**：直接复用 ``draft``，避免
  重复跑 5 件工具 + 重复调 LLM。体检 PACS 因网络抖动或同检查重推
  会在 5 分钟内二次入站同一张胸片。
- **5 分钟 TTL（spec.md §5 维度 5 硬约束）**：覆盖「同检查短时间重复
  入站」场景；过期视为失效（业务方重新生成）。
- **驳回时清空缓存**（``invalidate_study``）：医师驳回 → 状态回到
  ``draft_pending`` → 旧草稿连带阳性定位框一起失效，避免医师看到
  已驳回的旧结果。

与 persistence.IdempotencyStore 的区别：

1. **默认 tool_version**：state 层默认用当前产品 ``__version__``，
   避免调用方误传版本号错用别版本的草稿（持久化层默认让调用方必传）。
2. **默认 TTL 5 分钟**：state 层硬编码（spec 约束）；持久化层默认
   5 分钟但允许测试 override。
3. **state 视角的失效接口**：
   - ``invalidate_study(study_uid)`` — 一次失效该 Study 全部版本，
     用于驳回到 ``draft_pending`` 时一并清空。
   - ``invalidate(study_uid, tool_version=...)`` — 失效单条（暴露
     tool_version 默认值，与持久化层同名）。

持久化落在 ``draft_idempotency`` 表（Task 3 已建）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .. import __version__
from ..persistence.idempotency import IDEMPOTENCY_TTL_SECONDS, IdempotencyStore
from ..persistence.sqlite import Database

logger = logging.getLogger(__name__)

# state 层硬约束 5 分钟 TTL（spec.md §5 维度 5）
STATE_IDEMPOTENCY_TTL_SECONDS = IDEMPOTENCY_TTL_SECONDS  # 复用持久化层常量（5 * 60）

# 默认 tool_version：当前产品版本号
DEFAULT_TOOL_VERSION = __version__


class StateIdempotencyStore:
    """state 视角的幂等缓存 — 包装 ``persistence.IdempotencyStore``。

    Parameters
    ----------
    db:
        ``Database`` 实例（来自 ``cxr_triage.persistence``）。
    ttl_seconds:
        TTL 秒数；默认 5 分钟（spec.md §5 维度 5）。测试可注入小 TTL
        加速验证。
    """

    def __init__(
        self,
        db: Database,
        ttl_seconds: int = STATE_IDEMPOTENCY_TTL_SECONDS,
    ) -> None:
        self._store = IdempotencyStore(db=db, ttl_seconds=ttl_seconds)
        self.ttl_seconds = ttl_seconds
        logger.debug(
            "StateIdempotencyStore ready: ttl=%ds, default_tool_version=%s",
            ttl_seconds,
            DEFAULT_TOOL_VERSION,
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    async def cache_draft(
        self,
        study_uid: str,
        draft: dict[str, Any],
        tool_version: str = DEFAULT_TOOL_VERSION,
    ) -> None:
        """写入 / 覆盖 (study_uid, tool_version) 草稿。

        ``ON CONFLICT(study_uid, tool_version) DO UPDATE`` 保证：
        - 首次入站：INSERT；
        - 同 key 二次入站：刷新 inserted_at 与 draft_json。

        Parameters
        ----------
        study_uid:
            DICOM ``StudyInstanceUID``。
        draft:
            工作流产物（5 件工具结果聚合的 dict；调用方负责可序列化）。
        tool_version:
            草稿对应的产品版本；默认 ``__version__``。
        """
        await self._store.cache_draft(
            study_uid=study_uid,
            tool_version=tool_version,
            draft=draft,
        )

    # ------------------------------------------------------------------
    # 读取（命中且未过期才返回）
    # ------------------------------------------------------------------
    async def lookup(
        self,
        study_uid: str,
        tool_version: str = DEFAULT_TOOL_VERSION,
    ) -> Optional[dict[str, Any]]:
        """命中且未过期 → 返回 draft dict；否则 → None。

        用于 TriageTrigger 跑前查幂等；命中直接走 SSE 推送 + 入
        triage_queue，不重跑 5 件工具。
        """
        return await self._store.lookup(
            study_uid=study_uid,
            tool_version=tool_version,
        )

    async def lookup_with_meta(
        self,
        study_uid: str,
        tool_version: str = DEFAULT_TOOL_VERSION,
    ) -> Optional[dict[str, Any]]:
        """调试 / 监控用：返回 ``{"draft": dict, "inserted_at": str,
        "age_seconds": float, "expired": bool}``。

        即使过期也会返回 dict（带 ``expired=True``），方便前端展示
        「缓存命中但已过期」的状态。
        """
        return await self._store.lookup_with_meta(
            study_uid=study_uid,
            tool_version=tool_version,
        )

    # ------------------------------------------------------------------
    # 失效
    # ------------------------------------------------------------------
    async def invalidate(
        self,
        study_uid: str,
        tool_version: str = DEFAULT_TOOL_VERSION,
    ) -> int:
        """失效 (study_uid, tool_version) 单条；返回删除行数。

        用于产品升级 ``__version__`` 时清掉旧版本草稿。
        """
        return await self._store.invalidate(
            study_uid=study_uid,
            tool_version=tool_version,
        )

    async def invalidate_study(self, study_uid: str) -> int:
        """一次失效该 Study 全部 tool_version；返回删除行数。

        用于驳回到 ``draft_pending`` 时清掉旧草稿 — 旧阳性定位框和
        所见草稿都不应继续展示给医师，否则会出现「驳回了但旧结果
        仍可见」的状态不一致。

        业务示例：jr 看到阳性后驳回 → state_machine 走两步
        ``jr_confirmed → rejected → draft_pending``；同步调本方法
        清掉 (study_uid, *) 全部草稿；TriageTrigger 收到 draft_pending
        后重新跑 5 件工具覆盖。
        """
        deleted = await self._store.invalidate(
            study_uid=study_uid,
            tool_version=None,  # 持久化层 None = 失效该 study 全部
        )
        logger.debug(
            "invalidate_study: %s → deleted %d rows", study_uid, deleted
        )
        return deleted


__all__ = [
    "StateIdempotencyStore",
    "STATE_IDEMPOTENCY_TTL_SECONDS",
    "DEFAULT_TOOL_VERSION",
]
