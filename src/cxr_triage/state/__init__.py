"""state — 5 态状态机 + CAS 并发防护 + 驳回回退 + 幂等缓存。

公开入口：
- ``CxrState``                  5 态枚举（draft_pending / jr_confirmed /
                                sr_reviewed / sr_signed / rejected）
- ``ActorRole``                 操作者角色枚举（jr / reviewer / sr / system）
- ``StateTransitionError``      非法跃迁 / 角色不匹配 / 状态不匹配异常
- ``TransitionResult``          transition() 返回值（success / new_state /
                                audit_log_id / from_state）
- ``CxrStateMachine``           状态机主类（transition / reject / query_state /
                                get_or_init_state / list_by_state /
                                count_by_state / list_audit_for_study）
- ``StateIdempotencyStore``     state 视角的幂等缓存（草稿复用 + 驳回清空）
- ``STATE_IDEMPOTENCY_TTL_SECONDS``  5 分钟 TTL 常量
- ``DEFAULT_TOOL_VERSION``      当前产品版本号（默认 tool_version）

业务定位（spec.md §5 维度 3 + 5 + plan.md §5）：
- 体检 PACS 落片 → StateGraph 跑 5 件工具 → 入 triage_queue 时默认
  ``draft_pending``；医师在浏览器面板上点确认 / 复核 / 签发 / 驳回，
  走 ``CxrStateMachine.transition()`` 推进。
- **三角色流转**：jr → reviewer → sr；任一阶段可驳回。
- **驳回回退**：被驳回 → ``rejected`` 后**自动**回 ``draft_pending``，
  由 ``CxrStateMachine.reject()`` 一并完成（两次状态变更），并通过
  ``StateIdempotencyStore.invalidate_study()`` 清掉旧草稿。
- **CAS 并发防护**：两个医师同时点 sr_sign → 只有一条 UPDATE 影响行 = 1，
  另一条返回 ``False``；调用方拿到 ``False`` 后应当重新刷新界面。
- **幂等缓存**：TriageTrigger 跑前查 ``StateIdempotencyStore.lookup``；
  命中则跳过 5 件工具直接复用 draft（避免重复调 LLM）。
- **审计留痕**：每次 transition 同步写一条 ``audit_log`` 行，含
  ``actor_role + ts + from_state + to_state + study_uid + payload_json``。
"""

from __future__ import annotations

from .idempotency import (
    DEFAULT_TOOL_VERSION,
    STATE_IDEMPOTENCY_TTL_SECONDS,
    StateIdempotencyStore,
)
from .state_machine import (
    ActorRole,
    CxrState,
    CxrStateMachine,
    StateTransitionError,
    TransitionResult,
    VALID_TRANSITIONS,
)

__all__ = [
    # 状态机
    "CxrState",
    "ActorRole",
    "CxrStateMachine",
    "StateTransitionError",
    "TransitionResult",
    "VALID_TRANSITIONS",
    # 幂等
    "StateIdempotencyStore",
    "STATE_IDEMPOTENCY_TTL_SECONDS",
    "DEFAULT_TOOL_VERSION",
]
