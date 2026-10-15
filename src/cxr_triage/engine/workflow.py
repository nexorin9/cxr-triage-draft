"""workflow — LangGraph StateGraph 编排器（process + execute + has_tool_calls 条件边）。

把源仓库 ``medrax/agent/agent.py:91-100`` 的 ``process / execute`` 双节点 +
``has_tool_calls`` 条件边复用并改写：

- 触发器从 chat 文本改为 **StudyInstanceUID 事件**（体检 PACS 落片）。
- ``process`` 节点始终**先跑 classifier**（18 病理概率）拿 ``max_prob``；命中即
  标记 ``tools_to_execute`` 列表装填剩余 4 件工具，否则产物即用（阴性无需
  后续工具）。
- ``execute`` 节点**串行**跑列表里的工具，避免并发竞写同一份 draft；每跑完
  一件工具把工具名追加到 ``tools_executed``。
- ``has_tool_calls`` 条件边：
  - ``process`` → ``execute`` ：classifier_probs 存在且列表非空；
  - ``process`` → ``END`` ：classifier_probs 缺失 / 跑过且没有下一件工具；
  - ``execute`` → ``process`` ：tools_to_execute 与 tools_executed 差集非空；
  - ``execute`` → ``END``   ：全部跑完。

公开入口：

- ``CxrTriageWorkflow``   StateGraph 编译后的可执行 Runnable
- ``CxrTriageState``      TypedDict 状态字段
- ``ToolRegistry``        工具注册表（duck-typed：``.name`` + ``async run(state)``）
- ``DEFAULT_TOOL_NAMES``  5 件工具的默认注册顺序（classifier 必首位）
- ``WorkflowError``       状态非法 / 工具未注册异常

业务定位（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法）：

- 入站 ``StudyInstanceUID`` → ``run(study_uid, png_bytes, metadata)`` → 5 件
  工具协同跑通 → 浏览器三栏面板 SSE 推送 ``draft_ready`` 事件 → 医师三角色
  流转 → RIS / 体检主检外发。
- 本类**只**管 StateGraph 编排与状态字段；**不**入队（找 ``engine.TriageQueue``），
  也不**写** RIS / 主检（找 ``writers``）；**不**持久化草稿（找
  ``state.StateIdempotencyStore``）。
"""
from __future__ import annotations

import logging
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
)

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


# ---------- 工具注册顺序（与 spec.md §2 工作闭环一一对应）----------
# 顺序：classifier 必首位 → keyqa → finding_draft → positive_grounding → image_preview
DEFAULT_TOOL_NAMES: List[str] = [
    "classifier",
    "keyqa",
    "finding_draft",
    "positive_grounding",
    "image_preview",
]

# 工具调用契约（duck-typed）：每个工具实例须有
#   - ``name: str``
#   - ``async run(state: Dict[str, Any]) -> Dict[str, Any]``
# 引擎把 ``state`` 当作 dict 透传；工具返回需要 merge 回 state 的字段子集。

ToolAsyncFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class WorkflowError(RuntimeError):
    """StateGraph 编排期错误：工具未注册 / 状态非法 / 触发器参数缺失。"""


# ---------------------------------------------------------------------------
# 状态字段
# ---------------------------------------------------------------------------


class CxrTriageState(TypedDict, total=False):
    """StateGraph 流转状态字段。

    所有字段都 ``total=False`` —— 任意字段缺失均不抛；工具按需读写，
    ``process`` / ``execute`` 节点负责补齐关键字段并把合并结果返回。
    """

    # ---- 触发器 / 输入 ----
    study_uid: str
    png_bytes: bytes
    metadata: Dict[str, Any]

    # ---- 工具产物（每个工具负责写入自己名下的字段）----
    classifier_probs: List[Dict[str, Any]]      # TriageClassifier 输出
    classifier_max_prob: float                  # 18 病理概率中的最大值
    positive_flag: bool                         # max_prob ≥ 阈值
    key_qa_list: List[Dict[str, Any]]           # CXRKeyQA 输出
    finding_text: str                           # FindingDraft 中文所见
    finding_struct: Dict[str, Any]              # 结构化所见（pathologies/key_qa/grounding_boxes）
    grounding_boxes: List[Dict[str, Any]]       # PositiveGrounding 输出
    png_preview_b64: str                        # ImagePreview 输出 base64

    # ---- 编排控制字段 ----
    tools_to_execute: List[str]                 # 待执行工具名（process 节点填充）
    tools_executed: Annotated[List[str], _list_concat]  # 已执行工具名（execute 追加）
    degraded: bool                              # 是否进入降级（工具不可达）
    degradation_reasons: List[str]              # 降级原因清单


# 旧版 langgraph 不支持 list 累加 reducer；用 ``operator.add`` 兼容：
# typing 模块里 ``list`` 不可 ``operator.add``，退化为「execute 返回
# [tool_name]，由 StateGraph 默认 field merge 写入」。
def _list_concat(current, new):
    """tools_executed 累加 reducer（langgraph 旧版字段重写时需要显式 list）。"""
    return list(current or []) + list(new or [])


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------


class ToolRegistry:
    """工具注册表：duck-typed 协议 ``async run(state) -> state``。

    与 ``medrax/agent/agent.py:101`` 的 ``{t.name: t for t in tools}``
    同形；区别在于 MedRAX 工具继承 ``BaseTool``（与 langchain 紧耦合），
    本表只要求 ``.name`` 属性与 ``run()`` 方法，避免被强制绑进 langchain 工具
    抽象。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        """注册一件工具；重复名抛 WorkflowError。"""
        if not hasattr(tool, "name") or not isinstance(tool.name, str) or not tool.name:
            raise WorkflowError("tool must expose non-empty .name attribute")
        if not hasattr(tool, "run") or not callable(tool.run):
            raise WorkflowError(f"tool {tool.name!r} must expose async .run() method")
        if tool.name in self._tools:
            raise WorkflowError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Any:
        """按名取工具；不存在抛 WorkflowError（便于 StateGraph 早失败）。"""
        if name not in self._tools:
            raise WorkflowError(f"tool {name!r} not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# StateGraph 编排器
# ---------------------------------------------------------------------------


class CxrTriageWorkflow:
    """体检胸片所见草稿 StateGraph 编排器。

    使用方式::

        registry = ToolRegistry()
        registry.register(classifier_tool)
        registry.register(keyqa_tool)
        ...
        wf = CxrTriageWorkflow(registry)
        # 一次性 build_graph() 后才能调用 run()
        wf.build_graph()
        draft = await wf.run(study_uid, png_bytes, metadata)

    ``run`` 返回 ``dict[str, Any]``，即最终 state 快照，可直接喂给
    ``engine.TriageQueue.enqueue`` 与 ``state.StateIdempotencyStore.cache_draft``。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        positive_threshold: float = 0.5,
        max_cycles: int = 8,
    ) -> None:
        """``positive_threshold`` 是 classifier ``max_prob`` 触发后续工具的下限；
        ``max_cycles`` 防御性 cycle 上限（spec.md §6 部署：单张 30 秒内出结构化
        产物，``max_cycles=8`` 足以覆盖 5 件工具 + 状态收敛）。"""
        if not isinstance(registry, ToolRegistry):
            raise WorkflowError("registry must be a ToolRegistry instance")
        if not (0.0 <= float(positive_threshold) <= 1.0):
            raise WorkflowError("positive_threshold must be in [0, 1]")
        if int(max_cycles) < 1:
            raise WorkflowError("max_cycles must be >= 1")

        self.registry = registry
        self.positive_threshold = float(positive_threshold)
        self.max_cycles = int(max_cycles)

        # classifier 必须注册（process 节点强依赖）
        if not registry.has("classifier"):
            raise WorkflowError(
                "registry must contain 'classifier' (process node depends on it)"
            )

        self._compiled = None  # 编译后存这里；build_graph() 之前为 None

    # ---- 公开入口 ----

    def build_graph(self) -> Any:
        """编译 StateGraph；重复调用幂等（同一 registry 复用同一份编译结果）。"""
        if self._compiled is not None:
            return self._compiled

        graph = StateGraph(dict)  # 用 dict 而不是 CxrTriageState：避免 TypedDict
                                  # 字段 total=False 时 langgraph 校验报"missing
                                  # required field"；运行时按 state 字段语义
                                  # 透传即可。

        # ---- 节点装配 ----
        graph.add_node("process", self._process_node)
        graph.add_node("execute", self._execute_node)

        # ---- 条件边 ----
        # process → execute / END（classifier_probs 是否已就绪）
        graph.add_conditional_edges(
            "process",
            self._has_tool_calls_after_process,
            {True: "execute", False: END},
        )

        # execute → process / END（剩余待执行工具是否为空）
        graph.add_conditional_edges(
            "execute",
            self._has_tool_calls_after_execute,
            {True: "process", False: END},
        )

        graph.set_entry_point("process")

        self._compiled = graph.compile()
        return self._compiled

    async def run(
        self,
        study_uid: str,
        png_bytes: bytes,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """触发一次完整 StateGraph 跑通，返回最终 state 快照。

        入口处做最小入参校验：``study_uid`` 非空字符串、``png_bytes`` 非空
        bytes、``metadata`` 是 dict（缺字段由各工具自行兜底）。
        """
        if not isinstance(study_uid, str) or not study_uid:
            raise WorkflowError("study_uid must be a non-empty string")
        if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) == 0:
            raise WorkflowError("png_bytes must be non-empty bytes")
        if not isinstance(metadata, dict):
            raise WorkflowError("metadata must be a dict")

        compiled = self.build_graph()
        init_state: Dict[str, Any] = {
            "study_uid": study_uid,
            "png_bytes": bytes(png_bytes),
            "metadata": dict(metadata),
            "tools_executed": [],
            "tools_to_execute": [],
            "degraded": False,
            "degradation_reasons": [],
        }

        # 防御性 cycle 上限：传入 recursion_limit 给递归/循环场景留余量
        config = {"recursion_limit": self.max_cycles * 2}
        result = await compiled.ainvoke(init_state, config=config)

        if not isinstance(result, dict):
            raise WorkflowError(
                f"StateGraph returned non-dict result: {type(result).__name__}"
            )
        return result

    # ---- 节点实现（私有）----

    async def _process_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """``process`` 节点：始终先跑 classifier；之后决定剩余工具列表。

        设计意图（与源 ``medrax/agent/agent.py:104-118`` 对照）：

        - 源 ``process`` 让 LLM 决策调哪个工具；本产品把决策换成**确定性规则**
          （max_prob ≥ 阈值 → 跑剩余 4 件工具，否则只出 classifier 产物）。
        - 这是医院语境下的合理收紧：避免模型在临床路径上「自由发挥」。
        """
        out: Dict[str, Any] = dict(state)  # 浅拷，工具自填字段会按需覆盖
        executed = list(out.get("tools_executed") or [])

        # ---- 1. 跑 classifier（必跑）----
        if not _has_state(out, "classifier_probs"):
            classifier = self.registry.get("classifier")
            try:
                result = await classifier.run(out)
            except Exception as exc:  # noqa: BLE001 —— 降级兜底
                # classifier 不可达 → 走 degraded；标记原因；剩余工具不调度
                logger.warning(
                    "classifier raised in process_node for %s: %s",
                    out.get("study_uid"), exc,
                )
                out["degraded"] = True
                out["degradation_reasons"] = list(out.get("degradation_reasons") or [])
                out["degradation_reasons"].append(f"classifier_unreachable:{exc}")
                out["classifier_probs"] = []
                out["classifier_max_prob"] = 0.0
                out["positive_flag"] = False
                out["tools_to_execute"] = []  # 不再调度后续工具
                if "classifier" not in executed:
                    executed.append("classifier")
                out["tools_executed"] = executed
                return out

            # merge result fields back
            for k, v in (result or {}).items():
                out[k] = v
            if "classifier_probs" not in out:
                # 工具返回不规范 → 不抛，降级即可
                logger.warning(
                    "classifier returned no classifier_probs for %s; degrading",
                    out.get("study_uid"),
                )
                out["classifier_probs"] = []
                out["classifier_max_prob"] = 0.0
                out["positive_flag"] = False
                out["degraded"] = True
                out["degradation_reasons"] = list(out.get("degradation_reasons") or [])
                out["degradation_reasons"].append("classifier_no_probs_in_result")
                out["tools_to_execute"] = []
                if "classifier" not in executed:
                    executed.append("classifier")
                out["tools_executed"] = executed
                return out

            # ---- 2. 由 max_prob 决定剩余工具列表 ----
            max_prob = float(out.get("classifier_max_prob", 0.0) or 0.0)
            positive = max_prob >= self.positive_threshold
            out["positive_flag"] = positive

            # 无论阴阳，classifier 已被执行；记入 executed（与 execute_node 行为一致，
            # 让 tools_executed 始终反映「计划过 / 跑过 / 跳过」的完整集合）
            if "classifier" not in executed:
                executed.append("classifier")
            out["tools_executed"] = executed

            if positive:
                # 阳性 / 边界：跑剩余 4 件工具（按 DEFAULT_TOOL_NAMES 顺序，跳过 classifier）
                # 不预过滤 registry.has —— 缺工具由 execute_node 走「跳过 + 标记
                # degraded」分支（与降级兜底语义一致）
                out["tools_to_execute"] = list(DEFAULT_TOOL_NAMES[1:])
            else:
                # 阴性：classifier 之外**仍**跑 image_preview（医师也需要看窗宽
                # 窗位预览）；keyqa / finding_draft / grounding 在阴性场景
                # 下不必要
                out["tools_to_execute"] = ["image_preview"]

        return out

    async def _execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """``execute`` 节点：从 ``tools_to_execute`` 取第一件工具跑；标记已完成。"""
        out: Dict[str, Any] = dict(state)

        pending = list(out.get("tools_to_execute") or [])
        executed = list(out.get("tools_executed") or [])

        if not pending:
            # 防御性：条件边理论上不会到此处；万一到了，安静退出
            return out

        next_tool_name = pending[0]
        remaining = pending[1:]

        if not self.registry.has(next_tool_name):
            # 工具未注册 → 跳过、记录原因、不抛（避免阻断主路径）
            logger.warning(
                "execute_node: tool %r not registered for %s; skipping",
                next_tool_name, out.get("study_uid"),
            )
            out["degraded"] = True
            reasons = list(out.get("degradation_reasons") or [])
            reasons.append(f"tool_not_registered:{next_tool_name}")
            out["degradation_reasons"] = reasons
            executed.append(next_tool_name)
        else:
            tool = self.registry.get(next_tool_name)
            try:
                result = await tool.run(out)
                for k, v in (result or {}).items():
                    out[k] = v
                executed.append(next_tool_name)
            except Exception as exc:  # noqa: BLE001 —— 降级兜底
                logger.warning(
                    "execute_node: tool %r failed for %s: %s; degrading",
                    next_tool_name, out.get("study_uid"), exc,
                )
                out["degraded"] = True
                reasons = list(out.get("degradation_reasons") or [])
                reasons.append(f"{next_tool_name}_unreachable:{exc}")
                out["degradation_reasons"] = reasons
                executed.append(next_tool_name)  # 仍记已执行，避免死循环

        out["tools_to_execute"] = remaining
        out["tools_executed"] = executed
        return out

    # ---- 条件边函数 ----

    def _has_tool_calls_after_process(
        self, state: Dict[str, Any]
    ) -> bool:
        """``process → execute / END`` 判定：

        - classifier_probs 已就绪 **且**
        ``tools_to_execute`` 非空 → ``True``（去 execute）；
        - 否则（阴性 + 仅 image_preview、degraded、无可执行工具）→ ``False``（去 END）。
        """
        return _has_state(state, "classifier_probs") and bool(
            state.get("tools_to_execute") or []
        )

    def _has_tool_calls_after_execute(
        self, state: Dict[str, Any]
    ) -> bool:
        """``execute → process / END`` 判定：剩余 ``tools_to_execute`` 非空 → True。"""
        return bool(state.get("tools_to_execute") or [])


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _has_state(state: Dict[str, Any], key: str) -> bool:
    """state[key] 存在且非空（list/dict 检查 len，bytes 检查 len）。"""
    if key not in state:
        return False
    value = state[key]
    if value is None:
        return False
    if isinstance(value, (list, dict, tuple, str, bytes, bytearray)):
        return len(value) > 0
    return True


def build_workflow(
    registry: ToolRegistry,
    *,
    positive_threshold: float = 0.5,
    max_cycles: int = 8,
) -> CxrTriageWorkflow:
    """工厂函数：构造 + 编译 StateGraph 一步到位。"""
    wf = CxrTriageWorkflow(
        registry,
        positive_threshold=positive_threshold,
        max_cycles=max_cycles,
    )
    wf.build_graph()
    return wf


# ---------------------------------------------------------------------------
# 模块导出
# ---------------------------------------------------------------------------


__all__ = [
    "CxrTriageState",
    "CxrTriageWorkflow",
    "DEFAULT_TOOL_NAMES",
    "ToolRegistry",
    "WorkflowError",
    "build_workflow",
]