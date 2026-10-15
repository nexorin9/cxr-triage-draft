"""tests/test_state_workflow.py — Task 11 LangGraph StateGraph 编排器单测。

目标（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法）：

1. ``ToolRegistry`` 注册 / 取用 / 重复名拒绝 / 未注册拒绝。
2. ``CxrTriageWorkflow`` 5 件 mock 工具注册 → happy-path 跑通，状态字段正确。
3. 条件边：
   - ``process → END``（classifier_max_prob < 阈值 → 阴性 → 只出 image_preview）；
   - ``execute → END``（tools_to_execute 空 → 退出）；
4. 降级路径：classifier 抛异常 / 工具未注册 / 工具抛异常 均走 ``degraded=True``。
5. 入参校验：study_uid 空 / png_bytes 空 / metadata 非 dict → WorkflowError。
6. ``build_workflow`` 工厂一步编译。
"""
import asyncio
import pytest

from cxr_triage.engine import (
    DEFAULT_TOOL_NAMES,
    CxrTriageWorkflow,
    ToolRegistry,
    WorkflowError,
    build_workflow,
)


# ---------------------------------------------------------------------------
# 工具 mock 工厂
# ---------------------------------------------------------------------------


class MockTool:
    """duck-typed 工具 stub；``name`` + ``async run(state)``。"""

    def __init__(
        self,
        name: str,
        outputs: dict | None = None,
        raises: Exception | None = None,
    ):
        self.name = name
        self._outputs = dict(outputs or {})
        self._raises = raises
        self.calls: list[dict] = []

    async def run(self, state: dict) -> dict:
        self.calls.append(dict(state))
        if self._raises is not None:
            raise self._raises
        # 返回 outputs 的浅拷，避免 caller 修改影响 mock 自身
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self._outputs.items()}


def _build_full_registry():
    """5 件工具按 spec.md §2 顺序注册，classifier 输出阳性（max_prob=0.85）。"""
    reg = ToolRegistry()
    reg.register(MockTool("classifier", outputs={
        "classifier_probs": [
            {"name": "Pneumothorax", "prob": 0.85, "present": True},
            {"name": "Effusion", "prob": 0.42, "present": False},
        ],
        "classifier_max_prob": 0.85,
    }))
    reg.register(MockTool("keyqa", outputs={
        "key_qa_list": [
            {"q": "心影", "a": "心影未见明显增大"},
            {"q": "纵隔", "a": "纵隔居中"},
        ],
    }))
    reg.register(MockTool("finding_draft", outputs={
        "finding_text": "右下肺野见少量气胸征象（草稿），余肺野清晰。",
        "finding_struct": {"pathologies": [], "key_qa": [], "grounding_boxes": []},
    }))
    reg.register(MockTool("positive_grounding", outputs={
        "grounding_boxes": [
            {"x": 0.12, "y": 0.18, "w": 0.15, "h": 0.10, "label": "右下肺", "prob": 0.81},
        ],
    }))
    reg.register(MockTool("image_preview", outputs={
        "png_preview_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    }))
    return reg


# ---------------------------------------------------------------------------
# ToolRegistry 测试
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = MockTool("classifier")
        reg.register(tool)
        assert reg.get("classifier") is tool
        assert "classifier" in reg
        assert len(reg) == 1
        assert reg.names() == ["classifier"]

    def test_register_missing_name_raises(self):
        reg = ToolRegistry()

        class BadTool:
            async def run(self, state):
                return {}

        with pytest.raises(WorkflowError, match="non-empty .name"):
            reg.register(BadTool())

    def test_register_missing_run_raises(self):
        reg = ToolRegistry()

        class NoRun:
            name = "x"

        with pytest.raises(WorkflowError, match="async .run"):
            reg.register(NoRun())

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(MockTool("a"))
        with pytest.raises(WorkflowError, match="already registered"):
            reg.register(MockTool("a"))

    def test_get_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(WorkflowError, match="not registered"):
            reg.get("nope")

    def test_has_returns_bool(self):
        reg = ToolRegistry()
        assert not reg.has("k")
        reg.register(MockTool("k"))
        assert reg.has("k")


# ---------------------------------------------------------------------------
# 构造 / 入参校验
# ---------------------------------------------------------------------------


class TestWorkflowConstruction:
    def test_must_have_classifier_registered(self):
        reg = ToolRegistry()
        reg.register(MockTool("keyqa"))
        with pytest.raises(WorkflowError, match="'classifier'"):
            CxrTriageWorkflow(reg)

    def test_threshold_out_of_range(self):
        reg = ToolRegistry()
        reg.register(MockTool("classifier"))
        with pytest.raises(WorkflowError, match="positive_threshold"):
            CxrTriageWorkflow(reg, positive_threshold=1.5)
        with pytest.raises(WorkflowError, match="positive_threshold"):
            CxrTriageWorkflow(reg, positive_threshold=-0.1)

    def test_max_cycles_must_be_positive(self):
        reg = ToolRegistry()
        reg.register(MockTool("classifier"))
        with pytest.raises(WorkflowError, match="max_cycles"):
            CxrTriageWorkflow(reg, max_cycles=0)

    def test_registry_must_be_toolregistry(self):
        with pytest.raises(WorkflowError, match="ToolRegistry"):
            CxrTriageWorkflow(object())  # type: ignore[arg-type]

    def test_default_tool_names_order(self):
        # DEFAULT_TOOL_NAMES 顺序锁定（spec.md §2）：classifier 必首位
        assert DEFAULT_TOOL_NAMES[0] == "classifier"
        assert "keyqa" in DEFAULT_TOOL_NAMES
        assert "finding_draft" in DEFAULT_TOOL_NAMES
        assert "positive_grounding" in DEFAULT_TOOL_NAMES
        assert "image_preview" in DEFAULT_TOOL_NAMES


# ---------------------------------------------------------------------------
# Happy path（5 件工具跑通）
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_pipeline_runs_all_five_tools(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)

        result = await wf.run(
            study_uid="1.2.826.0.1.3680043.8.498.TEST",
            png_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
            metadata={"patient_id_hash": "abc123", "modality": "CR"},
        )

        # classifier 输出落到 result
        assert result["study_uid"] == "1.2.826.0.1.3680043.8.498.TEST"
        assert result["classifier_max_prob"] == 0.85
        assert result["positive_flag"] is True
        # 5 件工具全部跑过
        assert sorted(result["tools_executed"]) == sorted(DEFAULT_TOOL_NAMES)
        # 关键产物字段都填上
        assert len(result["classifier_probs"]) == 2
        assert len(result["key_qa_list"]) == 2
        assert "气胸征象" in result["finding_text"]
        assert len(result["grounding_boxes"]) == 1
        assert result["png_preview_b64"].startswith("iVBORw0KGgo")
        # 没降级
        assert result["degraded"] is False
        assert result["degradation_reasons"] == []
        # png_bytes / metadata 透传
        assert result["png_bytes"] == b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        assert result["metadata"]["modality"] == "CR"

    @pytest.mark.asyncio
    async def test_negative_short_circuit_runs_only_image_preview(self):
        """classifier_max_prob < 阈值 → 阴性 → 仅跑 image_preview。"""
        reg = ToolRegistry()
        reg.register(MockTool("classifier", outputs={
            "classifier_probs": [{"name": "Pneumothorax", "prob": 0.10, "present": False}],
            "classifier_max_prob": 0.10,
        }))
        keyqa = MockTool("keyqa", outputs={"key_qa_list": [{"q": "x", "a": "y"}]})
        reg.register(keyqa)
        reg.register(MockTool("finding_draft", outputs={"finding_text": "不应跑"}))
        reg.register(MockTool("positive_grounding", outputs={"grounding_boxes": []}))
        reg.register(MockTool("image_preview", outputs={"png_preview_b64": "iVBOR"}))

        wf = build_workflow(reg, positive_threshold=0.5)
        result = await wf.run(
            study_uid="neg.001",
            png_bytes=b"\x89PNG",
            metadata={},
        )

        assert result["positive_flag"] is False
        assert result["classifier_max_prob"] == 0.10
        # 阴性路径只跑 image_preview；keyqa/finding_draft/positive_grounding 不跑
        # 但 classifier 必跑并记入
        assert result["tools_executed"] == ["classifier", "image_preview"]
        assert keyqa.calls == []
        # finding_text 应是未设置（finding_draft 没跑），不是"不应跑"
        assert "finding_text" not in result or result.get("finding_text") is None


# ---------------------------------------------------------------------------
# 降级 / 失败路径
# ---------------------------------------------------------------------------


class TestDegradation:
    @pytest.mark.asyncio
    async def test_classifier_raises_degrades_without_running_other_tools(self):
        reg = ToolRegistry()
        classifier = MockTool(
            "classifier",
            outputs={"classifier_probs": [], "classifier_max_prob": 0.0},
            raises=RuntimeError("classifier model unavailable"),
        )
        reg.register(classifier)
        keyqa = MockTool("keyqa")
        reg.register(keyqa)
        reg.register(MockTool("finding_draft"))
        reg.register(MockTool("positive_grounding"))
        reg.register(MockTool("image_preview", outputs={"png_preview_b64": "iVBOR"}))

        wf = build_workflow(reg)
        result = await wf.run(
            study_uid="degraded.001",
            png_bytes=b"\x89PNG",
            metadata={},
        )

        assert result["degraded"] is True
        assert result["positive_flag"] is False
        assert result["classifier_max_prob"] == 0.0
        # classifier 失败 → 不进入 execute；但 classifier 自身已被记入 executed
        assert result["tools_executed"] == ["classifier"]
        # 至少一条原因包含 classifier_unreachable
        assert any(
            "classifier_unreachable" in r for r in result["degradation_reasons"]
        )
        # keyqa 不应该被调用
        assert keyqa.calls == []

    @pytest.mark.asyncio
    async def test_classifier_returns_no_probs_degrades(self):
        reg = ToolRegistry()
        reg.register(MockTool("classifier", outputs={"wrong_field": 1}))
        reg.register(MockTool("image_preview", outputs={"png_preview_b64": "iVBOR"}))

        wf = build_workflow(reg)
        result = await wf.run(
            study_uid="bad.classifier",
            png_bytes=b"\x89PNG",
            metadata={},
        )

        assert result["degraded"] is True
        assert "classifier_no_probs_in_result" in result["degradation_reasons"]

    @pytest.mark.asyncio
    async def test_execute_tool_raises_marks_degraded_and_continues(self):
        """execute 节点工具抛异常 → 标记 degraded + 仍记入 executed，避免死循环。"""
        reg = ToolRegistry()
        reg.register(MockTool("classifier", outputs={
            "classifier_probs": [{"name": "Pneumothorax", "prob": 0.85, "present": True}],
            "classifier_max_prob": 0.85,
        }))
        reg.register(MockTool("keyqa", raises=RuntimeError("vqa timeout")))
        reg.register(MockTool("finding_draft", outputs={"finding_text": "草稿（keyqa 失败）"}))
        reg.register(MockTool("positive_grounding", outputs={"grounding_boxes": []}))
        reg.register(MockTool("image_preview", outputs={"png_preview_b64": "iVBOR"}))

        wf = build_workflow(reg)
        result = await wf.run(
            study_uid="partial.001",
            png_bytes=b"\x89PNG",
            metadata={},
        )

        # 5 件工具都被记入 executed（含 keyqa 失败那次），不卡死
        assert sorted(result["tools_executed"]) == sorted(DEFAULT_TOOL_NAMES)
        assert result["degraded"] is True
        assert any(
            "keyqa_unreachable" in r for r in result["degradation_reasons"]
        )
        # keyqa 失败后，finding_draft 仍跑（产物结构按字段名覆盖）
        assert "草稿" in result.get("finding_text", "")
        assert result["png_preview_b64"] == "iVBOR"

    @pytest.mark.asyncio
    async def test_tool_not_registered_skips_silently(self):
        """registry 未注册某件 → execute 节点跳过 + 记 degraded，不阻断主路径。"""
        reg = ToolRegistry()
        reg.register(MockTool("classifier", outputs={
            "classifier_probs": [{"name": "Pneumothorax", "prob": 0.85, "present": True}],
            "classifier_max_prob": 0.85,
        }))
        # 不注册 keyqa：分类器阳性但 registry 缺该工具
        reg.register(MockTool("finding_draft", outputs={"finding_text": "草稿"}))
        reg.register(MockTool("image_preview", outputs={"png_preview_b64": "iVBOR"}))

        wf = build_workflow(reg)
        result = await wf.run(
            study_uid="missing.tool",
            png_bytes=b"\x89PNG",
            metadata={},
        )

        # keyqa 被跳过，finding_draft + image_preview 跑过
        assert "keyqa" in result["tools_executed"]
        assert "finding_draft" in result["tools_executed"]
        assert "image_preview" in result["tools_executed"]
        assert result["degraded"] is True
        assert any(
            "tool_not_registered:keyqa" in r for r in result["degradation_reasons"]
        )


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_study_uid_rejected(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        with pytest.raises(WorkflowError, match="study_uid"):
            await wf.run(study_uid="", png_bytes=b"\x89PNG", metadata={})

    @pytest.mark.asyncio
    async def test_non_string_study_uid_rejected(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        with pytest.raises(WorkflowError, match="study_uid"):
            await wf.run(study_uid=123, png_bytes=b"\x89PNG", metadata={})

    @pytest.mark.asyncio
    async def test_empty_png_bytes_rejected(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        with pytest.raises(WorkflowError, match="png_bytes"):
            await wf.run(study_uid="x", png_bytes=b"", metadata={})

    @pytest.mark.asyncio
    async def test_non_dict_metadata_rejected(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        with pytest.raises(WorkflowError, match="metadata"):
            await wf.run(study_uid="x", png_bytes=b"\x89PNG", metadata="bad")


# ---------------------------------------------------------------------------
# 工厂 / 复用
# ---------------------------------------------------------------------------


class TestBuildWorkflowFactory:
    def test_factory_compiles_immediately(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        # build_graph() 已被工厂调用
        assert wf._compiled is not None

    @pytest.mark.asyncio
    async def test_build_graph_idempotent(self):
        reg = _build_full_registry()
        wf = build_workflow(reg)
        first = wf.build_graph()
        second = wf.build_graph()
        # 同一 registry 重复编译返回同一份 compiled 对象
        assert first is second

    @pytest.mark.asyncio
    async def test_max_cycles_zero_rejected(self):
        reg = _build_full_registry()
        with pytest.raises(WorkflowError, match="max_cycles"):
            build_workflow(reg, max_cycles=0)