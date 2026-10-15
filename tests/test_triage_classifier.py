"""tests/test_triage_classifier.py — Task 12 TriageClassifier 18 病理概率单测。

目标（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 + Task 12/13 边界）：

1. **构造期校验**：model_name 非空 / prob_threshold ∈ [0, 1]
2. **常量与对齐**：18 病理枚举与 ``xrv.datasets.default_pathologies`` 顺序一致
3. **18 病理输出**：注入 mock 模型 → 跑通 3 例 fixtures（阳/阴/边界）→ 字段完整
4. **max_prob 计算正确**：与注入概率的最大值一致
5. **present 字段**：prob ≥ prob_threshold 时为 True
6. **rule_max_prob 静态方法**：与 process 节点现有 ``max_prob >= threshold``
   语义一致（Task 13 拆 baseline 对比复用）
7. **normalize_probs_to_dict 静态方法**：压成 ``{name: prob}``
8. **StateGraph 字段契约**：``run()`` 返回 ``{classifier_probs, classifier_max_prob}``
   且与 ``process_node`` 解析键名严格对齐
9. **占位 / 空 / 异常路径**：state["png_bytes"] 缺失/非 bytes 抛 TriageClassifierError；
   mock 模型抛异常时 ``run()`` 不抛（降级语义）—— 注：实际由 process 节点捕获
10. **设备回退**：CUDA 不可用自动回 CPU
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from cxr_triage.engine import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    PATHOLOGY_NAMES,
    POSITIVE_PROB_HEURISTIC,
    TriageClassifier,
    TriageClassifierError,
    TriageClassifierResult,
    build_triage_classifier,
)


# ---------------------------------------------------------------------------
# Mock 模型 / Transform
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 64, height: int = 64, value: int = 128) -> bytes:
    """生成单色 8-bit PNG bytes（用于推理入参）。"""
    img = Image.new("L", (width, height), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockModel:
    """Duck-typed DenseNet 替身；按概率 dict 顺序返回 18 概率 list。"""

    def __init__(self, probs_by_name: dict[str, float], raise_exc: Exception | None = None):
        self._probs_by_name = dict(probs_by_name)
        self._raise = raise_exc
        self.call_count = 0
        self.last_input = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, tensor):
        self.call_count += 1
        self.last_input = tensor
        if self._raise is not None:
            raise self._raise
        # 按 PATHOLOGY_NAMES 顺序返 list
        return [self._probs_by_name.get(name, 0.0) for name in PATHOLOGY_NAMES]


class MockTransform:
    """Identity transform：透传 tensor。"""

    def __init__(self, raise_exc: Exception | None = None):
        self._raise = raise_exc
        self.call_count = 0

    def __call__(self, arr):
        self.call_count += 1
        if self._raise is not None:
            raise self._raise
        return arr


def _make_classifier(probs: dict[str, float], **kwargs) -> TriageClassifier:
    """构造注入 mock 模型的 TriageClassifier，避免真实模型下载/推理。"""
    return TriageClassifier(
        model=MockModel(probs),
        transform=MockTransform(),
        device="cpu",
        **kwargs,
    )


# 18 病理：阳性（气胸主），阴性（全低），边界（心影刚好 0.51）
PROBS_POSITIVE = {"Pneumothorax": 0.85, "Effusion": 0.42, "Cardiomegaly": 0.30}
PROBS_NEGATIVE = {name: 0.05 for name in PATHOLOGY_NAMES}
PROBS_BORDERLINE = {"Cardiomegaly": 0.51, "Pneumothorax": 0.49}


# ---------------------------------------------------------------------------
# 常量层
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_model_name(self):
        assert DEFAULT_MODEL_NAME == "densenet121-res224-all"

    def test_default_device(self):
        assert DEFAULT_DEVICE == "cpu"

    def test_positive_prob_heuristic_default(self):
        assert POSITIVE_PROB_HEURISTIC == 0.5

    def test_pathology_names_count_18(self):
        assert len(PATHOLOGY_NAMES) == 18

    def test_pathology_names_known_set(self):
        expected_subset = {
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
            "Pleural_Thickening", "Pneumonia", "Pneumothorax", "Lung Lesion",
            "Lung Opacity", "Fracture", "Enlarged Cardiomediastinum",
        }
        assert set(PATHOLOGY_NAMES) == expected_subset

    def test_pathology_names_unique(self):
        assert len(set(PATHOLOGY_NAMES)) == 18

    def test_pathology_names_no_spaces_in_label(self):
        # 与 xrv.datasets.default_pathologies 顺序对齐（多词病理带空格）
        assert "Lung Lesion" in PATHOLOGY_NAMES
        assert "Pleural_Thickening" in PATHOLOGY_NAMES


# ---------------------------------------------------------------------------
# 构造期校验
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_attributes(self):
        clf = _make_classifier(PROBS_POSITIVE)
        assert clf.name == "classifier"
        assert clf.model_name == DEFAULT_MODEL_NAME
        assert clf.prob_threshold == POSITIVE_PROB_HEURISTIC

    def test_custom_model_name(self):
        clf = TriageClassifier(
            model=MockModel(PROBS_POSITIVE),
            transform=MockTransform(),
            model_name="densenet121-res224-rsna",
        )
        assert clf.model_name == "densenet121-res224-rsna"

    def test_custom_prob_threshold(self):
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.7)
        assert clf.prob_threshold == 0.7

    def test_invalid_model_name_raises(self):
        with pytest.raises(TriageClassifierError, match="model_name"):
            TriageClassifier(model_name="", model=MockModel(PROBS_POSITIVE))

    def test_invalid_prob_threshold_above_1(self):
        with pytest.raises(TriageClassifierError, match="prob_threshold"):
            _make_classifier(PROBS_POSITIVE, prob_threshold=1.5)

    def test_invalid_prob_threshold_below_0(self):
        with pytest.raises(TriageClassifierError, match="prob_threshold"):
            _make_classifier(PROBS_POSITIVE, prob_threshold=-0.1)

    def test_factory_function(self):
        clf = build_triage_classifier(
            model=MockModel(PROBS_POSITIVE),
            transform=MockTransform(),
        )
        assert isinstance(clf, TriageClassifier)
        assert clf.name == "classifier"

    def test_device_cuda_falls_back_to_cpu(self):
        # macOS 无 CUDA → 应回退
        clf = TriageClassifier(
            model=MockModel(PROBS_POSITIVE),
            transform=MockTransform(),
            device="cuda",
        )
        assert clf.device == "cpu"


# ---------------------------------------------------------------------------
# run() 公开契约
# ---------------------------------------------------------------------------


class TestRunContract:
    @pytest.mark.asyncio
    async def test_run_returns_state_patch(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png, "metadata": {"StudyInstanceUID": "1.2.3"}}
        patch = await clf.run(state)
        assert "classifier_probs" in patch
        assert "classifier_max_prob" in patch
        assert isinstance(patch["classifier_probs"], list)
        assert isinstance(patch["classifier_max_prob"], float)

    @pytest.mark.asyncio
    async def test_run_outputs_18_pathologies(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        assert len(patch["classifier_probs"]) == 18

    @pytest.mark.asyncio
    async def test_run_probabilities_order_matches_pathology_names(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        names_in_order = [p["name"] for p in patch["classifier_probs"]]
        assert names_in_order == list(PATHOLOGY_NAMES)

    @pytest.mark.asyncio
    async def test_run_max_prob_equals_max_in_vector(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # 0.85（Pneumothorax）应为最大
        assert patch["classifier_max_prob"] == pytest.approx(0.85, abs=1e-6)
        # 注入的概率与输出逐项对齐
        for p in patch["classifier_probs"]:
            assert "prob" in p
            assert "present" in p
            assert "name" in p

    @pytest.mark.asyncio
    async def test_run_present_field_for_high_prob(self):
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.5)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["present"] is True  # 0.85 >= 0.5

    @pytest.mark.asyncio
    async def test_run_present_field_for_low_prob(self):
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.5)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # 注入的所有非 Pneumothorax 都 < 0.5 → present=False
        for p in patch["classifier_probs"]:
            if p["name"] != "Pneumothorax":
                assert p["present"] is False

    @pytest.mark.asyncio
    async def test_run_with_custom_prob_threshold(self):
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.9)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # 0.9 阈值下 Pneumothorax 0.85 < 0.9 → present=False
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["present"] is False

    @pytest.mark.asyncio
    async def test_run_png_bytearray_accepted(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": bytearray(png)}
        patch = await clf.run(state)
        assert patch["classifier_max_prob"] == pytest.approx(0.85, abs=1e-6)

    @pytest.mark.asyncio
    async def test_run_state_unchanged_after_call(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png, "metadata": {"a": 1}}
        _ = await clf.run(state)
        # run() 应只返 patch，不修改入参 state
        assert "classifier_probs" not in state
        assert "classifier_max_prob" not in state
        assert state["metadata"] == {"a": 1}


# ---------------------------------------------------------------------------
# 3 例 fixtures（阳 / 阴 / 边界）
# ---------------------------------------------------------------------------


class TestThreeFixtures:
    @pytest.mark.asyncio
    async def test_positive_case_pneumothorax(self):
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # max_prob 0.85 → 阳性
        assert patch["classifier_max_prob"] > POSITIVE_PROB_HEURISTIC
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["prob"] == pytest.approx(0.85, abs=1e-6)
        assert pneumothorax["present"] is True

    @pytest.mark.asyncio
    async def test_negative_case_all_low(self):
        clf = _make_classifier(PROBS_NEGATIVE)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # max_prob 0.05 → 阴性
        assert patch["classifier_max_prob"] == pytest.approx(0.05, abs=1e-6)
        # 全部 present=False
        for p in patch["classifier_probs"]:
            assert p["present"] is False

    @pytest.mark.asyncio
    async def test_borderline_case_cardiomegaly_above_threshold(self):
        clf = _make_classifier(PROBS_BORDERLINE, prob_threshold=0.5)
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        # 0.51 > 0.5 → 边界阳性（max_prob 刚好 0.51）
        assert patch["classifier_max_prob"] == pytest.approx(0.51, abs=1e-6)
        cardio = next(p for p in patch["classifier_probs"] if p["name"] == "Cardiomegaly")
        assert cardio["present"] is True  # 0.51 >= 0.5
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["present"] is False  # 0.49 < 0.5

    @pytest.mark.asyncio
    async def test_borderline_case_with_dcm_fixture(self):
        """用真实 DICOM fixture（阳性）跑通 8-bit PNG → 18 概率。"""
        clf = _make_classifier(PROBS_POSITIVE)
        # 把 DICOM 走 DicomProcessor → PNG bytes（与生产路径一致）
        from cxr_triage.pacs.dicom_processor import build_dicom_processor
        with open("tests/fixtures/sample_cxr_positive.dcm", "rb") as f:
            dcm_bytes = f.read()
        proc = build_dicom_processor()
        result = proc.process(dcm_bytes)
        png_bytes = result.png_bytes
        state = {"png_bytes": png_bytes}
        patch = await clf.run(state)
        assert len(patch["classifier_probs"]) == 18
        assert patch["classifier_max_prob"] == pytest.approx(0.85, abs=1e-6)


# ---------------------------------------------------------------------------
# rule_max_prob / normalize_probs_to_dict 静态方法
# ---------------------------------------------------------------------------


class TestRuleMaxProb:
    def test_max_prob_above_threshold_returns_true(self):
        probs = [{"name": "Pneumothorax", "prob": 0.85, "present": True}]
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_max_prob_below_threshold_returns_false(self):
        probs = [{"name": "Pneumothorax", "prob": 0.10, "present": False}]
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False

    def test_max_prob_at_threshold_returns_true(self):
        probs = [{"name": "Pneumothorax", "prob": 0.5, "present": True}]
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_empty_probs_returns_false(self):
        assert TriageClassifier.rule_max_prob([], threshold=0.5) is False

    def test_invalid_threshold_raises(self):
        with pytest.raises(TriageClassifierError, match="threshold"):
            TriageClassifier.rule_max_prob([{"name": "X", "prob": 0.5}], threshold=1.5)

    def test_invalid_probs_type_raises(self):
        with pytest.raises(TriageClassifierError, match="probs must be"):
            TriageClassifier.rule_max_prob("not a list", threshold=0.5)

    def test_skip_non_dict_items(self):
        # 混入 None / 字符串 / 数字 → 跳过
        probs = [None, "x", 42, {"name": "P", "prob": 0.7}]
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_skip_non_numeric_prob(self):
        probs = [{"name": "P", "prob": "0.9"}, {"name": "Q", "prob": 0.1}]
        # 字符串 prob 跳过 → max=0.1 < 0.5
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False


class TestNormalizeProbsToDict:
    def test_basic(self):
        probs = [
            {"name": "A", "prob": 0.5},
            {"name": "B", "prob": 0.7},
        ]
        out = TriageClassifier.normalize_probs_to_dict(probs)
        assert out == {"A": 0.5, "B": 0.7}

    def test_skip_missing_name(self):
        probs = [{"name": "", "prob": 0.5}, {"prob": 0.7}]
        out = TriageClassifier.normalize_probs_to_dict(probs)
        assert out == {}  # name 空串跳过 / name 缺失跳过

    def test_skip_non_numeric_prob(self):
        probs = [{"name": "A", "prob": "x"}, {"name": "B", "prob": 0.5}]
        out = TriageClassifier.normalize_probs_to_dict(probs)
        assert out == {"B": 0.5}

    def test_skip_non_dict(self):
        probs = [None, "x", {"name": "A", "prob": 0.5}]
        out = TriageClassifier.normalize_probs_to_dict(probs)
        assert out == {"A": 0.5}

    def test_bool_excluded_from_prob(self):
        # bool 是 int 子类 → 必须排除，避免 True/False 误判
        probs = [{"name": "A", "prob": True}, {"name": "B", "prob": 0.5}]
        out = TriageClassifier.normalize_probs_to_dict(probs)
        assert out == {"B": 0.5}


# ---------------------------------------------------------------------------
# 异常 / 降级
# ---------------------------------------------------------------------------


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_run_missing_png_bytes_raises(self):
        clf = _make_classifier(PROBS_POSITIVE)
        with pytest.raises(TriageClassifierError, match="png_bytes"):
            await clf.run({"metadata": {}})

    @pytest.mark.asyncio
    async def test_run_empty_png_bytes_raises(self):
        clf = _make_classifier(PROBS_POSITIVE)
        with pytest.raises(TriageClassifierError, match="png_bytes"):
            await clf.run({"png_bytes": b""})

    @pytest.mark.asyncio
    async def test_run_non_bytes_png_raises(self):
        clf = _make_classifier(PROBS_POSITIVE)
        with pytest.raises(TriageClassifierError, match="png_bytes"):
            await clf.run({"png_bytes": "not bytes"})

    @pytest.mark.asyncio
    async def test_model_raises_during_call_returns_empty(self):
        """模型推理抛异常 → 不抛，降级返空 probs（与 process_node 现有降级语义对齐）。"""
        mock = MockModel({}, raise_exc=RuntimeError("model failure"))
        clf = TriageClassifier(model=mock, transform=MockTransform(), device="cpu")
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        assert patch["classifier_probs"] == []
        assert patch["classifier_max_prob"] == 0.0

    @pytest.mark.asyncio
    async def test_transform_raises_returns_empty(self):
        """Transform 失败 → 走 _infer 降级兜底（异常在外层 try 捕获），返空 probs。

        设计选择：与「模型推理失败」同走降级语义，不让 preprocess 阶段异常
        击穿 StateGraph process 节点；这与 TriageClassifier 不抛的整体风格
        一致（异常由 process_node 捕获并写 degradation_reasons）。
        """
        mock_model = MockModel({})
        mock_transform = MockTransform(raise_exc=ValueError("transform fail"))
        clf = TriageClassifier(model=mock_model, transform=mock_transform, device="cpu")
        png = _make_png_bytes()
        state = {"png_bytes": png}
        patch = await clf.run(state)
        assert patch["classifier_probs"] == []
        assert patch["classifier_max_prob"] == 0.0


# ---------------------------------------------------------------------------
# 模型输出长度不一致 / 钳位
# ---------------------------------------------------------------------------


class TestOutputLengthAndClamping:
    @pytest.mark.asyncio
    async def test_18_pathologies_locked(self):
        """PATHOLOGY_NAMES 顺序锁定；模型输出 18 项必须对齐。"""
        clf = _make_classifier(PROBS_POSITIVE)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        names = [p["name"] for p in patch["classifier_probs"]]
        assert names == list(PATHOLOGY_NAMES)

    @pytest.mark.asyncio
    async def test_probability_clamped_to_unit_interval(self):
        """模型输出偶尔越界（softmax 前 logit）→ 钳到 [0, 1]。"""
        probs = {name: 0.05 for name in PATHOLOGY_NAMES}
        probs["Pneumothorax"] = 1.5  # 越界
        clf = _make_classifier(probs)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["prob"] == 1.0  # 钳到上限

    @pytest.mark.asyncio
    async def test_probability_clamped_to_lower_bound(self):
        probs = {name: 0.05 for name in PATHOLOGY_NAMES}
        probs["Pneumothorax"] = -0.3
        clf = _make_classifier(probs)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        pneumothorax = next(p for p in patch["classifier_probs"] if p["name"] == "Pneumothorax")
        assert pneumothorax["prob"] == 0.0


# ---------------------------------------------------------------------------
# StateGraph 集成
# ---------------------------------------------------------------------------


class TestStateGraphIntegration:
    @pytest.mark.asyncio
    async def test_register_in_registry(self):
        """TriageClassifier 可直接注册到 ToolRegistry（duck-typed 契约）。"""
        from cxr_triage.engine import ToolRegistry
        reg = ToolRegistry()
        clf = _make_classifier(PROBS_POSITIVE)
        reg.register(clf)
        assert "classifier" in reg
        assert reg.get("classifier") is clf

    @pytest.mark.asyncio
    async def test_workflow_with_real_classifier(self):
        """完整 StateGraph：真实 TriageClassifier + 4 件 mock 工具跑通。"""
        from cxr_triage.engine import (
            CxrTriageWorkflow, ToolRegistry, build_workflow,
        )
        reg = ToolRegistry()
        reg.register(_make_classifier(PROBS_POSITIVE))
        # 其余 4 件用 mock 补齐
        class StubTool:
            def __init__(self, name, outputs):
                self.name = name
                self._outputs = outputs
            async def run(self, state):
                return dict(self._outputs)
        reg.register(StubTool("keyqa", {"key_qa_list": [{"q": "心影", "a": "未见明显异常"}]}))
        reg.register(StubTool("finding_draft", {"finding_text": "草稿所见"}))
        reg.register(StubTool("positive_grounding", {"grounding_boxes": []}))
        reg.register(StubTool("image_preview", {"png_preview_b64": "aGVsbG8="}))

        wf = build_workflow(reg, positive_threshold=0.5)
        png = _make_png_bytes()
        result = await wf.run(
            study_uid="1.2.3.4",
            png_bytes=png,
            metadata={"StudyInstanceUID": "1.2.3.4"},
        )
        # classifier_max_prob 来自 TriageClassifier 而非 mock
        assert result["classifier_max_prob"] == pytest.approx(0.85, abs=1e-6)
        assert result["positive_flag"] is True
        assert "classifier" in result["tools_executed"]
        assert "keyqa" in result["tools_executed"]
        assert result["finding_text"] == "草稿所见"
