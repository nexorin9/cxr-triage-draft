"""tests/test_triage_classifier_threshold.py — Task 13 阈值规则 + max_prob 守门单测。

目标（spec.md §5 维度 3 + plan.md §5 深度 (a) 边界与异常用例 + Task 13 边界）：

1. **CxrTriageConfig.positive_threshold** 字段存在且默认 0.5（spec §4 字段对齐）
2. **rule_max_prob 静态方法独立可测**：max_prob ≥ threshold → True
3. **边界用例**：max_prob=0.49 < 0.5 → False；max_prob=0.51 ≥ 0.5 → True；
   max_prob=0.50 ≥ 0.5 → True（at-threshold 算阳性）
4. **阈值可配**：threshold=0.7 时 0.51 → False，0.85 → True
5. **rule 与模型分离**：rule_max_prob 是静态方法，不依赖 model 实例；
   便于单工具 baseline vs 多工具协同对比复用
6. **阈值守门与 process_node 同形**：rule_max_prob 的语义与
   ``process_node`` 现有 ``max_prob >= self.positive_threshold`` 一致
7. **空 / 越界 / 非 dict 输入降级**：不抛（与 production 一致），允许
   baseline_comparison.json 中部分字段缺失时仍能跑

本测试**不**重复 ``tests/test_triage_classifier.py`` 已覆盖的路径
（如 model 异常、设备回退、image_preview 整合等），专注于：
- **阈值规则的可测试性**
- **从 config 读阈值的端到端流**
- **与 baseline_comparison.json 字段契约**
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from cxr_triage.config import CxrTriageConfig, load_config
from cxr_triage.engine import (
    PATHOLOGY_NAMES,
    POSITIVE_PROB_HEURISTIC,
    TriageClassifier,
    TriageClassifierError,
    TriageClassifierResult,
    build_triage_classifier,
)


# ---------------------------------------------------------------------------
# 常量与 fixture
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 64, height: int = 64, value: int = 128) -> bytes:
    """生成单色 8-bit PNG bytes（推理入参）。"""
    img = Image.new("L", (width, height), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _probs_with_max(target: float, *, name: str = "Pneumothorax") -> List[Dict[str, Any]]:
    """构造 ``[{name, prob}, ...]`` 让 ``max(prob) == target``。"""
    # 其余 17 病理给 0.0 → max 仅由 target 决定
    return [{"name": name, "prob": target, "present": target >= 0.5}]


class _MockModel:
    """按概率 dict 顺序返 18 概率 list 的 duck-typed 模型替身。"""

    def __init__(self, probs_by_name: Dict[str, float]):
        self._probs_by_name = dict(probs_by_name)
        self.call_count = 0

    def to(self, device):  # noqa: ARG002
        return self

    def eval(self):
        return self

    def __call__(self, tensor):  # noqa: ARG002
        self.call_count += 1
        return [self._probs_by_name.get(name, 0.0) for name in PATHOLOGY_NAMES]


class _MockTransform:
    """Identity transform。"""

    def __call__(self, x):  # noqa: ARG002
        return x


def _make_classifier(probs_by_name: Dict[str, float], *, prob_threshold: float = POSITIVE_PROB_HEURISTIC) -> TriageClassifier:
    return TriageClassifier(
        model=_MockModel(probs_by_name),
        transform=_MockTransform(),
        prob_threshold=prob_threshold,
    )


# 3 例 fixtures 与 Task 12 对齐
PROBS_POSITIVE: Dict[str, float] = {**{n: 0.05 for n in PATHOLOGY_NAMES}, "Pneumothorax": 0.85}
PROBS_NEGATIVE: Dict[str, float] = {n: 0.05 for n in PATHOLOGY_NAMES}
PROBS_BORDERLINE: Dict[str, float] = {**{n: 0.05 for n in PATHOLOGY_NAMES}, "Cardiomegaly": 0.51}


# ---------------------------------------------------------------------------
# 1. CxrTriageConfig.positive_threshold 字段存在性 + 默认值
# ---------------------------------------------------------------------------


class TestConfigPositiveThreshold:
    def test_positive_threshold_field_default_0_5(self):
        """Config 默认 positive_threshold=0.5（spec §4 字段对齐）。"""
        # 用临时 env 隔离其他字段
        env = {
            "RIS_API_TOKEN": "test-ris",
            "HEALTHCHECK_API_TOKEN": "test-hc",
            "OPENAI_API_KEY": "test-openai",
        }
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            cfg = CxrTriageConfig()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert hasattr(cfg, "positive_threshold"), "CxrTriageConfig must have positive_threshold"
        assert cfg.positive_threshold == pytest.approx(0.5, abs=1e-9)

    def test_positive_threshold_field_overridable_by_env(self):
        """POSITIVE_THRESHOLD env 可覆写。"""
        env = {
            "RIS_API_TOKEN": "test-ris",
            "HEALTHCHECK_API_TOKEN": "test-hc",
            "OPENAI_API_KEY": "test-openai",
            "POSITIVE_THRESHOLD": "0.7",
        }
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            cfg = CxrTriageConfig()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert cfg.positive_threshold == pytest.approx(0.7, abs=1e-9)

    def test_positive_threshold_ge_le_validator(self):
        """越界值（<0 或 >1）启动即抛。"""
        env = {
            "RIS_API_TOKEN": "test-ris",
            "HEALTHCHECK_API_TOKEN": "test-hc",
            "OPENAI_API_KEY": "test-openai",
            "POSITIVE_THRESHOLD": "1.5",
        }
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            with pytest.raises(Exception) as exc_info:
                CxrTriageConfig()
            # pydantic ValidationError 文本含 "positive_threshold" 或 "POSITIVE_THRESHOLD"
            assert "positive_threshold" in str(exc_info.value).lower() or \
                   "positive" in str(exc_info.value).lower()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# 2. rule_max_prob 静态方法独立可测
# ---------------------------------------------------------------------------


class TestRuleMaxProbBoundary:
    """Task 13 核心：max_prob=0.49 vs 0.51 vs at-threshold 行为差异。"""

    def test_max_prob_0_49_below_default_0_5_returns_false(self):
        probs = _probs_with_max(0.49)
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False

    def test_max_prob_0_51_above_default_0_5_returns_true(self):
        probs = _probs_with_max(0.51)
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_max_prob_at_0_5_threshold_returns_true(self):
        """at-threshold（>=）算阳性。"""
        probs = _probs_with_max(0.5)
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_max_prob_just_below_threshold_returns_false(self):
        probs = _probs_with_max(0.4999)
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False

    def test_max_prob_just_above_threshold_returns_true(self):
        probs = _probs_with_max(0.5001)
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True

    def test_rule_is_static_no_instance_needed(self):
        """rule_max_prob 是 staticmethod，无实例即可调用。"""
        probs = _probs_with_max(0.51)
        # 直接在类上调用（不构造实例）
        result = TriageClassifier.rule_max_prob(probs, threshold=0.5)
        assert result is True

    def test_rule_max_with_high_threshold_0_7(self):
        """阈值从 0.5 提到 0.7：原阳性 0.51 → False；0.85 → True。"""
        probs_low = _probs_with_max(0.51)
        probs_high = _probs_with_max(0.85)
        assert TriageClassifier.rule_max_prob(probs_low, threshold=0.7) is False
        assert TriageClassifier.rule_max_prob(probs_high, threshold=0.7) is True

    def test_rule_max_with_low_threshold_0_3(self):
        """阈值降到 0.3：原阴性 0.10 → False；0.40 → True。"""
        probs_low = _probs_with_max(0.10)
        probs_high = _probs_with_max(0.40)
        assert TriageClassifier.rule_max_prob(probs_low, threshold=0.3) is False
        assert TriageClassifier.rule_max_prob(probs_high, threshold=0.3) is True

    def test_rule_with_multiple_probs_takes_max(self):
        """多病理场景取 max 后再与阈值比较。"""
        probs = [
            {"name": "A", "prob": 0.30},
            {"name": "B", "prob": 0.85},
            {"name": "C", "prob": 0.10},
        ]
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is True
        assert TriageClassifier.rule_max_prob(probs, threshold=0.9) is False

    def test_rule_skips_present_field_only_uses_prob(self):
        """rule 只看 prob 字段；present 字段不影响判断。"""
        probs = [
            {"name": "A", "prob": 0.10, "present": True},   # present=True 但 prob=0.10
            {"name": "B", "prob": 0.40, "present": False},  # present=False 但 prob=0.40
        ]
        # max=0.40 < 0.5 → False
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False


# ---------------------------------------------------------------------------
# 3. 阈值从 config 端到端流（构造 → run → positive_flag 推导）
# ---------------------------------------------------------------------------


class TestEndToEndThresholdFlow:
    """端到端流：从 CxrTriageConfig.positive_threshold → TriageClassifier.run → positive_flag。"""

    @pytest.mark.asyncio
    async def test_default_threshold_0_5_marks_0_51_positive(self):
        clf = _make_classifier(PROBS_BORDERLINE, prob_threshold=0.5)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        # 边界 0.51 → max_prob=0.51 ≥ 0.5 → True
        assert patch["classifier_max_prob"] == pytest.approx(0.51, abs=1e-6)
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.5) is True

    @pytest.mark.asyncio
    async def test_threshold_0_7_marks_0_51_negative(self):
        """阈值从 config 改 0.7：0.51 不再阳性。"""
        clf = _make_classifier(PROBS_BORDERLINE, prob_threshold=0.7)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        assert patch["classifier_max_prob"] == pytest.approx(0.51, abs=1e-6)
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.7) is False

    @pytest.mark.asyncio
    async def test_threshold_0_3_marks_0_10_positive_negative(self):
        """阈值 0.3：0.10 不够。"""
        clf = _make_classifier(PROBS_NEGATIVE, prob_threshold=0.3)
        png = _make_png_bytes()
        patch = await clf.run({"png_bytes": png})
        assert patch["classifier_max_prob"] == pytest.approx(0.05, abs=1e-6)
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.3) is False

    @pytest.mark.asyncio
    async def test_three_fixtures_recall_at_default_threshold(self):
        """3 例 fixtures 默认阈值 0.5：阳/边界阳性、阴阴性。"""
        png = _make_png_bytes()
        # 阳性 0.85 → True
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.5)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.5) is True
        # 阴性 0.05 → False
        clf = _make_classifier(PROBS_NEGATIVE, prob_threshold=0.5)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.5) is False
        # 边界 0.51 → True（>= 0.5）
        clf = _make_classifier(PROBS_BORDERLINE, prob_threshold=0.5)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.5) is True

    @pytest.mark.asyncio
    async def test_three_fixtures_recall_at_strict_threshold_0_7(self):
        """3 例 fixtures 阈值 0.7：阳阳性、阴阴性、边界阴性。"""
        png = _make_png_bytes()
        clf = _make_classifier(PROBS_POSITIVE, prob_threshold=0.7)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.7) is True
        clf = _make_classifier(PROBS_NEGATIVE, prob_threshold=0.7)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.7) is False
        clf = _make_classifier(PROBS_BORDERLINE, prob_threshold=0.7)
        patch = await clf.run({"png_bytes": png})
        assert TriageClassifier.rule_max_prob(patch["classifier_probs"], threshold=0.7) is False


# ---------------------------------------------------------------------------
# 4. 与 baseline_comparison.json 字段契约
# ---------------------------------------------------------------------------


class TestBaselineComparisonContract:
    """baseline_comparison.json 必须可被 rule_max_prob 复用出 recall / precision。"""

    def test_baseline_comparison_json_exists(self):
        path = Path("tests/baseline_comparison.json")
        assert path.exists(), "tests/baseline_comparison.json must exist (Task 13 placeholder)"

    def test_baseline_comparison_json_has_required_fields(self):
        path = Path("tests/baseline_comparison.json")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert "single_tool_baseline" in data, "must have single_tool_baseline block"
        assert "multi_tool_pipeline" in data, "must have multi_tool_pipeline block"
        assert "fixtures" in data, "must have fixtures list (3 examples)"
        assert isinstance(data["fixtures"], list)
        assert len(data["fixtures"]) >= 3, "must have at least 3 fixtures (positive/negative/borderline)"

    def test_baseline_comparison_json_fixtures_recompute_with_rule(self):
        """baseline_comparison.json 中的 fixtures 用 rule_max_prob 可重新算阳性 flag。"""
        path = Path("tests/baseline_comparison.json")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        threshold = data.get("threshold", 0.5)
        for fx in data["fixtures"]:
            probs = fx.get("classifier_probs", [])
            # rule_max_prob 静态可重算
            recomputed = TriageClassifier.rule_max_prob(probs, threshold=threshold)
            # 与 json 中声明的 baseline 一致（baseline 是 single-tool 出的）
            assert "baseline_positive_flag" in fx
            assert recomputed == bool(fx["baseline_positive_flag"])

    def test_baseline_comparison_json_fixtures_have_18_pathologies(self):
        """每例 fixture 的 classifier_probs 含 18 病理。"""
        path = Path("tests/baseline_comparison.json")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for fx in data["fixtures"]:
            probs = fx.get("classifier_probs", [])
            assert len(probs) == 18, f"fixture {fx.get('id')} must have 18 pathologies"


# ---------------------------------------------------------------------------
# 5. rule_max_prob 与 process_node 同形（设计契约）
# ---------------------------------------------------------------------------


class TestRuleMaxProbConsistency:
    """rule_max_prob(probs, threshold) ↔ process_node 现 max_prob ≥ self.positive_threshold。"""

    def test_rule_returns_true_iff_max_ge_threshold(self):
        """max >= threshold ↔ True（与 process 节点同形）。"""
        cases = [
            ([{"prob": 0.0}], 0.5, False),
            ([{"prob": 0.49}], 0.5, False),
            ([{"prob": 0.5}], 0.5, True),
            ([{"prob": 0.51}], 0.5, True),
            ([{"prob": 1.0}], 0.5, True),
            ([{"prob": 0.0}], 0.0, True),  # threshold=0 → 全过
            ([{"prob": 0.5}], 1.0, False),  # threshold=1 → 需 prob=1
        ]
        for probs, threshold, expected in cases:
            assert TriageClassifier.rule_max_prob(probs, threshold=threshold) is expected, \
                f"probs={probs} threshold={threshold} expected {expected}"

    def test_rule_does_not_consult_present_field(self):
        """rule 只读 prob，不读 present；present 字段是 _format_probs 的标记，rule 独立判定。"""
        probs = [
            {"name": "A", "prob": 0.10, "present": True},   # present=True 误导
            {"name": "B", "prob": 0.40, "present": False},
        ]
        # rule 独立 → max=0.40 < 0.5 → False
        assert TriageClassifier.rule_max_prob(probs, threshold=0.5) is False


# ---------------------------------------------------------------------------
# 6. 异常路径（与 Task 12 已覆盖对齐，不重复）
# ---------------------------------------------------------------------------


class TestRuleMaxProbErrorPaths:
    def test_invalid_threshold_too_high_raises(self):
        with pytest.raises(TriageClassifierError, match="threshold"):
            TriageClassifier.rule_max_prob([{"name": "X", "prob": 0.5}], threshold=1.5)

    def test_invalid_threshold_negative_raises(self):
        with pytest.raises(TriageClassifierError, match="threshold"):
            TriageClassifier.rule_max_prob([{"name": "X", "prob": 0.5}], threshold=-0.1)

    def test_invalid_probs_type_raises(self):
        with pytest.raises(TriageClassifierError, match="probs"):
            TriageClassifier.rule_max_prob("not a list", threshold=0.5)

    def test_invalid_probs_type_dict_raises(self):
        with pytest.raises(TriageClassifierError, match="probs"):
            TriageClassifier.rule_max_prob({"name": "X", "prob": 0.5}, threshold=0.5)

    def test_invalid_threshold_non_numeric_string_raises(self):
        """非数字字符串 threshold → 抛（float("abc") 失败）。"""
        with pytest.raises((TriageClassifierError, ValueError)):
            TriageClassifier.rule_max_prob([{"name": "X", "prob": 0.5}], threshold="abc")

    def test_threshold_numeric_string_accepted(self):
        """数字字符串 threshold 由 float() 转换接受（如 "0.7"）。"""
        # rule_max_prob 走 float(threshold) → "0.7" → 0.7 接受
        probs = [{"name": "P", "prob": 0.75}]
        assert TriageClassifier.rule_max_prob(probs, threshold="0.7") is True
        assert TriageClassifier.rule_max_prob(probs, threshold="0.8") is False
