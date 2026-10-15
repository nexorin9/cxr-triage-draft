"""tests/test_positive_grounding.py — Task 16 PositiveGrounding 阳性定位框工具单测。

目标（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 + Task 16 细则）：

1. **构造期校验**：
   - localization_map 类型错 / 键空 / 值空 → PositiveGroundingError
   - bbox_by_region 类型错 / 中心越界 / 尺寸越界 / 非 4-tuple / 非数值 → fail-fast
   - fallback_bbox 类型错 / 越界 → fail-fast
   - prob_heuristic 越界 / 非数值 → fail-fast
   - max_boxes 非 int / ≤ 0 → fail-fast

2. **常量层契约**：
   - DEFAULT_BBOX_BY_REGION 至少含「心影 / 纵隔 / 双肺野 / 右下肺野 / 右侧胸腔」
   - DEFAULT_FALLBACK_BBOX 中心 (0.5, 0.5)、宽高 (0.4, 0.4)
   - POSITIVE_GROUNDING_PROB_HEURISTIC = 0.5
   - MAX_GROUNDING_BOXES = 6
   - PATHOLOGY_NAMES 18 项顺序锁定

3. **bbox 生成契约**：
   - 阳性 pathologies（prob ≥ 阈值）→ grounding_boxes 非空
   - 阴性 / 空列表 → grounding_boxes = []
   - 坐标归一化到 [0, 1] + x+w ≤ 1 + y+h ≤ 1
   - label 与 18 病理对位 + prob 字段真实值
   - 跨 finding_draft.DEFAULT_LOCALIZATION_BY_PATHOLOGY 18 项全覆盖

4. **优先级排序**：
   - 多阳性时按 prob DESC 排序输出
   - 超过 MAX_GROUNDING_BOXES 时截断 + truncated=True

5. **降级路径**：
   - classifier_probs 缺 / 非 list / 全空 → 空 boxes + degraded=True +
     degradation_reasons 含 positive_grounding_*
   - 未知 label → 跳过 + audit / 排除
   - 定位文案未命中 bbox 表 → fallback + fallback_region_used 记录
   - 不抛任何异常（入参容错）

6. **确定性**：同输入多次调用 → 同一输出（飞检可重复）

7. **StateGraph 集成**：
   - ToolRegistry 注册（name="positive_grounding"）
   - 5 件 mock 工具全跑通 → grounding_boxes 出现在最终 state
   - 阴性路径不调 positive_grounding（仅 classifier + image_preview）

8. **to_finding_struct 静态工具**：
   - 坐标钳位 [0, 1] + x+w ≤ 1
   - 非 dict / 字段缺失 / 非数值 → 跳过
   - prob 缺省时省略该字段
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional

import pytest

from cxr_triage.engine.finding_draft import DEFAULT_LOCALIZATION_BY_PATHOLOGY
from cxr_triage.engine.positive_grounding import (
    DEFAULT_BBOX_BY_REGION,
    DEFAULT_FALLBACK_BBOX,
    MAX_GROUNDING_BOXES,
    PATHOLOGY_NAMES,
    POSITIVE_GROUNDING_PROB_HEURISTIC,
    PositiveGrounding,
    PositiveGroundingError,
    PositiveGroundingResult,
    build_positive_grounding,
)
from cxr_triage.engine.workflow import DEFAULT_TOOL_NAMES, ToolRegistry, build_workflow


# ---------------------------------------------------------------------------
# 3 例 fixtures — 与 test_finding_draft 的 fixture 对位
# ---------------------------------------------------------------------------


# 阳性：右下肺气胸（Pneumothorax=0.85）→ 期望 1 个 bbox
PROBS_POSITIVE: List[Dict[str, Any]] = [
    {"name": "Pneumothorax", "prob": 0.85, "present": True},
    {"name": "Effusion", "prob": 0.42, "present": False},
    {"name": "Cardiomegaly", "prob": 0.30, "present": False},
]

# 阴性：所有 prob < 0.5 → 期望空 boxes
PROBS_NEGATIVE: List[Dict[str, Any]] = [
    {"name": "Atelectasis", "prob": 0.10, "present": False},
    {"name": "Cardiomegaly", "prob": 0.20, "present": False},
    {"name": "Effusion", "prob": 0.15, "present": False},
]

# 边界 + 多阳性：Pneumothorax=0.55 + Cardiomegaly=0.51 → 期望 2 个 bbox
PROBS_BORDERLINE_MULTI: List[Dict[str, Any]] = [
    {"name": "Pneumothorax", "prob": 0.55, "present": True},
    {"name": "Cardiomegaly", "prob": 0.51, "present": True},
    {"name": "Effusion", "prob": 0.30, "present": False},
]

# 全阳性（≥ 0.5）病理清单（用于截断测试，7 条 → MAX_GROUNDING_BOXES=6 截断）
PROBS_MANY_POSITIVE: List[Dict[str, Any]] = [
    {"name": "Pneumothorax", "prob": 0.95, "present": True},
    {"name": "Effusion", "prob": 0.90, "present": True},
    {"name": "Cardiomegaly", "prob": 0.85, "present": True},
    {"name": "Nodule", "prob": 0.80, "present": True},
    {"name": "Mass", "prob": 0.75, "present": True},
    {"name": "Consolidation", "prob": 0.70, "present": True},
    {"name": "Pneumonia", "prob": 0.65, "present": True},
]


def _state_with_probs(
    probs: List[Dict[str, Any]],
    *,
    study_uid: str = "1.2.826.0.1.3680043.8.498.<EXAMPLE_T16>",
) -> Dict[str, Any]:
    """构造给 PositiveGrounding.run() 的最小 state。"""
    # 仅取 dict 且 prob 是数值的项算 max；非 dict / 非数值 / 非 prob 键的项
    # 会被 PositiveGrounding 静默过滤，不应让 helper 崩
    valid_probs = [
        p for p in probs
        if isinstance(p, dict)
        and isinstance(p.get("prob"), (int, float))
        and not isinstance(p.get("prob"), bool)
    ]
    max_prob = max((p["prob"] for p in valid_probs), default=0.0)
    return {
        "study_uid": study_uid,
        "png_bytes": b"\x89PNG\r\n\x1a\n",  # 最小 PNG magic，run() 不读
        "metadata": {},
        "classifier_probs": probs,
        "classifier_max_prob": max_prob,
        "positive_flag": True,
    }


# ---------------------------------------------------------------------------
# 1. 构造期校验
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_default_construction_succeeds(self) -> None:
        tool = PositiveGrounding()
        assert tool.name == "positive_grounding"
        assert tool.localization_map == DEFAULT_LOCALIZATION_BY_PATHOLOGY
        assert tool.bbox_by_region == DEFAULT_BBOX_BY_REGION
        assert tool.fallback_bbox == DEFAULT_FALLBACK_BBOX
        assert tool.prob_heuristic == 0.5
        assert tool.max_boxes == MAX_GROUNDING_BOXES == 6

    def test_localization_map_must_be_dict(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(localization_map=["not", "a", "dict"])  # type: ignore[arg-type]

    def test_localization_map_empty_key_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(localization_map={"": "value"})

    def test_localization_map_empty_value_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(localization_map={"Pneumothorax": ""})

    def test_bbox_by_region_must_be_dict(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region=[("a", "b", "c", "d")])  # type: ignore[arg-type]

    def test_bbox_by_region_wrong_length_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region={"心影": (0.4, 0.6, 0.2)})  # type: ignore[dict-item]

    def test_bbox_by_region_center_out_of_range_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region={"心影": (1.5, 0.5, 0.2, 0.2)})

    def test_bbox_by_region_size_zero_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region={"心影": (0.4, 0.5, 0.0, 0.2)})

    def test_bbox_by_region_size_over_one_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region={"心影": (0.4, 0.5, 1.5, 0.2)})

    def test_bbox_by_region_non_numeric_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(bbox_by_region={"心影": ("a", 0.5, 0.2, 0.2)})  # type: ignore[dict-item]

    def test_fallback_bbox_wrong_length_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(fallback_bbox=(0.5, 0.5, 0.4))  # type: ignore[arg-type]

    def test_fallback_bbox_non_numeric_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(fallback_bbox=(0.5, 0.5, 0.4, "0.4"))  # type: ignore[arg-type]

    def test_fallback_bbox_center_out_of_range_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(fallback_bbox=(1.5, 0.5, 0.4, 0.4))

    def test_fallback_bbox_size_zero_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(fallback_bbox=(0.5, 0.5, 0.4, 0.0))

    def test_prob_heuristic_out_of_range_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(prob_heuristic=1.5)
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(prob_heuristic=-0.1)

    def test_prob_heuristic_non_numeric_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(prob_heuristic="abc")  # type: ignore[arg-type]

    def test_prob_heuristic_numeric_string_accepted(self) -> None:
        """与 finding_draft 一致：数字字符串「0.5」被接受（避免配置 yaml/toml 误拒）。"""
        tool = PositiveGrounding(prob_heuristic="0.7")  # type: ignore[arg-type]
        assert tool.prob_heuristic == 0.7

    def test_max_boxes_non_int_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(max_boxes="6")  # type: ignore[arg-type]

    def test_max_boxes_zero_rejected(self) -> None:
        with pytest.raises(PositiveGroundingError):
            PositiveGrounding(max_boxes=0)

    def test_build_factory_returns_instance(self) -> None:
        tool = build_positive_grounding()
        assert isinstance(tool, PositiveGrounding)
        assert tool.name == "positive_grounding"


# ---------------------------------------------------------------------------
# 2. 常量层契约
# ---------------------------------------------------------------------------


class TestConstantsContract:
    def test_default_bbox_by_region_covers_required_regions(self) -> None:
        """体检胸片常用 5 个解剖区必须存在。"""
        for region in ("心影", "纵隔", "双肺野", "右下肺野", "右侧胸腔"):
            assert region in DEFAULT_BBOX_BY_REGION, f"missing region {region!r}"

    def test_default_bbox_by_region_all_in_unit_range(self) -> None:
        """所有 region 的 bbox 必须 ``(cx, cy, w, h) ∈ [0, 1]`` 且 w/h > 0。"""
        for region, bbox in DEFAULT_BBOX_BY_REGION.items():
            cx, cy, w, h = bbox
            assert 0.0 <= cx <= 1.0, f"{region}: cx {cx} out of [0, 1]"
            assert 0.0 <= cy <= 1.0, f"{region}: cy {cy} out of [0, 1]"
            assert 0.0 < w <= 1.0, f"{region}: w {w} out of (0, 1]"
            assert 0.0 < h <= 1.0, f"{region}: h {h} out of (0, 1]"

    def test_default_fallback_bbox_centered(self) -> None:
        cx, cy, w, h = DEFAULT_FALLBACK_BBOX
        assert (cx, cy, w, h) == (0.5, 0.5, 0.4, 0.4)

    def test_positive_grounding_prob_heuristic_is_half(self) -> None:
        assert POSITIVE_GROUNDING_PROB_HEURISTIC == 0.5

    def test_max_grounding_boxes_is_six(self) -> None:
        assert MAX_GROUNDING_BOXES == 6

    def test_pathology_names_eighteen_unique(self) -> None:
        assert len(PATHOLOGY_NAMES) == 18
        assert len(set(PATHOLOGY_NAMES)) == 18, "duplicate pathology names"

    def test_localization_map_covers_all_pathologies(self) -> None:
        """finding_draft 的 DEFAULT_LOCALIZATION_BY_PATHOLOGY 必须覆盖 18 病理。"""
        for p in PATHOLOGY_NAMES:
            assert p in DEFAULT_LOCALIZATION_BY_PATHOLOGY, (
                f"pathology {p!r} missing from localization_map"
            )


# ---------------------------------------------------------------------------
# 3. bbox 生成契约
# ---------------------------------------------------------------------------


class TestBoundingBoxGeneration:
    @pytest.mark.asyncio
    async def test_positive_case_emits_one_bbox(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        assert "grounding_boxes" in patch
        assert isinstance(patch["grounding_boxes"], list)
        assert len(patch["grounding_boxes"]) == 1

    @pytest.mark.asyncio
    async def test_positive_case_bbox_coords_normalized(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        box = patch["grounding_boxes"][0]
        for key in ("x", "y", "w", "h"):
            v = box[key]
            assert 0.0 <= v <= 1.0, f"{key}={v} out of [0, 1]"
        assert box["x"] + box["w"] <= 1.0 + 1e-9
        assert box["y"] + box["h"] <= 1.0 + 1e-9

    @pytest.mark.asyncio
    async def test_positive_case_bbox_label_and_prob(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        box = patch["grounding_boxes"][0]
        assert box["label"] == "Pneumothorax"
        assert abs(box["prob"] - 0.85) < 1e-9

    @pytest.mark.asyncio
    async def test_negative_case_empty_boxes(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_NEGATIVE))
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True
        assert any(
            r.startswith("positive_grounding_")
            for r in patch["degradation_reasons"]
        )

    @pytest.mark.asyncio
    async def test_pneumothorax_bbox_in_right_lower_zone(self) -> None:
        """Pneumothorax 默认定位文案「右下肺野」，bbox 中心应在中下偏右。"""
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        box = patch["grounding_boxes"][0]
        # 右下肺野 bbox (cx=0.70, cy=0.68, w=0.24, h=0.24) → (x=0.58, y=0.56)
        assert 0.55 <= box["x"] <= 0.62
        assert 0.55 <= box["y"] <= 0.62

    @pytest.mark.asyncio
    async def test_cardiomegaly_bbox_in_cardiac_zone(self) -> None:
        """Cardiomegaly 默认定位文案「心影」，bbox 应偏左下。"""
        tool = PositiveGrounding()
        state = _state_with_probs([{"name": "Cardiomegaly", "prob": 0.85, "present": True}])
        patch = await tool.run(state)
        box = patch["grounding_boxes"][0]
        # 心影 (cx=0.40, cy=0.65, w=0.22, h=0.24) → (x=0.29, y=0.53)
        assert box["x"] < 0.5, "心影 bbox 应偏左"
        assert box["y"] > 0.5, "心影 bbox 应偏下"

    @pytest.mark.asyncio
    async def test_pneumonia_bbox_in_both_lung_fields(self) -> None:
        """Pneumonia 默认定位文案「双肺野」，bbox 应居中且宽。"""
        tool = PositiveGrounding()
        state = _state_with_probs([{"name": "Pneumonia", "prob": 0.85, "present": True}])
        patch = await tool.run(state)
        box = patch["grounding_boxes"][0]
        # 双肺野 (cx=0.5, cy=0.55, w=0.6, h=0.5) → (x=0.2, y=0.3)
        assert box["w"] >= 0.5
        assert box["h"] >= 0.4

    @pytest.mark.asyncio
    async def test_borderline_multi_emits_two_boxes(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_BORDERLINE_MULTI))
        assert len(patch["grounding_boxes"]) == 2
        labels = sorted([b["label"] for b in patch["grounding_boxes"]])
        assert labels == ["Cardiomegaly", "Pneumothorax"]

    @pytest.mark.asyncio
    async def test_all_eighteen_pathologies_have_a_bbox(self) -> None:
        """18 病理全跑通；每条都至少有一个 bbox。"""
        tool = PositiveGrounding()
        for pathology in PATHOLOGY_NAMES:
            state = _state_with_probs(
                [{"name": pathology, "prob": 0.95, "present": True}]
            )
            patch = await tool.run(state)
            assert len(patch["grounding_boxes"]) >= 1, (
                f"pathology {pathology!r} produced no bbox"
            )
            box = patch["grounding_boxes"][0]
            assert box["label"] == pathology


# ---------------------------------------------------------------------------
# 4. 优先级排序 + 截断
# ---------------------------------------------------------------------------


class TestPriorityAndTruncation:
    @pytest.mark.asyncio
    async def test_boxes_sorted_by_prob_desc(self) -> None:
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_BORDERLINE_MULTI))
        boxes = patch["grounding_boxes"]
        probs = [b["prob"] for b in boxes]
        assert probs == sorted(probs, reverse=True), (
            f"boxes not sorted by prob DESC: {probs}"
        )

    @pytest.mark.asyncio
    async def test_truncation_when_exceed_max_boxes(self) -> None:
        """7 条全阳性 + 默认 max_boxes=6 → 截断到 6 条 + truncated=True。"""
        tool = PositiveGrounding()  # max_boxes=6
        patch = await tool.run(_state_with_probs(PROBS_MANY_POSITIVE))
        assert len(patch["grounding_boxes"]) == 6
        meta = patch["_grounding_meta"]
        assert meta["truncated"] is True
        assert meta["box_count"] == 6

    @pytest.mark.asyncio
    async def test_truncation_keeps_highest_probs(self) -> None:
        """截断时保留 prob 最高的 6 条（Pneumonia=0.65 应被丢弃）。"""
        tool = PositiveGrounding()
        patch = await tool.run(_state_with_probs(PROBS_MANY_POSITIVE))
        kept_labels = [b["label"] for b in patch["grounding_boxes"]]
        assert "Pneumonia" not in kept_labels, "Pneumonia=0.65 应被截断"
        assert "Pneumothorax" in kept_labels, "Pneumothorax=0.95 应保留"
        assert "Effusion" in kept_labels, "Effusion=0.90 应保留"

    @pytest.mark.asyncio
    async def test_max_boxes_override(self) -> None:
        tool = PositiveGrounding(max_boxes=3)
        patch = await tool.run(_state_with_probs(PROBS_MANY_POSITIVE))
        assert len(patch["grounding_boxes"]) == 3


# ---------------------------------------------------------------------------
# 5. 降级路径
# ---------------------------------------------------------------------------


class TestDegradationPaths:
    @pytest.mark.asyncio
    async def test_missing_classifier_probs(self) -> None:
        tool = PositiveGrounding()
        state = _state_with_probs([])
        patch = await tool.run(state)
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True
        assert "positive_grounding_no_positive_pathologies" in (
            patch["degradation_reasons"]
        )

    @pytest.mark.asyncio
    async def test_classifier_probs_not_a_list(self) -> None:
        tool = PositiveGrounding()
        state = _state_with_probs(PROBS_POSITIVE)  # type: ignore[arg-type]
        state["classifier_probs"] = "not-a-list"
        patch = await tool.run(state)
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True
        assert "positive_grounding_classifier_probs_missing_or_invalid" in (
            patch["degradation_reasons"]
        )

    @pytest.mark.asyncio
    async def test_classifier_probs_contains_non_dict_items(self) -> None:
        tool = PositiveGrounding()
        state = _state_with_probs([
            {"name": "Pneumothorax", "prob": 0.85, "present": True},
            "bad-string-item",
            None,
            {"name": "NoName", "prob": "non-numeric"},
            {"no_name_key": "Pneumothorax", "prob": 0.99},
        ])
        patch = await tool.run(state)
        # 唯一有效项是 Pneumothorax
        assert len(patch["grounding_boxes"]) == 1
        assert patch["grounding_boxes"][0]["label"] == "Pneumothorax"
        assert patch["degraded"] is False

    @pytest.mark.asyncio
    async def test_unknown_label_filtered(self) -> None:
        """未知 label（不在 18 病理枚举）应被过滤；不计入 boxes。"""
        tool = PositiveGrounding()
        state = _state_with_probs([
            {"name": "Pneumothorax", "prob": 0.85, "present": True},
            {"name": "Unknown_Pathology_X", "prob": 0.99, "present": True},
        ])
        patch = await tool.run(state)
        labels = [b["label"] for b in patch["grounding_boxes"]]
        assert "Pneumothorax" in labels
        assert "Unknown_Pathology_X" not in labels

    @pytest.mark.asyncio
    async def test_fallback_used_when_region_not_in_bbox_table(self) -> None:
        """定位文案不在 bbox 表里 → 走 DEFAULT_FALLBACK_BBOX + 记录。"""
        # 自定义一个全新 region 不在 DEFAULT_BBOX_BY_REGION 里
        custom_localization = {"Pneumothorax": "未知区域TEST"}
        custom_bbox_table = {}  # 让「未知区域TEST」未命中
        tool = PositiveGrounding(
            localization_map=custom_localization,
            bbox_by_region=custom_bbox_table,
        )
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        # 还是应该有一个 bbox（fallback），但 source=fallback
        assert len(patch["grounding_boxes"]) == 1
        meta = patch["_grounding_meta"]
        assert meta["source"] == "fallback"
        assert "未知区域TEST" in meta["fallback_region_used"]

    @pytest.mark.asyncio
    async def test_unknown_label_only_yields_empty(self) -> None:
        """所有 label 都未知 → 空 boxes + degraded=True。"""
        tool = PositiveGrounding()
        state = _state_with_probs([
            {"name": "Made_Up_Pathology", "prob": 0.85, "present": True},
        ])
        patch = await tool.run(state)
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True
        assert "positive_grounding_no_known_pathology_labels" in (
            patch["degradation_reasons"]
        )

    @pytest.mark.asyncio
    async def test_label_without_localization_skipped(self) -> None:
        """label 在 PATHOLOGY_NAMES 但 localization_map 没有 → 跳过。"""
        # 自定义 localization_map，去掉 Pneumothorax
        custom = {k: v for k, v in DEFAULT_LOCALIZATION_BY_PATHOLOGY.items()
                  if k != "Pneumothorax"}
        tool = PositiveGrounding(localization_map=custom)
        patch = await tool.run(_state_with_probs(PROBS_POSITIVE))
        # Pneumothorax 在 PATHOLOGY_NAMES 但 localization_map 缺 → 跳过
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True
        assert "positive_grounding_no_region_match_for_positive_pathologies" in (
            patch["degradation_reasons"]
        )

    @pytest.mark.asyncio
    async def test_prob_clamped_to_unit_range(self) -> None:
        """prob 越界值（> 1 / < 0）应被钳位。"""
        tool = PositiveGrounding()
        state = _state_with_probs([
            {"name": "Pneumothorax", "prob": 1.5, "present": True},
            {"name": "Cardiomegaly", "prob": -0.2, "present": False},
        ])
        patch = await tool.run(state)
        box = patch["grounding_boxes"][0]
        assert box["prob"] == 1.0  # 钳位到 1.0

    @pytest.mark.asyncio
    async def test_does_not_raise_on_invalid_state(self) -> None:
        """入参完全非法也不抛（与 finding_draft 不同；grounding 是辅助）。"""
        tool = PositiveGrounding()
        # 空 state
        patch = await tool.run({})
        assert patch["grounding_boxes"] == []
        assert patch["degraded"] is True


# ---------------------------------------------------------------------------
# 6. 确定性（飞检可重复）
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_same_input_same_output(self) -> None:
        """同输入多次调用必须返回同一组 bbox。"""
        tool = PositiveGrounding()
        state = _state_with_probs(PROBS_BORDERLINE_MULTI)
        p1 = await tool.run(state)
        p2 = await tool.run(state)
        p3 = await tool.run(state)
        assert p1["grounding_boxes"] == p2["grounding_boxes"]
        assert p2["grounding_boxes"] == p3["grounding_boxes"]


# ---------------------------------------------------------------------------
# 7. StateGraph 集成
# ---------------------------------------------------------------------------


class TestStateGraphIntegration:
    @pytest.mark.asyncio
    async def test_tool_registry_name_aligns_with_workflow(self) -> None:
        """name 必须等于 'positive_grounding'，与 DEFAULT_TOOL_NAMES[3] 对齐。"""
        assert PositiveGrounding().name == "positive_grounding"
        assert "positive_grounding" in DEFAULT_TOOL_NAMES

    @pytest.mark.asyncio
    async def test_registry_registration(self) -> None:
        registry = ToolRegistry()
        tool = PositiveGrounding()
        registry.register(tool)
        assert registry.has("positive_grounding")
        assert registry.get("positive_grounding") is tool

    @pytest.mark.asyncio
    async def test_full_workflow_positive_path(self) -> None:
        """5 件工具：classifier 输出阳性 + PositiveGrounding 真跑 → boxes 出现在最终 state。"""
        from tests.test_state_workflow import MockTool  # type: ignore

        # classifier 输出阳性（Pneumothorax=0.85）
        mock_classifier = MockTool("classifier", outputs={
            "classifier_probs": [
                {"name": "Pneumothorax", "prob": 0.85, "present": True},
                {"name": "Effusion", "prob": 0.42, "present": False},
            ],
            "classifier_max_prob": 0.85,
        })
        mock_keyqa = MockTool("keyqa", outputs={
            "key_qa_list": [{"q": "心影", "a": "心影未见明显增大"}],
        })
        mock_finding_draft = MockTool("finding_draft", outputs={
            "finding_text": "右下肺野见少量气胸征象（草稿），余肺野清晰。",
            "finding_struct": {"pathologies": [], "key_qa": [], "grounding_boxes": []},
        })
        # 真 PositiveGrounding 工具实例（替换 mock）→ 验证 StateGraph 集成
        real_grounding = PositiveGrounding()
        mock_image_preview = MockTool("image_preview", outputs={
            "png_preview_b64": "iVBORw0KGgo=",
        })

        registry = ToolRegistry()
        registry.register(mock_classifier)
        registry.register(mock_keyqa)
        registry.register(mock_finding_draft)
        registry.register(real_grounding)
        registry.register(mock_image_preview)

        wf = build_workflow(registry, positive_threshold=0.5)
        result = await wf.run(
            study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_T16_WF>",
            png_bytes=b"\x89PNG\r\n\x1a\n",
            metadata={"PatientSex": "F", "PatientAge": "050Y"},
        )
        assert "grounding_boxes" in result
        assert isinstance(result["grounding_boxes"], list)
        assert len(result["grounding_boxes"]) == 1
        assert result["grounding_boxes"][0]["label"] == "Pneumothorax"
        assert abs(result["grounding_boxes"][0]["prob"] - 0.85) < 1e-9
        # 全部 5 件工具都被执行
        assert "positive_grounding" in result["tools_executed"]
        assert "image_preview" in result["tools_executed"]


# ---------------------------------------------------------------------------
# 8. to_finding_struct 静态工具
# ---------------------------------------------------------------------------


class TestToFindingStruct:
    def test_passthrough_normalized(self) -> None:
        boxes = [
            {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": "Pneumothorax", "prob": 0.85},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert out == [
            {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": "Pneumothorax", "prob": 0.85},
        ]

    def test_non_list_returns_empty(self) -> None:
        assert PositiveGrounding.to_finding_struct(None) == []  # type: ignore[arg-type]
        assert PositiveGrounding.to_finding_struct("not-list") == []  # type: ignore[arg-type]

    def test_non_dict_items_filtered(self) -> None:
        boxes = [
            {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": "X"},
            "bad-string",
            None,
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert len(out) == 1

    def test_coords_clamped_to_unit_range(self) -> None:
        boxes = [
            {"x": 1.5, "y": -0.1, "w": 0.5, "h": 0.5, "label": "X"},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert out[0]["x"] == 1.0
        assert out[0]["y"] == 0.0

    def test_w_h_clamped_to_xw_le_one(self) -> None:
        boxes = [
            {"x": 0.5, "y": 0.5, "w": 1.5, "h": 1.5, "label": "X"},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert out[0]["x"] + out[0]["w"] <= 1.0 + 1e-9
        assert out[0]["y"] + out[0]["h"] <= 1.0 + 1e-9

    def test_missing_prob_omits_field(self) -> None:
        boxes = [
            {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": "X"},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert "prob" not in out[0]

    def test_invalid_field_types_filtered(self) -> None:
        boxes = [
            {"x": "not-numeric", "y": 0.5, "w": 0.2, "h": 0.2, "label": "X"},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert out == []

    def test_non_string_label_coerced_to_str(self) -> None:
        boxes = [
            {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "label": 123},
        ]
        out = PositiveGrounding.to_finding_struct(boxes)
        assert out[0]["label"] == "123"


# ---------------------------------------------------------------------------
# 9. 模块导出
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_public_symbols(self) -> None:
        from cxr_triage.engine import positive_grounding as pg

        for name in (
            "DEFAULT_BBOX_BY_REGION",
            "DEFAULT_FALLBACK_BBOX",
            "MAX_GROUNDING_BOXES",
            "POSITIVE_GROUNDING_PROB_HEURISTIC",
            "PATHOLOGY_NAMES",
            "PositiveGrounding",
            "PositiveGroundingError",
            "PositiveGroundingResult",
            "build_positive_grounding",
        ):
            assert hasattr(pg, name), f"missing export: {name}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))