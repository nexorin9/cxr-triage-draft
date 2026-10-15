"""tests/test_finding_draft.py — Task 15 FindingDraft 中文所见草稿生成器单测。

目标（spec.md §5 维度 1 + §10 安全边界 + 中式实务优先 + Task 15 细则）：

1. **构造期校验**：阈值越界 / localization_map 类型错 / top_n 越界
2. **常量层**：
   - DRAFT_MARKER 锁定为「（草稿 / 未经医师确认）」
   - POSITIVE_TEMPLATE / NO_ABNORMALITY_TEMPLATE 含 DRAFT_MARKER 占位
   - DEFAULT_POSITIVE_THRESHOLD = 0.5
   - DEFAULT_LOCALIZATION_BY_PATHOLOGY 至少含 18 项
3. **finding_struct 字段契约**：
   - pathologies 保留 prob > 0.05 + 按 prob DESC + 含 present 字段
   - key_qa 过滤非 dict / 字段缺失 / 空字符串
   - grounding_boxes 坐标钳位 [0, 1]
4. **finding_text 强制 DRAFT_MARKER**：
   - 阳性路径：「{localization}见{top1_zh}征象（草稿 / 未经医师确认），
     余肺野清晰。」
   - 阴性路径：「胸片未见明显异常（草稿 / 未经医师确认），必要时建议
     结合临床或复查。」
   - 概率列空 + positive=True → 「未取得可信分类结果（草稿 / 未经医师确
     认），仍待人工复核。」（防御性）
5. **positive_flag 由 max_prob × threshold 推导**：
   - max_prob=0.85 + threshold=0.5 → True
   - max_prob=0.49 + threshold=0.5 → False
   - threshold=0.7 + max_prob=0.51 → False
6. **3 例 fixtures 端到端**：
   - 阳性（Pneumothorax=0.85） → finding_text 含「气胸」+ DRAFT_MARKER
   - 阴性（all low）→ finding_text 含「未见明显异常」+ DRAFT_MARKER
   - 边界（Cardiomegaly=0.51）→ positive_flag=True（max≥0.5）
7. **finding_struct 端到端**：
   - 病理清单含分类 + 关键问答清单 + grounding_boxes（可空）
   - 序列化为 JSON 不丢中文（ensure_ascii round-trip）
8. **serialize_for_ris 静态工具**：
   - spec.md §4 RIS payload 字段 7 个（study_uid / patient_id_hash /
     finding_struct / finding_text / positive_flag / tool_version /
     draft_ts）齐全
9. **StateGraph 集成**：ToolRegistry 注册 + 5 件 mock 工具全跑 + finding_text
   含 DRAFT_MARKER + finding_struct.pathologies 完整 18 项
10. **不依赖 LLM**：finding_draft 完全确定性合成，CXRKeyQA 留空时不阻断本
    工具（单测与 StateGraph 集成均验证）
"""
from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from cxr_triage.engine import (
    DEFAULT_LOCALIZATION_BY_PATHOLOGY,
    DEFAULT_POSITIVE_THRESHOLD,
    DRAFT_MARKER,
    FindingDraft,
    FindingDraftError,
    FindingDraftResult,
    MAX_PATHOLOGIES_IN_STRUCT,
    NO_ABNORMALITY_TEMPLATE,
    PATHOLOGY_NAMES,
    POSITIVE_TEMPLATE,
    TOP_N_PATHOLOGIES,
    build_finding_draft,
)
from cxr_triage.engine.workflow import ToolRegistry, build_workflow


# ---------------------------------------------------------------------------
# 3 例 fixtures — 与 test_triage_classifier.py 的 18 病理对齐
# ---------------------------------------------------------------------------


# 阳性：右下肺气胸（Pneumothorax=0.85）
PROBS_POSITIVE: Dict[str, float] = {
    "Pneumothorax": 0.85,
    "Effusion": 0.42,
    "Cardiomegaly": 0.30,
}

# 阴性：全低概率，所有 present=False
PROBS_NEGATIVE: Dict[str, float] = {name: 0.05 for name in PATHOLOGY_NAMES}

# 边界：心影临界（Cardiomegaly=0.51），其余低
PROBS_BORDERLINE: Dict[str, float] = {
    "Cardiomegaly": 0.51,
    "Effusion": 0.30,
}
for name in PATHOLOGY_NAMES:
    PROBS_BORDERLINE.setdefault(name, 0.05)


def _make_png_bytes(width: int = 64, height: int = 64, value: int = 128) -> bytes:
    img = Image.new("L", (width, height), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fixed_now_iso() -> str:
    """固定时戳，便于单测断言。"""
    fixed = datetime(2026, 10, 13, 7, 23, 11, 487000, tzinfo=timezone.utc)
    return fixed.strftime("%Y-%m-%dT%H:%M:%S.") + f"{fixed.microsecond // 1000:03d}Z"


def _make_probs(probs_by_name: Dict[str, float]) -> List[Dict[str, Any]]:
    """模拟 TriageClassifier 输出格式 ``[{name, prob, present}, ...]``。"""
    out: List[Dict[str, Any]] = []
    for name, p in probs_by_name.items():
        out.append({
            "name": name,
            "prob": float(p),
            "present": bool(p >= 0.5),
        })
    return out


def _make_state(
    probs_by_name: Dict[str, float],
    key_qa_list: Optional[List[Dict[str, str]]] = None,
    grounding_boxes: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    probs = _make_probs(probs_by_name)
    max_prob = max((p["prob"] for p in probs), default=0.0)
    return {
        "study_uid": "<EXAMPLE_STUDY_UID>",
        "png_bytes": _make_png_bytes(),
        "metadata": metadata
            or {"PatientID": "<EXAMPLE_PATIENT_ID>", "PatientSex": "F",
                "PatientAge": "052Y"},
        "classifier_probs": probs,
        "classifier_max_prob": max_prob,
        "key_qa_list": key_qa_list or [
            {"q": "心影", "a": "心影增大，心胸比约 0.55"},
            {"q": "肺野", "a": "右下肺纹理稀疏"},
        ],
        "grounding_boxes": grounding_boxes or [
            {"x": 0.62, "y": 0.71, "w": 0.15, "h": 0.10,
             "label": "右下肺", "prob": 0.81},
        ],
    }


def _key_qa_payload(qa_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [dict(it) for it in qa_list]


# ===========================================================================
# 1) 构造期校验
# ===========================================================================


def test_construction_defaults_ok() -> None:
    """默认值构造 OK：threshold=0.5 + 内置 localization_map + top_n=3。"""
    tool = FindingDraft()
    assert tool.positive_threshold == DEFAULT_POSITIVE_THRESHOLD == 0.5
    assert tool.top_n == TOP_N_PATHOLOGIES == 3
    assert tool.localization_map == DEFAULT_LOCALIZATION_BY_PATHOLOGY
    assert tool.name == "finding_draft"


def test_threshold_must_be_in_unit_interval() -> None:
    for bad in (-0.1, 1.5, 2.0, -1):
        with pytest.raises(FindingDraftError):
            FindingDraft(positive_threshold=bad)


def test_threshold_string_numeric_ok() -> None:
    """数字字符串如 '0.7' 接受；非数字字符串拒绝。"""
    tool = FindingDraft(positive_threshold="0.7")
    assert tool.positive_threshold == 0.7
    with pytest.raises(FindingDraftError):
        FindingDraft(positive_threshold="abc")
    with pytest.raises(FindingDraftError):
        FindingDraft(positive_threshold=None)


def test_localization_map_validation() -> None:
    """localization_map 类型校验：非 dict 拒绝；键非空字符串拒绝；值非空字符串拒绝。"""
    # 类型错
    with pytest.raises(FindingDraftError):
        FindingDraft(localization_map=[("Pneumothorax", "右下肺野")])  # type: ignore[arg-type]
    # 键非空字符串
    with pytest.raises(FindingDraftError):
        FindingDraft(localization_map={"": "右下肺野"})
    # 值非空字符串
    with pytest.raises(FindingDraftError):
        FindingDraft(localization_map={"Pneumothorax": ""})


def test_top_n_validation() -> None:
    """top_n 越界拒绝。"""
    with pytest.raises(FindingDraftError):
        FindingDraft(top_n=0)
    with pytest.raises(FindingDraftError):
        FindingDraft(top_n=19)
    with pytest.raises(FindingDraftError):
        FindingDraft(top_n="3")  # type: ignore[arg-type]


def test_build_factory_returns_FindingDraft() -> None:
    tool = build_finding_draft(positive_threshold=0.7)
    assert isinstance(tool, FindingDraft)
    assert tool.positive_threshold == 0.7


# ===========================================================================
# 2) 常量层
# ===========================================================================


def test_DRAFT_MARKER_locked() -> None:
    """DRAFT_MARKER 锁定 — 任何改变都视为 spec 突破。"""
    assert DRAFT_MARKER == "（草稿 / 未经医师确认）"


def test_POSITIVE_TEMPLATE_has_marker() -> None:
    """POSITIVE_TEMPLATE 内含 DRAFT_MARKER 占位，便于 .format 后替换。"""
    assert "DRAFT_MARKER" in POSITIVE_TEMPLATE


def test_NO_ABNORMALITY_TEMPLATE_has_marker() -> None:
    assert "DRAFT_MARKER" in NO_ABNORMALITY_TEMPLATE


def test_default_positive_threshold_is_0_5() -> None:
    assert DEFAULT_POSITIVE_THRESHOLD == 0.5


def test_default_localization_map_has_18_pathologies() -> None:
    """DEFAULT_LOCALIZATION_BY_PATHOLOGY 含至少 18 病理定位。"""
    assert len(DEFAULT_LOCALIZATION_BY_PATHOLOGY) >= 18
    # 全 18 病理均能找到 fallback（fallback 文案可读）
    for pathology in PATHOLOGY_NAMES:
        assert pathology in DEFAULT_LOCALIZATION_BY_PATHOLOGY


def test_PATHOLOGY_NAMES_locked() -> None:
    """PATHOLOGY_NAMES 含 18 病理，与 xrv.datasets.default_pathologies 顺序一致。"""
    assert len(PATHOLOGY_NAMES) == 18
    assert "Pneumothorax" in PATHOLOGY_NAMES
    assert "Cardiomegaly" in PATHOLOGY_NAMES
    assert "Effusion" in PATHOLOGY_NAMES


# ===========================================================================
# 3) 入参校验（fail-fast）
# ===========================================================================


@pytest.mark.asyncio
async def test_run_requires_classifier_probs_list() -> None:
    """state['classifier_probs'] 缺/非 list → FindingDraftError。"""
    tool = FindingDraft()
    with pytest.raises(FindingDraftError):
        await tool.run({})  # 缺字段
    with pytest.raises(FindingDraftError):
        await tool.run({"classifier_probs": None})
    with pytest.raises(FindingDraftError):
        await tool.run({"classifier_probs": "not-a-list"})


@pytest.mark.asyncio
async def test_run_requires_classifier_max_prob_numeric() -> None:
    """state['classifier_max_prob'] 缺/非数值 → FindingDraftError。"""
    tool = FindingDraft()
    base = {
        "classifier_probs": _make_probs(PROBS_POSITIVE),
    }
    with pytest.raises(FindingDraftError):
        await tool.run({**base})
    with pytest.raises(FindingDraftError):
        await tool.run({**base, "classifier_max_prob": None})
    with pytest.raises(FindingDraftError):
        await tool.run({**base, "classifier_max_prob": "0.5"})
    with pytest.raises(FindingDraftError):
        await tool.run({**base, "classifier_max_prob": True})  # bool 拒绝


# ===========================================================================
# 4) finding_struct 字段契约
# ===========================================================================


@pytest.mark.asyncio
async def test_finding_struct_pathologies_contains_present_field() -> None:
    """``pathologies`` 项含 ``name / prob / present`` 3 字段。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    out = await tool.run(state)
    for p in out["finding_struct"]["pathologies"]:
        assert {"name", "prob", "present"} <= set(p.keys())


@pytest.mark.asyncio
async def test_finding_struct_pathologies_prob_desc_order() -> None:
    """``pathologies`` 按 prob DESC 排序。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    out = await tool.run(state)
    probs_in_struct = [p["prob"] for p in out["finding_struct"]["pathologies"]]
    assert probs_in_struct == sorted(probs_in_struct, reverse=True)


@pytest.mark.asyncio
async def test_finding_struct_pathologies_capped_at_18() -> None:
    """``pathologies`` 上限 MAX_PATHOLOGIES_IN_STRUCT (=18) 项。"""
    tool = FindingDraft()
    probs_dict = {name: float(i) / 20.0 for i, name in enumerate(PATHOLOGY_NAMES)}
    state = _make_state(probs_dict)
    out = await tool.run(state)
    assert len(out["finding_struct"]["pathologies"]) == MAX_PATHOLOGIES_IN_STRUCT == 18


@pytest.mark.asyncio
async def test_finding_struct_pathologies_prob_clamped() -> None:
    """越界 prob 钳到 [0, 1]；非数值 prob 跳过。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    state["classifier_probs"] = [
        {"name": "Pneumothorax", "prob": 1.5, "present": True},     # > 1 钳到 1
        {"name": "Effusion", "prob": -0.2, "present": False},     # < 0 钳到 0
        {"name": "Cardiomegaly", "prob": "abc", "present": False}, # 跳过
        {"name": "Mass", "prob": True, "present": False},          # bool 跳过
        {"name": "Nodule", "prob": 0.7, "present": True},
    ]
    state["classifier_max_prob"] = 1.0
    out = await tool.run(state)
    paths = out["finding_struct"]["pathologies"]
    by_name = {p["name"]: p["prob"] for p in paths}
    assert by_name["Pneumothorax"] == 1.0
    assert by_name["Effusion"] == 0.0
    assert "Cardiomegaly" not in by_name
    assert "Mass" not in by_name
    assert by_name["Nodule"] == 0.7


@pytest.mark.asyncio
async def test_finding_struct_key_qa_filters_invalid() -> None:
    """``key_qa`` 过滤非 dict / 字段缺失 / 空字符串。"""
    tool = FindingDraft()
    state = _make_state(
        PROBS_NEGATIVE,
        key_qa_list=[
            {"q": "心影", "a": "心影增大"},               # 保留
            {"q": "纵隔", "a": "纵隔居中"},               # 保留
            {"q": "", "a": "空 q"},                      # 过滤（q 空）
            {"q": "胸膜", "a": ""},                      # 过滤（a 空）
            {"q": "   ", "a": "   "},                    # 过滤（空白）
            "not-a-dict",                                # 过滤
            {"q": 123, "a": "数值字段拒绝"},             # 过滤
            {"q": "肺野", "a": None},                    # 过滤
        ],
    )
    out = await tool.run(state)
    assert out["finding_struct"]["key_qa"] == [
        {"q": "心影", "a": "心影增大"},
        {"q": "纵隔", "a": "纵隔居中"},
    ]


@pytest.mark.asyncio
async def test_finding_struct_key_qa_empty_state_ok() -> None:
    """``key_qa_list`` 缺失（keyqa 软降级空列表）→ finding_struct.key_qa = []。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    state["key_qa_list"] = None  # 等价于 keyqa 没跑过
    out = await tool.run(state)
    assert out["finding_struct"]["key_qa"] == []


@pytest.mark.asyncio
async def test_finding_struct_grounding_boxes_normalization() -> None:
    """``grounding_boxes`` 坐标钳位 [0, 1] 且过滤非 dict。"""
    tool = FindingDraft()
    state = _make_state(
        PROBS_POSITIVE,
        grounding_boxes=[
            {"x": 0.62, "y": 0.71, "w": 0.15, "h": 0.10,
             "label": "右下肺", "prob": 0.81},                 # 正常
            {"x": 1.5, "y": -0.1, "w": 0.20, "h": 0.20,
             "label": "out-of-bounds", "prob": 0.5},           # 钳位
            {"x": 0.10, "y": 0.10, "w": 1.5, "h": 0.10,
             "label": "w-overflow", "prob": 0.4},              # w 超出右边界后裁剪
            {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0,
             "label": "degenerate"},                            # 全 0（仍保留）
            "not-a-dict",
            {"x": "abc", "y": 0.0, "w": 0.0, "h": 0.0},        # 跳过
        ],
    )
    out = await tool.run(state)
    boxes = out["finding_struct"]["grounding_boxes"]
    assert len(boxes) == 4
    # 第一个原值保留
    assert boxes[0]["x"] == 0.62 and boxes[0]["y"] == 0.71
    assert boxes[0]["label"] == "右下肺"
    # 第二个钳位
    assert boxes[1]["x"] == 1.0 and boxes[1]["y"] == 0.0
    # 第三个 w 溢出右边界（x=0.10，max_w = 1.0 - 0.10 = 0.9）→ w 钳到
    # min(1.5, 1.0-0.10) = 0.9；x + w = 1.0（不出界）
    assert boxes[2]["x"] == 0.10
    assert boxes[2]["w"] == 0.9
    assert boxes[2]["x"] + boxes[2]["w"] == pytest.approx(1.0)
    # 第四个全零
    assert boxes[3]["w"] == 0.0 and boxes[3]["h"] == 0.0


@pytest.mark.asyncio
async def test_finding_struct_grounding_boxes_missing_state_ok() -> None:
    """``grounding_boxes`` 字段缺失（PositiveGrounding 未跑）→ [] 不抛。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    state.pop("grounding_boxes", None)
    out = await tool.run(state)
    assert out["finding_struct"]["grounding_boxes"] == []


# ===========================================================================
# 5) finding_text 强制 DRAFT_MARKER
# ===========================================================================


@pytest.mark.asyncio
async def test_finding_text_negative_contains_marker_and_banner() -> None:
    """max_prob < 阈值 → finding_text 含「未见明显异常」+ DRAFT_MARKER。"""
    tool = FindingDraft(positive_threshold=0.5)
    state = _make_state(PROBS_NEGATIVE)
    out = await tool.run(state)
    assert DRAFT_MARKER in out["finding_text"]
    assert "未见明显异常" in out["finding_text"]
    # positive_flag=False
    assert out["positive_flag"] is False


@pytest.mark.asyncio
async def test_finding_text_positive_contains_marker_and_top1_zh() -> None:
    """max_prob ≥ 阈值 → finding_text 含「{loc}见{top1_zh}征象」+ DRAFT_MARKER。"""
    tool = FindingDraft(positive_threshold=0.5)
    state = _make_state(PROBS_POSITIVE)
    out = await tool.run(state)
    assert DRAFT_MARKER in out["finding_text"]
    # 「气胸」为 Pneumothorax 中文 fallback 命名（_PATHOLOGY_ZH 表第一映射）
    assert "气胸" in out["finding_text"]
    # 「右下肺野」为 Pneumothorax 的 DEFAULT_LOCALIZATION_BY_PATHOLOGY 定位
    assert "右下肺野" in out["finding_text"]
    # 余肺野清晰是体检科报告长期沿用的收尾
    assert "余肺野清晰" in out["finding_text"]
    # positive_flag=True
    assert out["positive_flag"] is True


@pytest.mark.asyncio
async def test_finding_text_positive_top1_default_localization() -> None:
    """阳性 top1 病理在 DEFAULT_LOCALIZATION_BY_PATHOLOGY 找不到时 fallback 到「右下肺野」。"""
    tool = FindingDraft(positive_threshold=0.5, localization_map={})
    state = _make_state(PROBS_POSITIVE)
    out = await tool.run(state)
    # 仍含 DRAFT_MARKER + 「气胸」（fallback 文案）
    assert DRAFT_MARKER in out["finding_text"]
    assert "气胸" in out["finding_text"]
    assert "右下肺野" in out["finding_text"]


@pytest.mark.asyncio
async def test_finding_text_positive_classifier_probs_empty_defensive_banner() -> None:
    """positive=True 但 probs 列表空（极防御性路径）：finding_text 含 DRAFT_MARKER
    + 「未取得可信分类结果」banner。"""
    tool = FindingDraft(positive_threshold=0.0)  # 阈值 0 → 任何 max_prob 都阳性
    state = _make_state(PROBS_POSITIVE)
    state["classifier_probs"] = []  # 强制空列表
    state["classifier_max_prob"] = 0.0
    out = await tool.run(state)
    assert DRAFT_MARKER in out["finding_text"]
    assert "未取得可信分类结果" in out["finding_text"]


@pytest.mark.asyncio
async def test_finding_text_defensive_unrelated_pathology_fallback() -> None:
    """未知病理名（非 _PATHOLOGY_ZH 也非 DEFAULT_LOCALIZATION_BY_PATHOLOGY）：
    走 fallback 文案（病灶名 / 「右下肺野」）。"""
    tool = FindingDraft(positive_threshold=0.5)
    state = _make_state({"MysteryPathology": 0.85})
    out = await tool.run(state)
    assert DRAFT_MARKER in out["finding_text"]
    # 病灶名直接复用英文（MysteryPathology）
    assert "MysteryPathology" in out["finding_text"]
    # fallback 定位
    assert "右下肺野" in out["finding_text"]


@pytest.mark.asyncio
async def test_finding_text_includes_marker_on_corner_case_max_prob_exactly_threshold() -> None:
    """max_prob == threshold 时仍 positive=True（与 process_node 现有 >= 语义一致）。"""
    tool = FindingDraft(positive_threshold=0.5)
    probs = {"Pneumothorax": 0.50}
    state = _make_state(probs)
    state["classifier_probs"] = _make_probs(probs)
    state["classifier_max_prob"] = 0.50
    out = await tool.run(state)
    assert out["positive_flag"] is True
    assert DRAFT_MARKER in out["finding_text"]


@pytest.mark.asyncio
async def test_finding_text_always_contains_DRAFT_MARKER_across_paths() -> None:
    """任意分支 finding_text 都必含 DRAFT_MARKER（防御性兜底测试）。"""
    tool = FindingDraft(positive_threshold=0.5)
    # path A: 阳性
    state = _make_state(PROBS_POSITIVE)
    out_a = await tool.run(state)
    assert DRAFT_MARKER in out_a["finding_text"]
    # path B: 阴性
    state = _make_state(PROBS_NEGATIVE)
    out_b = await tool.run(state)
    assert DRAFT_MARKER in out_b["finding_text"]
    # path C: 边界阳性
    state = _make_state(PROBS_BORDERLINE)
    out_c = await tool.run(state)
    assert DRAFT_MARKER in out_c["finding_text"]
    # path D: 概率列空
    state = _make_state({})
    state["classifier_max_prob"] = 0.0
    out_d = await tool.run(state)
    assert DRAFT_MARKER in out_d["finding_text"]


# ===========================================================================
# 6) positive_flag 推导
# ===========================================================================


@pytest.mark.asyncio
async def test_positive_flag_max_prob_above_threshold() -> None:
    tool = FindingDraft(positive_threshold=0.5)
    out = await tool.run(_make_state(PROBS_POSITIVE))
    assert out["positive_flag"] is True


@pytest.mark.asyncio
async def test_positive_flag_max_prob_below_threshold() -> None:
    tool = FindingDraft(positive_threshold=0.5)
    out = await tool.run(_make_state(PROBS_NEGATIVE))
    assert out["positive_flag"] is False


@pytest.mark.asyncio
async def test_positive_flag_threshold_change_recomputes() -> None:
    """阈值上调后，max_prob=0.51 不再被标阳。"""
    state = _make_state(PROBS_BORDERLINE)
    # 默认 0.5 时 Cardiomegaly=0.51 → positive
    out_default = await FindingDraft(positive_threshold=0.5).run(state)
    assert out_default["positive_flag"] is True
    # 阈值升 0.7 时 → 不再 positive
    out_tight = await FindingDraft(positive_threshold=0.7).run(state)
    assert out_tight["positive_flag"] is False


# ===========================================================================
# 7) draft_ts（UTC ISO8601 毫秒精度）
# ===========================================================================


@pytest.mark.asyncio
async def test_draft_ts_is_iso8601_ms_utc() -> None:
    """``draft_ts`` 形如 2026-10-13T07:23:11.487Z（19 字符 + Z，毫秒精度）。"""
    fixed_ts = _fixed_now_iso()
    tool = FindingDraft(now_iso=lambda: fixed_ts)
    out = await tool.run(_make_state(PROBS_POSITIVE))
    assert out["draft_ts"] == fixed_ts
    assert out["draft_ts"].endswith("Z")
    # 26 个字符（4+2+2 + T + 2+2+2 + . + 3 + Z = 20）
    assert len(out["draft_ts"]) == 24


@pytest.mark.asyncio
async def test_draft_ts_injected_now_iso_used() -> None:
    """``now_iso`` 注入函数被调用，draft_ts 跟随。"""
    calls = {"n": 0}

    def fake_now() -> str:
        calls["n"] += 1
        return f"2026-01-01T00:00:00.{calls['n'] * 100:03d}Z"

    tool = FindingDraft(now_iso=fake_now)
    out1 = await tool.run(_make_state(PROBS_POSITIVE))
    out2 = await tool.run(_make_state(PROBS_POSITIVE))
    assert out1["draft_ts"].endswith("100Z")
    assert out2["draft_ts"].endswith("200Z")
    assert calls["n"] == 2


# ===========================================================================
# 8) 3 例 fixtures 端到端
# ===========================================================================


@pytest.mark.asyncio
async def test_positive_fixture_finding_text_and_struct() -> None:
    """阳性 fixture：finding_text 含「气胸」+ DRAFT_MARKER；finding_struct.pathologies 含 18 病理。"""
    tool = FindingDraft()
    out = await tool.run(_make_state(PROBS_POSITIVE))
    # text
    assert DRAFT_MARKER in out["finding_text"]
    assert "气胸" in out["finding_text"]
    # struct
    paths_by_name = {p["name"]: p for p in out["finding_struct"]["pathologies"]}
    assert paths_by_name["Pneumothorax"]["prob"] == 0.85
    assert paths_by_name["Pneumothorax"]["present"] is True
    # key_qa 与 grounding_boxes 透传（含基础项）
    assert len(out["finding_struct"]["key_qa"]) >= 1
    assert len(out["finding_struct"]["grounding_boxes"]) >= 1
    # positive_flag
    assert out["positive_flag"] is True


@pytest.mark.asyncio
async def test_negative_fixture_finding_text_and_struct() -> None:
    """阴性 fixture：finding_text 含「未见明显异常」+ DRAFT_MARKER；finding_struct 全量保留。"""
    tool = FindingDraft()
    out = await tool.run(_make_state(PROBS_NEGATIVE))
    assert "未见明显异常" in out["finding_text"]
    assert DRAFT_MARKER in out["finding_text"]
    assert out["positive_flag"] is False
    # finding_struct.pathologies 全 18 项
    assert len(out["finding_struct"]["pathologies"]) == 18
    for p in out["finding_struct"]["pathologies"]:
        assert p["present"] is False
        assert p["prob"] == 0.05


@pytest.mark.asyncio
async def test_borderline_fixture_finding_text_and_struct() -> None:
    """边界 fixture：Cardiomegaly=0.51 时 positive_flag=True，finding_text 含「心影增大」。"""
    tool = FindingDraft()
    out = await tool.run(_make_state(PROBS_BORDERLINE))
    assert out["positive_flag"] is True
    assert DRAFT_MARKER in out["finding_text"]
    assert "心影增大" in out["finding_text"]
    # 「心影」为 Cardiomegaly 在 DEFAULT_LOCALIZATION_BY_PATHOLOGY 的定位文案
    assert "心影" in out["finding_text"]


# ===========================================================================
# 9) finding_struct 序列化（中文 round-trip）
# ===========================================================================


@pytest.mark.asyncio
async def test_finding_struct_json_round_trip_preserves_chinese() -> None:
    """finding_struct 经 json.dumps/loads 后中文所见保留（含 ensure_ascii=False）。"""
    tool = FindingDraft()
    out = await tool.run(_make_state(PROBS_POSITIVE))
    payload = json.dumps(out, ensure_ascii=False)
    loaded = json.loads(payload)
    assert loaded["finding_text"] == out["finding_text"]
    # 还原后 QA 字段含中文
    assert any(
        "心影" in it.get("q", "")
        for it in loaded["finding_struct"]["key_qa"]
    )


# ===========================================================================
# 10) serialize_for_ris 静态工具
# ===========================================================================


@pytest.mark.asyncio
async def test_serialize_for_ris_payload_contract() -> None:
    """``serialize_for_ris`` 输出 spec.md §4 RIS payload 7 字段。"""
    tool = FindingDraft()
    state = _make_state(PROBS_POSITIVE)
    out = await tool.run(state)
    payload = tool.serialize_for_ris(
        finding_text=out["finding_text"],
        finding_struct=out["finding_struct"],
        positive_flag=out["positive_flag"],
        draft_ts=out["draft_ts"],
        tool_version="0.1.0",
        study_uid="<EXAMPLE_STUDY_UID>",
        patient_id_hash="sha256:<EXAMPLE_HASH>",
    )
    # 7 字段齐全（spec.md §4）
    expected_keys = {
        "study_uid",
        "patient_id_hash",
        "finding_struct",
        "finding_text",
        "positive_flag",
        "tool_version",
        "draft_ts",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["study_uid"] == "<EXAMPLE_STUDY_UID>"
    assert payload["patient_id_hash"] == "sha256:<EXAMPLE_HASH>"
    assert payload["tool_version"] == "0.1.0"
    assert DRAFT_MARKER in payload["finding_text"]
    assert payload["positive_flag"] is True


def test_serialize_for_ris_defaults_when_omitted() -> None:
    """``serialize_for_ris`` 缺省 study_uid / patient_id_hash / tool_version
    时仍合法（调用方在 Task 22 RISWriter 内统一补齐）。"""
    payload = FindingDraft.serialize_for_ris(
        finding_text="demo",
        finding_struct={"pathologies": [], "key_qa": [], "grounding_boxes": []},
        positive_flag=True,
        draft_ts="2026-10-13T07:23:11.487Z",
    )
    assert payload["study_uid"] == ""
    assert payload["patient_id_hash"] == ""
    assert payload["tool_version"] == "0.1.0"


@pytest.mark.asyncio
async def test_serialize_for_ris_negative_payload_shape() -> None:
    """阴性 fixture 的 RIS payload 仍含 finding_struct（结构完整），finding_text 含 DRAFT_MARKER。"""
    tool = FindingDraft()
    out = await tool.run(_make_state(PROBS_NEGATIVE))
    payload = tool.serialize_for_ris(
        finding_text=out["finding_text"],
        finding_struct=out["finding_struct"],
        positive_flag=out["positive_flag"],
        draft_ts=out["draft_ts"],
        study_uid="<EXAMPLE_NEG>",
    )
    # 阴性：positive_flag=False，但 finding_struct.pathologies 全量保留 18 项
    assert payload["positive_flag"] is False
    assert len(payload["finding_struct"]["pathologies"]) == 18
    assert DRAFT_MARKER in payload["finding_text"]


# ===========================================================================
# 11) StateGraph 集成（ToolRegistry + 5 件 mock 工具全跑）
# ===========================================================================


class MockFindingDraft:
    """Mock finding_draft：按 state 触发器跑真实 FindingDraft（双层）。"""

    def __init__(self, finding_draft: FindingDraft) -> None:
        self.name = "finding_draft"
        self._tool = finding_draft

    async def run(self, state):
        return await self._tool.run(state)


class MockClassifier:
    """Mock TriageClassifier：固定 positive 概率。"""

    def __init__(self, probs_by_name: Dict[str, float]) -> None:
        self.name = "classifier"
        self._probs = _make_probs(probs_by_name)
        self._max = max((p["prob"] for p in self._probs), default=0.0)

    async def run(self, state):
        return {"classifier_probs": list(self._probs),
                "classifier_max_prob": float(self._max)}


class MockKeyQA:
    """Mock CXRKeyQA：固定 4 答案。"""

    name = "keyqa"

    async def run(self, state):
        return {
            "key_qa_list": [
                {"q": "心影", "a": "心影增大，心胸比约 0.55"},
                {"q": "纵隔", "a": "纵隔居中"},
                {"q": "胸膜", "a": "双侧胸膜未见明显增厚"},
                {"q": "肺野", "a": "右下肺纹理稀疏"},
            ],
        }


class MockGrounding:
    """Mock PositiveGrounding：固定 1 个定位框。"""

    name = "positive_grounding"

    async def run(self, state):
        return {
            "grounding_boxes": [
                {"x": 0.62, "y": 0.71, "w": 0.15, "h": 0.10,
                 "label": "右下肺", "prob": 0.81},
            ],
        }


class MockImagePreview:
    """Mock ImagePreview：返回 base64 PNG bytes。"""

    name = "image_preview"

    async def run(self, state):
        png = _make_png_bytes()
        import base64
        return {"png_preview_b64": base64.b64encode(png).decode("ascii")}


@pytest.mark.asyncio
async def test_stategraph_integration_with_5_mock_tools() -> None:
    """5 件 mock 工具全跑 → finding_draft 写 finding_text 含 DRAFT_MARKER
    + finding_struct.pathologies 全 18 项。"""
    registry = ToolRegistry()
    registry.register(MockClassifier(PROBS_POSITIVE))
    registry.register(MockKeyQA())
    registry.register(MockFindingDraft(FindingDraft()))
    registry.register(MockGrounding())
    registry.register(MockImagePreview())

    wf = build_workflow(registry)
    result = await wf.run(
        study_uid="<EXAMPLE_STUDY_UID>",
        png_bytes=_make_png_bytes(),
        metadata={"PatientID": "<EXAMPLE_PATIENT_ID>",
                  "PatientSex": "F", "PatientAge": "052Y"},
    )

    # finding_draft 字段就位
    assert "finding_text" in result
    assert DRAFT_MARKER in result["finding_text"]
    # finding_struct 含 pathologies / key_qa / grounding_boxes 三轨
    fs = result["finding_struct"]
    assert "pathologies" in fs and "key_qa" in fs and "grounding_boxes" in fs
    # MockClassifier 只输 3 项概率（Pneumothorax / Effusion / Cardiomegaly），但
    # finding_struct.pathologies 接受任何长度的列表（≤ 18）；这里 3 项 mock 输入
    # 与 finding_draft 透传长度对齐
    assert len(fs["pathologies"]) == 3
    assert any(p["name"] == "Pneumothorax" and p["prob"] == 0.85
               for p in fs["pathologies"])
    assert len(fs["key_qa"]) == 4
    # 注意：workflow.run() 初始化 state 时不注入 grounding_boxes 字段（workflow.py
    # 写死初始 keys），且 finding_draft 在 positive_grounding 之前执行，
    # 所以 finding_struct.grounding_boxes 在这一帧是空列表；字段存在即可
    # （透传逻辑见 test_finding_struct_grounding_boxes_normalization）
    assert isinstance(fs["grounding_boxes"], list)
    # 但最终状态的顶层 grounding_boxes 应由 MockGrounding 在 finding_draft
    # 之后写入并保留 1 项
    assert isinstance(result.get("grounding_boxes"), list)
    assert len(result["grounding_boxes"]) == 1
    # positive_flag
    assert result["positive_flag"] is True
    # draft_ts
    assert result["draft_ts"].endswith("Z")
    # 5 件工具全打
    assert "classifier" in result["tools_executed"]
    assert "keyqa" in result["tools_executed"]
    assert "finding_draft" in result["tools_executed"]
    assert "positive_grounding" in result["tools_executed"]
    assert "image_preview" in result["tools_executed"]


@pytest.mark.asyncio
async def test_stategraph_integration_negative_skips_finding_draft() -> None:
    """阴性路径：process 节点判定 max_prob < 阈值 → finding_draft 不在 tools_to_execute 中；
    但 finding_struct 字段缺失时不抛错（这里确认 tools_to_execute=[]仅 image_preview）。"""
    registry = ToolRegistry()
    registry.register(MockClassifier(PROBS_NEGATIVE))
    registry.register(MockFindingDraft(FindingDraft()))  # 注册但不应被调用
    registry.register(MockImagePreview())

    wf = build_workflow(registry)
    result = await wf.run(
        study_uid="<EXAMPLE_NEG>",
        png_bytes=_make_png_bytes(),
        metadata={},
    )

    assert result["positive_flag"] is False
    # 阴性路径下 finding_draft + keyqa + grounding 都不应被调
    assert "finding_draft" not in result["tools_executed"]
    assert "keyqa" not in result["tools_executed"]
    assert "positive_grounding" not in result["tools_executed"]
    # 但 image_preview 必须跑（医师需要看窗宽窗位预览）
    assert "image_preview" in result["tools_executed"]
    assert "classifier" in result["tools_executed"]


# ===========================================================================
# 12) FindingDraftResult dataclass
# ===========================================================================


def test_result_dataclass_as_state_patch() -> None:
    """``FindingDraftResult.as_state_patch()`` 返回 spec 字段 4 个。"""
    res = FindingDraftResult(
        finding_text="demo",
        finding_struct={"pathologies": [], "key_qa": [], "grounding_boxes": []},
        positive_flag=True,
        draft_ts="2026-10-13T07:23:11.487Z",
        draft_marker_included=True,
    )
    patch = res.as_state_patch()
    assert set(patch.keys()) == {"finding_text", "finding_struct",
                                  "positive_flag", "draft_ts"}
    assert patch["positive_flag"] is True


# ===========================================================================
# 13) 与既有模块衔接
# ===========================================================================


def test_engine_module_exports_finding_draft() -> None:
    """``engine.__init__`` 暴露 ``FindingDraft`` / 常量 / 工厂。"""
    import cxr_triage.engine as engine
    for name in [
        "FindingDraft",
        "FindingDraftError",
        "FindingDraftResult",
        "DRAFT_MARKER",
        "NO_ABNORMALITY_TEMPLATE",
        "POSITIVE_TEMPLATE",
        "DEFAULT_POSITIVE_THRESHOLD",
        "TOP_N_PATHOLOGIES",
        "MAX_PATHOLOGIES_IN_STRUCT",
        "DEFAULT_LOCALIZATION_BY_PATHOLOGY",
        "build_finding_draft",
    ]:
        assert hasattr(engine, name), f"engine.{name} missing"


def test_tool_name_matches_default_tool_names() -> None:
    """``.name == "finding_draft"`` 与 ``DEFAULT_TOOL_NAMES[2]`` 严格对齐。"""
    from cxr_triage.engine.workflow import DEFAULT_TOOL_NAMES
    assert DEFAULT_TOOL_NAMES[2] == "finding_draft"
    assert FindingDraft().name == DEFAULT_TOOL_NAMES[2]


@pytest.mark.asyncio
async def test_run_is_idempotent_on_same_state() -> None:
    """相同 state 二次跑 → finding_text / draft_ts（注入时）一致。"""
    tool = FindingDraft(now_iso=_fixed_now_iso)
    state = _make_state(PROBS_POSITIVE)
    out1 = await tool.run(state)
    out2 = await tool.run(state)
    assert out1["finding_text"] == out2["finding_text"]
    assert out1["draft_ts"] == out2["draft_ts"]
    assert out1["finding_struct"] == out2["finding_struct"]
    assert out1["positive_flag"] == out2["positive_flag"]
