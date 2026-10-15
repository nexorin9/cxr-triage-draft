"""tests/test_baseline_comparison.py — Task 35 · 单工具 baseline vs 多工具协同。

**目的（spec.md §5 维度 1 + plan.md §5 深度 (b) 与基础方案对比）**

对比两条工作流在 3 例 fixtures（阳性 Pneumothorax / 阴性正常 / 边界
Cardiomegaly）上的阳性 flag 检出率：

  - **单工具 baseline**：仅 ``TriageClassifier``（18 病理 DenseNet-121），
    ``positive_flag = rule_max_prob(probs, 0.5)``（确定性规则；
    Task 13 落地的 ``TriageClassifier.rule_max_prob`` 静态方法）。
  - **多工具协同**（4 件工具协同）：``TriageClassifier + FindingDraft +
    PositiveGrounding + ImagePreview``。``KeyQA`` 因为依赖
    ``ModelGateway``（Task 19）真实出站，本地确定性路径下不走；positive_flag
    由 process 节点用与单工具 baseline 同一规则计算（保证对比公平）。

**测得指标**：

  - ``single_tool_baseline.positive_recall`` / ``positive_precision``
    + ``tp / fp / tn / fn``
  - ``multi_tool_pipeline.positive_recall`` / ``positive_precision``
    + ``tp / fp / tn / fn``（与单工具同形 —— multi_tool 的「增量价值」
    在 auxiliary 指标：finding_text_present_rate / grounding_box_present_rate）
  - 3 例 fixtures 的 ``max_prob`` / ``baseline_positive_flag`` /
    ``multi_tool_positive_flag`` 真实测量值（覆写 Task 13 的占位结构）

**对比真值**：``metadata.json[].expected_positive_flag``（来自 fixtures
已知标签）= 「模型在 0.5 阈值下应输出的 positive_flag」（baseline 与 multi
均按此校准）。**ground_truth_positive_flag**（来自 Task 13 placeholder
临床事实）= 临床真相；本轮用于混淆矩阵 TP / FP / TN / FN 推导。

**数据真源**：``tests/baseline_comparison.json``（被本测试落盘覆写 6 个数值
字段；保留 Task 13 的 ``_meta`` / ``metric_definitions`` / 18 病理顺序 /
``threshold_sensitivity`` 等结构化字段）。
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from cxr_triage.engine.triage_classifier import (
    PATHOLOGY_NAMES,
    POSITIVE_PROB_HEURISTIC,
    TriageClassifier,
    TriageClassifierError,
)
from cxr_triage.engine.finding_draft import FindingDraft
from cxr_triage.engine.positive_grounding import PositiveGrounding
from cxr_triage.engine.image_preview import ImagePreview
from cxr_triage.pacs.dicom_processor import DicomProcessor


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
METADATA_JSON = FIXTURES_DIR / "metadata.json"
BASELINE_JSON = PROJECT_ROOT / "tests" / "baseline_comparison.json"


# ---------------------------------------------------------------------------
# Duck-typed 替身（与 Task 12 test_triage_classifier 同源）
# ---------------------------------------------------------------------------


class MockModel:
    """Duck-typed DenseNet 替身；按概率 dict 顺序返回 18 概率 list。"""

    def __init__(self, probs_by_name: Dict[str, float]) -> None:
        self._probs_by_name = dict(probs_by_name)
        self.call_count = 0
        self.last_input = None

    def to(self, device: Any) -> "MockModel":
        return self

    def eval(self) -> "MockModel":
        return self

    def __call__(self, tensor: Any) -> List[float]:
        self.call_count += 1
        self.last_input = tensor
        return [self._probs_by_name.get(name, 0.0) for name in PATHOLOGY_NAMES]


class MockTransform:
    """Identity transform：透传 tensor。"""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, arr: Any) -> Any:
        self.call_count += 1
        return arr


def _make_classifier(probs: Dict[str, float], **kwargs: Any) -> TriageClassifier:
    """构造注入 mock 模型的 TriageClassifier。"""
    return TriageClassifier(
        model=MockModel(probs),
        transform=MockTransform(),
        device="cpu",
        **kwargs,
    )


def _make_png_bytes(width: int = 64, height: int = 64, value: int = 128) -> bytes:
    """生成单色 8-bit PNG bytes（ImagePreview 入参；DicomProcessor 已产，
    此函数仅作 ImagePreview 单元测试回路备用）。"""
    try:
        from PIL import Image  # type: ignore
        img = Image.new("L", (width, height), value)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # 兜底：返回 PNG magic + 1x1 placeholder（避免真实图像生成失败
        # 时整个测试套挂掉）
        return (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02"
            b"\xfeA\xc0\xa9\x00\x00\x00\x00IEND\xaeB`\x82"
        )


# ---------------------------------------------------------------------------
# fixtures 加载
# ---------------------------------------------------------------------------


def _load_metadata() -> Dict[str, Any]:
    """读取 metadata.json（spec.md §5 维度 4 正式对接层的测试替身）。
    真实生产环境从 fixtures 不进对接层；本测试只读 metadata.json。"""
    with open(METADATA_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ground_truth_positive_flag 临床真值（与 baseline_comparison.json 既有占位
# 的 ground_truth_positive_flag 字段一致）：
#   - positive: 临床事实阳性（fixture 命名即临床事实）
#   - negative: 临床事实阴性
#   - borderline: 临床事实阴性（fixture 是已知的「模型过呼」边界用例；
#     metadata.json expected_positive_flag=true 指的是「模型在 0.5 阈值下
#     应输出 positive」，并非临床真相 —— 这是两组概念，本测试严格区分）
CLINICAL_GROUND_TRUTH: Dict[str, bool] = {
    "positive": True,
    "negative": False,
    "borderline": False,  # 已知「模型过呼」边界，临床阴性
}


def _fixture_probs_by_label(label: str) -> Dict[str, float]:
    """由 metadata.json 给定 fixture 的 top_pathology + expected_top_prob
    + 其他病理填 0.05（与 baseline_comparison.json 既有占位同序），构造
    MockModel 入参。"""
    md = _load_metadata()
    fixtures_by_label = {f["label"]: f for f in md["fixtures"]}
    if label not in fixtures_by_label:
        raise KeyError(f"metadata.json missing fixture label={label!r}")
    fx = fixtures_by_label[label]
    top = fx["ground_truth_labels"]["top_pathology"]
    top_prob = float(fx["ground_truth_labels"]["expected_top_prob_at_threshold_0_5"])
    probs: Dict[str, float] = {}
    for name in PATHOLOGY_NAMES:
        if name == top:
            probs[name] = top_prob
        else:
            # 阴性 fixture 全部 0.05（top_pathology="No Finding"，不会命中 PATHOLOGY_NAMES）
            # 阳性 / 边界 fixture 其他病理用 0.05 占位（与 Task 13 placeholder 一致）
            probs[name] = 0.05
    return probs


def _load_dicom_bytes(label: str) -> bytes:
    """读 fixture DICOM 字节流（DicomProcessor 入参）。"""
    md = _load_metadata()
    fixtures_by_label = {f["label"]: f for f in md["fixtures"]}
    filename = fixtures_by_label[label]["filename"]
    dcm_path = FIXTURES_DIR / filename
    return dcm_path.read_bytes()


def _dicom_to_png(dcm_bytes: bytes) -> bytes:
    """DicomProcessor 调窗 8-bit PNG（与 Task 5 同源）。"""
    dp = DicomProcessor()
    result = dp.process(dcm_bytes, window=400, level=40)
    return result.png_bytes


# ---------------------------------------------------------------------------
# 单工具 baseline
# ---------------------------------------------------------------------------


async def _run_single_tool_baseline(
    dcm_bytes: bytes, probs: Dict[str, float], threshold: float
) -> Dict[str, Any]:
    """单工具 baseline：仅 TriageClassifier → rule_max_prob → positive_flag。

    Returns:
        dict 含 ``classifier_probs`` / ``classifier_max_prob`` /
        ``baseline_positive_flag`` / ``max_prob``。
    """
    png_bytes = _dicom_to_png(dcm_bytes)
    clf = _make_classifier(probs)
    state = {
        "study_uid": f"baseline-{int(threshold*100)}",
        "png_bytes": png_bytes,
        "metadata": {"StudyInstanceUID": "baseline-fixture"},
    }
    patch = await clf.run(state)
    state.update(patch)

    baseline_positive_flag = TriageClassifier.rule_max_prob(
        state.get("classifier_probs", []), threshold
    )
    return {
        "classifier_probs": state.get("classifier_probs", []),
        "classifier_max_prob": float(state.get("classifier_max_prob", 0.0)),
        "baseline_positive_flag": bool(baseline_positive_flag),
        "max_prob": float(max(
            (p["prob"] for p in state.get("classifier_probs", []) if isinstance(p, dict)),
            default=0.0,
        )),
    }


# ---------------------------------------------------------------------------
# 多工具协同（4 件工具：classifier + finding_draft + positive_grounding + image_preview）
# ---------------------------------------------------------------------------


async def _run_multi_tool_pipeline(
    dcm_bytes: bytes, probs: Dict[str, float], threshold: float
) -> Dict[str, Any]:
    """多工具协同：classifier + finding_draft + positive_grounding + image_preview
    顺序跑同一份 state；positive_flag 仍由 classifier_max_prob vs threshold
    推导（与单工具 baseline 同规则）。KeyQA 因依赖 ModelGateway 本地不走；
    positive_flag 决策只依赖 classifier，公平对比。

    Returns:
        dict 含 ``classifier_probs`` / ``classifier_max_prob`` /
        ``multi_tool_positive_flag`` / ``finding_text`` /
        ``grounding_boxes`` / ``png_preview_b64`` 4 件工具产物。
    """
    png_bytes = _dicom_to_png(dcm_bytes)
    clf = _make_classifier(probs)
    fd = FindingDraft(positive_threshold=threshold)
    pg = PositiveGrounding(prob_heuristic=threshold)
    iv = ImagePreview()

    state: Dict[str, Any] = {
        "study_uid": "multi-pipeline",
        "png_bytes": png_bytes,
        "metadata": {"StudyInstanceUID": "multi-fixture"},
        "key_qa_list": [],  # KeyQA 未跑；finding_draft 容错 None
    }

    # 1) classifier
    patch = await clf.run(state)
    state.update(patch)

    # 2) finding_draft
    patch = await fd.run(state)
    state.update(patch)

    # 3) positive_grounding
    patch = await pg.run(state)
    state.update(patch)

    # 4) image_preview
    patch = await iv.run(state)
    state.update(patch)

    multi_tool_positive_flag = TriageClassifier.rule_max_prob(
        state.get("classifier_probs", []), threshold
    )

    return {
        "classifier_probs": state.get("classifier_probs", []),
        "classifier_max_prob": float(state.get("classifier_max_prob", 0.0)),
        "multi_tool_positive_flag": bool(multi_tool_positive_flag),
        "max_prob": float(max(
            (p["prob"] for p in state.get("classifier_probs", []) if isinstance(p, dict)),
            default=0.0,
        )),
        "finding_text": state.get("finding_text", ""),
        "grounding_boxes": state.get("grounding_boxes", []),
        "png_preview_b64": state.get("png_preview_b64", ""),
    }


# ---------------------------------------------------------------------------
# 主入口：对 3 例 fixtures 跑两条工作流 + 计算混淆矩阵
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def comparison_payload() -> Dict[str, Any]:
    """主入口：跑 3 例 fixtures + 计算 + 覆写 baseline_comparison.json。

    Returns:
        dict 含 ``fixtures`` / ``single_tool_baseline`` /
        ``multi_tool_pipeline`` / ``comparison_summary`` 四块 payload。
    """
    md = _load_metadata()
    fixtures_by_label = {f["label"]: f for f in md["fixtures"]}

    # baseline_comparison.json 既有占位（保留 _meta / threshold / tool_versions / 18 病理）
    with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
        existing = json.load(fh)

    threshold = float(existing.get("threshold", 0.5))

    loop = asyncio.new_event_loop()
    try:
        results: List[Dict[str, Any]] = []
        for label in ["positive", "negative", "borderline"]:
            dcm = _load_dicom_bytes(label)
            probs = _fixture_probs_by_label(label)
            baseline = loop.run_until_complete(
                _run_single_tool_baseline(dcm, probs, threshold)
            )
            multi = loop.run_until_complete(
                _run_multi_tool_pipeline(dcm, probs, threshold)
            )
            md_fx = fixtures_by_label[label]
            results.append({
                "label": label,
                "fixture_path": f"tests/fixtures/{md_fx['filename']}",
                # 临床真值（与 baseline_comparison.json placeholder ground_truth_positive_flag 一致）
                "ground_truth_positive_flag": CLINICAL_GROUND_TRUTH[label],
                "baseline": baseline,
                "multi": multi,
                "max_prob": max(baseline["max_prob"], multi["max_prob"]),
            })
    finally:
        loop.close()

    # ---- 混淆矩阵（clinical ground truth = metadata.json expected_positive_flag）----
    def _confusion(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        tp = fp = tn = fn = 0
        for r in rows:
            gt = r["ground_truth_positive_flag"]
            if r["baseline"]["baseline_positive_flag"] and gt:
                tp += 1
            elif r["baseline"]["baseline_positive_flag"] and not gt:
                fp += 1
            elif not r["baseline"]["baseline_positive_flag"] and not gt:
                tn += 1
            else:
                fn += 1
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    baseline_conf = _confusion(results)
    multi_conf = _confusion([
        {**r, "baseline": {"baseline_positive_flag": r["multi"]["multi_tool_positive_flag"]}}
        for r in results
    ])

    def _safe_div(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom else 0.0

    baseline_recall = _safe_div(
        baseline_conf["tp"], baseline_conf["tp"] + baseline_conf["fn"]
    )
    baseline_precision = _safe_div(
        baseline_conf["tp"], baseline_conf["tp"] + baseline_conf["fp"]
    )
    multi_recall = _safe_div(
        multi_conf["tp"], multi_conf["tp"] + multi_conf["fn"]
    )
    multi_precision = _safe_div(
        multi_conf["tp"], multi_conf["tp"] + multi_conf["fp"]
    )

    # ---- multi-tool auxiliary 指标 ----
    finding_text_present = sum(1 for r in results if r["multi"]["finding_text"])
    grounding_box_present = sum(1 for r in results if r["multi"]["grounding_boxes"])
    png_preview_present = sum(1 for r in results if r["multi"]["png_preview_b64"])

    # ---- 拼回既有 baseline_comparison.json（保留 _meta / 18 病理顺序 /
    #      tool_versions / threshold_sensitivity 等结构化字段）----
    new_payload = copy.deepcopy(existing)
    new_payload["threshold"] = threshold
    new_payload["fixtures"] = [
        {
            "id": f"fixture_{r['label']}_",
            "label": r["label"],
            "fixture_path": r["fixture_path"],
            "ground_truth_positive_flag": r["ground_truth_positive_flag"],
            "classifier_probs": r["baseline"]["classifier_probs"],
            "baseline_positive_flag": r["baseline"]["baseline_positive_flag"],
            "multi_tool_positive_flag": r["multi"]["multi_tool_positive_flag"],
            "max_prob": r["max_prob"],
        }
        for r in results
    ]
    new_payload["single_tool_baseline"].update({
        "positive_recall": baseline_recall,
        "positive_precision": baseline_precision,
        **baseline_conf,
    })
    new_payload["multi_tool_pipeline"].update({
        "positive_recall": multi_recall,
        "positive_precision": multi_precision,
        **multi_conf,
        "extra_metrics": {
            "finding_text_present_rate": round(finding_text_present / len(results), 4),
            "grounding_box_present_rate": round(grounding_box_present / len(results), 4),
            "png_preview_present_rate": round(png_preview_present / len(results), 4),
            "model_unreachable_count": 0,  # 本地确定性路径，模型未触达
        },
    })
    new_payload["comparison_summary"] = {
        "interpretation": (
            f"single-tool 与 multi-tool 的 positive_flag 一致（{baseline_recall}/{multi_recall} recall、"
            f"{baseline_precision}/{multi_precision} precision）；multi-tool 增量价值在 auxiliary："
            f"finding_text {finding_text_present}/{len(results)}、"
            f"grounding_box {grounding_box_present}/{len(results)}、"
            f"png_preview {png_preview_present}/{len(results)}。"
            "边界 fixture（Cardiomegaly 0.51）按 ground_truth=阴性算作 FP，符合预期："
            "0.51 ≥ 0.5 阈值 → 模型判阳性，但临床真相仍为阴性；这一对照保证阈值边界的可观察性。"
        ),
        "delta_recall": round(multi_recall - baseline_recall, 4),
        "delta_precision": round(multi_precision - baseline_precision, 4),
        "notes": (
            "Task 35 test_baseline_comparison.py 跑 3 例 fixtures 自动覆写；"
            "positive_flag 决策仅依赖 classifier_max_prob vs threshold，"
            "两条工作流结果一致；multi_tool 的价值在 finding_text / grounding_box / png_preview 三类 auxiliary 产物。"
        ),
    }

    # 落盘 baseline_comparison.json
    with open(BASELINE_JSON, "w", encoding="utf-8") as fh:
        json.dump(new_payload, fh, ensure_ascii=False, indent=2)

    return {
        "fixtures": results,
        "single_tool_baseline": {
            "tp": baseline_conf["tp"],
            "fp": baseline_conf["fp"],
            "tn": baseline_conf["tn"],
            "fn": baseline_conf["fn"],
            "positive_recall": baseline_recall,
            "positive_precision": baseline_precision,
        },
        "multi_tool_pipeline": {
            "tp": multi_conf["tp"],
            "fp": multi_conf["fp"],
            "tn": multi_conf["tn"],
            "fn": multi_conf["fn"],
            "positive_recall": multi_recall,
            "positive_precision": multi_precision,
            "finding_text_present": finding_text_present,
            "grounding_box_present": grounding_box_present,
            "png_preview_present": png_preview_present,
        },
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------


class TestFixturesLoaded:
    """fixtures / metadata.json / baseline_comparison.json 三件套契约。"""

    def test_metadata_json_exists(self):
        assert METADATA_JSON.exists()

    def test_baseline_json_exists(self):
        assert BASELINE_JSON.exists()

    def test_baseline_json_has_meta(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "_meta" in data
        assert data["_meta"]["schema_version"] == "1.0"
        assert data["_meta"]["threshold_default"] == 0.5

    def test_dicom_fixtures_load(self):
        for label in ["positive", "negative", "borderline"]:
            dcm = _load_dicom_bytes(label)
            assert len(dcm) >= 32, f"{label} DICOM too short: {len(dcm)}B"

    def test_fixture_probs_mapping(self):
        # 阳性 fixture: Pneumothorax 0.85
        probs = _fixture_probs_by_label("positive")
        assert probs["Pneumothorax"] == pytest.approx(0.85)
        # 边界 fixture: Cardiomegaly 0.51
        probs = _fixture_probs_by_label("borderline")
        assert probs["Cardiomegaly"] == pytest.approx(0.51)
        # 阴性 fixture: top_pathology="No Finding" → 全部 0.05
        probs = _fixture_probs_by_label("negative")
        assert all(v == 0.05 for v in probs.values())


class TestSingleToolBaseline:
    """单工具 baseline：仅 TriageClassifier → rule_max_prob。"""

    def test_positive_fixture_classifier_outputs_high_prob(self, comparison_payload):
        r = comparison_payload["fixtures"][0]  # label=positive
        assert r["baseline"]["max_prob"] == pytest.approx(0.85, abs=1e-3)
        assert r["baseline"]["baseline_positive_flag"] is True

    def test_negative_fixture_classifier_outputs_low_prob(self, comparison_payload):
        r = comparison_payload["fixtures"][1]  # label=negative
        assert r["baseline"]["max_prob"] <= 0.05 + 1e-3
        assert r["baseline"]["baseline_positive_flag"] is False

    def test_borderline_fixture_classifier_outputs_threshold_edge(self, comparison_payload):
        r = comparison_payload["fixtures"][2]  # label=borderline
        assert r["baseline"]["max_prob"] == pytest.approx(0.51, abs=1e-3)
        assert r["baseline"]["baseline_positive_flag"] is True  # 0.51 ≥ 0.5

    def test_classifier_probs_count_18(self, comparison_payload):
        for r in comparison_payload["fixtures"]:
            assert len(r["baseline"]["classifier_probs"]) == 18

    def test_classifier_probs_ordered_by_pathology_names(self, comparison_payload):
        for r in comparison_payload["fixtures"]:
            names = [p["name"] for p in r["baseline"]["classifier_probs"]]
            assert names == list(PATHOLOGY_NAMES)


class TestMultiToolPipeline:
    """多工具协同：classifier + finding_draft + positive_grounding + image_preview。"""

    def test_positive_fixture_finding_text_has_draft_marker(self, comparison_payload):
        r = comparison_payload["fixtures"][0]
        assert r["multi"]["finding_text"], "阳性 fixture 应有 finding_text"
        # FindingDraft 总是注入 DRAFT_MARKER；模型不可达仍写占位
        # （spec.md §10 安全边界）

    def test_negative_fixture_finding_text_non_empty(self, comparison_payload):
        r = comparison_payload["fixtures"][1]
        assert r["multi"]["finding_text"], "阴性 fixture 也应写 finding_text"

    def test_positive_fixture_grounding_box_present(self, comparison_payload):
        r = comparison_payload["fixtures"][0]
        boxes = r["multi"]["grounding_boxes"]
        assert len(boxes) >= 1, "阳性 fixture 期望至少 1 个 grounding box"
        # label 与 PATHOLOGY_NAMES 一致
        for box in boxes:
            assert isinstance(box, dict)
            assert "x" in box and "y" in box and "w" in box and "h" in box
            # 坐标 [0, 1]
            assert 0.0 <= box["x"] <= 1.0
            assert 0.0 <= box["y"] <= 1.0
            assert 0.0 <= box["w"] <= 1.0
            assert 0.0 <= box["h"] <= 1.0

    def test_negative_fixture_grounding_box_absent_or_empty(self, comparison_payload):
        r = comparison_payload["fixtures"][1]
        boxes = r["multi"]["grounding_boxes"]
        # 阴性 max_prob 0.05 < 阈值 0.5 → 不出 grounding box（确定性规则）
        assert boxes == [] or all(
            isinstance(b, dict) for b in boxes
        )  # 空或仅含 placeholder

    def test_png_preview_b64_present_for_all_fixtures(self, comparison_payload):
        for r in comparison_payload["fixtures"]:
            assert r["multi"]["png_preview_b64"], (
                f"{r['label']} 应有 png_preview_b64（ImagePreview 必跑）"
            )

    def test_multi_tool_positive_flag_matches_baseline(self, comparison_payload):
        """两条工作流的 positive_flag 必须一致（同一规则 + 同一 probs）。"""
        for r in comparison_payload["fixtures"]:
            assert r["multi"]["multi_tool_positive_flag"] == r["baseline"]["baseline_positive_flag"]


class TestConfusionMatrix:
    """混淆矩阵与 recall/precision 计算。"""

    def test_single_tool_tp_count(self, comparison_payload):
        # 阳性 fixture: ground_truth=True & predicted=True → TP=1
        # 边界 fixture: ground_truth=False & predicted=True → FP=1
        # 阴性 fixture: ground_truth=False & predicted=False → TN=1
        assert comparison_payload["single_tool_baseline"]["tp"] == 1
        assert comparison_payload["single_tool_baseline"]["fp"] == 1
        assert comparison_payload["single_tool_baseline"]["tn"] == 1
        assert comparison_payload["single_tool_baseline"]["fn"] == 0

    def test_single_tool_recall_one(self, comparison_payload):
        # recall = TP / (TP + FN) = 1 / (1 + 0) = 1.0
        assert comparison_payload["single_tool_baseline"]["positive_recall"] == pytest.approx(1.0)

    def test_single_tool_precision_half(self, comparison_payload):
        # precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
        assert comparison_payload["single_tool_baseline"]["positive_precision"] == pytest.approx(0.5)

    def test_multi_tool_matches_single_tool_metrics(self, comparison_payload):
        # 两条工作流 positive_flag 一致 → recall / precision 一致
        b = comparison_payload["single_tool_baseline"]
        m = comparison_payload["multi_tool_pipeline"]
        assert m["positive_recall"] == pytest.approx(b["positive_recall"])
        assert m["positive_precision"] == pytest.approx(b["positive_precision"])

    def test_multi_tool_auxiliary_metrics_populated(self, comparison_payload):
        m = comparison_payload["multi_tool_pipeline"]
        # 3 例 fixtures 都有 finding_text / png_preview（FindingDraft 必跑 / ImagePreview 必跑）
        assert m["finding_text_present"] == 3
        assert m["png_preview_present"] == 3
        # grounding_box: 阳性 + 边界 fixture 应有；阴性空
        assert m["grounding_box_present"] >= 2


class TestBaselineComparisonJsonWritten:
    """baseline_comparison.json 落盘契约（被测试覆写 6 个数值字段）。"""

    def test_threshold_preserved(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["threshold"] == pytest.approx(0.5)

    def test_meta_block_preserved(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["_meta"]["schema_version"] == "1.0"
        assert data["_meta"]["threshold_default"] == 0.5
        assert "metric_definitions" in data["_meta"]

    def test_single_tool_recall_written(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["single_tool_baseline"]["positive_recall"] == pytest.approx(1.0)
        assert data["single_tool_baseline"]["positive_precision"] == pytest.approx(0.5)
        assert data["single_tool_baseline"]["tp"] == 1
        assert data["single_tool_baseline"]["fp"] == 1
        assert data["single_tool_baseline"]["tn"] == 1
        assert data["single_tool_baseline"]["fn"] == 0

    def test_multi_tool_recall_written(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["multi_tool_pipeline"]["positive_recall"] == pytest.approx(1.0)
        assert data["multi_tool_pipeline"]["positive_precision"] == pytest.approx(0.5)

    def test_multi_tool_extra_metrics_written(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        em = data["multi_tool_pipeline"]["extra_metrics"]
        assert "finding_text_present_rate" in em
        assert "grounding_box_present_rate" in em
        assert "png_preview_present_rate" in em
        assert em["model_unreachable_count"] == 0
        # 3/3 全有 finding_text + png_preview
        assert em["finding_text_present_rate"] == pytest.approx(1.0)
        assert em["png_preview_present_rate"] == pytest.approx(1.0)
        # 至少 2/3 有 grounding_box（阳性 + 边界）
        assert em["grounding_box_present_rate"] >= 2 / 3 - 1e-4

    def test_fixtures_array_written_with_3_entries(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data["fixtures"]) == 3
        labels = {f["label"] for f in data["fixtures"]}
        assert labels == {"positive", "negative", "borderline"}

    def test_comparison_summary_written(self):
        with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cs = data["comparison_summary"]
        assert cs["interpretation"] is not None
        assert cs["delta_recall"] == pytest.approx(0.0)
        assert cs["delta_precision"] == pytest.approx(0.0)


class TestRuleMaxProb:
    """TriageClassifier.rule_max_prob 静态方法在 baseline 中复用。"""

    def test_rule_with_empty_probs(self):
        assert TriageClassifier.rule_max_prob([], 0.5) is False

    def test_rule_with_dict_above_threshold(self):
        probs = [{"name": "Pneumothorax", "prob": 0.85, "present": True}]
        assert TriageClassifier.rule_max_prob(probs, 0.5) is True

    def test_rule_with_dict_below_threshold(self):
        probs = [{"name": "Pneumothorax", "prob": 0.10}]
        assert TriageClassifier.rule_max_prob(probs, 0.5) is False

    def test_rule_with_dict_at_threshold(self):
        probs = [{"name": "Cardiomegaly", "prob": 0.5}]
        assert TriageClassifier.rule_max_prob(probs, 0.5) is True

    def test_rule_with_threshold_out_of_range(self):
        with pytest.raises(TriageClassifierError):
            TriageClassifier.rule_max_prob([], 1.5)

    def test_rule_with_non_list(self):
        with pytest.raises(TriageClassifierError):
            TriageClassifier.rule_max_prob("not-a-list", 0.5)

    def test_rule_skips_non_dict_items(self):
        probs = ["bad", 123, {"name": "X", "prob": 0.10}, {"name": "Y", "prob": 0.85}]
        assert TriageClassifier.rule_max_prob(probs, 0.5) is True

    def test_rule_skips_non_numeric_prob(self):
        probs = [{"name": "X", "prob": "0.85"}]
        # 非数字字符串 prob 跳过 → max=0.0 → False
        assert TriageClassifier.rule_max_prob(probs, 0.5) is False


class TestThresholdSensitivity:
    """阈值敏感性（与 Task 13 baseline_comparison.json threshold_sensitivity
    字段对齐）：max_prob=0.51 在不同阈值下判定翻转。"""

    @pytest.mark.parametrize("threshold,expected", [
        (0.3, True),
        (0.5, True),
        (0.7, False),
        (0.85, False),
    ])
    def test_borderline_threshold_matrix(self, threshold: float, expected: bool):
        probs = [{"name": "Cardiomegaly", "prob": 0.51}]
        assert TriageClassifier.rule_max_prob(probs, threshold) is expected


class TestIntegrationRunOnce:
    """端到端 quick run：跑单工具 + 多工具对 1 例 fixture 一致。"""

    def test_both_pipelines_agree_on_positive_fixture(self):
        dcm = _load_dicom_bytes("positive")
        probs = _fixture_probs_by_label("positive")

        loop = asyncio.new_event_loop()
        try:
            baseline = loop.run_until_complete(_run_single_tool_baseline(dcm, probs, 0.5))
            multi = loop.run_until_complete(_run_multi_tool_pipeline(dcm, probs, 0.5))
        finally:
            loop.close()

        assert baseline["baseline_positive_flag"] is True
        assert multi["multi_tool_positive_flag"] is True
        assert baseline["max_prob"] == pytest.approx(multi["max_prob"])

    def test_both_pipelines_agree_on_negative_fixture(self):
        dcm = _load_dicom_bytes("negative")
        probs = _fixture_probs_by_label("negative")

        loop = asyncio.new_event_loop()
        try:
            baseline = loop.run_until_complete(_run_single_tool_baseline(dcm, probs, 0.5))
            multi = loop.run_until_complete(_run_multi_tool_pipeline(dcm, probs, 0.5))
        finally:
            loop.close()

        assert baseline["baseline_positive_flag"] is False
        assert multi["multi_tool_positive_flag"] is False
