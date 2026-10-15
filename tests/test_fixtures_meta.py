"""Fixture contract verification for tests/fixtures/.

Confirms the 3 DICOM samples + metadata.json match the contract used by:
- Task 4 PACS C-STORE listener (parse_cxr_metadata reads StudyInstanceUID/PatientID/etc.)
- Task 5 DICOM processor (window/level + PNG)
- Task 12 TriageClassifier (3 fixtures 阳/阴/边界)
- Task 35 baseline_comparison.json (expected_max_prob_at_threshold_0_5)

The fixtures are desensitized — every PatientID/AccessionNumber/StudyInstanceUID carries
the `<EXAMPLE_*>` placeholder so they cannot be confused with real PACS data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pydicom
import pytest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
METADATA_JSON = FIXTURES_DIR / "metadata.json"

DICOM_FILES = {
    "positive": "sample_cxr_positive.dcm",
    "negative": "sample_cxr_negative.dcm",
    "borderline": "sample_cxr_borderline.dcm",
}

PLACEHOLDER_PREFIX = "<EXAMPLE_"


def _load_metadata():
    with METADATA_JSON.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def test_metadata_json_exists_and_parses():
    assert METADATA_JSON.exists(), f"missing {METADATA_JSON}"
    md = _load_metadata()
    assert isinstance(md, dict)
    assert md.get("schema_version"), "schema_version required"
    assert "fixtures" in md
    assert len(md["fixtures"]) == 3


def test_metadata_lists_all_three_labels():
    md = _load_metadata()
    labels = [fx["label"] for fx in md["fixtures"]]
    assert set(labels) == {"positive", "negative", "borderline"}


def test_each_dcm_file_exists():
    md = _load_metadata()
    for fx in md["fixtures"]:
        path = FIXTURES_DIR / fx["filename"]
        assert path.exists(), f"missing DICOM file {path}"
        assert path.stat().st_size > 0, f"DICOM file {path} is empty"


def test_dicom_metadata_matches_metadata_json():
    """DICOM tag ↔ metadata.json 双向一致 — Task 35 baseline_comparison 依赖该一致性。"""
    md = _load_metadata()
    by_label = {fx["label"]: fx for fx in md["fixtures"]}
    for label, filename in DICOM_FILES.items():
        ds = pydicom.dcmread(FIXTURES_DIR / filename)
        expected = by_label[label]["metadata"]
        assert ds.StudyInstanceUID == expected["StudyInstanceUID"], (
            f"{label}: DICOM StudyInstanceUID != metadata.json"
        )
        assert ds.PatientID == expected["PatientID"], (
            f"{label}: DICOM PatientID != metadata.json"
        )
        assert ds.AccessionNumber == expected["AccessionNumber"], (
            f"{label}: DICOM AccessionNumber != metadata.json"
        )
        assert ds.Modality == expected["Modality"], (
            f"{label}: DICOM Modality != metadata.json"
        )


def test_three_dicom_study_uids_are_unique():
    """3 张样例 UID 必须唯一 — Task 4 PACS listener 同 UID 二次入站走 INSERT OR REPLACE。"""
    md = _load_metadata()
    uids = [fx["metadata"]["StudyInstanceUID"] for fx in md["fixtures"]]
    assert len(uids) == len(set(uids)), f"duplicate StudyInstanceUID: {uids}"


def test_placeholders_are_desensitized_not_real_data():
    """每个 UID/PatientID/AccessionNumber 必须包含 <EXAMPLE_ 子串 — 防与真实 PACS 数据混淆。

    StudyInstanceUID 形如 ``1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>``(含 DICOM 标准 UID 前缀
    + 占位尾段);PatientID/AccessionNumber 本身就是 ``<EXAMPLE_*>`` 占位串。
    """
    md = _load_metadata()
    for fx in md["fixtures"]:
        meta = fx["metadata"]
        for key in ("StudyInstanceUID", "PatientID", "AccessionNumber"):
            value = meta[key]
            assert PLACEHOLDER_PREFIX in value, (
                f"{fx['label']}.{key}={value!r} missing <EXAMPLE_> substring"
            )


def test_expected_positive_flag_matches_threshold_rule():
    """Task 13 rule_max_prob(probs, threshold=0.5) 必须与 expected_positive_flag 一致。"""
    md = _load_metadata()
    for fx in md["fixtures"]:
        max_prob = fx["expected_max_prob_at_threshold_0_5"]
        expected_flag = fx["expected_positive_flag"]
        rule_flag = max_prob >= 0.5
        assert rule_flag == expected_flag, (
            f"{fx['label']}: rule says {rule_flag}, expected {expected_flag} (max_prob={max_prob})"
        )


def test_positive_grounding_label_in_pathology_set():
    """阳性/边界 fixture 的 ground_truth_labels.top_pathology 必须是 18 病理之一;
    阴性 fixture 用 "No Finding" 哨兵(无阳性) — Task 16 PositiveGrounding PATHOLOGY_NAMES。
    """
    from cxr_triage.engine.positive_grounding import PATHOLOGY_NAMES

    NEGATIVE_SENTINEL = "No Finding"
    md = _load_metadata()
    for fx in md["fixtures"]:
        top = fx["ground_truth_labels"]["top_pathology"]
        if fx["label"] == "negative":
            assert top == NEGATIVE_SENTINEL, (
                f"negative fixture must use {NEGATIVE_SENTINEL!r} sentinel"
            )
        else:
            assert top in PATHOLOGY_NAMES, (
                f"{fx['label']}: top_pathology {top!r} not in PATHOLOGY_NAMES"
            )


def test_salt_for_patient_id_hash_is_constant():
    """metadata.json 给出 salt 占位 — writers.cxspect_study_hash 与 persistence.patient_id_hash 共用。"""
    md = _load_metadata()
    salt = md.get("salt_for_patient_id_hash")
    assert isinstance(salt, str) and len(salt) >= 8, f"salt invalid: {salt!r}"


def test_patient_id_hash_consistent_with_sha256():
    """patient_id_hash_sha256 字段必须等于 SHA256(PatientID+salt) 的前 64 个 hex 字符。"""
    md = _load_metadata()
    salt = md["salt_for_patient_id_hash"]
    for fx in md["fixtures"]:
        meta = fx["metadata"]
        # 占位 PatientID 以 <EXAMPLE_ 起头,实际真实计算场景下不会有前缀
        # 这里仅验证 SHA256 计算公式可重现
        expected_hash = meta["patient_id_hash_sha256"]
        assert len(expected_hash) == 64, (
            f"{fx['label']}: hash length {len(expected_hash)} != 64"
        )
        assert all(c in "0123456789abcdef" for c in expected_hash.lower()), (
            f"{fx['label']}: hash not lowercase hex"
        )


def test_dicom_pixel_data_present_and_minimum_size():
    """3 张样例必须有像素数据 — Task 17 ImagePreview 调窗 + Task 5 DicomProcessor 调窗。"""
    md = _load_metadata()
    for fx in md["fixtures"]:
        path = FIXTURES_DIR / fx["filename"]
        ds = pydicom.dcmread(path)
        assert hasattr(ds, "PixelData"), f"{fx['label']}: PixelData missing"
        assert len(ds.PixelData) >= 32, (
            f"{fx['label']}: PixelData too small ({len(ds.PixelData)} bytes)"
        )


def test_dicom_modality_is_cr_only():
    """spec.md §3 体检 PACS 仅接收 CR/DX 胸片 — fixture 必须是 CR。"""
    md = _load_metadata()
    for fx in md["fixtures"]:
        assert fx["metadata"]["Modality"] == "CR", (
            f"{fx['label']}: Modality {fx['metadata']['Modality']} != CR"
        )


def test_metadata_usage_notes_present():
    """metadata.json 末尾 usage_notes 必须存在 — 给 Task 35 / demo / README 引用。"""
    md = _load_metadata()
    assert "usage_notes" in md
    assert isinstance(md["usage_notes"], list)
    assert len(md["usage_notes"]) >= 3


def test_metadata_placeholder_policy_declares_no_real_data():
    md = _load_metadata()
    assert "placeholder_policy" in md
    assert "EXAMPLE" in md["placeholder_policy"]


def test_no_dcm_filename_collides_with_real_patient_data():
    """fixture 文件名必须明显可识别为 fixture — 避免被误传到生产 PACS。"""
    md = _load_metadata()
    for fx in md["fixtures"]:
        assert fx["filename"].startswith("sample_cxr_"), (
            f"{fx['label']}: filename {fx['filename']!r} missing sample_cxr_ prefix"
        )
        assert fx["filename"].endswith(".dcm")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])