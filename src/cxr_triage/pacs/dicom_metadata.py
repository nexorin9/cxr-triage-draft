"""DICOM metadata 解析 — 体检胸片 PACS 入站时提取 StudyInstanceUID 等关键字段。

设计原则（spec.md §4 + §5 维度 2 关键实体绑定）：

1. **只解析、不渲染**。本文件不做窗宽窗位 / PNG 转换；只把 DICOM 元数据
   抽成 dict，给 ``incoming_studies`` 入条用。
2. **缺字段兜底**。PHI 字段（PatientID / AccessionNumber 等）允许为空
   字符串，``StudyInstanceUID`` / ``Modality`` 缺失则抛 ``DicomMetadataError``，
   因为它们是状态机的关键锚点（缺一就拒绝入站）。
3. **Modality 白名单**。只接受 ``CR`` / ``DX`` 胸片；其他 modality 一律
   返回 ``None``，由 ``PacsListener`` 记 audit 后丢弃。
4. **PatientID hash**。``patient_id_hash`` 是 ``SHA256(PatientID+salt)``，
   落库后不再回退到明文；salt 由 ``PacsListener`` 从 env 注入。

参考：源仓库 ``medrax/tools/dicom.py:51-94``（metadata dict 提取模式）。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# 体检 PACS 落片只接收 CR / DX；CT / MR / US 等由其它子系统处理。
ALLOWED_MODALITIES: frozenset[str] = frozenset({"CR", "DX"})


class DicomMetadataError(ValueError):
    """关键 DICOM 字段缺失或类型异常时抛。"""


def _get_str(ds, tag_name: str, default: str = "") -> str:
    """安全取 DICOM 字符串字段；缺字段返回 default（空串）。"""
    val = getattr(ds, tag_name, None)
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        # DICOM 多值字段（如 WindowCenter）取第一项
        val = val[0] if val else default
    return str(val).strip()


def patient_id_hash(patient_id: str, salt: str) -> str:
    """``SHA256(PatientID + salt)``。

    用于 ``patient_id_hash`` 落库；明文 PatientID 仅在元数据 dict 中保留
    一次（用于审计），落库后只存 hash。
    """
    if not patient_id:
        return ""
    return hashlib.sha256(f"{patient_id}{salt}".encode("utf-8")).hexdigest()


def parse_cxr_metadata(
    ds,
    *,
    salt: str = "cxr-triage-salt",
) -> Optional[dict[str, Any]]:
    """DICOM ``Dataset`` → 体检胸片入站元数据 dict。

    Parameters
    ----------
    ds:
        ``pydicom.Dataset`` 或 ``pynetdicom`` 事件中的 ``dataset`` 字段。
    salt:
        ``PatientID`` hash 用的盐；与 ``PacsListener`` 启动时一致。

    Returns
    -------
    ``None`` 表示 modality 过滤失败（不是 CR/DX）；其它情况返回 dict。
    关键字段缺失会抛 ``DicomMetadataError``，由 ``PacsListener`` 写 audit 后丢弃。

    字段约定与 ``incoming_studies`` 表列对齐：

    - ``study_uid``         str   PRIMARY KEY
    - ``patient_id``        str
    - ``patient_id_hash``   str   SHA256(PatientID+salt)
    - ``accession_number``  str
    - ``modality``          str   CR / DX
    - ``study_date``        str
    - ``patient_sex``       str
    - ``patient_age``       str
    """
    study_uid = _get_str(ds, "StudyInstanceUID")
    modality = _get_str(ds, "Modality").upper()

    if not study_uid:
        raise DicomMetadataError("StudyInstanceUID 缺失；无法入站")
    if not modality:
        raise DicomMetadataError(f"Modality 缺失（study_uid={study_uid}）")
    if modality not in ALLOWED_MODALITIES:
        logger.info("Modality 过滤：丢弃 modality=%s study_uid=%s", modality, study_uid)
        return None

    patient_id = _get_str(ds, "PatientID")
    return {
        "study_uid": study_uid,
        "patient_id": patient_id,
        "patient_id_hash": patient_id_hash(patient_id, salt) if patient_id else "",
        "accession_number": _get_str(ds, "AccessionNumber"),
        "modality": modality,
        "study_date": _get_str(ds, "StudyDate"),
        "patient_sex": _get_str(ds, "PatientSex"),
        "patient_age": _get_str(ds, "PatientAge"),
    }


def metadata_to_insert_row(
    meta: Mapping[str, Any],
    *,
    source: str = "pacs_cstore",
    status: str = "pending",
) -> tuple:
    """``parse_cxr_metadata`` 的 dict → ``incoming_studies`` INSERT 元组。

    与 ``schema.SCHEMA_DDL`` 列顺序对齐。
    """
    return (
        meta["study_uid"],
        meta.get("patient_id", ""),
        meta.get("patient_id_hash", ""),
        meta.get("accession_number", ""),
        meta["modality"],
        meta.get("study_date", ""),
        meta.get("patient_sex", ""),
        meta.get("patient_age", ""),
        source,
        status,
        0.0,  # max_prob 默认
    )


__all__ = [
    "ALLOWED_MODALITIES",
    "DicomMetadataError",
    "parse_cxr_metadata",
    "patient_id_hash",
    "metadata_to_insert_row",
]
