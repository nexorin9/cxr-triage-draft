"""tests/test_dicom_processor.py — DICOM 调窗 + PNG 编码。

覆盖（CLAUDE.md Phase 4 验收 · Task 5）：

1. **happy path 阳性 fixture**：已知阳性胸片 → process() 返回 PNG bytes +
   完整 metadata dict；
2. **happy path 阴性 fixture**：已知阴性胸片 → 同样返回合法 PNG；
3. **手动构造 16-bit DICOM**：覆盖 RescaleSlope / WindowCenter / WindowWidth
   自带场景；
4. **窗宽窗位自定义**：override window / level → 输出 PNG 受该窗口约束；
5. **元数据字段完整性**：StudyInstanceUID / Modality / Rows / Columns /
   WindowWidth / WindowLevel / PixelSpacing 都在；
6. **PNG 字节流合法性**：开头 ``\\x89PNG\\r\\n\\x1a\\n`` magic + 可被 PIL 读回；
7. **失败语义**：空 bytes / 全零无效数据 → ``DicomProcessError``；
8. **窗宽 ≤ 0** → ``DicomProcessError``；
9. **窗口 default**：DICOM 内 WindowCenter/WindowWidth 缺失时退回肺窗默认值；
10. **DICOM 内置 DS 多值**：WindowCenter 列表 → 取第一个。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
import pytest
import pytest_asyncio
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    PYDICOM_IMPLEMENTATION_UID,
)

from cxr_triage.pacs import (
    DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH,
    DicomProcessError,
    DicomProcessor,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def processor() -> DicomProcessor:
    """默认肺窗 (W/L=400/40)。"""
    return DicomProcessor()


def _make_cxr_dataset_bytes(
    *,
    study_uid: str,
    pixel: np.ndarray,
    modality: str = "CR",
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    bits_stored: int = 16,
    bits_allocated: int = 16,
    rescale_slope: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    window_center: Optional[Any] = None,
    window_width: Optional[Any] = None,
    photometric: str = "MONOCHROME2",
) -> bytes:
    """把内存 numpy 像素数组打包成最小可解析 DICOM 字节流。"""
    height = int(rows or pixel.shape[0])
    width = int(cols or pixel.shape[1])

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(
        filename_or_obj="<memory>",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.StudyInstanceUID = study_uid
    ds.Modality = modality
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    if rescale_slope is not None:
        ds.RescaleSlope = rescale_slope
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    ds.PixelSpacing = [0.143, 0.143]
    ds.PatientID = "<EXAMPLE_PATIENT>"
    ds.AccessionNumber = "<EXAMPLE_ACC>"

    # PixelData 16-bit
    pixel_u16 = pixel.astype(np.uint16)
    ds.PixelData = pixel_u16.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def _fixture_positive_pixels(seed: int = 42) -> np.ndarray:
    """模拟阳性胸片：右下肺野一片稍暗（高 max_prob）区域。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 800, size=(256, 256), dtype=np.int32)  # 基础噪声
    # 右下肺野「病灶」区域
    img[180:230, 130:200] = 50  # 暗（气胸 = 暗区）
    return img.astype(np.int32)


def _fixture_negative_pixels(seed: int = 7) -> np.ndarray:
    """模拟阴性胸片：均匀肺纹理，无明显高密度病灶。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(400, 700, size=(256, 256), dtype=np.int32)
    return img.astype(np.int32)


# ======================================================================
# 1. happy path — 阳性 fixture
# ======================================================================
def test_dicom_processor_handles_positive_fixture(processor):
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    result = processor.process(dcm_bytes)
    # PNG 字节流合法
    assert result.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    # 尺寸对得上 DICOM
    assert result.png_height == 256
    assert result.png_width == 256
    # metadata 字段齐
    md = result.metadata
    assert md["study_uid"] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
    assert md["modality"] == "CR"
    assert md["rows"] == 256
    assert md["columns"] == 256
    assert md["bits_stored"] == 16
    assert md["photometric"] == "MONOCHROME2"
    assert md["pixel_spacing"] == [0.143, 0.143]
    # 默认肺窗
    assert md["window_width"] == DEFAULT_WINDOW_WIDTH
    assert md["window_level"] == DEFAULT_WINDOW_LEVEL


# ======================================================================
# 2. happy path — 阴性 fixture
# ======================================================================
def test_dicom_processor_handles_negative_fixture(processor):
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>",
        pixel=_fixture_negative_pixels(),
        modality="DX",
    )
    result = processor.process(dcm_bytes)
    assert result.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert result.metadata["modality"] == "DX"


# ======================================================================
# 3. 手动构造：DICOM 内置窗宽窗位被优先使用
# ======================================================================
def test_dicom_processor_honors_dicom_window_when_set():
    """DICOM ``WindowCenter=100`` / ``WindowWidth=200`` → 用它，不退回默认肺窗。"""
    proc = DicomProcessor()  # 默认肺窗
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_W>",
        pixel=_fixture_negative_pixels(seed=99),
        window_center=100,
        window_width=200,
    )
    result = proc.process(dcm_bytes)
    # DICOM 内置窗口覆盖默认肺窗
    assert result.metadata["window_width"] == 200.0
    assert result.metadata["window_level"] == 100.0


# ======================================================================
# 4. override 窗宽窗位 → 严格覆盖
# ======================================================================
def test_dicom_processor_override_window():
    proc = DicomProcessor()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_OVR>",
        pixel=_fixture_positive_pixels(seed=1),
        window_center=500,
        window_width=1500,
    )
    # override 强制改
    result = proc.process(dcm_bytes, window=200, level=30)
    assert result.metadata["window_width"] == 200.0
    assert result.metadata["window_level"] == 30.0


# ======================================================================
# 5. 元数据完整性（用 dataclass 字典）
# ======================================================================
def test_dicom_processor_metadata_complete(processor):
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_META>",
        pixel=_fixture_negative_pixels(seed=11),
    )
    result = processor.process(dcm_bytes)
    md = result.metadata
    expected_keys = {
        "study_uid",
        "patient_id_hash",
        "modality",
        "rows",
        "columns",
        "bits_stored",
        "pixel_spacing",
        "photometric",
        "window_width",
        "window_level",
        "rescale_slope",
        "rescale_intercept",
    }
    assert expected_keys.issubset(set(md.keys()))


# ======================================================================
# 6. PNG 字节流合法性 + 可被 PIL 读回
# ======================================================================
def test_dicom_processor_png_is_loadable_by_pil(processor):
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_PIL>",
        pixel=_fixture_negative_pixels(seed=3),
    )
    result = processor.process(dcm_bytes)
    # PIL 直接读字节流
    img = Image.open(io.BytesIO(result.png_bytes))
    img.load()
    assert img.format == "PNG"
    assert img.size == (256, 256)
    # 灰度图
    assert img.mode in ("L", "LA", "P")


# ======================================================================
# 7. 失败语义：空 bytes / 非 DICOM 字节流
# ======================================================================
def test_dicom_processor_rejects_empty_bytes(processor):
    with pytest.raises(DicomProcessError) as exc:
        processor.process(b"")
    assert "为空" in str(exc.value)


def test_dicom_processor_rejects_garbage_bytes(processor):
    with pytest.raises(DicomProcessError):
        processor.process(b"\x00\x01\x02 not a dicom stream at all \xff\xff")


# ======================================================================
# 8. 窗宽 ≤ 0
# ======================================================================
def test_dicom_processor_rejects_nonpositive_window():
    proc = DicomProcessor()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_NPW>",
        pixel=_fixture_negative_pixels(seed=5),
        window_center=40,
        window_width=0,  # 强制窗宽 0
    )
    with pytest.raises(DicomProcessError):
        proc.process(dcm_bytes)


# ======================================================================
# 9. 默认窗口：DICOM 缺 WindowCenter/Width → 用肺窗
# ======================================================================
def test_dicom_processor_uses_default_when_dicom_missing_window():
    """不设 DICOM WindowCenter/Width → 退回 ``DEFAULT_*``。"""
    proc = DicomProcessor()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_DEF>",
        pixel=_fixture_negative_pixels(seed=8),
        # 不传 window_* → DICOM 没这些字段
    )
    result = proc.process(dcm_bytes)
    assert result.metadata["window_width"] == DEFAULT_WINDOW_WIDTH
    assert result.metadata["window_level"] == DEFAULT_WINDOW_LEVEL


# ======================================================================
# 10. DICOM 内置 DS 多值 WindowCenter（部分设备会写列表）
# ======================================================================
def test_dicom_processor_handles_multi_valued_window():
    """DICOM WindowCenter = [60, 200, 1500] → 取第一个（肺窗）。"""
    proc = DicomProcessor()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_MULTI>",
        pixel=_fixture_negative_pixels(seed=13),
        window_center=[60, 200, 1500],  # 多值
        window_width=[400, 800, 2500],
    )
    result = proc.process(dcm_bytes)
    assert result.metadata["window_level"] == 60
    assert result.metadata["window_width"] == 400
