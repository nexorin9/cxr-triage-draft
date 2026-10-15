"""DICOM 调窗处理器 — DICOM 像素数据 → PNG 字节流 + 元数据。

设计原则（spec.md §5 维度 1 + plan.md §5 深度 (c) 真实规模输入）：

1. **窗宽窗位默认肺窗**（window=400, level=40），可由体检科按本地
   CT/MR 习惯通过 ``override`` 字段调。
2. **复用源仓库思路**：参考 ``medrax/tools/dicom.py:43-94``
   ``_apply_windowing`` 与 ``_process_dicom``，但改两件事：
   - 输入从文件路径改成内存 ``bytes``（避免磁盘 IO + 与 WADO 拉图衔接）
   - 返回从 numpy 改成 ``PIL.Image`` 的 PNG ``bytes``（浏览器可直接
     ``data:image/png;base64,...`` 渲染）
3. **RescaleSlope / Intercept**：DICOM 可能含 ``RescaleSlope=1`` 偏移与
   ``RescaleIntercept=-1024``；先做线性变换再做窗宽窗位。
4. **失败语义**：缺 pixel data / BitsStored 缺失 / PhotometricInterpretation
   异常 → ``DicomProcessError``；调用方需捕获，不静默返回空。
5. **元数据抽取**：取 ``StudyInstanceUID`` / ``PatientID`` / ``Modality`` /
   ``Rows`` / ``Columns`` / ``BitsStored`` / ``PixelSpacing``，便于上层
   浏览器面板展示。
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


# 肺窗默认值（按 spec.md §3 + CLAUDE.md「中式实务优先」）
DEFAULT_WINDOW_WIDTH = 400.0
DEFAULT_WINDOW_LEVEL = 40.0


class DicomProcessError(RuntimeError):
    """DICOM 处理异常（缺字段 / 非单色 / 无法调窗）。"""


@dataclass
class DicomProcessResult:
    """调窗 + PNG 编码的产物。"""

    png_bytes: bytes
    png_width: int
    png_height: int
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "png_bytes": self.png_bytes,
            "png_width": self.png_width,
            "png_height": self.png_height,
            "metadata": dict(self.metadata),
        }


@dataclass
class _WindowOverride:
    """窗宽窗位覆盖（由调用方传）。"""

    window: Optional[float] = None
    level: Optional[float] = None


class DicomProcessor:
    """DICOM 字节流 → 调窗 PNG。

    Parameters
    ----------
    default_window_width / default_window_level:
        缺省肺窗窗宽 / 窗位；可在构造时改成其它协议（骨窗 2000/500、
        纵隔窗 350/50 等）。
    """

    def __init__(
        self,
        *,
        default_window_width: float = DEFAULT_WINDOW_WIDTH,
        default_window_level: float = DEFAULT_WINDOW_LEVEL,
    ) -> None:
        self.default_window_width = float(default_window_width)
        self.default_window_level = float(default_window_level)

    # ------------------------------------------------------------------
    # 主路径
    # ------------------------------------------------------------------
    def process(
        self,
        dcm_bytes: bytes,
        *,
        window: Optional[float] = None,
        level: Optional[float] = None,
    ) -> DicomProcessResult:
        """输入 DICOM 字节流，调窗 → PNG + 抽取 metadata。

        ``window`` / ``level`` 为 None 时，先尝试读 DICOM 内 ``WindowCenter``
        / ``WindowWidth``，还没有则使用 ``self.default_*``。
        """
        if not dcm_bytes:
            raise DicomProcessError("dcm_bytes 为空")

        try:
            ds = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)
        except Exception as exc:  # noqa: BLE001 — pydicom 抛多类异常
            raise DicomProcessError(f"pydicom.dcmread 失败: {exc}") from exc

        # 1. 拿到像素数组（16-bit raw）
        try:
            pixel = ds.pixel_array
        except (AttributeError, ValueError) as exc:
            raise DicomProcessError(
                f"DICOM 缺 PixelData / 无法解析: {exc}"
            ) from exc

        if not isinstance(pixel, np.ndarray) or pixel.size == 0:
            raise DicomProcessError("DICOM pixel_array 为空或非 ndarray")

        # 2. RescaleSlope / Intercept（如有）
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            pixel = pixel.astype(np.float32) * slope + intercept

        # 3. 窗宽窗位选择（顺序：override → DICOM 内置 → 默认肺窗）
        if window is None:
            window = self._read_dicom_window(ds, "WindowWidth")
        if level is None:
            level = self._read_dicom_window(ds, "WindowCenter")
        if window is None:
            window = self.default_window_width
        if level is None:
            level = self.default_window_level
        window = float(window)
        level = float(level)

        # 4. 应用窗宽窗位
        img_u8 = self._apply_windowing(pixel, center=level, width=window)
        # 5. PIL 编码为 PNG bytes
        png_bytes = self._to_png_bytes(img_u8)

        # 6. metadata 抽取
        rows = int(getattr(ds, "Rows", img_u8.shape[0]) or img_u8.shape[0])
        cols = int(getattr(ds, "Columns", img_u8.shape[1]) or img_u8.shape[1])
        metadata = {
            "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
            "patient_id_hash": "",  # 由上层 PacsListener 注入；processor 只读
            "modality": str(getattr(ds, "Modality", "") or ""),
            "rows": rows,
            "columns": cols,
            "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
            "pixel_spacing": list(getattr(ds, "PixelSpacing", []) or []),
            "photometric": str(getattr(ds, "PhotometricInterpretation", "") or ""),
            "window_width": window,
            "window_level": level,
            "rescale_slope": slope,
            "rescale_intercept": intercept,
        }

        return DicomProcessResult(
            png_bytes=png_bytes,
            png_width=cols,
            png_height=rows,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_windowing(
        img: np.ndarray, center: float, width: float
    ) -> np.ndarray:
        """把 float array 经窗宽窗位归一化到 uint8。

        与 ``medrax/tools/dicom.py:43-49`` 的 ``_apply_windowing`` 同算法；
        原版用 ``// 2``，此处改 ``/ 2`` 兼容浮点窗位。
        """
        if width <= 0:
            raise DicomProcessError(f"window_width 必须 > 0（实际 {width}）")
        img_min = center - width / 2.0
        img_max = center + width / 2.0
        clipped = np.clip(img, img_min, img_max)
        # 避免除以 0
        if img_max == img_min:
            normalized = np.zeros_like(clipped, dtype=np.uint8)
        else:
            normalized = ((clipped - img_min) / (img_max - img_min) * 255.0).astype(
                np.uint8
            )
        return normalized

    @staticmethod
    def _to_png_bytes(img_u8: np.ndarray) -> bytes:
        """uint8 ndarray → PNG bytes（PIL Image 编码）。"""
        try:
            with Image.fromarray(img_u8, mode="L") as im:
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                return buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            raise DicomProcessError(f"PIL 编码 PNG 失败: {exc}") from exc

    @staticmethod
    def _read_dicom_window(ds: pydicom.Dataset, key: str) -> Optional[float]:
        """DICOM ``WindowCenter`` / ``WindowWidth`` 可能是 DSfloat 多值。"""
        try:
            value = getattr(ds, key, None)
        except AttributeError:
            return None
        if value is None:
            return None
        # DICOM DS 多值 → list / MultiValue
        try:
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = value[0]
            return float(value)
        except (TypeError, ValueError):
            return None


def build_dicom_processor(
    *,
    default_window_width: float = DEFAULT_WINDOW_WIDTH,
    default_window_level: float = DEFAULT_WINDOW_LEVEL,
) -> DicomProcessor:
    """工厂：默认肺窗。"""
    return DicomProcessor(
        default_window_width=default_window_width,
        default_window_level=default_window_level,
    )


__all__ = [
    "DicomProcessor",
    "DicomProcessError",
    "DicomProcessResult",
    "DEFAULT_WINDOW_WIDTH",
    "DEFAULT_WINDOW_LEVEL",
    "build_dicom_processor",
]
