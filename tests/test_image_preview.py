"""tests/test_image_preview.py — ImagePreview 调窗 PNG 预览 + base64 data URL。

覆盖（CLAUDE.md Phase 4 验收 · Task 17）：

1. **常量层**：DEFAULT_WINDOW_WIDTH=400 / DEFAULT_WINDOW_LEVEL=40；
   PNG_MAGIC 与 ``\\x89PNG\\r\\n\\x1a\\n`` 一致；PNG_DATA_URL_PREFIX
   = ``data:image/png;base64,``；PREVIEW_MAX_BASE64_BYTES = 350_000。
2. **构造期校验**：default_window_width / level 接受 int / float /
   数字字符串，拒绝 bool / None / 非数字字符串 / ≤ 0；
   ``max_base64_bytes`` 拒绝非 int / < 1。
3. **静态工具 ``encode_png_to_b64``**：合法 PNG bytes → 合法 base64 串；
   非 bytes / 空 / 非 PNG magic 抛 ``ImagePreviewError``。
4. **静态工具 ``build_data_url``**：合法 b64 + png mime →
   ``data:image/png;base64,<...>``；b64 含非 base64 字符 / 空 / 非 str
   抛 ``ImagePreviewError``；MIME 支持 png / jpeg / jpg / gif / webp，
   拒绝其它类型。
5. **happy path 3 例 fixtures（阳/阴/边界）**：经 DicomProcessor 调窗
   → PNG bytes → ImagePreview.run → 返回 ``png_preview_b64`` /
   ``png_data_url`` / ``png_byte_len``；``png_data_url`` 前缀严格等于
   ``data:image/png;base64,``；base64 解码回 PNG 字节流合法；
   ``<img src=...>`` 渲染所需的合法 MIME 前缀齐备。
6. **缺 png_bytes / dcm_bytes** → ``ImagePreviewError``；state 给 dcm_bytes
   + override → 重新调窗（re_windowed=True）；窗口代填合法 → 出图。
7. **窗口 override 副作用**（dcm_bytes 路径）：override window 与 default
   不同 → output 不报 default；dcm_bytes 缺 + override 有 → 仍抛
   ``ImagePreviewError``。
8. **state 字段透出**：``png_width`` / ``png_height`` / ``window_width`` /
   ``window_level`` 都在 out dict；``png_preview_b64`` / ``png_data_url``
   串长在 ``png_byte_len`` 一致。
9. **StateGraph 集成**：ToolRegistry 注册 5 件工具（含真 ImagePreview +
   4 件 mock tools）→ workflow 跑通 happy-path；
   tools_executed 含 ``"image_preview"``；png_preview_b64 / png_data_url
   都产出。
10. **降级行为**：base64 长度超 max_base64_bytes → ``degraded=True`` +
    ``degradation_reasons`` 含 ``image_preview_oversized``；tool_calls
    不阻断。
11. **工具名契约** ``.name = "image_preview"`` 与 ``DEFAULT_TOOL_NAMES[4]``
    严格对齐。
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    PYDICOM_IMPLEMENTATION_UID,
)

from cxr_triage.engine import (
    DEFAULT_TOOL_NAMES,
    ImagePreview,
    ImagePreviewError,
    ImagePreviewResult,
    PNG_DATA_URL_PREFIX,
    PNG_MAGIC,
    PREVIEW_MAX_BASE64_BYTES,
    build_data_url,
    build_image_preview,
    encode_png_to_b64,
)
from cxr_triage.engine.workflow import ToolRegistry, build_workflow
from cxr_triage.pacs import (
    DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH,
    DicomProcessor,
)


# ---------------------------------------------------------------------------
# Helpers — 与 test_dicom_processor.py 共享的 _make_cxr_dataset_bytes
# ---------------------------------------------------------------------------


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
    window_center: Optional[object] = None,
    window_width: Optional[object] = None,
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

    pixel_u16 = pixel.astype(np.uint16)
    ds.PixelData = pixel_u16.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def _fixture_positive_pixels(seed: int = 42) -> np.ndarray:
    """阳性 fixture：右下肺野气胸征象暗区。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 800, size=(256, 256), dtype=np.int32)
    img[180:230, 130:200] = 50
    return img.astype(np.int32)


def _fixture_negative_pixels(seed: int = 7) -> np.ndarray:
    """阴性 fixture：均匀肺纹理，无明显病灶。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(400, 700, size=(256, 256), dtype=np.int32)
    return img.astype(np.int32)


def _fixture_borderline_pixels(seed: int = 21) -> np.ndarray:
    """边界 fixture：心影临界（密度 600-650 → 触发 0.51 边界）。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(300, 800, size=(256, 256), dtype=np.int32)
    img[100:170, 80:180] = 620
    return img.astype(np.int32)


@pytest.fixture
def processor() -> DicomProcessor:
    return DicomProcessor()


@pytest.fixture
def image_preview() -> ImagePreview:
    return ImagePreview()


# ---------------------------------------------------------------------------
# 1. 常量层
# ---------------------------------------------------------------------------


def test_default_window_constants_match_lung_window():
    """默认肺窗窗宽 400 / 窗位 40（与 pacs.DicomProcessor 一致）。"""
    assert DEFAULT_WINDOW_WIDTH == 400.0
    assert DEFAULT_WINDOW_LEVEL == 40.0


def test_png_magic_bytes_match_png_spec():
    """PNG magic = b'\\x89PNG\\r\\n\\x1a\\n'。"""
    assert PNG_MAGIC == b"\x89PNG\r\n\x1a\n"
    assert len(PNG_MAGIC) == 8


def test_data_url_prefix_is_png():
    """data URL 前缀 = 'data:image/png;base64,'（浏览器 <img> 标准）。"""
    assert PNG_DATA_URL_PREFIX == "data:image/png;base64,"


def test_max_base64_bytes_cap_constant():
    """上限 350000（约 262KB PNG 原始）。"""
    assert PREVIEW_MAX_BASE64_BYTES == 350_000
    assert isinstance(PREVIEW_MAX_BASE64_BYTES, int)
    assert PREVIEW_MAX_BASE64_BYTES > 0


# ---------------------------------------------------------------------------
# 2. 构造期校验
# ---------------------------------------------------------------------------


def test_image_preview_default_construction():
    """默认构造用肺窗。"""
    tool = ImagePreview()
    assert tool.name == "image_preview"
    assert tool.default_window_width == 400.0
    assert tool.default_window_level == 40.0
    assert tool.max_base64_bytes == PREVIEW_MAX_BASE64_BYTES


def test_image_preview_accepts_numeric_strings():
    """窗宽 / 窗位接受数字字符串（"400" / "40"）。"""
    tool = ImagePreview(default_window_width="400", default_window_level="40")
    assert tool.default_window_width == 400.0
    assert tool.default_window_level == 40.0


def test_image_preview_accepts_custom_windows():
    """自定义窗（如骨窗 2000/500 / 纵隔窗 350/50）。"""
    tool = ImagePreview(default_window_width=2000, default_window_level=500)
    assert tool.default_window_width == 2000.0
    assert tool.default_window_level == 500.0


@pytest.mark.parametrize("bad_value", [True, False, None, "abc", ""])
def test_image_preview_rejects_non_numeric_window_width(bad_value):
    with pytest.raises(ImagePreviewError):
        ImagePreview(default_window_width=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_image_preview_rejects_non_positive_window_width(bad_value):
    with pytest.raises(ImagePreviewError):
        ImagePreview(default_window_width=bad_value)


@pytest.mark.parametrize("bad_value", [True, False, None, "abc", ""])
def test_image_preview_rejects_non_numeric_window_level(bad_value):
    with pytest.raises(ImagePreviewError):
        ImagePreview(default_window_level=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, "350", 1.5, None, True])
def test_image_preview_rejects_bad_max_base64_bytes(bad_value):
    with pytest.raises(ImagePreviewError):
        ImagePreview(max_base64_bytes=bad_value)


def test_build_image_preview_factory():
    tool = build_image_preview()
    assert isinstance(tool, ImagePreview)
    assert tool.default_window_width == 400.0
    assert tool.default_window_level == 40.0


# ---------------------------------------------------------------------------
# 3. 静态工具 encode_png_to_b64
# ---------------------------------------------------------------------------


def test_encode_png_to_b64_happy_path():
    """最小合法 PNG → base64 串。"""
    img = np.zeros((8, 8), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = encode_png_to_b64(png_bytes)
    assert isinstance(b64, str)
    # 可被反向 decode 验证
    decoded = base64.b64decode(b64)
    assert decoded == png_bytes


def test_encode_png_to_b64_with_3_fixtures(processor):
    """3 例 fixtures（阳/阴/边界）→ encode 后 base64 可被解码回原 PNG。"""
    fixtures = [
        ("1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>", _fixture_positive_pixels(), "CR"),
        ("1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>", _fixture_negative_pixels(), "DX"),
        ("1.2.826.0.1.3680043.8.498.<EXAMPLE_BORDER>", _fixture_borderline_pixels(), "CR"),
    ]
    for uid, pix, mod in fixtures:
        dcm_bytes = _make_cxr_dataset_bytes(
            study_uid=uid, pixel=pix, modality=mod
        )
        png_result = processor.process(dcm_bytes)
        b64 = encode_png_to_b64(png_result.png_bytes)
        decoded = base64.b64decode(b64)
        assert decoded == bytes(png_result.png_bytes)
        # 浏览器友好的字符集
        assert set(b64) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        )


def test_encode_png_to_b64_rejects_non_bytes():
    with pytest.raises(ImagePreviewError):
        encode_png_to_b64("not bytes")  # type: ignore[arg-type]


def test_encode_png_to_b64_rejects_empty():
    with pytest.raises(ImagePreviewError):
        encode_png_to_b64(b"")


def test_encode_png_to_b64_rejects_non_png_magic():
    """不带 PNG magic 头 → 抛错。"""
    with pytest.raises(ImagePreviewError):
        encode_png_to_b64(b"NOTPNG\x00\x01\x02\x03")


def test_encode_png_to_b64_rejects_truncated():
    """不足 PNG magic 长度 → 抛错。"""
    with pytest.raises(ImagePreviewError):
        encode_png_to_b64(b"\x89PN\x00")


# ---------------------------------------------------------------------------
# 4. 静态工具 build_data_url
# ---------------------------------------------------------------------------


def test_build_data_url_png_happy_path():
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    url = build_data_url(b64)
    assert url == f"data:image/png;base64,{b64}"
    assert url.startswith("data:image/png;base64,")


def test_build_data_url_jpeg_alias():
    """jpeg 和 jpg 都被规范化为 image/jpeg。"""
    b64 = "abcd"
    assert build_data_url(b64, mime="jpeg") == "data:image/jpeg;base64,abcd"
    assert build_data_url(b64, mime="jpg") == "data:image/jpeg;base64,abcd"


@pytest.mark.parametrize("mime", ["gif", "webp"])
def test_build_data_url_gif_webp(mime):
    b64 = "abcd"
    url = build_data_url(b64, mime=mime)
    assert url == f"data:image/{mime};base64,{b64}"


def test_build_data_url_rejects_empty_b64():
    with pytest.raises(ImagePreviewError):
        build_data_url("")


def test_build_data_url_rejects_non_string_b64():
    with pytest.raises(ImagePreviewError):
        build_data_url(b"abc")  # type: ignore[arg-type]


def test_build_data_url_rejects_non_base64_chars():
    """含非 base64 字符（如 '!'）→ 抛错。"""
    with pytest.raises(ImagePreviewError):
        build_data_url("hello!world")


def test_build_data_url_rejects_unsupported_mime():
    with pytest.raises(ImagePreviewError):
        build_data_url("abcd", mime="bmp")


# ---------------------------------------------------------------------------
# 5. happy path 3 例 fixtures (run())
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_preview_run_with_positive_fixture(image_preview, processor):
    """阳性 fixture：run() 返回 base64 + data_url，浏览器可渲。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    png_result = processor.process(dcm_bytes)
    state = {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>",
        "metadata": {"modality": "CR"},
        "png_bytes": bytes(png_result.png_bytes),
    }
    out = await image_preview.run(state)
    assert "png_preview_b64" in out
    assert "png_data_url" in out
    assert out["png_preview_b64"] == base64.b64encode(png_result.png_bytes).decode()
    assert out["png_data_url"] == f"data:image/png;base64,{out['png_preview_b64']}"
    assert out["png_data_url"].startswith(PNG_DATA_URL_PREFIX)
    assert out["png_byte_len"] == len(out["png_preview_b64"])
    assert out["png_width"] == 256
    assert out["png_height"] == 256
    assert out["window_width"] == DEFAULT_WINDOW_WIDTH
    assert out["window_level"] == DEFAULT_WINDOW_LEVEL
    assert out["re_windowed"] is False


@pytest.mark.asyncio
async def test_image_preview_run_with_negative_fixture(image_preview, processor):
    """阴性 fixture：同样能出合法 preview（医师需看图复核）。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>",
        pixel=_fixture_negative_pixels(),
        modality="DX",
    )
    png_result = processor.process(dcm_bytes)
    state = {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>",
        "metadata": {"modality": "DX"},
        "png_bytes": bytes(png_result.png_bytes),
    }
    out = await image_preview.run(state)
    assert out["png_data_url"].startswith(PNG_DATA_URL_PREFIX)
    # base64 解码回去能 PIL.Image.open
    decoded = base64.b64decode(out["png_preview_b64"])
    Image.open(io.BytesIO(decoded)).load()
    assert out["re_windowed"] is False


@pytest.mark.asyncio
async def test_image_preview_run_with_borderline_fixture(image_preview, processor):
    """边界 fixture：仍走 image_preview（医师需要看图判断 max_prob 边界）。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BORDER>",
        pixel=_fixture_borderline_pixels(),
        modality="CR",
    )
    png_result = processor.process(dcm_bytes)
    state = {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_BORDER>",
        "metadata": {"modality": "CR"},
        "png_bytes": bytes(png_result.png_bytes),
    }
    out = await image_preview.run(state)
    assert out["png_data_url"].startswith(PNG_DATA_URL_PREFIX)
    decoded = base64.b64decode(out["png_preview_b64"])
    assert decoded[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# 6. 入参校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_preview_run_raises_when_state_empty(image_preview):
    """state 既无 png_bytes 也无 dcm_bytes → ImagePreviewError。"""
    with pytest.raises(ImagePreviewError):
        await image_preview.run({"study_uid": "1.2.826.<EXAMPLE>"})


@pytest.mark.asyncio
async def test_image_preview_run_raises_when_png_bytes_empty(image_preview):
    """png_bytes 为空 bytes → 抛错（不让图就静默出空 base64）。"""
    with pytest.raises(ImagePreviewError):
        await image_preview.run({"study_uid": "1.2.826.<EXAMPLE>", "png_bytes": b""})


@pytest.mark.asyncio
async def test_image_preview_run_raises_when_png_bytes_invalid_magic(image_preview):
    """png_bytes 非 PNG magic → 抛错。"""
    with pytest.raises(ImagePreviewError):
        await image_preview.run(
            {"study_uid": "1.2.826.<EXAMPLE>", "png_bytes": b"NOTPNG1234"}
        )


@pytest.mark.asyncio
async def test_image_preview_run_rejects_non_bytes_png(image_preview):
    """png_bytes 非 bytes 类型（字符串）→ 抛错。"""
    with pytest.raises(ImagePreviewError):
        await image_preview.run({"study_uid": "1.2.826.<EXAMPLE>", "png_bytes": "abc"})


# ---------------------------------------------------------------------------
# 7. dcm_bytes + override 重新调窗
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_preview_rewindows_dcm_bytes_when_no_png_bytes():
    """state 只给 dcm_bytes（无 png_bytes）→ ImagePreview 内部重新调窗。"""
    tool = ImagePreview()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_REW>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    out = await tool.run({"study_uid": "1.2.826.<EXAMPLE_REW>", "dcm_bytes": dcm_bytes})
    assert out["re_windowed"] is True
    decoded = base64.b64decode(out["png_preview_b64"])
    assert decoded[:8] == PNG_MAGIC


@pytest.mark.asyncio
async def test_image_preview_rewindow_with_override_takes_effect():
    """state 给 dcm_bytes + window override → 工具内用 override 调窗。"""
    tool = ImagePreview(default_window_width=2000, default_window_level=500)
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BONE>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    out = await tool.run(
        {
            "study_uid": "1.2.826.<EXAMPLE_BONE>",
            "dcm_bytes": dcm_bytes,
            "window_width": 1500,
            "window_level": 400,
        }
    )
    assert out["re_windowed"] is True
    # override 反映在返回的 window_* 字段上
    assert out["window_width"] == 2000.0  # tool 自己的 default
    # 实际预览反映了重调窗的图（PNG 字节流合法）
    assert out["png_data_url"].startswith(PNG_DATA_URL_PREFIX)


@pytest.mark.asyncio
async def test_image_preview_rejects_dcm_bytes_with_bad_override():
    """dcm_bytes 路径下 override window 非数字字符串 → ImagePreviewError。"""
    tool = ImagePreview()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BAD_OVR>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    with pytest.raises(ImagePreviewError):
        await tool.run(
            {
                "study_uid": "1.2.826.<EXAMPLE_BAD_OVR>",
                "dcm_bytes": dcm_bytes,
                "window_width": "abc",
            }
        )


@pytest.mark.asyncio
async def test_image_preview_rejects_bool_override():
    """override window 为 bool → ImagePreviewError。"""
    tool = ImagePreview()
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BOOL>",
        pixel=_fixture_positive_pixels(),
        modality="CR",
    )
    with pytest.raises(ImagePreviewError):
        await tool.run(
            {
                "study_uid": "1.2.826.<EXAMPLE_BOOL>",
                "dcm_bytes": dcm_bytes,
                "window_width": True,
            }
        )


# ---------------------------------------------------------------------------
# 8. State 字段透出
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_preview_run_out_keys_match_spec(image_preview, processor):
    """out dict 字段完整且类型正确。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_KEYS>",
        pixel=_fixture_positive_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    out = await image_preview.run(
        {"study_uid": "1.2.826.<EXAMPLE_KEYS>", "png_bytes": bytes(png_result.png_bytes)}
    )
    expected_keys = {
        "png_preview_b64", "png_data_url", "png_byte_len",
        "window_width", "window_level", "png_width", "png_height",
        "re_windowed",
    }
    assert set(out.keys()) >= expected_keys
    assert isinstance(out["png_preview_b64"], str)
    assert isinstance(out["png_data_url"], str)
    assert isinstance(out["png_byte_len"], int)
    assert isinstance(out["window_width"], float)
    assert isinstance(out["window_level"], float)
    assert isinstance(out["png_width"], int)
    assert isinstance(out["png_height"], int)
    assert isinstance(out["re_windowed"], bool)


@pytest.mark.asyncio
async def test_image_preview_byte_len_matches_b64_length(image_preview, processor):
    """png_byte_len == len(png_preview_b64)（监控 SSE payload）。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_LEN>",
        pixel=_fixture_negative_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    out = await image_preview.run(
        {"study_uid": "1.2.826.<EXAMPLE_LEN>", "png_bytes": bytes(png_result.png_bytes)}
    )
    assert out["png_byte_len"] == len(out["png_preview_b64"])


# ---------------------------------------------------------------------------
# 9. StateGraph 集成
# ---------------------------------------------------------------------------


class _StubTool:
    """StateGraph 集成测试用 mock 工具（与既有测试同形）。"""

    def __init__(self, name: str, outputs: dict):
        self.name = name
        self._outputs = outputs

    async def run(self, state):
        return dict(self._outputs)


@pytest.mark.asyncio
async def test_image_preview_in_tool_registry_with_other_tools(processor):
    """ImagePreview + 4 mock tools 注册到 ToolRegistry → build_workflow 跑通。"""
    reg = ToolRegistry()
    # Mock classifier with positive prob
    reg.register(
        _StubTool(
            "classifier",
            {
                "classifier_probs": [
                    {"name": "Pneumothorax", "prob": 0.85, "present": True}
                ],
                "classifier_max_prob": 0.85,
            },
        )
    )
    reg.register(_StubTool("keyqa", {"key_qa_list": []}))
    reg.register(
        _StubTool(
            "finding_draft",
            {
                "finding_text": "右下肺野见气胸征象（草稿）。",
                "finding_struct": {"pathologies": [], "key_qa": [], "grounding_boxes": []},
            },
        )
    )
    reg.register(_StubTool("positive_grounding", {"grounding_boxes": []}))

    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_REG>",
        pixel=_fixture_positive_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    reg.register(build_image_preview())

    workflow = build_workflow(registry=reg, positive_threshold=0.5, max_cycles=20)

    state = await workflow.run(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_REG>",
        png_bytes=bytes(png_result.png_bytes),
        metadata={"modality": "CR"},
    )
    assert "image_preview" in state["tools_executed"]
    assert state["png_data_url"].startswith(PNG_DATA_URL_PREFIX)
    assert state["png_byte_len"] == len(state["png_preview_b64"])


@pytest.mark.asyncio
async def test_image_preview_runs_only_in_negative_path(processor):
    """阴性路径只跑 classifier + image_preview（image_preview 总要出图）。"""
    reg = ToolRegistry()

    # 用 _NoOp 工具 — 若被误调，会写一个特殊字段供测试发现
    class _NoOp:
        def __init__(self, name): self.name = name

        async def run(self, state):
            # 阴性路径不应该被调到这里；记录一个信号
            return {"_should_not_be_called": self.name}

    reg.register(_NoOp("classifier"))  # placeholder — 真正的 classifier 用 _StubTool
    # 替换 classifier 为产生 0.05 max_prob 的 stub
    try:
        reg._tools.pop("classifier")
    except KeyError:
        pass
    reg.register(
        _StubTool(
            "classifier",
            {
                "classifier_probs": [{"name": "Pneumothorax", "prob": 0.05}],
                "classifier_max_prob": 0.05,
            },
        )
    )
    # 剩下的工具用 _NoOp — 阴性路径不应被调用
    reg.register(_NoOp("keyqa"))
    reg.register(_NoOp("finding_draft"))
    reg.register(_NoOp("positive_grounding"))

    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG_PATH>",
        pixel=_fixture_negative_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    reg.register(build_image_preview())

    workflow = build_workflow(registry=reg, positive_threshold=0.5, max_cycles=20)

    state = await workflow.run(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG_PATH>",
        png_bytes=bytes(png_result.png_bytes),
        metadata={"modality": "DX"},
    )
    assert state["tools_executed"] == ["classifier", "image_preview"]
    assert state["png_data_url"].startswith(PNG_DATA_URL_PREFIX)
    # 阴性路径不应有 should_not_be_called 信号（3 个 _NoOp 都没被调用）
    assert "_should_not_be_called" not in state


# ---------------------------------------------------------------------------
# 10. 降级行为
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_preview_oversized_sets_degraded_banner(image_preview, processor, monkeypatch):
    """base64 超 max_base64_bytes → degraded + degradation_reasons。"""
    # 用极小上限触发过大告警
    tool = ImagePreview(max_base64_bytes=10)
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BIG>",
        pixel=_fixture_positive_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    out = await tool.run(
        {"study_uid": "1.2.826.<EXAMPLE_BIG>", "png_bytes": bytes(png_result.png_bytes)}
    )
    assert out.get("degraded") is True
    assert "image_preview_oversized" in out.get("degradation_reasons", [])
    # 但仍产出完整 base64（不阻断）
    assert out["png_data_url"].startswith(PNG_DATA_URL_PREFIX)


@pytest.mark.asyncio
async def test_image_preview_oversized_appends_reasons_without_duplicates(image_preview, processor):
    """已有 degradation_reasons 时追加，不重复。"""
    tool = ImagePreview(max_base64_bytes=10)
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_DUP>",
        pixel=_fixture_positive_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    out = await tool.run(
        {
            "study_uid": "1.2.826.<EXAMPLE_DUP>",
            "png_bytes": bytes(png_result.png_bytes),
            "degradation_reasons": ["image_preview_oversized"],  # 已存在
        }
    )
    reasons = out.get("degradation_reasons", [])
    assert reasons.count("image_preview_oversized") == 1


# ---------------------------------------------------------------------------
# 11. 工具名契约
# ---------------------------------------------------------------------------


def test_image_preview_name_matches_workflow_constant():
    """ImagePreview.name 必须 = 'image_preview' = DEFAULT_TOOL_NAMES[4]。"""
    tool = ImagePreview()
    assert tool.name == "image_preview"
    assert tool.name == DEFAULT_TOOL_NAMES[4]


def test_image_preview_factory_name_inherits():
    """工厂 build_image_preview 产出的实例 .name 也对齐。"""
    tool = build_image_preview()
    assert tool.name == "image_preview"


# ---------------------------------------------------------------------------
# 12. dataclass / 工具契约
# ---------------------------------------------------------------------------


def test_image_preview_result_as_state_patch_keys():
    """ImagePreviewResult.as_state_patch 字段名与 workflow 对齐。"""
    res = ImagePreviewResult(
        png_preview_b64="abcd",
        png_data_url="data:image/png;base64,abcd",
        png_byte_len=4,
        window_width=400.0,
        window_level=40.0,
        png_width=256,
        png_height=256,
        re_windowed=False,
    )
    patch = res.as_state_patch()
    expected = {
        "png_preview_b64", "png_data_url", "png_byte_len",
        "window_width", "window_level",
    }
    assert set(patch.keys()) == expected


def test_image_preview_result_is_frozen():
    """frozen=True → 字段不可变。"""
    res = ImagePreviewResult(
        png_preview_b64="abcd",
        png_data_url="data:image/png;base64,abcd",
        png_byte_len=4,
        window_width=400.0,
        window_level=40.0,
        png_width=256,
        png_height=256,
        re_windowed=False,
    )
    with pytest.raises((AttributeError, Exception)):
        res.png_preview_b64 = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 13. 默认肺窗 400/40 与 pacs.DicomProcessor 一致性
# ---------------------------------------------------------------------------


def test_image_preview_default_window_matches_pacs_processor():
    """ImagePreview 默认窗与 pacs.DicomProcessor 默认窗严格一致。"""
    from cxr_triage.pacs import DEFAULT_WINDOW_WIDTH as PACS_WW
    from cxr_triage.pacs import DEFAULT_WINDOW_LEVEL as PACS_WL

    assert DEFAULT_WINDOW_WIDTH == PACS_WW
    assert DEFAULT_WINDOW_LEVEL == PACS_WL


@pytest.mark.asyncio
async def test_image_preview_state_includes_window_width_level(image_preview, processor):
    """out["window_width"] / out["window_level"] 反映工具默认值。"""
    dcm_bytes = _make_cxr_dataset_bytes(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_WW>",
        pixel=_fixture_positive_pixels(),
    )
    png_result = processor.process(dcm_bytes)
    out = await image_preview.run(
        {"study_uid": "1.2.826.<EXAMPLE_WW>", "png_bytes": bytes(png_result.png_bytes)}
    )
    assert out["window_width"] == 400.0
    assert out["window_level"] == 40.0
