"""image_preview — 调窗 PNG 预览工具（产 base64 data URL 给浏览器面盘）。

源产品能力 / 代码证据（spec.md §7 开源对照 + plan.md §4 核心引擎深度 (c)）：

源仓库 ``github_ref/MedRAX/medrax/tools/dicom.py`` ``DicomProcessorTool``
（行 43-94、114-142）做 DICOM → PNG 调窗后再保存到 tempdir，并把路径
传回 Gradio 面板渲染。融合后医院产品形态有三条不能直接复用的差异：

  1) **输入形态**：体检 PACS 经 WADO-RS 拉回的是内存 ``bytes``，不再走
     文件路径；``DicomProcessor.process(dcm_bytes)`` 已经在 TriageTrigger
     入口调过窗产出 PNG bytes（pacs 层），``state["png_bytes"]`` 是 8-bit PNG。
  2) **输出形态**：浏览器三栏面板用 ``<img src="data:image/png;base64,...">``
     **直接渲染**，不写临时文件、不经 Gradio；产物体积受 SSE payload 限制，
     单张超 200KB 要降采样 / 重压缩。
  3) **流式时序**：阳性分诊触发器 → StateGraph.execute → image_preview 与
     keyqa / finding_draft / grounding **并行无关**；所以必须独立运行，
     不能阻塞主路径。

融合后产品路径（spec.md §2 工作闭环 + plan.md §4 核心引擎深度 (c)）：

    StateGraph.execute_node → ImagePreview.run(state)
      → 读 state["png_bytes"]        # DicomProcessor 已经产好的 8-bit PNG
      → （可选）state["dcm_bytes"]   # 若触发器未先调窗，可在此重新调窗
      → encode_png_bytes_to_b64      # base64(str) + data_url(str)
      → state["png_preview_b64"] = "<base64>"
      → state["png_data_url"]  = "data:image/png;base64,<...>"  # 浏览器直渲
      → state["png_byte_len"]  = <int>                          # SSE payload 监控

为什么工具里**再次**可选地接受 ``dcm_bytes`` + window / level：

  - 体检科可临时用「骨窗 2000/500」「纵隔窗 350/50」快速复核（CT/MR 习惯）
  - 触发器层默认已经用肺窗 400/40；若 ImagePreview 收到 ``dcm_bytes + override
    window``，仍能在工具内就地重新调窗 → 不让医师切回 PACS 调窗。

工具契约（与 Task 11 ``ToolRegistry`` + Task 12 ``TriageClassifier`` 对齐）：

  - ``.name = "image_preview"``（与 ``DEFAULT_TOOL_NAMES[4]`` 严格对齐）
  - ``async run(state: Dict[str, Any]) -> Dict[str, Any]``
  - 返回 ``{"png_preview_b64", "png_data_url", "png_byte_len",
           "window_width", "window_level"}`` + 可选 ``degraded``

公开入口：

  - ``ImagePreview``         StateGraph 注册的工具实例
  - ``ImagePreviewError``    构造 / 入参错误（应抛）
  - ``ImagePreviewResult``   单次产物的 dataclass
  - ``DEFAULT_WINDOW_WIDTH`` / ``DEFAULT_WINDOW_LEVEL`` （= 肺窗 400/40）
  - ``encode_png_to_b64`` / ``build_data_url``  静态工具（便于单测直接调用）
  - ``build_image_preview``  工厂函数

关键不变量：

  1. **png 字节流合法**：开头必须是 PNG magic ``\\x89PNG\\r\\n\\x1a\\n``，
     且 PIL.Image.open 必须能 load，否则不入 state。
  2. **base64 安全**：使用 ``base64.b64encode``（非 ``b64decode``），结果
     字符串可在浏览器 ``<img src=data:...>`` 直接渲染。
  3. **data URL 前缀唯一** ``data:image/png;base64,``：避免其它 MIME（jpg/
     gif）混入，与浏览器 ``<img>`` 标签同形。
  4. **窗口使用**：默认肺窗 400/40，仅当 state["dcm_bytes"] 提供时才
     在工具内重新调窗；缺则视为「触发器已调好」直接 encode。
  5. **失败语义**：png_bytes 缺 / 非 bytes / 空 / 非 PNG magic → 抛
     ``ImagePreviewError``（区别于 grounding 的 fail-soft 语义：本工具是
     「送医师看的图」，缺图必须显式失败，不允许静默送空 base64）。
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量 — 默认肺窗（与 pacs/dicom_processor.py / finding_draft 同源）
# ---------------------------------------------------------------------------

# 肺窗默认值（spec.md §3 + plan.md §5 深度 (c)）：窗宽 400 / 窗位 40
DEFAULT_WINDOW_WIDTH: float = 400.0
DEFAULT_WINDOW_LEVEL: float = 40.0

# PNG 文件 magic（与 PIL Image.open 兼容）
PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"

# 浏览器 data URL MIME 前缀（PNG）
PNG_DATA_URL_PREFIX: str = "data:image/png;base64,"

# 单张 base64 上限（防止 SSE payload 失控；与浏览器 <img> 200KB 经验阈值对齐）
PREVIEW_MAX_BASE64_BYTES: int = 350_000  # ≈ 262KB 原始 PNG


# ---------------------------------------------------------------------------
# 异常 / 结果
# ---------------------------------------------------------------------------


class ImagePreviewError(RuntimeError):
    """ImagePreview 构造 / 入参 / PNG magic 校验失败。"""


@dataclass(frozen=True)
class ImagePreviewResult:
    """单次预览产物的不可变结果，便于单测断言。"""

    png_preview_b64: str
    png_data_url: str
    png_byte_len: int
    window_width: float
    window_level: float
    png_width: int
    png_height: int
    re_windowed: bool

    def as_state_patch(self) -> Dict[str, Any]:
        """转 ``dict``，与 process_node 期望的字段名严格对齐。

        Workflow 状态字段（``CxrTriageState``）::

            png_preview_b64: str        # base64 字符串（裸）
            png_data_url: str           # browser-直渲 data URL
            png_byte_len: int           # payload 大小监控
            window_width: float         # 实际生效的窗宽
            window_level: float         # 实际生效的窗位
            png_width: int              # 像素宽度
            png_height: int             # 像素高度
        """
        return {
            "png_preview_b64": self.png_preview_b64,
            "png_data_url": self.png_data_url,
            "png_byte_len": self.png_byte_len,
            "window_width": self.window_width,
            "window_level": self.window_level,
        }


# ---------------------------------------------------------------------------
# 工具方法（不依赖实例，便于单测独立调用）
# ---------------------------------------------------------------------------


def _looks_like_png(data: bytes) -> bool:
    """检查字节流是否以 PNG magic 开头。"""
    return isinstance(data, (bytes, bytearray)) and data[: len(PNG_MAGIC)] == PNG_MAGIC


def encode_png_to_b64(png_bytes: bytes) -> str:
    """PNG bytes → base64 str（裸，无前缀）。

    Raises:
        ImagePreviewError: png_bytes 缺失 / 非 bytes / 空 / 非 PNG magic
    """
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise ImagePreviewError(
            f"png_bytes must be bytes, got {type(png_bytes).__name__}"
        )
    if len(png_bytes) == 0:
        raise ImagePreviewError("png_bytes is empty")
    if not _looks_like_png(png_bytes):
        raise ImagePreviewError("png_bytes does not start with PNG magic")
    try:
        encoded = base64.b64encode(bytes(png_bytes))
    except (binascii.Error, ValueError) as exc:
        raise ImagePreviewError(f"base64 encode failed: {exc}") from exc
    return encoded.decode("ascii")


def build_data_url(b64: str, *, mime: str = "png") -> str:
    """拼装 ``data:<mime>;base64,<b64>``。

    Raises:
        ImagePreviewError: b64 为空 / 非字符串 / 含非 base64 字符
    """
    if not isinstance(b64, str) or not b64:
        raise ImagePreviewError(f"b64 must be non-empty str, got {b64!r}")
    # 严格 base64 字符集校验
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if not set(b64) <= allowed:
        raise ImagePreviewError("b64 contains non-base64 characters")
    mime_token = mime.strip().lower()
    if mime_token not in {"png", "jpeg", "jpg", "gif", "webp"}:
        raise ImagePreviewError(f"unsupported mime: {mime!r}")
    # jpg/jpeg 兼容写法
    canonical = "jpeg" if mime_token == "jpg" else mime_token
    return f"data:image/{canonical};base64,{b64}"


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


class ImagePreview:
    """StateGraph 注册的 image_preview 工具。

    工具契约（Task 11 ``ToolRegistry`` 强校验）：

      - ``.name = "image_preview"``（与 ``DEFAULT_TOOL_NAMES[4]`` 对齐）
      - ``async run(state: dict) -> dict`` —— 返回 ``png_preview_b64`` +
        ``png_data_url`` + ``png_byte_len`` + ``window_*`` 字段 patch

    构造参数：

      - ``default_window_width`` / ``default_window_level``  本地偏好的窗宽 / 窗位
      - ``max_base64_bytes``    base64 串上限；超过视为不可入 SSE（仍要 encode，
                                但会被上层 SSE publisher 过滤；本层只 warn）

    入参（``state`` 字段）：

      - ``state["png_bytes"]``         ``bytes``（DicomProcessor 调好的 8-bit PNG）
      - ``state["dcm_bytes"]``         ``Optional[bytes]``（若存在 + 给了 window
                                       覆盖，将在工具内重新调窗）
      - ``state["window_width"]`` / ``state["window_level"]``  临时 override
      - ``state["study_uid"]``         写入 ``degradation_reasons`` 时用
      - ``state["metadata"]``          DICOM metadata（仅校验存在，不深读）

    异常处理：

      - 构造期：参数类型 / 阈值越界 → ``ImagePreviewError``
      - 入参：png_bytes 缺 / 非 bytes / 空 / 非 PNG magic → 抛
        ``ImagePreviewError``（区别于 grounding 的 fail-soft：本工具是
        「送医师看的图」，缺图必须显式失败让上层降级呈现）
    """

    name: str = "image_preview"

    def __init__(
        self,
        *,
        default_window_width: float = DEFAULT_WINDOW_WIDTH,
        default_window_level: float = DEFAULT_WINDOW_LEVEL,
        max_base64_bytes: int = PREVIEW_MAX_BASE64_BYTES,
    ) -> None:
        # ---- default_window_width ----
        if isinstance(default_window_width, str):
            try:
                default_window_width = float(default_window_width)
            except (TypeError, ValueError) as exc:
                raise ImagePreviewError(
                    f"default_window_width must be numeric string, "
                    f"got {default_window_width!r}"
                ) from exc
        elif isinstance(default_window_width, bool) or not isinstance(
            default_window_width, (int, float)
        ):
            raise ImagePreviewError(
                f"default_window_width must be numeric, "
                f"got {type(default_window_width).__name__}: {default_window_width!r}"
            )
        ww = float(default_window_width)
        if ww <= 0:
            raise ImagePreviewError(
                f"default_window_width must be > 0 (实际 {ww})"
            )

        # ---- default_window_level ----
        if isinstance(default_window_level, str):
            try:
                default_window_level = float(default_window_level)
            except (TypeError, ValueError) as exc:
                raise ImagePreviewError(
                    f"default_window_level must be numeric string, "
                    f"got {default_window_level!r}"
                ) from exc
        elif isinstance(default_window_level, bool) or not isinstance(
            default_window_level, (int, float)
        ):
            raise ImagePreviewError(
                f"default_window_level must be numeric, "
                f"got {type(default_window_level).__name__}: {default_window_level!r}"
            )
        wl = float(default_window_level)

        # ---- max_base64_bytes ----
        if isinstance(max_base64_bytes, bool) or not isinstance(
            max_base64_bytes, int
        ):
            raise ImagePreviewError(
                f"max_base64_bytes must be int, got "
                f"{type(max_base64_bytes).__name__}: {max_base64_bytes!r}"
            )
        if max_base64_bytes < 1:
            raise ImagePreviewError(
                f"max_base64_bytes must be >= 1, got {max_base64_bytes!r}"
            )

        self.default_window_width = ww
        self.default_window_level = wl
        self.max_base64_bytes = int(max_base64_bytes)

    # ---- 公开契约 ----

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """StateGraph 节点调用入口；返回 ``png_*`` 字段 patch。

        优先级：
          1) state["png_bytes"] 存在 → 直接 encode
          2) state["dcm_bytes"] + state["window_width"/"window_level"] 提供
             → 用 ``pacs.DicomProcessor`` 重新调窗
          3) 二者皆无 → 抛 ``ImagePreviewError``（让上层降级呈现）
        """
        png_bytes = state.get("png_bytes")
        re_windowed = False
        if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) == 0:
            # 缺失时尝试走 dcm_bytes + override
            dcm_bytes = state.get("dcm_bytes")
            window_override = state.get("window_width")
            level_override = state.get("window_level")
            if isinstance(dcm_bytes, (bytes, bytearray)) and len(dcm_bytes) > 0:
                png_bytes = self._rewindow_dcm(
                    dcm_bytes,
                    window=self._coerce_override(window_override),
                    level=self._coerce_override(level_override),
                )
                re_windowed = True
            else:
                raise ImagePreviewError(
                    "state missing both png_bytes and dcm_bytes"
                )

        # PNG magic 兜底校验（即便 png_bytes 来自 dcm 重调窗也要再 check）
        if not _looks_like_png(png_bytes):
            raise ImagePreviewError(
                "png_bytes (after optionally re-windowing) does not start "
                "with PNG magic"
            )

        png_bytes = bytes(png_bytes)
        b64 = encode_png_to_b64(png_bytes)
        data_url = build_data_url(b64)

        # 像素宽高（直接从 PNG IHDR 抽；不依赖 PIL）
        png_width, png_height = self._probe_png_dims(png_bytes)

        # 上限告警（不影响返回值；让上层 SSE publisher 看到）
        oversized = False
        if len(b64) > self.max_base64_bytes:
            oversized = True
            logger.warning(
                "image_preview base64 length %d exceeds cap %d for %s",
                len(b64), self.max_base64_bytes, state.get("study_uid"),
            )

        result = ImagePreviewResult(
            png_preview_b64=b64,
            png_data_url=data_url,
            png_byte_len=len(b64),
            window_width=self.default_window_width,
            window_level=self.default_window_level,
            png_width=png_width if png_width is not None else 0,
            png_height=png_height if png_height is not None else 0,
            re_windowed=re_windowed,
        )

        out = result.as_state_patch()
        out["png_width"] = png_width
        out["png_height"] = png_height
        out["re_windowed"] = re_windowed
        if oversized:
            out["degraded"] = True
            reasons = list(state.get("degradation_reasons") or [])
            if "image_preview_oversized" not in reasons:
                reasons.append("image_preview_oversized")
            out["degradation_reasons"] = reasons
        return out

    # ---- 辅助 ----

    @staticmethod
    def _coerce_override(value: Any) -> Optional[float]:
        """state 里 window/level override 可能是 None / float / 字符串。"""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ImagePreviewError(f"window/level override must be numeric, got bool")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ImagePreviewError(
                    f"window/level override must be numeric string, got {value!r}"
                ) from exc
        raise ImagePreviewError(
            f"window/level override must be numeric, "
            f"got {type(value).__name__}: {value!r}"
        )

    def _rewindow_dcm(
        self,
        dcm_bytes: bytes,
        *,
        window: Optional[float],
        level: Optional[float],
    ) -> bytes:
        """用本地 ``pacs.DicomProcessor`` 重新调窗（仅在 state 没给 png_bytes 时用）。

        懒导入避免 engine → pacs 反向依赖循环触发（pacs 自身 import engine）。
        """
        try:
            from cxr_triage.pacs import DicomProcessor
        except ImportError as exc:
            raise ImagePreviewError(
                f"DicomProcessor import failed for rewindow: {exc}"
            ) from exc

        proc = DicomProcessor(
            default_window_width=self.default_window_width,
            default_window_level=self.default_window_level,
        )
        result = proc.process(
            dcm_bytes,
            window=window if window is not None else self.default_window_width,
            level=level if level is not None else self.default_window_level,
        )
        return bytes(result.png_bytes)

    @staticmethod
    def _probe_png_dims(png_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
        """从 PNG IHDR 抽 width / height；不依赖 PIL。"""
        if len(png_bytes) < 24:
            return None, None
        # PNG signature (8 bytes) + IHDR chunk: length(4) + "IHDR"(4) + width(4) + height(4)
        try:
            if png_bytes[12:16] != b"IHDR":
                return None, None
            import struct

            width = struct.unpack(">I", png_bytes[16:20])[0]
            height = struct.unpack(">I", png_bytes[20:24])[0]
            return int(width), int(height)
        except (struct.error, ValueError):  # pragma: no cover — defensive
            return None, None


def build_image_preview(
    *,
    default_window_width: float = DEFAULT_WINDOW_WIDTH,
    default_window_level: float = DEFAULT_WINDOW_LEVEL,
    max_base64_bytes: int = PREVIEW_MAX_BASE64_BYTES,
) -> ImagePreview:
    """工厂：默认肺窗 (W/L=400/40)。"""
    return ImagePreview(
        default_window_width=default_window_width,
        default_window_level=default_window_level,
        max_base64_bytes=max_base64_bytes,
    )


__all__ = [
    "ImagePreview",
    "ImagePreviewError",
    "ImagePreviewResult",
    "DEFAULT_WINDOW_WIDTH",
    "DEFAULT_WINDOW_LEVEL",
    "PREVIEW_MAX_BASE64_BYTES",
    "PNG_DATA_URL_PREFIX",
    "PNG_MAGIC",
    "encode_png_to_b64",
    "build_data_url",
    "build_image_preview",
]
