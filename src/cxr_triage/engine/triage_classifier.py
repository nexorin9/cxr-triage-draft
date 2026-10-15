"""triage_classifier — 18 病理 DenseNet-121 分类工具。

源产品能力 / 代码证据（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法）：

源仓库 ``github_ref/MedRAX/medrax/tools/classification.py:24-153`` 的
``ChestXRayClassifierTool`` 走 ``torchxrayvision.models.DenseNet(weights=
"densenet121-res224-all")`` + ``xrv.datasets.XRayCenterCrop`` 预处理 +
``xrv.datasets.normalize`` 像素归一化；输出 ``xrv.datasets.default_pathologies``
与概率向量的 ``zip``，共 **18 病理**：

  0  Atelectasis              9  Pneumonia
  1  Consolidation           10  Pleural_Thickening
  2  Infiltration            11  Cardiomegaly
  3  Pneumothorax            12  Nodule
  4  Edema                   13  Mass
  5  Emphysema               14  Hernia
  6  Fibrosis                15  Lung Lesion
  7  Effusion                16  Fracture
  8  Pneumonia               17  Lung Opacity

融合后产品路径（spec.md §2 工作闭环 + plan.md §4 核心引擎深度 (a)）：

体检 PACS / watch_folder 落 DICOM 胸片
  → DicomProcessor 调窗 8-bit PNG bytes
  → CxrTriageWorkflow.process 节点（Task 11）
  → ``TriageClassifier.run(state)``
  → ``state["classifier_probs"] = [{name, prob, present}, ...]``（18 项）
  → ``state["classifier_max_prob"] = max(prob)``（确定性规则由 process 节点比较
    ``self.positive_threshold`` 决定 ``tools_to_execute``，与本类解耦）
  → 阳性 / 边界 → keyqa / finding_draft / positive_grounding / image_preview

实现取舍：

- **触发器**：从源 ``image_path`` 改为内存 PNG bytes（与 DicomProcessor 衔接，
  不再依赖磁盘临时文件）。模型权重从 ``~/.cache/torchxrayvision`` 自动下载
  （与源同源），可由 ``CXR_TRIAGE_MODEL_DIR`` 环境变量覆写。
- **依赖注入**：构造函数允许外部传入 ``model`` / ``transform`` 替身，便于
  单测避免真实模型下载与推理；同时保留默认路径的「真模型 + 真推理」端到端
  形态（生产配置 + 脱敏 fixture 可走）。
- **CPU/CUDA 双轨**：默认 ``device="cpu"``（开发与测试环境无 GPU；spec.md §6
  部署默认设备）；CUDA 不可用时自动回退 CPU，不静默丢数据。
- **可降级 / 不抛**：模型推理失败时返回 ``classifier_probs=[]`` +
  ``classifier_max_prob=0.0``（不抛）—— 与 ``process_node`` 现有降级语义
  对齐（避免双重异常）；异常本身由调用方 ``process_node`` 捕获并写
  ``degradation_reasons``。这一选择与源 ``ChestXRayClassifierTool._run`` 的
  ``except → {"error": ...}`` 行为**不同**（源会回 ``{error: ...}`` 字典），
  理由是：本产品走 StateGraph 编排，节点期待字段命名稳定（``classifier_probs``
  / ``classifier_max_prob``），让编排层决定降级比让工具静默改键名更可控。
- **图像预处理**复刻源算法：``PIL.Image.open(BytesIO(png_bytes)).convert("L")``
  → numpy → ``xrv.datasets.normalize(img, 255)`` → ``[None, :, :]`` 加 batch 维
  → ``XRayCenterCrop(224)`` → ``torch.from_numpy``。
- **空产物守门**：``classifier_probs`` 缺字段或长度为 0 → ``classifier_max_prob
  = 0.0``，``present`` 字段统一为 ``prob >= 0.5`` 启发式（与
  ``CxrTriageConfig.positive_threshold`` 默认 0.5 对齐；Task 13 显式拆
  ``rule_max_prob`` 静态方法）。
- **同步 + 异步一致**：源 ``_run`` 与 ``_arun`` 都是同步；本工具对外只暴露
  ``async run(state)``，内部用 ``asyncio.to_thread`` 包装同步 torch 推理
  （避免阻塞 FastAPI 事件循环）。
- **工具契约（与 Task 11 ToolRegistry 对齐）**：``.name = "classifier"`` +
  ``async run(state) -> dict``；返回值用 ``{"classifier_probs": [...],
  "classifier_max_prob": float}``，与 ``process_node`` 解析键名严格对齐。

公开入口：

- ``TriageClassifier``       StateGraph 注册的工具实例
- ``TriageClassifierError``  构造 / 入参 / 模型加载失败异常
- ``TriageClassifierResult`` 单次推理产物的 dataclass
- ``DEFAULT_MODEL_NAME``     ``"densenet121-res224-all"``（与源同源）
- ``DEFAULT_DEVICE``         ``"cpu"``
- ``POSITIVE_PROB_HEURISTIC`` 单病理 present 阈值（与 config 0.5 对齐）
- ``PATHOLOGY_NAMES``        18 病理枚举元组（与 ``xrv.datasets.default_pathologies`` 一致）
- ``build_triage_classifier`` 工厂函数
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量（与源 / spec 对齐）
# ---------------------------------------------------------------------------

# 与源 ChestXRayClassifierTool.__init__ 同源；torchxrayvision 0.x / 1.x 都
# 接受这个 weights 名（denseNet-121 224x224 18 病理全量训练权重）
DEFAULT_MODEL_NAME: str = "densenet121-res224-all"

# 开发与单测环境默认 CPU；spec.md §6 部署 默认设备；CUDA 可用时上层可覆写
DEFAULT_DEVICE: str = "cpu"

# 单病理 prob ≥ 该阈值 → 标 present=True（与 CxrTriageConfig.positive_threshold
# 默认 0.5 对齐；这是**单病理**层级的 present 标记，不是「整张是否阳性」——
# 整张阳性由 process 节点比 classifier_max_prob vs self.positive_threshold）
POSITIVE_PROB_HEURISTIC: float = 0.5

# 18 病理枚举（与 xrv.datasets.default_pathologies 顺序一致；冗余列出便于：
#   1) mock 工具生成同序概率；
#   2) spec §4 字段文档；
#   3) 单测不依赖 torchxrayvision import 也能校验顺序）
PATHOLOGY_NAMES: Tuple[str, ...] = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)


# ---------------------------------------------------------------------------
# 异常 / 结果
# ---------------------------------------------------------------------------


class TriageClassifierError(RuntimeError):
    """TriageClassifier 构造 / 入参 / 推理期错误。"""


@dataclass(frozen=True)
class TriageClassifierResult:
    """单次推理结果（便于单测与无 StateGraph 场景直接消费）。

    字段与 ``state`` 写入对齐：``probs`` 给 StateGraph、``max_prob`` 给
    process 节点、``present_count`` 给单测断言。
    """

    probs: List[Dict[str, Any]]  # [{name, prob, present}, ...] 18 项
    max_prob: float
    present_count: int
    model_name: str
    device: str
    elapsed_ms: int = 0

    def as_state_patch(self) -> Dict[str, Any]:
        """转 ``dict``，与 ``process_node`` 期望的字段名严格对齐。"""
        return {
            "classifier_probs": list(self.probs),
            "classifier_max_prob": float(self.max_prob),
        }


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


class TriageClassifier:
    """StateGraph 注册的 classifier 工具。

    工具契约（Task 11 ``ToolRegistry`` 强校验）：

    - ``.name = "classifier"``（与 ``DEFAULT_TOOL_NAMES[0]`` 对齐）
    - ``async run(state: dict) -> dict`` —— 返回 ``classifier_probs`` +
      ``classifier_max_prob`` 字段 patch

    构造参数：

    - ``model_name`` : torchxrayvision weights 名（默认 ``densenet121-res224-all``）
    - ``device``     : 推理设备（``"cpu"`` / ``"cuda"``；CUDA 不可用时自动回 CPU）
    - ``model``      : 注入的模型对象（默认 None → 内部加载 torchxrayvision）；
                       测试可注入 ``MockModel`` 替身
    - ``transform``  : 注入的图像预处理 compose（默认 None → 内部用
                       ``xrv.datasets.XRayCenterCrop``）；测试可注入 stub
    - ``prob_threshold`` : ``present`` 标记的启发式阈值（默认 0.5）
    """

    name: str = "classifier"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        *,
        model: Any = None,
        transform: Any = None,
        prob_threshold: float = POSITIVE_PROB_HEURISTIC,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise TriageClassifierError("model_name must be non-empty string")
        if not (0.0 <= float(prob_threshold) <= 1.0):
            raise TriageClassifierError(
                f"prob_threshold must be in [0, 1], got {prob_threshold}"
            )

        self.model_name = model_name.strip()
        self.prob_threshold = float(prob_threshold)

        # 设备回退：CUDA 不可用时静默回 CPU（避免在没 GPU 的开发机上一启动就崩）
        self.device = self._resolve_device(device)

        # 模型 + transform 加载（依赖注入或真模型）
        if model is not None:
            self.model = model
            self._owns_model = False
        else:
            self.model = self._load_model()
            self._owns_model = True

        if transform is not None:
            self.transform = transform
            self._owns_transform = False
        else:
            self.transform = self._load_transform()
            self._owns_transform = True

        # 把模型搬到目标设备（注入的替身可能不接受 .to()，需 duck-type 保护）
        try:
            if hasattr(self.model, "to") and callable(self.model.to):
                self.model = self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TriageClassifier: model.to(%s) failed: %s; using as-is",
                self.device, exc,
            )
        try:
            if hasattr(self.model, "eval") and callable(self.model.eval):
                self.model = self.model.eval()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TriageClassifier: model.eval() failed: %s", exc)

    # ---- 公开契约 ----

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """StateGraph 节点调用入口；返回 ``{classifier_probs, classifier_max_prob}``。

        入参：
        - ``state["png_bytes"]`` : 8-bit PNG 字节（DicomProcessor 产物）
        - ``state["metadata"]``   : DICOM 解析后的 metadata dict（保留供审计）

        异常处理（与 process_node 降级语义对齐）：
        - 任何异常由 ``process_node`` 捕获后写 ``degradation_reasons``；
        - 本方法**不抛**便于被 mock / stub 注入；如需 fail-fast 在上层处理。
        """
        png_bytes = state.get("png_bytes")
        if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) == 0:
            raise TriageClassifierError("state['png_bytes'] must be non-empty bytes")

        result = await self._infer(png_bytes)
        return result.as_state_patch()

    # ---- 静态规则（Task 13 复用）----

    @staticmethod
    def rule_max_prob(probs: Sequence[Dict[str, Any]], threshold: float) -> bool:
        """由 max_prob 与阈值推导 ``positive_flag`` 的**确定性规则**。

        静态方法以便 Task 13 单元测试与 baseline 对比复用；与 process 节点
        现有 ``max_prob >= self.positive_threshold`` 同形，但显式拆成单工具
        静态函数便于「单工具 baseline」与「多工具协同」对照。
        """
        if not isinstance(probs, (list, tuple)):
            raise TriageClassifierError("probs must be a list/tuple")
        if not (0.0 <= float(threshold) <= 1.0):
            raise TriageClassifierError(f"threshold must be in [0, 1], got {threshold}")
        max_p = 0.0
        for item in probs:
            if not isinstance(item, dict):
                continue
            p = item.get("prob")
            if isinstance(p, (int, float)) and not isinstance(p, bool):
                if float(p) > max_p:
                    max_p = float(p)
        return max_p >= float(threshold)

    @staticmethod
    def normalize_probs_to_dict(
        probs: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """把 ``[{name, prob}, ...]`` 压成 ``{name: prob}`` 便于去重 / 序列化。"""
        out: Dict[str, float] = {}
        for item in probs:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            p = item.get("prob")
            if isinstance(name, str) and name and isinstance(p, (int, float)) \
                    and not isinstance(p, bool):
                out[name] = float(p)
        return out

    # ---- 内部 ----

    async def _infer(self, png_bytes: bytes) -> TriageClassifierResult:
        """端到端推理：PNG bytes → 18 病理概率 → 结果 dataclass。"""
        # 同步 torch 推理走 to_thread 避免阻塞事件循环
        import time
        start = time.monotonic()

        # 拆「图像预处理」与「模型推理」两步；任一步失败都进降级
        try:
            tensor = await asyncio.to_thread(self._preprocess, png_bytes)
            raw_probs = await asyncio.to_thread(self._predict, tensor)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TriageClassifier inference failed: %s; returning empty probs",
                exc,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return TriageClassifierResult(
                probs=[],
                max_prob=0.0,
                present_count=0,
                model_name=self.model_name,
                device=str(self.device),
                elapsed_ms=elapsed_ms,
            )

        probs = self._format_probs(raw_probs)
        max_prob = max((p["prob"] for p in probs), default=0.0)
        present_count = sum(1 for p in probs if p["present"])

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return TriageClassifierResult(
            probs=probs,
            max_prob=float(max_prob),
            present_count=int(present_count),
            model_name=self.model_name,
            device=str(self.device),
            elapsed_ms=elapsed_ms,
        )

    def _preprocess(self, png_bytes: bytes) -> Any:
        """PNG bytes → torch tensor (1, 1, 224, 224)。

        复刻源 ``ChestXRayClassifierTool._process_image`` 算法：PIL ``"L"`` 模式
        → numpy ``float`` → ``xrv.datasets.normalize(img, 255)`` → 灰度取单通道
        → 加 batch 维 → ``XRayCenterCrop(224)`` → ``torch.from_numpy`` →
        ``unsqueeze(0)`` 加 batch 维。
        """
        # 延迟 import：让未装 torchxrayvision 的环境仍可 import 本模块（mock 用）
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        # xrv.datasets.normalize 接受 (img, max_value) → 把 0–255 像素线性归一到
        # torchxrayvision 训练分布。延迟 import；替身 transform 不需要这步
        if self._owns_transform:
            try:
                import torchxrayvision as xrv
                arr = xrv.datasets.normalize(arr, 255)
            except Exception as exc:  # noqa: BLE001
                logger.warning("xrv.normalize unavailable: %s; using arr/255", exc)
                arr = arr / 255.0
        else:
            arr = arr / 255.0

        if arr.ndim > 2:
            arr = arr[:, :, 0]
        arr = arr[None, :, :]  # (1, H, W)
        if self.transform is not None:
            try:
                arr = self.transform(arr)
            except Exception as exc:  # noqa: BLE001
                raise TriageClassifierError(f"transform failed: {exc}") from exc
        # torch.from_numpy; 替身 transform 可能直接返 tensor
        try:
            import torch
            if not isinstance(arr, torch.Tensor):
                arr = torch.from_numpy(arr)
            if arr.ndim == 3:
                arr = arr.unsqueeze(0)  # (1, 1, H, W)
        except Exception as exc:  # noqa: BLE001
            raise TriageClassifierError(f"tensor build failed: {exc}") from exc

        # 搬设备
        try:
            if hasattr(arr, "to") and callable(arr.to):
                arr = arr.to(self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tensor.to(%s) failed: %s; using cpu tensor", self.device, exc)
        return arr

    def _predict(self, tensor: Any) -> List[float]:
        """tensor → 18 病理概率 list[float]。

        兼容两类模型对象：
        1) 真 ``xrv.models.DenseNet`` ：``model(tensor).cpu()[0]`` 返 1D 概率向量
        2) 替身：``model(tensor)`` 返 list[float] / list[dict] / numpy 1D；
           见测试用 ``MockModel`` 形态
        """
        # 1) 真模型走 inference_mode
        try:
            import torch
            with torch.inference_mode():
                out = self.model(tensor)
            # DenseNet 输出 shape (1, 18)
            if hasattr(out, "cpu"):
                arr = out.cpu()
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                else:
                    arr = list(arr)
                if hasattr(arr, "ndim") and arr.ndim >= 2:
                    arr = arr[0]
                return [float(x) for x in list(arr)]
        except Exception as exc:  # noqa: BLE001
            # 替身可能不接受 inference_mode 或非 torch 输出
            logger.debug("torch-inference path failed: %s; trying duck-typed path", exc)

        # 2) 替身 / 通用 duck-typed 路径
        out = self.model(tensor)
        if isinstance(out, dict):
            # 替身直接返 {pathology: prob} 字典
            return [float(out.get(name, 0.0)) for name in PATHOLOGY_NAMES]
        if isinstance(out, (list, tuple)):
            if len(out) == len(PATHOLOGY_NAMES):
                return [float(x) for x in out]
            # 嵌套 list：取首行
            if len(out) > 0 and isinstance(out[0], (list, tuple)):
                return [float(x) for x in out[0]]
            return [float(x) for x in out]
        # numpy 1D
        try:
            import numpy as np
            if isinstance(out, np.ndarray):
                if out.ndim >= 2:
                    out = out[0]
                return [float(x) for x in out]
        except Exception:
            pass
        raise TriageClassifierError(
            f"unsupported model output type: {type(out).__name__}"
        )

    def _format_probs(self, raw_probs: Sequence[float]) -> List[Dict[str, Any]]:
        """18 概率 → ``[{name, prob, present}, ...]`` 18 项。"""
        if len(raw_probs) != len(PATHOLOGY_NAMES):
            raise TriageClassifierError(
                f"model output length mismatch: got {len(raw_probs)}, "
                f"expected {len(PATHOLOGY_NAMES)}"
            )
        probs: List[Dict[str, Any]] = []
        for name, p in zip(PATHOLOGY_NAMES, raw_probs):
            prob = float(p)
            # 钳到 [0, 1] 防止模型输出偶发越界（softmax 之前 logit 也能进）
            prob = max(0.0, min(1.0, prob))
            probs.append({
                "name": name,
                "prob": prob,
                "present": bool(prob >= self.prob_threshold),
            })
        return probs

    @staticmethod
    def _resolve_device(device: str) -> Any:
        """CUDA 不可用时静默回 CPU；device 字段允许 str 形式。"""
        if not isinstance(device, str):
            return device
        dev_str = device.strip().lower()
        if dev_str in ("", "cpu"):
            return "cpu"
        if dev_str.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return device
            except Exception:
                pass
            logger.warning(
                "TriageClassifier: CUDA requested but unavailable; falling back to CPU"
            )
            return "cpu"
        # 其他设备（如 mps）直接透传
        return device

    @staticmethod
    def _load_model() -> Any:
        """加载 ``xrv.models.DenseNet(weights=...)`` 真实模型。"""
        try:
            import torchxrayvision as xrv
        except Exception as exc:  # noqa: BLE001
            raise TriageClassifierError(
                f"torchxrayvision not available: {exc}; "
                f"install via `pip install torchxrayvision` or inject model=..."
            ) from exc
        try:
            return xrv.models.DenseNet(weights=DEFAULT_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001
            raise TriageClassifierError(
                f"failed to load DenseNet weights={DEFAULT_MODEL_NAME!r}: {exc}"
            ) from exc

    @staticmethod
    def _load_transform() -> Any:
        """``xrv.datasets.XRayCenterCrop(224)`` 中心裁剪到 224x224。"""
        try:
            import torchxrayvision as xrv
            return xrv.datasets.XRayCenterCrop()
        except Exception as exc:  # noqa: BLE001
            raise TriageClassifierError(
                f"failed to build XRayCenterCrop: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_triage_classifier(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    model: Any = None,
    transform: Any = None,
    prob_threshold: float = POSITIVE_PROB_HEURISTIC,
) -> TriageClassifier:
    """工厂：构造 ``TriageClassifier`` 一行完成。"""
    return TriageClassifier(
        model_name=model_name,
        device=device,
        model=model,
        transform=transform,
        prob_threshold=prob_threshold,
    )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "POSITIVE_PROB_HEURISTIC",
    "PATHOLOGY_NAMES",
    "TriageClassifier",
    "TriageClassifierError",
    "TriageClassifierResult",
    "build_triage_classifier",
]
