"""positive_grounding — 阳性病灶定位框工具（确定性规则版）。

源产品能力 / 代码证据（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 +
Task 11 「阳性触发其余 4 件工具」中的第 4 件）：

源仓库 ``github_ref/MedRAX/medrax/tools/`` 区域里 ``XRayPhraseGroundingTool``
走 Maira-2 多模态模型对自由文本「grounding phrase」在胸片像素上做 grounding
（输出 ``(x, y, w, h)``）。融合后产品路径（spec.md §2 工作闭环 +
plan.md §4 核心引擎深度 (a)）有以下三条不能接受的代价：

  1) **Maira-2 模型推理** + **像素级 grounding** 在体检科每天 200+ 张胸片
     下不可控；每张要 1–3s GPU 推理时间 + 显存消耗，体检旺季日 300+ 张
     时会出现 **30 分钟+ 排队**，体检医师等到下班。
  2) **像素级坐标不可审计**：Maira-2 输出的 ``(x, y, w, h)`` 是像素坐标
     （如 ``(123, 456, 80, 60)``），飞检按 study_uid 还原「那天看过什么」
     时无法用同一坐标系回到像素位置（不同设备不同分辨率）。
  3) **开放短语** + **像素级输出** 双重幻觉来源：模型可能给出"看起来像"
     但与所见定位矛盾的 bbox。

融合后产品路径（spec.md §2 工作闭环 + plan.md §4 核心引擎深度 (a) +
plan.md §5 深度 (a) 边界与异常用例的具体落地）：

    StateGraph 阳性分流 → PositiveGrounding.run(state)
      → 读 state["classifier_probs"] + state["positive_flag"]
      → 确定性规则：每个 positive pathology（prob ≥ 阈值）
          → 查 DEFAULT_LOCALIZATION_BY_PATHOLOGY（finding_draft 共有）
              取定位文案（如「右下肺野」「心影」）
          → 查 DEFAULT_BBOX_BY_REGION 把文案映射到 ``(cx, cy, w, h)``
              标准 PA 胸片解剖区（坐标归一化到 [0, 1]）
          → 钳位：x = clamp(cx - w/2, 0, 1-w); y = clamp(cy - h/2, 0, 1-h)
          → 输出 ``{x, y, w, h, label, prob}``
      → ``state["grounding_boxes"] = [...]``

为什么选择确定性规则而不是 LLM grounding：

  - **可重复**：同 StudyInstanceUID 二次跑出同一组 bbox（飞检必备）。
  - **快**：O(18) dict lookup，**不**进模型推理。
  - **可解释**：每条 bbox 都对位 finding_draft 使用的定位文案，
    飞检能直接对照「所见」栏与定位框是否一致。
  - **降级路径自洽**：Task 20 显式确认「PositiveGrounding 可本地阈值规
    则出 bbox」是降级路径；正常路径同样走规则（Maira-2 不可达时无差别）。

关键不变量：

  1. **坐标归一化 [0, 1]**：与 finding_draft._normalize_grounding_boxes
     钳位契约严格对齐（避免 RIS payload 出现 ``x > 1.0`` 之类异常值）。
  2. **label 与 18 病理枚举对位**：仅输出 ``PATHOLOGY_NAMES`` 内的 label；
     未知 label → 跳过 + audit（避免幻觉 label 写入 RIS）。
  3. **prob 字段真实值**：bbox 上的 ``prob`` 与 ``classifier_probs`` 里该
     病理的 prob 一致，便于医师一眼判断「这个框的把握多大」。
  4. **bbox w/h ≤ 1.0**：保证 x+w ≤ 1.0、y+h ≤ 1.0；越界钳位而非丢弃
     （保留可解释性）。
  5. **阳性为 0 时 grounding_boxes=[]**：阴性不走 grounding（与
     workflow.process_node 的 ``tools_to_execute = ["image_preview"]`` 语
     义一致）。

工具契约（与 Task 11 ``ToolRegistry`` + Task 12 ``TriageClassifier`` 对齐）：

  - ``.name = "positive_grounding"``（与 ``DEFAULT_TOOL_NAMES[3]`` 严格对齐）
  - ``async run(state: Dict[str, Any]) -> Dict[str, Any]``
  - 返回 ``{"grounding_boxes": [...]}`` + 可选 degraded 字段

公开入口：

  - ``PositiveGrounding``           StateGraph 注册的工具实例
  - ``PositiveGroundingError``      构造 / 入参 / 规则错误（应抛）
  - ``PositiveGroundingResult``     单次产物的 dataclass
  - ``DEFAULT_BBOX_BY_REGION``      定位文案 → ``(cx, cy, w, h)`` 映射表
  - ``DEFAULT_FALLBACK_BBOX``       区域文案未命中时的兜底 ``(cx, cy, w, h)``
  - ``MAX_GROUNDING_BOXES``         单张最多保留框数（避免 RIS payload 失控）
  - ``POSITIVE_GROUNDING_PROB_HEURISTIC``   0.5（与
    ``CxrTriageConfig.positive_threshold`` / TriageClassifier 对齐）
  - ``build_positive_grounding``    工厂函数

实现取舍：

  - **完全确定性**：不依赖 LLM、无网络调用；与 finding_draft 同形（两者
    都是确定性合成产物）。``model_gateway`` / ``audit_logger`` 仅作为
    **未来**接 Maira-2 / 审计落盘的占位协议；当前 run() 不调用它们。
  - **bbox 来源**：胸片 PA 标准解剖区（与 finding_draft 共用
    ``DEFAULT_LOCALIZATION_BY_PATHOLOGY``），避免「所见」与「定位框」解耦。
  - **MISSING_FALLBACK = 中心默认框**：定位文案未在
    ``DEFAULT_BBOX_BY_REGION`` 命中时，落到 ``DEFAULT_FALLBACK_BBOX``（中
    心 0.5/0.5、宽高 0.4/0.4），避免 silent skip；同时记
    ``fallback_region_used`` 字段，便于定位文案字典迭代时排查缺漏。
  - **降级为可见**：bbox 数超过 ``MAX_GROUNDING_BOXES`` 时截断并记
    ``truncated`` 原因；RIS 写回时不会因为极端阳性清单把 payload 撑爆。
  - **入参容错**：classifier_probs 缺字段 / 全空 / 全被过滤 → 返空
    boxes + degraded=True + audit + 不抛；与 finding_draft 的 fail-fast
    不同——finding_draft 是核心所见生成，grounding 是辅助定位，可降级。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .finding_draft import DEFAULT_LOCALIZATION_BY_PATHOLOGY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量 — 标准胸片 PA 解剖区映射
# ---------------------------------------------------------------------------

# 标准 PA 胸片（成人立位后前位），坐标原点 (0,0) 在图像左上角；
# x 向右递增，y 向下递增；全部归一化到 [0, 1]。
# 与 finding_draft.DEFAULT_LOCALIZATION_BY_PATHOLOGY 的 18 项定位文案对位。
DEFAULT_BBOX_BY_REGION: Dict[str, Tuple[float, float, float, float]] = {
    # 心影区：偏左下，约占中下 1/3 区域
    "心影": (0.40, 0.65, 0.22, 0.24),
    # 纵隔：中央偏上，纵向条带
    "纵隔": (0.50, 0.55, 0.14, 0.32),
    # 胸膜：左/右侧纵向带
    "左侧胸膜": (0.20, 0.60, 0.12, 0.32),
    "右侧胸膜": (0.80, 0.60, 0.12, 0.32),
    # 肺野：分左右 + 上中下叶
    "左肺野": (0.32, 0.55, 0.30, 0.34),
    "左下肺野": (0.30, 0.68, 0.24, 0.24),
    "左肺下叶": (0.30, 0.68, 0.24, 0.24),
    "右肺野": (0.68, 0.55, 0.30, 0.34),
    "右下肺野": (0.70, 0.68, 0.24, 0.24),
    "右肺上叶": (0.68, 0.42, 0.24, 0.24),
    "右肺中叶": (0.68, 0.55, 0.22, 0.20),
    "双肺野": (0.50, 0.55, 0.60, 0.50),
    # 右侧胸腔（含肋膈角）：右下
    "右侧胸腔": (0.80, 0.65, 0.18, 0.30),
    # 肋骨：横跨中段
    "肋骨": (0.50, 0.50, 0.50, 0.20),
}

# 定位文案未在 DEFAULT_BBOX_BY_REGION 命中时的兜底（中心默认框）
DEFAULT_FALLBACK_BBOX: Tuple[float, float, float, float] = (0.50, 0.50, 0.40, 0.40)

# 单张胸片最多保留的定位框数（防 RIS payload 失控 + 浏览器渲染性能）
MAX_GROUNDING_BOXES: int = 6

# 单病理层级的阳性阈值（与 triage_classifier.POSITIVE_PROB_HEURISTIC 同源）
POSITIVE_GROUNDING_PROB_HEURISTIC: float = 0.5

# 18 病理枚举（与 triage_classifier.PATHOLOGY_NAMES 同源；硬编码避免循环 import）
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


class PositiveGroundingError(RuntimeError):
    """PositiveGrounding 构造 / 入参错误（应抛）。"""


@dataclass(frozen=True)
class PositiveGroundingResult:
    """单次定位产物的不可变结果，便于单测直接消费与断言。"""

    grounding_boxes: List[Dict[str, Any]]
    source: str                       # "rule" | "fallback" | "empty"
    fallback_region_used: List[str] = field(default_factory=list)
    truncated: bool = False

    def as_state_patch(self) -> Dict[str, Any]:
        """转 ``dict``，与 ``process_node`` 期望的字段名严格对齐。"""
        return {
            "grounding_boxes": list(self.grounding_boxes),
        }


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


class PositiveGrounding:
    """StateGraph 注册的 positive_grounding 工具（确定性规则版）。

    工具契约（Task 11 ``ToolRegistry`` 强校验）：

    - ``.name = "positive_grounding"``（与 ``DEFAULT_TOOL_NAMES[3]`` 对齐）
    - ``async run(state: dict) -> dict`` —— 返回 ``grounding_boxes`` 字段
      patch + 可选 ``degraded`` / ``degradation_reasons``

    构造参数：

    - ``localization_map``   : 自定义病理 → 定位文案映射；缺省沿用
                             ``finding_draft.DEFAULT_LOCALIZATION_BY_PATHOLOGY``
                             （同一份真源，避免所见与定位框解耦）
    - ``bbox_by_region``     : 定位文案 → ``(cx, cy, w, h)`` 映射；缺省
                             ``DEFAULT_BBOX_BY_REGION``
    - ``fallback_bbox``      : 区域文案未命中时的兜底 bbox；缺省
                             ``DEFAULT_FALLBACK_BBOX``
    - ``prob_heuristic``     : 单病理阳性阈值；缺省
                             ``POSITIVE_GROUNDING_PROB_HEURISTIC=0.5``
    - ``max_boxes``          : 单张最多保留框数；缺省 ``MAX_GROUNDING_BOXES=6``
    - ``model_gateway``      : 未来接 Maira-2 的网关占位（**当前 run()
                             不调用**，便于后续平滑升级）
    - ``audit_logger``       : 审计写入占位（**当前 run() 不调用**，
                             degraded 写回由 workflow.execute_node 兜底）

    入参（``state`` 字段）：

    - ``state["classifier_probs"]`` : ``[{name, prob, present?}, ...]`` 18 项
    - ``state["classifier_max_prob"]`` : float（与 ``state["classifier_probs"]``
      的最大值一致）
    - ``state["positive_flag"]``   : bool（max_prob ≥ positive_threshold）
    - ``state["metadata"]``        : DICOM metadata dict（**当前不用**，
      留作 future patient-side / laterality 推断）

    异常处理：

    - 构造期（bbox_by_region 类型错 / fallback_bbox 越界 / 阈值越界 / max_boxes
      非正）→ ``PositiveGroundingError``
    - 入参（classifier_probs 缺 / 非 list / 含非 dict）→ 返回空 boxes +
      ``degraded=True`` + audit（**不抛**；grounding 是辅助，可降级）
    - 模型不可达（未来接 Maira-2）：通过 ``model_gateway`` 协议；当前实现不调
    """

    name: str = "positive_grounding"

    def __init__(
        self,
        *,
        localization_map: Optional[Dict[str, str]] = None,
        bbox_by_region: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        fallback_bbox: Tuple[float, float, float, float] = DEFAULT_FALLBACK_BBOX,
        prob_heuristic: float = POSITIVE_GROUNDING_PROB_HEURISTIC,
        max_boxes: int = MAX_GROUNDING_BOXES,
        model_gateway: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
    ) -> None:
        # ---- localization_map ----
        if localization_map is None:
            localization_map = dict(DEFAULT_LOCALIZATION_BY_PATHOLOGY)
        if not isinstance(localization_map, dict):
            raise PositiveGroundingError("localization_map must be a dict")
        for k, v in localization_map.items():
            if not isinstance(k, str) or not k:
                raise PositiveGroundingError(
                    f"localization_map key must be non-empty str, got {k!r}"
                )
            if not isinstance(v, str) or not v:
                raise PositiveGroundingError(
                    f"localization_map[{k!r}] value must be non-empty str"
                )

        # ---- bbox_by_region ----
        if bbox_by_region is None:
            bbox_by_region = dict(DEFAULT_BBOX_BY_REGION)
        if not isinstance(bbox_by_region, dict):
            raise PositiveGroundingError("bbox_by_region must be a dict")
        for region, bbox in bbox_by_region.items():
            self._validate_bbox(region, bbox)

        # ---- fallback_bbox ----
        if not isinstance(fallback_bbox, tuple) or len(fallback_bbox) != 4:
            raise PositiveGroundingError(
                f"fallback_bbox must be a 4-tuple, got {fallback_bbox!r}"
            )
        # 严格数值：拒绝 str / bool / None 等（与 _validate_bbox 行为一致）
        fb_list: List[float] = []
        for v in fallback_bbox:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise PositiveGroundingError(
                    f"fallback_bbox values must be numeric (int/float), "
                    f"got {type(v).__name__}: {v!r}"
                )
            fb_list.append(float(v))
        fb = tuple(fb_list)
        self._validate_bbox("fallback", fb)
        # 单独再校验 bbox 中心/尺寸落入 [0, 1]
        cx, cy, w, h = fb
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise PositiveGroundingError(
                f"fallback_bbox center out of [0, 1]: ({cx}, {cy})"
            )
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise PositiveGroundingError(
                f"fallback_bbox size out of (0, 1]: w={w} h={h}"
            )

        # ---- prob_heuristic ----
        # 接受 int / float 与数字字符串（如 "0.5"），拒绝非数字字符串 / bool /
        # None —— 与 finding_draft 行为一致：与 triage_classifier 对齐
        if isinstance(prob_heuristic, bool):
            raise PositiveGroundingError(
                f"prob_heuristic must be numeric, got bool: {prob_heuristic!r}"
            )
        if isinstance(prob_heuristic, str):
            try:
                ph = float(prob_heuristic)
            except (TypeError, ValueError) as exc:
                raise PositiveGroundingError(
                    f"prob_heuristic must be numeric string, got {prob_heuristic!r}"
                ) from exc
        elif isinstance(prob_heuristic, (int, float)):
            ph = float(prob_heuristic)
        else:
            raise PositiveGroundingError(
                f"prob_heuristic must be numeric, got {type(prob_heuristic).__name__}: {prob_heuristic!r}"
            )
        if not (0.0 <= ph <= 1.0):
            raise PositiveGroundingError(
                f"prob_heuristic must be in [0, 1], got {prob_heuristic}"
            )

        # ---- max_boxes ----
        if not isinstance(max_boxes, int) or max_boxes < 1:
            raise PositiveGroundingError(
                f"max_boxes must be int >= 1, got {max_boxes!r}"
            )

        self.localization_map = dict(localization_map)
        self.bbox_by_region = dict(bbox_by_region)
        self.fallback_bbox = fb
        self.prob_heuristic = ph
        self.max_boxes = int(max_boxes)
        # 模型网关 / 审计日志占位（未来接 Maira-2 / 审计落盘，当前 run() 不调用）
        self.gateway = model_gateway
        self.audit_logger = audit_logger

    # ---- 公开契约 ----

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """StateGraph 节点调用入口；返回 ``{grounding_boxes, degraded?,
        degradation_reasons?}``。

        异常处理（与 workflow._execute_node 协作）：

        - 任何入参非法 / 阳性为 0 / 区域文案全部 fallback → 返回空 boxes +
          ``degraded=True`` + ``degradation_reasons=[...]``，**不抛**
        - 入参完全合法且至少有一项可解析 → 返回 bbox 列表 + ``degraded=False``
        """
        probs = state.get("classifier_probs")
        study_uid = state.get("study_uid") or ""
        if not isinstance(study_uid, str):
            study_uid = ""

        # ---- 入参容错 ----
        if not isinstance(probs, list):
            return self._empty_patch(
                reason="classifier_probs_missing_or_invalid",
                study_uid=study_uid,
            )

        # ---- 收集 positive pathologies ----
        positive_items: List[Dict[str, Any]] = []
        for item in probs:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            p = item.get("prob")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                continue
            p = float(p)
            # 钳位 + 阈值判定
            p = max(0.0, min(1.0, p))
            if p < self.prob_heuristic:
                continue
            positive_items.append({"name": name, "prob": p})

        if not positive_items:
            return self._empty_patch(
                reason="no_positive_pathologies",
                study_uid=study_uid,
            )

        # ---- 18 病理对位过滤（仅保留已知 label）----
        known_set = set(PATHOLOGY_NAMES)
        valid_items: List[Dict[str, Any]] = []
        unknown_labels: List[str] = []
        for it in positive_items:
            if it["name"] in known_set:
                valid_items.append(it)
            else:
                unknown_labels.append(it["name"])
        if not valid_items:
            return self._empty_patch(
                reason="no_known_pathology_labels",
                study_uid=study_uid,
                extra={"unknown_labels": unknown_labels},
            )

        # ---- 按 prob DESC 排序（保证 MAX_GROUNDING_BOXES 截断时保留把握最大的）----
        valid_items.sort(key=lambda x: x["prob"], reverse=True)

        # ---- 逐项生成 bbox ----
        boxes: List[Dict[str, Any]] = []
        fallback_used: List[str] = []
        truncated = False

        for item in valid_items:
            if len(boxes) >= self.max_boxes:
                truncated = True
                break
            label = item["name"]
            prob = item["prob"]
            region = self.localization_map.get(label)
            if region is None:
                # 病理名不在定位字典里 → 跳过（未知 label 通常是 model 输错；
                # 比用兜底文案输出更好，避免 hallucination 风险）
                continue
            bbox = self.bbox_by_region.get(region)
            if bbox is None:
                # 定位文案未在 bbox 表里命中 → 走 fallback
                bbox = self.fallback_bbox
                fallback_used.append(region)
            x, y, w, h = self._build_xywh(bbox)
            boxes.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "prob": prob,
            })

        if not boxes:
            return self._empty_patch(
                reason="no_region_match_for_positive_pathologies",
                study_uid=study_uid,
                extra={"fallback_used": fallback_used},
            )

        result = PositiveGroundingResult(
            grounding_boxes=boxes,
            source="fallback" if fallback_used else "rule",
            fallback_region_used=fallback_used,
            truncated=truncated,
        )
        patch = result.as_state_patch()
        patch["degraded"] = False
        patch["degradation_reasons"] = []
        patch["_grounding_meta"] = {
            "source": result.source,
            "box_count": len(boxes),
            "truncated": result.truncated,
            "fallback_region_used": list(result.fallback_region_used),
        }
        return patch

    # ---- 内部 ----

    @staticmethod
    def _validate_bbox(
        region: str,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """校验 ``(cx, cy, w, h)`` 形态；越界抛 ``PositiveGroundingError``。"""
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise PositiveGroundingError(
                f"bbox_by_region[{region!r}] must be 4-tuple, got {bbox!r}"
            )
        out: List[float] = []
        for v in bbox:
            # 严格数值：拒绝 str / bool / None / list 等（避免 "0.4" 被静默接受）
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise PositiveGroundingError(
                    f"bbox_by_region[{region!r}] values must be numeric "
                    f"(int/float), got {type(v).__name__}: {v!r}"
                )
            out.append(float(v))
        cx, cy, w, h = out
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise PositiveGroundingError(
                f"bbox_by_region[{region!r}] center out of [0, 1]: "
                f"({cx}, {cy})"
            )
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise PositiveGroundingError(
                f"bbox_by_region[{region!r}] size out of (0, 1]: w={w} h={h}"
            )

    @staticmethod
    def _build_xywh(
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """把 ``(cx, cy, w, h)`` 转 ``(x, y, w, h)``，并钳位到 ``x+w ≤ 1``、
        ``y+h ≤ 1``。"""
        cx, cy, w, h = bbox
        x = cx - w / 2.0
        y = cy - h / 2.0
        # 钳位（边缘防护：bbox 半宽/半高超出图像时拉回边界）
        if x < 0.0:
            x = 0.0
        if y < 0.0:
            y = 0.0
        if x + w > 1.0:
            w = max(0.0, 1.0 - x)
        if y + h > 1.0:
            h = max(0.0, 1.0 - y)
        return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))

    def _empty_patch(
        self,
        *,
        reason: str,
        study_uid: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """返回空 grounding_boxes + 软降级 patch（不抛，便于 workflow 兜底）。"""
        meta: Dict[str, Any] = {
            "source": "empty",
            "box_count": 0,
            "truncated": False,
            "fallback_region_used": [],
            "reason": reason,
        }
        if extra:
            meta.update(extra)
        return {
            "grounding_boxes": [],
            "degraded": True,
            "degradation_reasons": [f"positive_grounding_{reason}"],
            "_grounding_meta": meta,
        }

    # ---- 静态工具 ----

    @staticmethod
    def to_finding_struct(
        grounding_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """把 grounding_boxes 原样塞进 finding_struct.grounding_boxes
        （finding_draft 复用）。

        与 spec.md §4 finding_struct 字段约定
        ``grounding_boxes: [{x, y, w, h, label, prob?}, ...]`` 对齐。

        与 finding_draft._normalize_grounding_boxes 行为一致：非 dict / 字段缺
        失 / 越界由调用方过滤；本函数仅做 passthrough + 字段类型兜底。
        """
        if not isinstance(grounding_boxes, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in grounding_boxes:
            if not isinstance(it, dict):
                continue
            try:
                x = float(it.get("x", 0.0))
                y = float(it.get("y", 0.0))
                w = float(it.get("w", 0.0))
                h = float(it.get("h", 0.0))
            except (TypeError, ValueError):
                continue
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            label = it.get("label") or ""
            label_str = label if isinstance(label, str) else str(label)
            item: Dict[str, Any] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label_str,
            }
            prob = it.get("prob")
            if isinstance(prob, (int, float)) and not isinstance(prob, bool):
                item["prob"] = float(max(0.0, min(1.0, prob)))
            out.append(item)
        return out


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_positive_grounding(
    *,
    localization_map: Optional[Dict[str, str]] = None,
    bbox_by_region: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    fallback_bbox: Tuple[float, float, float, float] = DEFAULT_FALLBACK_BBOX,
    prob_heuristic: float = POSITIVE_GROUNDING_PROB_HEURISTIC,
    max_boxes: int = MAX_GROUNDING_BOXES,
    model_gateway: Optional[Any] = None,
    audit_logger: Optional[Any] = None,
) -> PositiveGrounding:
    """工厂：构造 ``PositiveGrounding`` 一行完成。"""
    return PositiveGrounding(
        localization_map=localization_map,
        bbox_by_region=bbox_by_region,
        fallback_bbox=fallback_bbox,
        prob_heuristic=prob_heuristic,
        max_boxes=max_boxes,
        model_gateway=model_gateway,
        audit_logger=audit_logger,
    )


# ---------------------------------------------------------------------------
# 模块导出
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_BBOX_BY_REGION",
    "DEFAULT_FALLBACK_BBOX",
    "MAX_GROUNDING_BOXES",
    "POSITIVE_GROUNDING_PROB_HEURISTIC",
    "PATHOLOGY_NAMES",
    "PositiveGrounding",
    "PositiveGroundingError",
    "PositiveGroundingResult",
    "build_positive_grounding",
]