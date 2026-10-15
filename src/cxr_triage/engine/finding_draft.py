"""finding_draft — 体检胸片中文所见草稿生成器。

源产品能力 / 代码证据（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 +
Task 11 「阳性触发其余 4 件工具」中的第 3 件）：

源仓库 ``github_ref/MedRAX/medrax/tools/`` 区域里 ``ChestXRayReportGeneratorTool``
走 SwinV2 + Beam Search 生成自然语言报告。融合后产品路径（spec.md §2 工作闭环 +
plan.md §4 核心引擎深度 (a)）有以下两条不能接受的代价：

  1) **自由报告 + 幻觉**：SwinV2 的自然语言生成模型在医院语境下风险不可控；
     体检科医师需要的是「所见短句 + 结构化字段」双轨，**不**读诊断、不出治疗
     建议（spec.md §10 安全边界 + 中式实务优先）。
  2) **不可审计**：纯文本报告无字段切片，飞检按 study_uid 复原「那天看过什么」
     时无法对照 18 病理概率与关键问答的逐项来源。

融合后产品路径：

    StateGraph 阳性分流 → FindingDraft.run(state)
      → 读 state["classifier_probs"] + state["key_qa_list"]
      → 确定性文本合成（不依赖 LLM；模板拼接所见短句）
      → ``state["finding_text"]``        中文所见短句 + 「（草稿 / 未经医
                                          师确认）」强制标识
      → ``state["finding_struct"]``      {pathologies, key_qa, grounding_boxes}
      → ``state["positive_flag"]``       由 max_prob vs positive_threshold
      → ``state["draft_ts"]``            UTC ISO8601（毫秒精
                                          度，与 audit 对齐）

工具契约（与 Task 11 ``ToolRegistry`` + Task 12 ``TriageClassifier`` 对齐）：

  - ``.name = "finding_draft"``（与 ``DEFAULT_TOOL_NAMES[2]`` 严格对齐）
  - ``async run(state: Dict[str, Any]) -> Dict[str, Any]``
  - 返回 ``{"finding_text": str, "finding_struct": dict, "positive_flag": bool,
            "draft_ts": str, "draft_marker": str}``

关键不变量：

  1. ``finding_text`` **必含** ``DRAFT_MARKER``（"（草稿 / 未经医师确认）"），
     保证浏览器面板 + RIS 写回前医师一眼能识别「这是草稿、未签发」。
  2. ``positive_flag`` 由 ``classifier_max_prob >= positive_threshold`` 推
     导；阈值缺省 ``0.5``（与 ``CxrTriageConfig.positive_threshold`` 对齐）；
     <阈值时仍返回 finding_struct（结构完整），但 ``finding_text`` 走「未
     见明显异常」分支 + 显式 banner，**不**误导阳性。
  3. 模型与模板**完全分离**：本工具不调 LLM（避免幻觉循环），仅依据
     classifier 与 keyqa 的结构化产物合成所见。LLM 失败不会污染
     ``finding_text``，与 CXRKeyQA 的软降级语义形成「结构不变 + 自由文本可
     留白」的分层。
  4. ``draft_ts`` 与 ``audit_log.ts`` 同步（UTC ISO8601 毫秒）；便于飞检以
     ts 对齐「草稿就绪时刻」与「医师三角色流转时刻」。

公开入口：

  - ``FindingDraft``              StateGraph 注册的工具实例
  - ``FindingDraftError``         构造 / 入参 / 合成错误（应抛）
  - ``FindingDraftResult``        单次合成产物的 dataclass
  - ``DRAFT_MARKER``              "（草稿 / 未经医师确认）"
  - ``NO_ABNORMALITY_TEMPLATE``   max_prob < 阈值时所见模板
  - ``POSITIVE_TEMPLATE``         max_prob ≥ 阈值时所见模板
  - ``DEFAULT_POSITIVE_THRESHOLD`` 0.5
  - ``build_finding_draft``       工厂函数
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量 — 草稿 / 模板 / 默认阈值
# ---------------------------------------------------------------------------

# 强制标识：每一条 finding_text 必须包含该标记，飞检 / 医师一眼能识别「草稿
# / 未签发」。禁止在签发后保留；签发阶段（Task 22 RISWriter）会剥掉该标记
# 后再外发（DRAFT_MARKER 由 RISWriter 一次性剔除）。
DRAFT_MARKER: str = "（草稿 / 未经医师确认）"

# max_prob < 阈值时的所见模板：明确「未见明显异常」+ 强制标识 + banner
NO_ABNORMALITY_TEMPLATE: str = (
    "胸片未见明显异常（DRAFT_MARKER），"
    "必要时建议结合临床或复查。"
)

# max_prob ≥ 阈值时的所见模板：列出 top-1 病理 + 其余肺野「余肺野清晰」
# （与典型体检胸片报告「所见」一栏长期沿用的格式对齐）。
POSITIVE_TEMPLATE: str = (
    "{localization}见{top1_pathology}征象（DRAFT_MARKER），余肺野清晰。"
)

# 默认阳性阈值（与 CxrTriageConfig.positive_threshold 默认 0.5 对齐）
DEFAULT_POSITIVE_THRESHOLD: float = 0.5

# 列出 top-N 病理时默认取的病理数（与 xrv.datasets.default_pathologies 中
# 中文名出现的常见表述对齐；超过 3 项一般体检所见不会逐项罗列）
TOP_N_PATHOLOGIES: int = 3

# finding_struct.pathologies 字段最大保留项数（避免冗余 + 控制 payload）
MAX_PATHOLOGIES_IN_STRUCT: int = 18

# 中文定位锚点（按病理给出常用体检表述），缺省时 fallback 到「右下肺野」。
# 与源 MedRAX 报告里「right lower zone」「left upper zone」的位置描述相比，
# 本表贴近中国体检科医师口语。
DEFAULT_LOCALIZATION_BY_PATHOLOGY: Dict[str, str] = {
    "Pneumothorax": "右下肺野",
    "Effusion": "右侧胸腔",
    "Consolidation": "右下肺野",
    "Atelectasis": "左下肺野",
    "Infiltration": "左肺野",
    "Pneumonia": "双肺野",
    "Cardiomegaly": "心影",
    "Mass": "右肺上叶",
    "Nodule": "右上肺野",
    "Pleural_Thickening": "左侧胸膜",
    "Edema": "双肺野",
    "Emphysema": "双肺野",
    "Fibrosis": "右下肺野",
    "Hernia": "纵隔",
    "Lung Lesion": "左肺下叶",
    "Fracture": "肋骨",
    "Lung Opacity": "右肺中叶",
    "Enlarged Cardiomediastinum": "纵隔",
}


# ---------------------------------------------------------------------------
# 异常 / 结果
# ---------------------------------------------------------------------------


class FindingDraftError(RuntimeError):
    """FindingDraft 构造 / 入参 / 合成错误（应抛）。"""


@dataclass(frozen=True)
class FindingDraftResult:
    """单次合成产物的不可变结果，便于单测直接消费与断言。"""

    finding_text: str
    finding_struct: Dict[str, Any]
    positive_flag: bool
    draft_ts: str
    draft_marker_included: bool

    def as_state_patch(self) -> Dict[str, Any]:
        """转 ``dict``，与 ``process_node`` / StateGraph 解析键名严格对齐。"""
        return {
            "finding_text": self.finding_text,
            "finding_struct": self.finding_struct,
            "positive_flag": self.positive_flag,
            "draft_ts": self.draft_ts,
        }


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


class FindingDraft:
    """StateGraph 注册的 finding_draft 工具。

    工具契约（Task 11 ``ToolRegistry`` 强校验）：

    - ``.name = "finding_draft"``（与 ``DEFAULT_TOOL_NAMES[2]`` 对齐）
    - ``async run(state: dict) -> dict`` —— 返回 ``finding_text`` +
      ``finding_struct`` + ``positive_flag`` + ``draft_ts`` patch

    构造参数：

    - ``positive_threshold`` : 阳性阈值（max_prob ≥ 阈值 → positive_flag=True）
                             默认 0.5，与 ``CxrTriageConfig.positive_threshold``
                             对齐；可外部覆写
    - ``localization_map``   : 自定义病理 → 定位文案映射；缺省
                             ``DEFAULT_LOCALIZATION_BY_PATHOLOGY``
    - ``now_iso``            : 可注入 UTC ISO8601 字符串函数（单测用）；
                             缺省走 ``_default_now_iso()``
    - ``top_n``              : 见 ``finding_struct.pathologies`` 保留项数
                             （默认 ``TOP_N_PATHOLOGIES=3``，所见模板仅引
                             用 top1，整体 finding_struct 全量保留）

    入参（``state`` 字段）：

    - ``state["classifier_probs"]`` : ``[{name, prob, present?}, ...]``
    - ``state["classifier_max_prob"]`` : float
    - ``state["key_qa_list"]``      : ``[{q, a}, ...]``（CXRKeyQA 产物）
    - ``state["grounding_boxes"]``  : ``[{x, y, w, h, label, prob?}, ...]``
    - ``state["metadata"]``         : DICOM metadata dict

    异常处理：

    - 入参非法（classifier_probs 缺/非 list）→ ``FindingDraftError``
    - 阈值越界 → ``FindingDraftError``
    - **不**因模型不可达抛错（finding_draft 不依赖 LLM，无可降级语义）
    """

    name: str = "finding_draft"

    def __init__(
        self,
        *,
        positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
        localization_map: Optional[Dict[str, str]] = None,
        now_iso: Optional[Any] = None,
        top_n: int = TOP_N_PATHOLOGIES,
    ) -> None:
        # threshold 校验
        try:
            t = float(positive_threshold)
        except (TypeError, ValueError):
            raise FindingDraftError(
                f"positive_threshold must be numeric, got {positive_threshold!r}"
            )
        if not (0.0 <= t <= 1.0):
            raise FindingDraftError(
                f"positive_threshold must be in [0, 1], got {positive_threshold}"
            )

        if localization_map is not None:
            if not isinstance(localization_map, dict):
                raise FindingDraftError("localization_map must be a dict")
            for k, v in localization_map.items():
                if not isinstance(k, str) or not k:
                    raise FindingDraftError(
                        f"localization_map key must be non-empty str, got {k!r}"
                    )
                if not isinstance(v, str) or not v:
                    raise FindingDraftError(
                        f"localization_map[{k!r}] value must be non-empty str"
                    )

        if not isinstance(top_n, int) or top_n < 1 or top_n > 18:
            raise FindingDraftError(
                f"top_n must be int in [1, 18], got {top_n!r}"
            )

        self.positive_threshold = float(positive_threshold)
        self.localization_map = (
            dict(localization_map) if localization_map is not None
            else dict(DEFAULT_LOCALIZATION_BY_PATHOLOGY)
        )
        self._now_iso = now_iso if now_iso is not None else _default_now_iso
        self.top_n = int(top_n)

    # ---- 公开契约 ----

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """StateGraph 节点调用入口；返回 ``{finding_text, finding_struct,
        positive_flag, draft_ts}``。

        异常处理：

        - 入参非法（classifier_probs 缺/非 list / classifier_max_prob 缺失）
          → ``FindingDraftError``（fail-fast；finding_draft 不依赖 LLM，无
          软降级语义）
        - 阈值越界 → ``FindingDraftError``
        """
        probs = state.get("classifier_probs")
        max_prob = state.get("classifier_max_prob")
        if not isinstance(probs, list):
            raise FindingDraftError(
                "state['classifier_probs'] must be a list "
                "(TriageClassifier must run before FindingDraft)"
            )
        if not isinstance(max_prob, (int, float)) or isinstance(max_prob, bool):
            raise FindingDraftError(
                f"state['classifier_max_prob'] must be numeric, got {max_prob!r}"
            )

        key_qa_list = state.get("key_qa_list") or []
        if not isinstance(key_qa_list, list):
            # 软兜底：keyqa 列表被异常污染时不强失败，按空列表走
            key_qa_list = []

        grounding_boxes = state.get("grounding_boxes") or []
        if not isinstance(grounding_boxes, list):
            grounding_boxes = []

        positive = bool(float(max_prob) >= self.positive_threshold)
        draft_ts = self._now_iso()

        # ---- finding_struct（结构化所见；payload 字段对齐 spec.md §4）----
        finding_struct = self._build_finding_struct(
            probs=probs,
            key_qa_list=key_qa_list,
            grounding_boxes=grounding_boxes,
        )

        # ---- finding_text（中文所见短句 + 强制草稿标识）----
        finding_text = self._build_finding_text(
            probs=probs,
            max_prob=float(max_prob),
            positive=positive,
        )

        # 防御性：finding_text 必须包含 DRAFT_MARKER（即使模板里走非主分支）
        if DRAFT_MARKER not in finding_text:
            finding_text = finding_text + DRAFT_MARKER

        result = FindingDraftResult(
            finding_text=finding_text,
            finding_struct=finding_struct,
            positive_flag=positive,
            draft_ts=draft_ts,
            draft_marker_included=(DRAFT_MARKER in finding_text),
        )
        return result.as_state_patch()

    # ---- 内部：finding_struct ----

    def _build_finding_struct(
        self,
        *,
        probs: List[Dict[str, Any]],
        key_qa_list: List[Dict[str, Any]],
        grounding_boxes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """构造 ``finding_struct = {pathologies, key_qa, grounding_boxes}``。

        与 spec.md §4 RIS payload 字段严格对齐：

        - ``pathologies`` : ``[{name, prob, present?}, ...]``，保留 prob >
          0.05 的项（避免冗余且保留可解释性），上限
          ``MAX_PATHOLOGIES_IN_STRUCT = 18``
        - ``key_qa``      : 原样保留 CXRKeyQA 产物
        - ``grounding_boxes`` : 原样保留 PositiveGrounding 产物
        """
        pathologies = self._normalize_pathologies(probs)
        normalized_key_qa = self._normalize_key_qa(key_qa_list)
        normalized_boxes = self._normalize_grounding_boxes(grounding_boxes)

        return {
            "pathologies": pathologies,
            "key_qa": normalized_key_qa,
            "grounding_boxes": normalized_boxes,
        }

    @staticmethod
    def _normalize_pathologies(
        probs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """把 ``classifier_probs`` 序列化为 ``finding_struct.pathologies``。

        规则：

        - 过滤非 dict 项
        - prob 必须数值（不接受 bool）
        - prob ∈ [0, 1]（越界钳位）
        - present = prob ≥ 0.5（启发式；与 triage_classifier 默认
          ``POSITIVE_PROB_HEURISTIC`` 对齐）
        - 截到 ``MAX_PATHOLOGIES_IN_STRUCT = 18`` 项，按 prob DESC 排序
        """
        out: List[Dict[str, Any]] = []
        for item in probs:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            p = item.get("prob")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                continue
            p = max(0.0, min(1.0, float(p)))
            out.append({
                "name": name,
                "prob": p,
                "present": bool(p >= 0.5),
            })
        # 按 prob 降序
        out.sort(key=lambda x: x["prob"], reverse=True)
        return out[:MAX_PATHOLOGIES_IN_STRUCT]

    @staticmethod
    def _normalize_key_qa(
        key_qa_list: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """过滤非 dict / 字段缺失项；与 CXRKeyQA.to_finding_struct 行为一致。"""
        out: List[Dict[str, str]] = []
        for it in key_qa_list:
            if not isinstance(it, dict):
                continue
            q = it.get("q")
            a = it.get("a")
            if not isinstance(q, str) or not isinstance(a, str):
                continue
            q2 = q.strip()
            a2 = a.strip()
            if not q2 or not a2:
                continue
            out.append({"q": q2, "a": a2})
        return out

    @staticmethod
    def _normalize_grounding_boxes(
        boxes: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """规范化 grounding_boxes 项；坐标钳位到 [0, 1]。"""
        out: List[Dict[str, Any]] = []
        for it in boxes:
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
            prob = it.get("prob")
            if not isinstance(prob, (int, float)) or isinstance(prob, bool):
                prob_val: Optional[float] = None
            else:
                prob_val = float(max(0.0, min(1.0, prob)))
            item: Dict[str, Any] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label_str,
            }
            if prob_val is not None:
                item["prob"] = prob_val
            out.append(item)
        return out

    # ---- 内部：finding_text ----

    def _build_finding_text(
        self,
        *,
        probs: List[Dict[str, Any]],
        max_prob: float,
        positive: bool,
    ) -> str:
        """合成本次 finding_text。

        - ``positive=False``（max_prob < 阈值）：走 ``NO_ABNORMALITY_TEMPLATE``，
          显式 banner「未见明显异常」+ 强制 DRAFT_MARKER。
        - ``positive=True``  且 probs 全空：走 NO_ABNORMALITY_TEMPLATE + 全
          局 banner「概率列空；仅结构化标注，仍待签」（防御性，正常路径
          不应进入）。
        - ``positive=True``  且 probs 非空：用 ``POSITIVE_TEMPLATE`` 拼接
          ``{localization}{top1_pathology}征象``。
        """
        if not positive:
            # max_prob < 阈值 → 「未见明显异常」+ 强制 DRAFT_MARKER
            return NO_ABNORMALITY_TEMPLATE.replace("DRAFT_MARKER", DRAFT_MARKER)

        # 阳性 / 边界 → 取 top-1 病理
        top1 = self._select_top_pathology(probs)
        if top1 is None:
            # 防御：positive=True 但 probs 列表空 / 全被过滤
            return (
                "未取得可信分类结果（DRAFT_MARKER），"
                "仍待人工复核。"
            ).replace("DRAFT_MARKER", DRAFT_MARKER)

        pathology_zh = _PATHOLOGY_ZH.get(top1, top1.replace("_", " "))
        localization = self.localization_map.get(top1, "右下肺野")
        body = POSITIVE_TEMPLATE.format(
            localization=localization,
            top1_pathology=pathology_zh,
        )
        return body.replace("DRAFT_MARKER", DRAFT_MARKER)

    @staticmethod
    def _select_top_pathology(
        probs: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        """返回概率最高且 prob > 0.5 的病理名；找不到返回 None。"""
        best_name: Optional[str] = None
        best_prob: float = -1.0
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
            if p > best_prob:
                best_prob = p
                best_name = name
        if best_name is None:
            return None
        if best_prob < 0.5:
            # positive=True 但 top1 仍 < 0.5 → 不输出 top1（异常路径）
            return None
        return best_name

    # ---- 静态工具 ----

    @staticmethod
    def serialize_for_ris(
        finding_text: str,
        finding_struct: Dict[str, Any],
        positive_flag: bool,
        draft_ts: str,
        tool_version: str = "0.1.0",
        study_uid: str = "",
        patient_id_hash: str = "",
    ) -> Dict[str, Any]:
        """导出为 RIS payload（与 spec.md §4 RIS finding-draft 输出对齐）。

        调用方（Task 22 ``RISWriter``）传入 ``study_uid`` / ``patient_id_hash``
        后构造最终 HTTP body；本函数**只**做 dict 序列化，**不**做网络调用。
        """
        return {
            "study_uid": study_uid,
            "patient_id_hash": patient_id_hash,
            "finding_struct": finding_struct,
            "finding_text": finding_text,
            "positive_flag": bool(positive_flag),
            "tool_version": tool_version,
            "draft_ts": draft_ts,
        }


# 18 病理中英文对照（体检科医师所见中常用的中文表述；fallback 用 name.replace）
_PATHOLOGY_ZH: Dict[str, str] = {
    "Atelectasis": "肺不张",
    "Consolidation": "实变",
    "Infiltration": "浸润影",
    "Pneumothorax": "气胸",
    "Edema": "肺水肿",
    "Emphysema": "肺气肿",
    "Fibrosis": "纤维化",
    "Effusion": "胸腔积液",
    "Pneumonia": "肺炎",
    "Pleural_Thickening": "胸膜增厚",
    "Cardiomegaly": "心影增大",
    "Nodule": "结节",
    "Mass": "肿块",
    "Hernia": "膈疝",
    "Lung Lesion": "肺内病灶",
    "Fracture": "骨折",
    "Lung Opacity": "肺部密度增高",
    "Enlarged Cardiomediastinum": "心影纵隔增宽",
}


# ---------------------------------------------------------------------------
# 时间工具
# ---------------------------------------------------------------------------


def _default_now_iso() -> str:
    """UTC ISO8601 + 毫秒 + Z 后缀；与 audit.audit_log_insert.ts 对齐。"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_finding_draft(
    *,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    localization_map: Optional[Dict[str, str]] = None,
    now_iso: Optional[Any] = None,
    top_n: int = TOP_N_PATHOLOGIES,
) -> FindingDraft:
    """工厂：构造 ``FindingDraft`` 一行完成。"""
    return FindingDraft(
        positive_threshold=positive_threshold,
        localization_map=localization_map,
        now_iso=now_iso,
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# 模块导出
# ---------------------------------------------------------------------------


__all__ = [
    "DRAFT_MARKER",
    "NO_ABNORMALITY_TEMPLATE",
    "POSITIVE_TEMPLATE",
    "DEFAULT_POSITIVE_THRESHOLD",
    "TOP_N_PATHOLOGIES",
    "MAX_PATHOLOGIES_IN_STRUCT",
    "DEFAULT_LOCALIZATION_BY_PATHOLOGY",
    "FindingDraft",
    "FindingDraftError",
    "FindingDraftResult",
    "build_finding_draft",
]
