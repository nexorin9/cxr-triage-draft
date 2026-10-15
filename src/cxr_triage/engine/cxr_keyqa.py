"""cxr_keyqa — 胸片 4 类预设问答 VQA 工具。

源产品能力 / 代码证据（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 +
Task 12 「阳性触发其余 4 件工具」中的第 2 件）：

源仓库 ``github_ref/MedRAX/medrax/tools/`` 区域里有开放式 ``XRayVQATool``（走
CheXagent / LLaVA-Med 等本地权重 + 自由问答 prompt）。这种「想问什么问什么」
的形态在医院语境下有两条不能接受的代价：

  1) **开放幻觉**：模型自由解读容易给「未经标注的临床路径」（spec.md §6
     风险缓解）。体检科医师需要的是「心影/纵隔/胸膜/肺野」四个固定的、
     入院时就有答案的预设问答，不是开放式 chat。

  2) **不可审计**：自由问答的 prompt 与 response 完全无结构，飞检没法按
     study_uid 还原「那天 VQA 问了什么、答了什么」。

融合后产品路径（把开放 VQA 改成「预设模板 + 结构化输出」）：

    StateGraph 阳性分流 → CXRKeyQA.run(state)
      → 4 模板 prompt（心影 / 纵隔 / 胸膜 / 肺野）
      → ModelGateway.chat_completion(prompt, json_schema=KEY_QA_JSON_SCHEMA,
            study_uid=...) 强制 ``response_format=json_schema``
      → 解析 ``[{q, a}, ...]`` 4 项 → ``state["key_qa_list"] = ...``
    失败路径：
      - 模型不可达 / 结构化失败 → 写 ``audit_log`` 事件
        ``keyqa_model_unreachable`` 或 ``keyqa_schema_validation_failed`` → 返
        ``key_qa_list=[] + degraded=True + degradation_reasons=[...]``
      - 工作流 execute_node 不抛（异常已经沉到工具内）

工具契约（与 Task 11 ``ToolRegistry`` + Task 12 ``TriageClassifier`` 对齐）：

  - ``.name = "keyqa"``（与 ``DEFAULT_TOOL_NAMES[1]`` 严格对齐）
  - ``async run(state: Dict[str, Any]) -> Dict[str, Any]``
  - 返回 ``{"key_qa_list": [...]}`` + 可选 degraded 字段

公开入口：

  - ``CXRKeyQA``                StateGraph 注册的工具实例
  - ``KeyQAToolError``          构造 / 入参 / 解析错误（应抛）
  - ``ModelUnreachableError``   ModelGateway 退避后仍失败的异常
  - ``ModelGatewayProtocol``    CXRKeyQA 对 ModelGateway 的协议（实现见 Task 19）
  - ``AuditLoggerProtocol``     CXRKeyQA 对 AuditLogger 的协议（实现见 Task 9）
  - ``DEFAULT_KEY_QA_TEMPLATES`` 4 模板（临床预设问答）
  - ``KEY_QA_JSON_SCHEMA``       强制模型按 ``[{q, a}, ...]`` 4 项结构化输出
  - ``DEFAULT_TIMEOUT_SECONDS``  30s（与 ModelGateway 协议对齐）
  - ``build_cxr_keyqa``         工厂函数

实现取舍：

  - **预设模板而非自由问答**：4 类临床常规模板，避免开放幻觉；prompt 含
    DICOM metadata（PatientSex / PatientAge / StudyDescription）帮助模型缩小
    范围但不强制回答。
  - **强制结构化输出**：``response_format=json_schema`` 强约束模型按
    ``[{q: str, a: str}, ...]`` 返，便于解析 + 飞检审计。
  - **模型不可达 = 软降级**：返 ``key_qa_list=[] + degraded=True`` +
    ``degradation_reasons`` + audit_log（不抛），与 workflow _execute_node 的
    异常路径解耦 —— 避免让 ``keyqa_model_unreachable`` 与 ``keyqa_unreachable``
    双层原因叠加。
  - **结构化失败 = 软降级**：同上，audit + 空列表；不重试（避免幻觉循环）。
  - **审计写入**：audit_logger 接受 None 时跳过（便于单测），不为审计崩溃
    而阻断主路径。
  - **依赖注入**：构造函数接受 ``model_gateway`` 与 ``audit_logger``，未提供
    时 ``run()`` 直接抛 ``KeyQAToolError``（fail-fast，避免生产环境误用空
    gateway 把模型出口偷换成本地 stub）。
  - **异步兼容**：`ModelGatewayProtocol.chat_completion` 是 async；异步超时用
    ``asyncio.wait_for`` 保护，避免单次卡住整个 StateGraph。
  - **协议而非耦合**：与 Task 19（ModelGateway 实现）解耦；Task 19 只需保证
    ``chat_completion(prompt, *, study_uid, json_schema, timeout_seconds)``
    满足本协议即可无缝替换 mock。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量 — 临床预设问答模板（4 类 · 每个模板 = (问题分类, prompt 子句)）
# ---------------------------------------------------------------------------

# 体格检查 + 体检影像初筛的标准 4 视图：心影 / 纵隔 / 胸膜 / 肺野。
# 这些分类是体检胸片报告「所见」一栏长期沿用的分项；固定 prompt 避免
# 开放式 VQA 的临床幻觉。
DEFAULT_KEY_QA_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    (
        "心影",
        "请回答该胸片的心影表现（心影大小、心胸比、形态是否正常；正常时"
        "请明确答'未见明显异常'）。不要给出诊断或治疗建议。",
    ),
    (
        "纵隔",
        "请回答该胸片的纵隔表现（纵隔位置、宽度、气管居中与否；正常时"
        "请明确答'未见明显异常'）。不要给出诊断或治疗建议。",
    ),
    (
        "胸膜",
        "请回答该胸片的两侧胸膜表现（有无增厚、积液、气胸；正常时请"
        "明确答'未见明显异常'）。不要给出诊断或治疗建议。",
    ),
    (
        "肺野",
        "请回答该胸片的两肺野表现（透亮度、有无实变/结节/肿块/浸润"
        "影；正常时请明确答'未见明显异常'）。不要给出诊断或治疗建议。",
    ),
)

# 模型默认超时（与 ModelGateway 协议默认对齐）
DEFAULT_TIMEOUT_SECONDS: float = 30.0

# 4 项问答的最大生成长度（模型 prompt 末段 + JSON schema enforce 用）
MAX_KEY_QA_ITEMS: int = 8
MIN_KEY_QA_ITEMS: int = 1


# ---------------------------------------------------------------------------
# JSON schema — 强制模型按 ``[{q: str, a: str}, ...]`` 结构化输出
# ---------------------------------------------------------------------------

KEY_QA_JSON_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "key_qa_response",
        "strict": True,
        "schema": {
            "type": "array",
            "minItems": MIN_KEY_QA_ITEMS,
            "maxItems": MAX_KEY_QA_ITEMS,
            "items": {
                "type": "object",
                "required": ["q", "a"],
                "properties": {
                    "q": {"type": "string", "minLength": 1, "maxLength": 64},
                    "a": {"type": "string", "minLength": 1, "maxLength": 512},
                },
                "additionalProperties": False,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 异常 / 协议 / 结果
# ---------------------------------------------------------------------------


class KeyQAToolError(RuntimeError):
    """CXRKeyQA 构造 / 入参 / 解析错误（应抛）。"""


class ModelUnreachableError(RuntimeError):
    """ModelGateway 退避后仍无法访问（Connection / Timeout / 5xx）。"""


class KeyQAProtocol(Protocol):
    """CXRKeyQA 对 ``ModelGateway`` 的最低协议（Task 19 实现须满足）。

    Task 19 的 ``ModelGateway.chat_completion`` 实现签名::

        async def chat_completion(
            self,
            prompt: str,
            *,
            study_uid: str = "",
            json_schema: Optional[Dict[str, Any]] = None,
            timeout_seconds: Optional[float] = None,
        ) -> Dict[str, Any]: ...

    返回值形态：

    - 成功：``{"content": <parsed JSON 字典或原始字符串>, "raw": <OpenAI 风格
      response dict>, "model": <model_id>}``
    - 失败：抛 ``ModelUnreachableError``
    """

    def chat_completion(
        self,
        prompt: str,
        *,
        study_uid: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Awaitable[Dict[str, Any]]:
        ...


class AuditLoggerProtocol(Protocol):
    """CXRKeyQA 对 ``AuditLogger``（Task 9）的最低协议。"""

    def audit_log_insert(
        self,
        actor_role: str,
        ts: str,
        study_uid: str,
        event_type: str,
        payload: Any,
    ) -> Awaitable[None]:
        ...


@dataclass(frozen=True)
class KeyQAResult:
    """单次 VQA 调用的不可变结果，便于单测直接消费。"""

    key_qa_list: List[Dict[str, str]]   # [{"q": "...", "a": "..."}, ...]
    source: str                         # "model" | "fallback"
    elapsed_ms: int
    model_id: str = ""

    def as_state_patch(self) -> Dict[str, Any]:
        """转 ``dict``，与 ``process_node`` 期望的字段名严格对齐。"""
        return {"key_qa_list": list(self.key_qa_list)}


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


class CXRKeyQA:
    """StateGraph 注册的 keyqa 工具（4 模板 VQA + 结构化输出 + 软降级）。

    工具契约（Task 11 ``ToolRegistry`` 强校验）：

    - ``.name = "keyqa"``（与 ``DEFAULT_TOOL_NAMES[1]`` 对齐）
    - ``async run(state: dict) -> dict`` —— 返回 ``key_qa_list`` 字段 patch +
      可选 ``degraded`` / ``degradation_reasons``

    构造参数：

    - ``model_gateway`` : 实现 ``KeyQAProtocol`` 的对象（Task 19 ``ModelGateway``
      或单测 MockGateway）；未提供 → ``run()`` 抛 ``KeyQAToolError``
    - ``audit_logger``  : 实现 ``AuditLoggerProtocol``（Task 9 ``AuditLogger``）
                         的对象；缺省时跳过审计写入（不阻断主路径）
    - ``templates``     : 预设问答模板元组，默认 ``DEFAULT_KEY_QA_TEMPLATES``
    - ``timeout_seconds``: 单次 chat_completion 超时秒数（默认 30s）

    入参：

    - ``state["png_bytes"]`` : 8-bit PNG 字节（DicomProcessor 产物）
    - ``state["study_uid"]`` : 业务研究唯一标识（用于 audit + 模型调用元数据）
    - ``state["metadata"]``  : DICOM 解析后的 metadata dict（辅助 prompt）

    异常处理：

    - 入参非法（png_bytes 缺失）→ ``KeyQAToolError``（fail-fast）
    - 模型不可达 → soft-degrade（audit + 返 ``key_qa_list=[] + degraded=True``）
    - 结构化失败 / JSON 解析失败 → soft-degrade（audit + 返空 + degraded）
    - 模型返回超过 maxItems / 缺字段 → soft-degrade（保留能解析的项）
    """

    name: str = "keyqa"

    def __init__(
        self,
        model_gateway: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        *,
        templates: Tuple[Tuple[str, str], ...] = DEFAULT_KEY_QA_TEMPLATES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # templates 必须为非空元组
        if not isinstance(templates, tuple) or len(templates) == 0:
            raise KeyQAToolError(
                "templates must be a non-empty tuple of (category, prompt) pairs"
            )
        for i, t in enumerate(templates):
            if not isinstance(t, tuple) or len(t) != 2:
                raise KeyQAToolError(
                    f"templates[{i}] must be a 2-tuple of (category, prompt)"
                )
            cat, sub = t
            if not isinstance(cat, str) or not cat.strip():
                raise KeyQAToolError(f"templates[{i}][0] category must be non-empty string")
            if not isinstance(sub, str) or not sub.strip():
                raise KeyQAToolError(f"templates[{i}][1] prompt must be non-empty string")

        # timeout_seconds 校验
        try:
            ts = float(timeout_seconds)
        except (TypeError, ValueError):
            raise KeyQAToolError(f"timeout_seconds must be numeric, got {timeout_seconds!r}")
        if ts <= 0 or ts > 600:
            raise KeyQAToolError(
                f"timeout_seconds must be in (0, 600], got {timeout_seconds}"
            )

        self.templates = templates
        self.timeout_seconds = float(timeout_seconds)
        self.gateway = model_gateway
        self.audit_logger = audit_logger

        # 模板分类名集合（去空白 + 顺序锁定）——便于 build_prompt 与单测对照
        self.template_categories: Tuple[str, ...] = tuple(cat for cat, _ in templates)

    # ---- 公开契约 ----

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """StateGraph 节点调用入口；返回 ``{key_qa_list, degraded?, degradation_reasons?}``。

        异常处理（与 workflow._execute_node 协作）：

        - 任何模型不可达 / 结构化失败 → soft-degrade：返回 ``key_qa_list=[]``
          + ``degraded=True`` + ``degradation_reasons=[...]``，不抛
        - 入参非法（png_bytes 缺失 / gateway None）→ 直接抛 ``KeyQAToolError``
        """
        png_bytes = state.get("png_bytes")
        if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) == 0:
            raise KeyQAToolError("state['png_bytes'] must be non-empty bytes")
        study_uid = state.get("study_uid") or ""
        if not isinstance(study_uid, str):
            raise KeyQAToolError("state['study_uid'] must be a string")
        metadata = state.get("metadata") or {}

        if self.gateway is None:
            # 与 Task 19 ModelGateway 解耦时仍能加载本类；运行期直接抛
            # fail-fast，避免生产环境误用空网关
            raise KeyQAToolError(
                "CXRKeyQA.run() requires model_gateway; use build_cxr_keyqa(...) "
                "to inject one. Soft-degrade path also requires explicit gateway "
                "for audit consistency."
            )

        prompt = self._build_prompt(metadata)
        start = time.monotonic()

        # ---- 调模型 ----
        try:
            response = await asyncio.wait_for(
                self.gateway.chat_completion(
                    prompt,
                    study_uid=study_uid,
                    json_schema=KEY_QA_JSON_SCHEMA,
                    timeout_seconds=self.timeout_seconds,
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return await self._record_degradation(
                study_uid=study_uid,
                event_type="keyqa_model_unreachable",
                reason=f"timeout_after_{self.timeout_seconds}s",
                elapsed_ms=elapsed,
                extra={"category": "timeout"},
            )
        except ModelUnreachableError as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return await self._record_degradation(
                study_uid=study_uid,
                event_type="keyqa_model_unreachable",
                reason=str(exc),
                elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 —— 网络 / 协议 / 其它
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning("keyqa gateway call unexpected exception: %s", exc)
            return await self._record_degradation(
                study_uid=study_uid,
                event_type="keyqa_model_unreachable",
                reason=f"unexpected:{type(exc).__name__}",
                elapsed_ms=elapsed,
            )

        # ---- 解析模型结果 ----
        try:
            result = self._parse_response(response)
        except KeyQAToolError as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return await self._record_degradation(
                study_uid=study_uid,
                event_type="keyqa_schema_validation_failed",
                reason=str(exc),
                elapsed_ms=elapsed,
                extra={"raw_response_keys": list(response.keys()) if isinstance(response, dict) else None},
            )

        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "key_qa_list": result.key_qa_list,
            "degraded": False,
            "degradation_reasons": [],
            "_keyqa_meta": {
                "source": result.source,
                "elapsed_ms": elapsed,
                "model_id": result.model_id,
                "item_count": len(result.key_qa_list),
            },
        }

    # ---- 模板 prompt ----

    def _build_prompt(self, metadata: Dict[str, Any]) -> str:
        """构造 4 模板问答 prompt（含 metadata 上下文，便于模型缩小范围）。"""
        ctx_lines: List[str] = []
        pat_sex = metadata.get("PatientSex") or metadata.get("patient_sex")
        pat_age = metadata.get("PatientAge") or metadata.get("patient_age")
        study_desc = metadata.get("StudyDescription") or metadata.get("study_description")
        if pat_sex:
            ctx_lines.append(f"患者性别：{pat_sex}")
        if pat_age:
            ctx_lines.append(f"患者年龄：{pat_age}")
        if study_desc:
            ctx_lines.append(f"检查描述：{study_desc}")

        ctx_block = ("\n".join(ctx_lines) + "\n\n") if ctx_lines else ""

        sections: List[str] = ["你是一名体检胸片初筛助手，需严格按下列 4 模板回答：\n"]
        for i, (cat, sub) in enumerate(self.templates, start=1):
            sections.append(f"{i}. 【{cat}】{sub}")
        sections.append(
            "\n请严格按上述 4 模板顺序，输出 JSON 数组；每个元素包含"
            ' "q"（问题分类的简短复述）与 "a"（所见要点，不超 100 字）。'
            " 正常项请明确答'未见明显异常'。不要给出诊断或治疗建议。"
        )
        return ctx_block + "\n".join(sections)

    # ---- 响应解析 ----

    def _parse_response(self, response: Any) -> KeyQAResult:
        """从 ModelGateway 响应里抽 ``[{q, a}, ...]`` 4 项。

        兼容三种返回形态：
          1) ``response["content"]`` 已经是 ``list[dict]``（model gateway 已按
             json_schema 解析）
          2) ``response["content"]`` 是 ``str``（含 JSON 数组）
          3) ``response["parsed"]`` 字段（OpenAI structured output）
          4) ``response["choices"][0]["message"]["content"]``（OpenAI 原始格式）
        """
        raw: Any = None
        if not isinstance(response, dict):
            raise KeyQAToolError(
                f"gateway returned non-dict response: {type(response).__name__}"
            )
        if "content" in response:
            raw = response["content"]
        elif "parsed" in response:
            raw = response["parsed"]
        elif "choices" in response and isinstance(response["choices"], list) \
                and len(response["choices"]) > 0:
            try:
                raw = response["choices"][0]["message"]["content"]
            except (TypeError, KeyError):
                raw = None
        else:
            raise KeyQAToolError(
                f"gateway response missing 'content'/'parsed'/'choices' keys: "
                f"{list(response.keys())}"
            )

        items = self._coerce_to_qa_list(raw)

        # 截取前 MAX_KEY_QA_ITEMS 条
        items = items[:MAX_KEY_QA_ITEMS]
        if len(items) < MIN_KEY_QA_ITEMS:
            raise KeyQAToolError(
                f"parsed key_qa list too short: got {len(items)} items, "
                f"expected >= {MIN_KEY_QA_ITEMS}"
            )

        model_id = ""
        if isinstance(response, dict):
            model_id = str(response.get("model") or "")

        return KeyQAResult(
            key_qa_list=items,
            source="model",
            elapsed_ms=0,
            model_id=model_id,
        )

    def _coerce_to_qa_list(self, raw: Any) -> List[Dict[str, str]]:
        """把模型内容（list / str / dict）归一成 ``[{q, a}, ...]``。"""
        if raw is None:
            raise KeyQAToolError("gateway response content is null")

        # 1) list
        if isinstance(raw, list):
            return [self._coerce_item(it) for it in raw]

        # 2) dict 套在数组里（极少数模型输出 {"questions": [...]}）
        if isinstance(raw, dict):
            for key in ("questions", "key_qa", "items", "results"):
                if key in raw and isinstance(raw[key], list):
                    return [self._coerce_item(it) for it in raw[key]]
            raise KeyQAToolError(
                f"gateway response object has no recognized list key: {list(raw.keys())}"
            )

        # 3) str —— 试图解析为 JSON 数组
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                raise KeyQAToolError("gateway response content is empty string")
            # 先尝试整段解析
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [self._coerce_item(it) for it in parsed]
                if isinstance(parsed, dict):
                    for key in ("questions", "key_qa", "items", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            return [self._coerce_item(it) for it in parsed[key]]
                    raise KeyQAToolError(
                        f"parsed JSON object has no recognized list key: {list(parsed.keys())}"
                    )
                raise KeyQAToolError(
                    f"parsed JSON is {type(parsed).__name__}, expected list"
                )
            except json.JSONDecodeError:
                pass

            # 退而求其次：抽取首段 [...] 子串
            start = stripped.find("[")
            end = stripped.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = stripped[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return [self._coerce_item(it) for it in parsed]
                except json.JSONDecodeError as exc:
                    raise KeyQAToolError(
                        f"could not parse JSON array from string content: {exc}"
                    ) from exc

            raise KeyQAToolError(
                "gateway response string is neither JSON nor wrapped in [...]"
            )

        raise KeyQAToolError(
            f"unsupported response content type: {type(raw).__name__}"
        )

    @staticmethod
    def _coerce_item(item: Any) -> Dict[str, str]:
        """单条 ``{q, a}`` 规范化：非 dict 拒绝；缺字段拒绝；字段非字符串拒绝；
        字段空白拒绝；超长截断。"""
        if not isinstance(item, dict):
            raise KeyQAToolError(f"key_qa item must be dict, got {type(item).__name__}")
        q = item.get("q")
        a = item.get("a")
        if not isinstance(q, str) or not q.strip():
            raise KeyQAToolError("key_qa item missing or empty 'q' field")
        if not isinstance(a, str) or not a.strip():
            raise KeyQAToolError("key_qa item missing or empty 'a' field")
        return {
            "q": q.strip()[:64],
            "a": a.strip()[:512],
        }

    # ---- 软降级 + audit ----

    async def _record_degradation(
        self,
        *,
        study_uid: str,
        event_type: str,
        reason: str,
        elapsed_ms: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写 audit + 返软降级 patch。

        - ``event_type``: ``keyqa_model_unreachable`` / ``keyqa_schema_validation_failed``
        - audit_logger 缺省时**只 warn 不阻断**（便于测试）
        """
        logger.warning(
            "keyqa soft-degrade: study_uid=%s event=%s reason=%s elapsed_ms=%d",
            study_uid or "<none>", event_type, reason, elapsed_ms,
        )

        payload: Dict[str, Any] = {
            "reason": reason,
            "elapsed_ms": elapsed_ms,
            "templates_count": len(self.templates),
        }
        if extra:
            payload.update(extra)

        if self.audit_logger is not None:
            try:
                from cxr_triage.writers.audit import utc_now_iso
                await self.audit_logger.audit_log_insert(
                    actor_role="system",
                    ts=utc_now_iso(),
                    study_uid=study_uid or "<no_study_uid>",
                    event_type=event_type,
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001 —— 审计失败不阻断主路径
                logger.warning(
                    "keyqa audit_log_insert failed for %s (%s): %s",
                    study_uid, event_type, exc,
                )

        reasons = list(self.templates or [])
        return {
            "key_qa_list": [],
            "degraded": True,
            "degradation_reasons": [f"{event_type}:{reason}"],
            "_keyqa_meta": {
                "source": "fallback",
                "elapsed_ms": elapsed_ms,
                "event_type": event_type,
            },
        }

    # ---- 静态工具 ----

    @staticmethod
    def to_finding_struct(
        key_qa_list: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """把 key_qa 列表原样塞进 finding_struct.key_qa 字段（finding_draft 复用）。

        与 spec.md §4 finding_struct 字段约定 ``key_qa: [{q, a}, ...]`` 对齐。
        """
        if not isinstance(key_qa_list, list):
            return []
        return [
            {"q": str(it.get("q", "")), "a": str(it.get("a", ""))}
            for it in key_qa_list
            if isinstance(it, dict)
        ]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_cxr_keyqa(
    *,
    model_gateway: Optional[Any] = None,
    audit_logger: Optional[Any] = None,
    templates: Tuple[Tuple[str, str], ...] = DEFAULT_KEY_QA_TEMPLATES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CXRKeyQA:
    """工厂：构造 ``CXRKeyQA`` 一行完成。"""
    return CXRKeyQA(
        model_gateway=model_gateway,
        audit_logger=audit_logger,
        templates=templates,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# 模块导出
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_KEY_QA_TEMPLATES",
    "DEFAULT_TIMEOUT_SECONDS",
    "KEY_QA_JSON_SCHEMA",
    "MAX_KEY_QA_ITEMS",
    "MIN_KEY_QA_ITEMS",
    "CXRKeyQA",
    "KeyQAToolError",
    "ModelUnreachableError",
    "KeyQAResult",
    "KeyQAProtocol",
    "AuditLoggerProtocol",
    "build_cxr_keyqa",
]
