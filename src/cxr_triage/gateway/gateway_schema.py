"""gateway_schema — OpenAI-compatible Chat Completions 请求 / 响应 / JSON schema 包装。

设计原则（spec.md §5 维度 4 正式对接层 + 维度 6 审计留痕 + CLAUDE.md
「按需模型网关」/「OpenAI-compatible 一种协议」）：

1. **不绑厂商 SDK**：业务层只调 ``ModelGateway.chat_completion``，不直接
   接触 ``openai`` / ``langchain_openai`` / 任何厂商 SDK；本文件定义
   ``ChatRequest`` / ``ChatResponse`` 是给 ``ModelGateway`` 内部序列化
   出站 JSON + 解析入站 JSON 用。
2. **结构化输出**：``response_format=json_schema`` 强约束（与 cxr_keyqa 的
   ``KEY_QA_JSON_SCHEMA`` 同形）；schema 必须含 ``type / name / strict /
   schema`` 四字段，匹配 OpenAI 2024-08+ 规范。
3. **可审计调用元数据**：``ChatRequest`` 与 ``ChatResponse`` 都带
   ``study_uid`` / ``tool`` 字段，便于 ``ModelGateway`` 写
   ``tool_calls_<date>.jsonl`` 时回填；本文件只定义 pydantic 模型，**不**
   触发任何外发。
4. **不耦合 httpx**：纯数据模型；HTTP 由 ``model_gateway.py`` 负责。
5. **可空凭据**：``api_key`` / ``base_url`` 在本类允许为空字符串，校验交给
   ``ModelGateway`` 构造器（fail-fast）。

公开入口：

- ``ChatRequest``          OpenAI-compatible 出站请求体（pydantic 模型）
- ``ChatResponse``         OpenAI-compatible 入站响应体（pydantic 模型）
- ``ChatChoice``           单条 response.choice（content / role / finish_reason）
- ``ChatUsage``            token 用量（prompt_tokens / completion_tokens / total_tokens）
- ``ResponseFormat``       ``response_format=json_schema`` 包装
- ``RESPONSE_FORMAT_JSON_OBJECT``  简易 ``response_format={"type": "json_object"}`` 常量
- ``build_json_schema_response_format`` 工厂：把 schema 包成 OpenAI 形态
- ``MessageRole``          角色枚举常量
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 角色字面量（OpenAI 协议：system / user / assistant；tool 由结构化输出扩展）
MessageRole = Literal["system", "user", "assistant", "tool"]

# 简易 JSON 对象响应格式（不强制 schema，仅保证模型返回合法 JSON object）
RESPONSE_FORMAT_JSON_OBJECT: Dict[str, Any] = {"type": "json_object"}


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


class ResponseFormat(BaseModel):
    """``response_format`` 字段；目前仅支持 ``json_schema`` 强约束或 ``json_object``。"""

    type: Literal["json_object", "json_schema", "text"] = "text"

    # 当 type=json_schema 时必填；type=text/json_object 时可为 None
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenAI 2024-08+ json_schema envelope: {name, strict, schema}",
    )

    def to_openai(self) -> Dict[str, Any]:
        """转 OpenAI 协议 ``response_format`` 字段。"""
        if self.type == "json_schema":
            if not self.json_schema:
                raise ValueError("response_format.type=json_schema requires json_schema field")
            return {"type": "json_schema", "json_schema": self.json_schema}
        if self.type == "json_object":
            return RESPONSE_FORMAT_JSON_OBJECT
        return {"type": "text"}


def build_json_schema_response_format(
    name: str,
    schema: Dict[str, Any],
    *,
    strict: bool = True,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """工厂：把业务侧 ``schema`` 字典包成 OpenAI 形态的 ``response_format``。

    Parameters
    ----------
    name:
        schema 名（OpenAI 要求）；如 ``"key_qa_response"`` / ``"finding_draft"``。
    schema:
        业务侧的 JSON Schema 字典（包含 type / items / properties 等）。
    strict:
        OpenAI 2024-08+ 强约束；默认 ``True``（任何字段都不允许额外属性）。
    description:
        可选 description；便于模型理解。

    Returns
    -------
    dict
        OpenAI 协议 ``response_format`` 字段，可直接塞进 ``ChatRequest``。
    """
    envelope: Dict[str, Any] = {
        "name": name,
        "strict": strict,
        "schema": schema,
    }
    if description:
        envelope["description"] = description
    return {"type": "json_schema", "json_schema": envelope}


# ---------------------------------------------------------------------------
# 出站请求
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """单条 message（OpenAI 协议）。"""

    role: MessageRole = "user"
    content: str = Field(default="", description="消息文本内容；空串合法（系统提示可空）")


class ChatRequest(BaseModel):
    """OpenAI-compatible Chat Completions 出站请求体。

    与 OpenAI 2024-08+ 协议对齐；只暴露本系统会用的字段（其余走 ``extra``）。
    """

    model: str = Field(..., description="模型名（如 medrax-llava-med-7b）")
    messages: List[ChatMessage] = Field(..., min_length=1, description="消息列表；至少 1 条")

    # ---- 生成参数 ----
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    # ---- 结构化输出 ----
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="json_schema 强约束；与 cxr_keyqa.KEY_QA_JSON_SCHEMA 配套",
    )

    # ---- 业务侧元数据（写 tool_calls_*.jsonl 用）----
    study_uid: str = Field(default="", description="DICOM StudyInstanceUID；可空")
    tool: str = Field(default="", description="调用方工具名（keyqa / finding_draft）")

    # ---- 透传字段（stream / stop / presence_penalty 等本系统不主动用）----
    extra: Dict[str, Any] = Field(default_factory=dict, description="透传额外字段")

    def to_openai_payload(self) -> Dict[str, Any]:
        """转 OpenAI 协议 JSON body（不含 study_uid/tool — 那些只在 audit 用）。"""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.response_format is not None:
            body["response_format"] = self.response_format.to_openai()
        for k, v in self.extra.items():
            # 避免被业务侧元数据污染 OpenAI 协议字段
            if k not in body:
                body[k] = v
        return body


# ---------------------------------------------------------------------------
# 入站响应
# ---------------------------------------------------------------------------


class ChatUsage(BaseModel):
    """OpenAI usage 字段。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    """单条 ``choices[i]``。"""

    index: int = 0
    message: ChatMessage = Field(default_factory=lambda: ChatMessage(role="assistant"))
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """OpenAI-compatible Chat Completions 入站响应体。

    字段容错：缺失字段走默认（0 / 空串）；只取 ``choices[0].message.content``
    即可拿到模型文本回复。
    """

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = Field(default_factory=list)
    usage: ChatUsage = Field(default_factory=ChatUsage)

    # 业务侧元数据（与请求对齐）
    study_uid: str = ""
    tool: str = ""

    def get_content(self) -> str:
        """拉 ``choices[0].message.content``；空 choices → 空串。"""
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""

    def get_model_id(self) -> str:
        """拉 ``model`` 字段；空 → 空串。"""
        return self.model or ""


__all__ = [
    "MessageRole",
    "RESPONSE_FORMAT_JSON_OBJECT",
    "ResponseFormat",
    "build_json_schema_response_format",
    "ChatMessage",
    "ChatRequest",
    "ChatChoice",
    "ChatUsage",
    "ChatResponse",
]
