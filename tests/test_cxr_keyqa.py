"""tests/test_cxr_keyqa.py — Task 14 CXRKeyQA 4 模板 VQA 单测。

目标（spec.md §5 维度 1 + plan.md §5 深度 (a) 真实算法 + Task 14）：

1. **常量与模板**：4 模板锁定 + 顺序锁定 + 类别名锁定（心影 / 纵隔 /
   胸膜 / 肺野）+ JSON schema 必备字段
2. **构造期校验**：templates 空 / 单条格式错 / timeout_seconds 越界 / 缺
   gateway 时 ``run()`` 抛 ``KeyQAToolError``
3. **入参校验**：png_bytes 缺失 / 空 / 非 bytes 一律抛 ``KeyQAToolError``
4. **happy path**：mock ModelGateway 返有效 ``[{q, a}, ...]`` → ``key_qa_list``
   4 项 + ``degraded=False``
5. **3 例 fixtures**：阳性 / 阴性 / 边界 → mock gateway 返差异化答案 → 都
   正常解析
6. **JSON schema 校验失败**：gateway 返非 list 字符串 / 缺字段 / 字段空白 /
   字段非字符串 → soft-degrade（audit + 空列表 + degraded）
7. **模型不可达**：gateway 抛 ``ModelUnreachableError`` → soft-degrade +
   audit ``keyqa_model_unreachable``
8. **超时降级**：gateway 阻塞超 timeout_seconds → soft-degrade +
   audit ``keyqa_model_unreachable:timeout_after_Xs``
9. **响应格式兼容**：content 已 list / content 是 JSON 字符串 / choices
   OpenAI 风格 / parsed 字段 — 4 种解析路径
10. **JSON schema 协议契约**：传给 gateway 的 prompt / study_uid /
    json_schema / timeout_seconds 字段对齐
11. **审计写入**：audit_logger 接受 + event_type / actor_role / payload 字段
    全对；audit 抛异常不阻断主路径
12. **与 StateGraph 集成**：ToolRegistry 注册 + 与 classifier + finding_draft
    mock 工具跑通 happy-path
13. **去 finding_struct 复用**：to_finding_struct 静态方法
"""
from __future__ import annotations

import asyncio
import io
import json
from typing import Any, Dict, List, Optional

import pytest

from cxr_triage.engine import (
    DEFAULT_KEY_QA_TEMPLATES,
    DEFAULT_TIMEOUT_SECONDS,
    KEY_QA_JSON_SCHEMA,
    CXRKeyQA,
    CxrTriageWorkflow,
    KeyQAResult,
    KeyQAToolError,
    ModelUnreachableError,
    PATHOLOGY_NAMES,
    TriageClassifier,
    ToolRegistry,
    build_cxr_keyqa,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 64, height: int = 64, value: int = 128) -> bytes:
    """生成单色 8-bit PNG bytes（用于推理入参）。"""
    from PIL import Image
    img = Image.new("L", (width, height), value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# 默认 4 模板对应的（q, a）列表；用于 mock gateway 成功路径
DEFAULT_4_ANSWERS: List[Dict[str, str]] = [
    {"q": "心影", "a": "心影大小未见明显异常，心胸比正常。"},
    {"q": "纵隔", "a": "纵隔位置居中，气管居中，纵隔宽度未见明显异常。"},
    {"q": "胸膜", "a": "两侧胸膜未见增厚、胸腔积液或气胸。"},
    {"q": "肺野", "a": "两肺野透亮度正常，未见明显实变/结节/肿块影。"},
]


class MockGateway:
    """Mock CXRKeyQA KeyQAProtocol 实现；按预设响应 / 异常策略分支。"""

    def __init__(
        self,
        response_content: Any = None,
        raise_exc: Optional[BaseException] = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.response_content = response_content
        self._raise = raise_exc
        self._delay = delay_seconds
        self.call_count = 0
        self.last_call_kwargs: Dict[str, Any] = {}

    async def chat_completion(
        self,
        prompt: str,
        *,
        study_uid: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        self.last_call_kwargs = {
            "prompt": prompt,
            "study_uid": study_uid,
            "json_schema": json_schema,
            "timeout_seconds": timeout_seconds,
        }
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise is not None:
            raise self._raise
        if isinstance(self.response_content, BaseException):
            raise self.response_content
        return {
            "content": self.response_content,
            "model": "mock-gpt-4o",
            "raw": {"mock": True},
        }


class MockAuditLogger:
    """Mock AuditLoggerProtocol；记录每次 audit_log_insert 调用。"""

    def __init__(self, raise_exc: Optional[BaseException] = None) -> None:
        self.inserts: List[Dict[str, Any]] = []
        self._raise = raise_exc

    async def audit_log_insert(
        self,
        actor_role: str,
        ts: str,
        study_uid: str,
        event_type: str,
        payload: Any,
    ) -> None:
        if self._raise is not None:
            raise self._raise
        self.inserts.append({
            "actor_role": actor_role,
            "ts": ts,
            "study_uid": study_uid,
            "event_type": event_type,
            "payload": payload,
        })


def _new_state(study_uid: str = "1.2.826.0.1.3680043.8.498.<EXAMPLE_TEST>") -> Dict[str, Any]:
    """构造最小可跑 StateGraph state。"""
    return {
        "study_uid": study_uid,
        "png_bytes": _make_png_bytes(),
        "metadata": {
            "PatientSex": "F",
            "PatientAge": "052Y",
            "StudyDescription": "Chest PA (routine)",
            "Modality": "CR",
        },
        "tools_to_execute": ["keyqa"],
        "tools_executed": ["classifier"],
        "classifier_probs": [{"name": "Pneumothorax", "prob": 0.85, "present": True}],
        "classifier_max_prob": 0.85,
        "positive_flag": True,
        "degraded": False,
        "degradation_reasons": [],
    }


# ---------------------------------------------------------------------------
# 常量层
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_templates_count_is_4(self):
        assert len(DEFAULT_KEY_QA_TEMPLATES) == 4

    def test_default_templates_categories(self):
        cats = tuple(cat for cat, _ in DEFAULT_KEY_QA_TEMPLATES)
        assert cats == ("心影", "纵隔", "胸膜", "肺野")

    def test_default_templates_unique_categories(self):
        cats = [cat for cat, _ in DEFAULT_KEY_QA_TEMPLATES]
        assert len(set(cats)) == len(cats)

    def test_default_templates_prompts_nonempty(self):
        for cat, prompt in DEFAULT_KEY_QA_TEMPLATES:
            assert isinstance(cat, str) and cat.strip()
            assert isinstance(prompt, str) and prompt.strip()

    def test_default_templates_avoid_diagnosis_phrasing(self):
        """禁止出现「直接推荐诊断 / 推荐进一步检查」类措辞（spec.md §10
        安全边界：不读诊断也不出治疗建议）。允许出现「不要给出诊断或治疗建议」
        这种**反向约束**措辞。"""
        # 真正禁止：模型被 prompt 鼓励去做诊断动作的措辞
        forbidden_endorsement = (
            "建议做CT", "建议CT复查", "建议进一步检查", "建议手术", "建议用药",
            "考虑诊断", "考虑为", "怀疑", "可能是",
        )
        for _, prompt in DEFAULT_KEY_QA_TEMPLATES:
            for f in forbidden_endorsement:
                assert f not in prompt, f"template must not endorse {f!r}: {prompt!r}"

        # 反向约束措辞应被保留
        for _, prompt in DEFAULT_KEY_QA_TEMPLATES:
            assert "不要给出诊断或治疗建议" in prompt \
                or "不要给出" in prompt \
                or "不输出" in prompt, \
                f"template must explicitly negate diagnosis phrasing: {prompt!r}"

    def test_json_schema_envelope_type(self):
        assert KEY_QA_JSON_SCHEMA["type"] == "json_schema"

    def test_json_schema_inner_schema(self):
        schema = KEY_QA_JSON_SCHEMA["json_schema"]
        assert schema["name"] == "key_qa_response"
        assert schema["schema"]["type"] == "array"
        item_schema = schema["schema"]["items"]
        assert set(item_schema["required"]) == {"q", "a"}
        # 强制 type=string
        assert item_schema["properties"]["q"]["type"] == "string"
        assert item_schema["properties"]["a"]["type"] == "string"

    def test_default_timeout_seconds(self):
        assert DEFAULT_TIMEOUT_SECONDS == 30.0


# ---------------------------------------------------------------------------
# 构造期校验
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_build_factory(self):
        gw = MockGateway()
        tool = build_cxr_keyqa(model_gateway=gw)
        assert isinstance(tool, CXRKeyQA)
        assert tool.name == "keyqa"

    def test_default_templates_locked_when_omitted(self):
        tool = CXRKeyQA(model_gateway=MockGateway())
        assert tool.templates == DEFAULT_KEY_QA_TEMPLATES
        assert tool.template_categories == ("心影", "纵隔", "胸膜", "肺野")

    def test_empty_templates_rejected(self):
        with pytest.raises(KeyQAToolError, match="non-empty tuple"):
            CXRKeyQA(model_gateway=MockGateway(), templates=())

    def test_malformed_template_rejected(self):
        with pytest.raises(KeyQAToolError, match="2-tuple"):
            CXRKeyQA(model_gateway=MockGateway(), templates=(("only-cat",),))

    def test_blank_category_rejected(self):
        with pytest.raises(KeyQAToolError, match="non-empty string"):
            CXRKeyQA(model_gateway=MockGateway(), templates=(("", "x"),))

    def test_blank_prompt_rejected(self):
        with pytest.raises(KeyQAToolError, match="prompt must be non-empty"):
            CXRKeyQA(model_gateway=MockGateway(), templates=(("cat", "  "),))

    def test_negative_timeout_rejected(self):
        with pytest.raises(KeyQAToolError, match="timeout_seconds"):
            CXRKeyQA(model_gateway=MockGateway(), timeout_seconds=-1)

    def test_excessive_timeout_rejected(self):
        with pytest.raises(KeyQAToolError, match="timeout_seconds"):
            CXRKeyQA(model_gateway=MockGateway(), timeout_seconds=700)

    def test_non_numeric_timeout_rejected(self):
        with pytest.raises(KeyQAToolError, match="timeout_seconds"):
            CXRKeyQA(model_gateway=MockGateway(), timeout_seconds="abc")


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


class TestStateValidation:
    @pytest.mark.asyncio
    async def test_missing_png_bytes_raises(self):
        tool = CXRKeyQA(model_gateway=MockGateway())
        with pytest.raises(KeyQAToolError, match="png_bytes"):
            await tool.run({"study_uid": "uid-1"})

    @pytest.mark.asyncio
    async def test_empty_png_bytes_raises(self):
        tool = CXRKeyQA(model_gateway=MockGateway())
        with pytest.raises(KeyQAToolError, match="png_bytes"):
            await tool.run({"study_uid": "uid-1", "png_bytes": b""})

    @pytest.mark.asyncio
    async def test_bytearray_accepted(self):
        tool = CXRKeyQA(model_gateway=MockGateway(response_content=DEFAULT_4_ANSWERS))
        state = _new_state()
        state["png_bytes"] = bytearray(state["png_bytes"])
        patch = await tool.run(state)
        assert "key_qa_list" in patch
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_non_string_study_uid_raises(self):
        tool = CXRKeyQA(model_gateway=MockGateway())
        with pytest.raises(KeyQAToolError, match="study_uid"):
            await tool.run({"png_bytes": _make_png_bytes(), "study_uid": 123})


# ---------------------------------------------------------------------------
# Happy path — 4 模板问答
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_default_4_answers(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())

        assert patch["key_qa_list"] == DEFAULT_4_ANSWERS
        assert patch["degraded"] is False
        assert patch["degradation_reasons"] == []
        assert gw.call_count == 1

    @pytest.mark.asyncio
    async def test_call_kwargs_passed_correctly(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw, timeout_seconds=15.0)
        await tool.run(_new_state())

        kw = gw.last_call_kwargs
        assert kw["study_uid"] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_TEST>"
        assert kw["json_schema"] == KEY_QA_JSON_SCHEMA
        assert kw["timeout_seconds"] == 15.0
        # prompt 应包含元数据 + 4 模板分类
        assert "心影" in kw["prompt"]
        assert "纵隔" in kw["prompt"]
        assert "胸膜" in kw["prompt"]
        assert "肺野" in kw["prompt"]
        assert "F" in kw["prompt"]  # PatientSex in metadata
        assert "052Y" in kw["prompt"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_meta_field", [None, "PatientSex", "PatientAge", "StudyDescription"])
    async def test_prompt_handles_missing_metadata(self, missing_meta_field):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        state = _new_state()
        if missing_meta_field is not None:
            state["metadata"].pop(missing_meta_field, None)
        # 不应抛
        await tool.run(state)
        assert gw.call_count == 1

    @pytest.mark.asyncio
    async def test_metadata_none_fallback(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        state = _new_state()
        state["metadata"] = None
        patch = await tool.run(state)
        assert patch["degraded"] is False

    @pytest.mark.asyncio
    async def test_metadata_dict_passthrough(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        state = _new_state()
        state["metadata"] = {"Modality": "DX", "PatientSex": "M"}
        patch = await tool.run(state)
        assert patch["degraded"] is False

    @pytest.mark.asyncio
    async def test_meta_block_contains_elapsed(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        meta = patch["_keyqa_meta"]
        assert meta["source"] == "model"
        assert meta["model_id"] == "mock-gpt-4o"
        assert meta["item_count"] == 4
        assert isinstance(meta["elapsed_ms"], int) and meta["elapsed_ms"] >= 0


# ---------------------------------------------------------------------------
# 3 例 fixtures（阳 / 阴 / 边界）
# ---------------------------------------------------------------------------


class TestThreeFixtures:
    @pytest.mark.asyncio
    async def test_positive_pneumothorax_fixture(self):
        """阳性 fixture：mock gateway 给出气胸相关的描写。"""
        answers = [
            {"q": "心影", "a": "心影大小未见明显异常。"},
            {"q": "纵隔", "a": "纵隔居中，未见明显异常。"},
            {"q": "胸膜", "a": "右侧胸腔见少量气胸征象，请结合临床。"},
            {"q": "肺野", "a": "右下肺野透亮度增高，符合气胸表现；余肺野清晰。"},
        ]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert "气胸" in patch["key_qa_list"][2]["a"]
        assert "气胸" in patch["key_qa_list"][3]["a"]

    @pytest.mark.asyncio
    async def test_negative_normal_fixture(self):
        answers = [
            {"q": "心影", "a": "心影大小未见明显异常，心胸比正常。"},
            {"q": "纵隔", "a": "纵隔居中，气管居中，未见明显异常。"},
            {"q": "胸膜", "a": "双侧胸膜未见明显异常。"},
            {"q": "肺野", "a": "两肺野未见明显异常。"},
        ]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        # 至少 3 项含「未见明显异常」
        cnt = sum(
            1 for item in patch["key_qa_list"]
            if "未见明显异常" in item["a"]
            or "未见异常" in item["a"]
        )
        assert cnt >= 3, f"expected ≥3 '未见异常' answers, got {cnt}"

    @pytest.mark.asyncio
    async def test_borderline_cardiomegaly_fixture(self):
        answers = [
            {"q": "心影", "a": "心影临界增大，心胸比约 0.52，建议结合临床。"},
            {"q": "纵隔", "a": "纵隔居中，未见明显异常。"},
            {"q": "胸膜", "a": "双侧胸膜未见明显异常。"},
            {"q": "肺野", "a": "两肺野清晰，未见明显异常。"},
        ]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert "临界" in patch["key_qa_list"][0]["a"]


# ---------------------------------------------------------------------------
# JSON schema 校验失败 → soft-degrade
# ---------------------------------------------------------------------------


class TestSchemaValidationFailures:
    @pytest.mark.asyncio
    async def test_response_not_dict_soft_degrade(self):
        """gateway 返 list 而非 dict → soft-degrade。"""
        class BadGw:
            async def chat_completion(self, prompt, **kw):
                return ["not", "a", "dict"]

        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=BadGw(), audit_logger=audit)
        patch = await tool.run(_new_state())

        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True
        assert any("non-dict" in r or "schema_validation_failed" in r
                   for r in patch["degradation_reasons"])
        assert audit.inserts[0]["event_type"] == "keyqa_schema_validation_failed"

    @pytest.mark.asyncio
    async def test_content_is_nondict_response(self):
        class BadGw:
            async def chat_completion(self, prompt, **kw):
                return "string-not-dict"

        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=BadGw(), audit_logger=audit)
        patch = await tool.run(_new_state())

        # 字符串 content 走解析分支：string parsed as JSON → 失败 → soft-degrade
        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True
        assert audit.inserts[0]["event_type"] == "keyqa_schema_validation_failed"

    @pytest.mark.asyncio
    async def test_content_missing_recognized_keys(self):
        class BadGw:
            async def chat_completion(self, prompt, **kw):
                return {"unexpected_key": "value"}

        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=BadGw(), audit_logger=audit)
        patch = await tool.run(_new_state())

        assert patch["degraded"] is True
        assert any("missing 'content'" in r or "schema_validation_failed" in r
                   for r in patch["degradation_reasons"])

    @pytest.mark.asyncio
    async def test_content_is_object_with_unknown_keys(self):
        class BadGw:
            async def chat_completion(self, prompt, **kw):
                return {"content": {"foo": "bar"}}

        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=BadGw(), audit_logger=audit)
        patch = await tool.run(_new_state())

        # content 是 dict 且不含 recognized list key → soft-degrade
        assert patch["degraded"] is True

    @pytest.mark.asyncio
    async def test_content_is_list_with_invalid_items(self):
        """content 是 list 但不是 dict 元素 → KeyQAToolError（schema 校验）。"""
        bad_items = ["not", "dict", "items"]
        gw = MockGateway(response_content=bad_items)
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())

        # 软降级
        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True
        assert any("schema_validation_failed" in r for r in patch["degradation_reasons"])
        # audit 落
        assert len(audit.inserts) == 1
        assert audit.inserts[0]["event_type"] == "keyqa_schema_validation_failed"

    @pytest.mark.asyncio
    async def test_q_field_empty(self):
        gw = MockGateway(response_content=[{"q": "", "a": "x"}])
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True

    @pytest.mark.asyncio
    async def test_a_field_empty(self):
        gw = MockGateway(response_content=[{"q": "心影", "a": "  "}])
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True

    @pytest.mark.asyncio
    async def test_q_field_not_string(self):
        gw = MockGateway(response_content=[{"q": 123, "a": "x"}])
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True

    @pytest.mark.asyncio
    async def test_a_field_truncated_to_512(self):
        long_a = "x" * 1000
        answers = [{"q": "心影", "a": long_a}, {"q": "纵隔", "a": "正常"},
                   {"q": "胸膜", "a": "正常"}, {"q": "肺野", "a": "正常"}]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"][0]["a"]) == 512

    @pytest.mark.asyncio
    async def test_q_field_truncated_to_64(self):
        long_q = "心影" * 100
        answers = [{"q": long_q, "a": "正常"}, {"q": "纵隔", "a": "正常"},
                   {"q": "胸膜", "a": "正常"}, {"q": "肺野", "a": "正常"}]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["key_qa_list"][0]["q"] == long_q[:64]

    @pytest.mark.asyncio
    async def test_too_few_items_rejected(self):
        answers = [{"q": "心影", "a": "正常"}]  # 只有 1 项，>=1 但 yield bad
        # 实际 MIN_KEY_QA_ITEMS = 1，所以 1 项会通过；改成 0 项
        gw = MockGateway(response_content=[])
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        # 0 项 → soft-degrade
        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True

    @pytest.mark.asyncio
    async def test_too_many_items_truncated(self):
        answers = [
            {"q": f"模板{i}", "a": f"答案{i}"} for i in range(20)
        ]
        gw = MockGateway(response_content=answers)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert len(patch["key_qa_list"]) == 8  # MAX_KEY_QA_ITEMS


# ---------------------------------------------------------------------------
# 模型不可达 → soft-degrade
# ---------------------------------------------------------------------------


class TestModelUnreachable:
    @pytest.mark.asyncio
    async def test_unreachable_raises_soft_degrade(self):
        gw = MockGateway(raise_exc=ModelUnreachableError("connection refused"))
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())

        assert patch["key_qa_list"] == []
        assert patch["degraded"] is True
        assert any("model_unreachable" in r for r in patch["degradation_reasons"])
        assert any("connection refused" in r for r in patch["degradation_reasons"])

        # audit 落
        assert len(audit.inserts) == 1
        ins = audit.inserts[0]
        assert ins["actor_role"] == "system"
        assert ins["event_type"] == "keyqa_model_unreachable"
        assert ins["payload"]["reason"] == "connection refused"

    @pytest.mark.asyncio
    async def test_unexpected_exception_caught(self):
        gw = MockGateway(raise_exc=RuntimeError("boom"))
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True
        assert any("RuntimeError" in r for r in patch["degradation_reasons"])

    @pytest.mark.asyncio
    async def test_timeout_soft_degrade(self):
        """gateway 阻塞超过 timeout → soft-degrade + 写 timeout 原因。"""
        # 让 gateway 阻塞 0.3s
        gw = MockGateway(delay_seconds=0.3)
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit, timeout_seconds=0.05)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True
        assert any("timeout" in r for r in patch["degradation_reasons"])
        # audit 落
        assert len(audit.inserts) == 1
        assert audit.inserts[0]["event_type"] == "keyqa_model_unreachable"

    @pytest.mark.asyncio
    async def test_no_audit_logger_silent_degrade(self):
        """audit_logger 缺省时 → soft-degrade 但不抛。"""
        gw = MockGateway(raise_exc=ModelUnreachableError("conn refused"))
        tool = CXRKeyQA(model_gateway=gw)  # 无 audit
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True
        assert patch["key_qa_list"] == []

    @pytest.mark.asyncio
    async def test_audit_logger_exception_not_blocking(self):
        gw = MockGateway(raise_exc=ModelUnreachableError("x"))
        audit = MockAuditLogger(raise_exc=RuntimeError("audit broken"))
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        # 不应抛，audit 异常静默吞
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True


# ---------------------------------------------------------------------------
# 响应格式兼容
# ---------------------------------------------------------------------------


class TestResponseFormatCompat:
    @pytest.mark.asyncio
    async def test_content_already_list(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_content_string_json_array(self):
        gw = MockGateway(response_content=json.dumps(DEFAULT_4_ANSWERS, ensure_ascii=False))
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_content_string_with_prose_around_array(self):
        prose = (
            "下面是 JSON：\n```json\n"
            + json.dumps(DEFAULT_4_ANSWERS, ensure_ascii=False)
            + "\n```\n以上。"
        )
        gw = MockGateway(response_content=prose)
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_content_choices_openai_style(self):
        class ChoicesGw:
            async def chat_completion(self, prompt, **kw):
                return {
                    "choices": [
                        {"message": {"content": json.dumps(DEFAULT_4_ANSWERS, ensure_ascii=False)}}
                    ]
                }
        tool = CXRKeyQA(model_gateway=ChoicesGw())
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_content_parsed_openai_structured(self):
        class ParsedGw:
            async def chat_completion(self, prompt, **kw):
                return {"parsed": DEFAULT_4_ANSWERS}
        tool = CXRKeyQA(model_gateway=ParsedGw())
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_content_dict_with_questions_key(self):
        gw = MockGateway(response_content={"questions": DEFAULT_4_ANSWERS})
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_string_object_with_questions_key(self):
        gw = MockGateway(
            response_content=json.dumps({"questions": DEFAULT_4_ANSWERS}, ensure_ascii=False)
        )
        tool = CXRKeyQA(model_gateway=gw)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is False
        assert len(patch["key_qa_list"]) == 4

    @pytest.mark.asyncio
    async def test_string_with_garbage_returns_soft_degrade(self):
        gw = MockGateway(response_content="not json, no brackets")
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        assert patch["degraded"] is True
        assert audit.inserts[0]["event_type"] == "keyqa_schema_validation_failed"


# ---------------------------------------------------------------------------
# 审计写入契约
# ---------------------------------------------------------------------------


class TestAuditIntegration:
    @pytest.mark.asyncio
    async def test_audit_event_fields_complete(self):
        gw = MockGateway(raise_exc=ModelUnreachableError("oops"))
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        await tool.run(_new_state(study_uid="<EXAMPLE_UID>"))

        assert len(audit.inserts) == 1
        ins = audit.inserts[0]
        assert ins["actor_role"] == "system"
        assert isinstance(ins["ts"], str) and ins["ts"].endswith("Z")
        assert ins["study_uid"] == "<EXAMPLE_UID>"
        assert ins["event_type"] == "keyqa_model_unreachable"
        assert isinstance(ins["payload"], dict)
        assert "elapsed_ms" in ins["payload"]
        assert ins["payload"]["reason"] == "oops"

    @pytest.mark.asyncio
    async def test_audit_payload_contains_templates_count(self):
        gw = MockGateway(raise_exc=ModelUnreachableError("x"))
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        await tool.run(_new_state())
        assert audit.inserts[0]["payload"]["templates_count"] == 4

    @pytest.mark.asyncio
    async def test_audit_meta_for_schema_failure(self):
        gw = MockGateway(response_content=[{"q": "心影", "a": "正常"},
                                          {"q": "纵隔", "a": None}])  # a=None
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        await tool.run(_new_state())
        assert len(audit.inserts) == 1
        assert audit.inserts[0]["event_type"] == "keyqa_schema_validation_failed"
        # raw_response_keys 应被写入 payload（便于飞检还原「那天拿到了什么形态」
        assert "raw_response_keys" in audit.inserts[0]["payload"]

    @pytest.mark.asyncio
    async def test_no_audit_on_happy_path(self):
        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)
        audit = MockAuditLogger()
        tool = CXRKeyQA(model_gateway=gw, audit_logger=audit)
        patch = await tool.run(_new_state())
        # 成功路径不写审计
        assert len(audit.inserts) == 0
        assert patch["degraded"] is False


# ---------------------------------------------------------------------------
# 与 StateGraph 集成
# ---------------------------------------------------------------------------


class TestStateGraphIntegration:
    @pytest.mark.asyncio
    async def test_tool_registry_register_and_run(self):
        from cxr_triage.engine.workflow import build_workflow

        gw = MockGateway(response_content=DEFAULT_4_ANSWERS)

        # mock classifier 工具（强制 max_prob=0.85 让阳性路径生效）
        class MockClassifier:
            name = "classifier"
            async def run(self, state):
                return {
                    "classifier_probs": [
                        {"name": "Pneumothorax", "prob": 0.85, "present": True}
                    ],
                    "classifier_max_prob": 0.85,
                }

        # 其他 3 件工具的最小占位（不让 execute_node 走 "tool_not_registered"）
        class MockFindingDraft:
            name = "finding_draft"
            async def run(self, state):
                return {"finding_text": "右下肺野见少量气胸征象（草稿 / 未经医师确认）。"}

        class MockPositiveGrounding:
            name = "positive_grounding"
            async def run(self, state):
                return {"grounding_boxes": [
                    {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1, "label": "气胸", "prob": 0.9}
                ]}

        class MockImagePreview:
            name = "image_preview"
            async def run(self, state):
                return {"png_preview_b64": "aGVsbG8="}

        registry = ToolRegistry()
        registry.register(MockClassifier())
        registry.register(CXRKeyQA(model_gateway=gw))
        registry.register(MockFindingDraft())
        registry.register(MockPositiveGrounding())
        registry.register(MockImagePreview())
        wf = build_workflow(registry, positive_threshold=0.5)

        result = await wf.run(
            study_uid="<EXAMPLE_WF>",
            png_bytes=_make_png_bytes(),
            metadata={"PatientSex": "F", "PatientAge": "052Y"},
        )

        assert result["classifier_max_prob"] == 0.85
        assert len(result["key_qa_list"]) == 4
        assert result["degraded"] is False
        assert "keyqa" in result["tools_executed"]
        assert "finding_draft" in result["tools_executed"]
        assert "positive_grounding" in result["tools_executed"]
        assert "image_preview" in result["tools_executed"]

    @pytest.mark.asyncio
    async def test_workflow_unreachable_keeps_state(self):
        """模型不可达时 workflow 不抛，只是 degraded=True。"""
        from cxr_triage.engine.workflow import build_workflow

        gw = MockGateway(raise_exc=ModelUnreachableError("boom"))

        class MockClassifier:
            name = "classifier"
            async def run(self, state):
                return {
                    "classifier_probs": [
                        {"name": "Pneumothorax", "prob": 0.85, "present": True}
                    ],
                    "classifier_max_prob": 0.85,
                }

        class MockFindingDraft:
            name = "finding_draft"
            async def run(self, state):
                return {"finding_text": "草稿"}

        class MockPositiveGrounding:
            name = "positive_grounding"
            async def run(self, state):
                return {"grounding_boxes": []}

        class MockImagePreview:
            name = "image_preview"
            async def run(self, state):
                return {"png_preview_b64": "aGVsbG8="}

        registry = ToolRegistry()
        registry.register(MockClassifier())
        registry.register(CXRKeyQA(model_gateway=gw))
        registry.register(MockFindingDraft())
        registry.register(MockPositiveGrounding())
        registry.register(MockImagePreview())
        wf = build_workflow(registry, positive_threshold=0.5)

        result = await wf.run(
            study_uid="<EXAMPLE_WF>",
            png_bytes=_make_png_bytes(),
            metadata={},
        )
        # keyqa soft-degrade：返空 list
        assert result["key_qa_list"] == []
        # keyqa 自身 soft-degrade（写入 degraded=True），并非 execute_node 异常
        # 路径。execute_node 不会捕获（keyqa 自吞），所以 degradation_reasons
        # 只含 keyqa 自己写的 keyqa_model_unreachable 原因
        assert result["degraded"] is True
        assert any(
            "keyqa_model_unreachable" in r
            for r in result["degradation_reasons"]
        )


# ---------------------------------------------------------------------------
# to_finding_struct 静态工具
# ---------------------------------------------------------------------------


class TestToFindingStruct:
    def test_passthrough(self):
        items = [{"q": "心影", "a": "正常"}, {"q": "纵隔", "a": "居中"}]
        out = CXRKeyQA.to_finding_struct(items)
        assert out == items

    def test_non_list_returns_empty(self):
        assert CXRKeyQA.to_finding_struct(None) == []
        assert CXRKeyQA.to_finding_struct("foo") == []
        assert CXRKeyQA.to_finding_struct(123) == []

    def test_filters_non_dict_items(self):
        items = [{"q": "心影", "a": "正常"}, "bad", 123, None,
                 {"q": "纵隔", "a": "居中"}]
        out = CXRKeyQA.to_finding_struct(items)
        assert out == [{"q": "心影", "a": "正常"}, {"q": "纵隔", "a": "居中"}]

    def test_stringifies_missing_fields(self):
        items = [{"q": "心影"}, {"a": "居中"}, {}]
        out = CXRKeyQA.to_finding_struct(items)
        assert out == [
            {"q": "心影", "a": ""},
            {"q": "", "a": "居中"},
            {"q": "", "a": ""},
        ]


# ---------------------------------------------------------------------------
# Gateway 缺省时
# ---------------------------------------------------------------------------


class TestGatewayNone:
    @pytest.mark.asyncio
    async def test_no_gateway_raises(self):
        tool = CXRKeyQA(model_gateway=None)
        with pytest.raises(KeyQAToolError, match="model_gateway"):
            await tool.run(_new_state())
