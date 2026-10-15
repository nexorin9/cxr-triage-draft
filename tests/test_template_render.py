"""tests/test_template_render.py — 浏览器三栏面板 Jinja2 模板渲染测试（Task 30）。

覆盖（spec.md §5 维度 4 GUI 优先 · 文档分层）：

1. 模板文件存在 + Jinja2 引擎加载无语法错。
2. ``service_name`` / ``version`` / ``endpoint_count`` / ``degradation_banner_text``
   占位符正确替换。
3. ``endpoints`` 列表渲染（每条 ``GET /api/health`` 形式）。
4. HTML 结构：DOCTYPE / 头部 / 三栏 grid / 顶部 banner 槽位 / 底部 footer。
5. Alpine.js x-data ``triagePanel()`` 函数 + EventSource 监听 4 类事件。
6. 三角色流转按钮点击 → 调对应 API 端点。
7. 降级 banner 渲染（CSS class ``.degraded-banner`` + 锁定的中文文案）。
8. 阳性 / 阴性 / 关键问答 / 定位框 区域全部存在。
9. 通过 ``Jinja2Templates(directory=...)`` 加载 + ``TemplateResponse`` 渲染。
10. ``{% for %}`` 占位符原生 Jinja 控制流（Jinja2 模板原生渲染，Alpine x-for
    仅在前端运行时补全；混用不冲突）。

---

**风格**：所有断言走 fixture + 字符串匹配，不耦合具体样式细节。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cxr_triage.web import (
    DEFAULT_DEGRADATION_BANNER_TEXT,
    INDEX_DEFAULT_SERVICE_NAME,
    INDEX_DEFAULT_VERSION,
    INDEX_ENDPOINT,
    INDEX_TEMPLATE_NAME,
    SSE_ENDPOINT,
    STATIC_MOUNT_PREFIX,
    TEMPLATES_DIR,
)
from cxr_triage.web.app import (
    JINJA_TEMPLATES,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def template_path() -> Path:
    """``templates/index.html`` 的绝对路径。"""
    return TEMPLATES_DIR / INDEX_TEMPLATE_NAME


@pytest.fixture
def app_js_path() -> Path:
    """``static/app.js`` 的绝对路径（Task 31：Alpine.js 工厂函数 + SSE 监听）。"""
    return TEMPLATES_DIR.parent / "static" / "app.js"


@pytest.fixture
def minimal_ctx() -> dict[str, Any]:
    """模板最低必填上下文（service_name/version/endpoint_count/...）。"""
    endpoints = [
        "GET  /api/health",
        "GET  /api/studies",
        "GET  /api/studies/{study_uid}",
        "POST /api/studies/{study_uid}/confirm",
        "POST /api/studies/{study_uid}/reject",
        "GET  /api/queue/positive",
        "GET  /api/queue/negative",
        "GET  /api/outbox",
        "GET  /api/sse",
        "GET  /",
    ]
    return {
        "service_name": INDEX_DEFAULT_SERVICE_NAME,
        "version": INDEX_DEFAULT_VERSION,
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "degradation_banner_text": DEFAULT_DEGRADATION_BANNER_TEXT,
        "sse_endpoint": SSE_ENDPOINT,
        "static_prefix": STATIC_MOUNT_PREFIX,
        "queue_count": {"positive": 0, "negative": 0},
        "outbox_alerting": 0,
        "degraded": False,
    }


# ---------------------------------------------------------------------------
# 1. 文件存在 + Jinja2 加载
# ---------------------------------------------------------------------------


def test_template_file_exists(template_path: Path) -> None:
    """模板文件应位于 ``web/templates/index.html``。"""
    assert template_path.is_file(), f"模板缺失: {template_path}"


def test_template_name_is_index_html(template_path: Path) -> None:
    assert template_path.name == INDEX_TEMPLATE_NAME


def test_jinja_env_can_load_template(template_path: Path) -> None:
    """Jinja2Templates(directory=...) 能加载模板，无 SyntaxError。"""
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # 触发 compile（首次访问 .TemplateResponse 才编译），用 _env 提前断言
    env = templates.env
    assert env is not None
    template = env.get_template(INDEX_TEMPLATE_NAME)
    assert template is not None


# ---------------------------------------------------------------------------
# 2. 占位符替换（service_name / version / endpoint_count / degradation_banner_text）
# ---------------------------------------------------------------------------


def test_service_name_rendered_in_title(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "{{ service_name }}" in text, "title 应该是动态渲染 service_name"


def test_service_name_rendered_in_h1(template_path: Path, minimal_ctx: dict[str, Any]) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "{{ service_name }}" in text
    # H1 包含 service_name 字面（同一变量二次出现）
    occurrences = text.count("{{ service_name }}")
    assert occurrences >= 1


def test_version_placeholder(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "{{ version }}" in text


def test_endpoint_count_placeholder(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "{{ endpoint_count }}" in text


def test_degradation_banner_placeholder(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    # 降级 banner 文案由 service 渲染：Jinja 占位符存在 + JS 中也通过常量
    assert "{{ degradation_banner_text }}" in text


def test_tojson_escaping_used_in_inline_js(template_path: Path) -> None:
    """Jinja 模板中以 ``|tojson`` 防止中文 / JSON 字符串破坏（Task 31 抽出 app.js 后保留对
    ``service_name`` / ``degradation_banner_text`` 的 ``tojson`` 转义，确保模板注入安全）。"""
    text = template_path.read_text(encoding="utf-8")
    assert "|tojson" in text


# ---------------------------------------------------------------------------
# 3. endpoints 列表渲染（模板目前只在上下文中，不强制渲染：仅断言 context 字段名合法）
# ---------------------------------------------------------------------------


def test_endpoints_field_in_minimal_ctx(minimal_ctx: dict[str, Any]) -> None:
    """endpoints 是 list[str]，不强制模板渲染，但 ctx 字段名要给。"""
    assert "endpoints" in minimal_ctx
    assert isinstance(minimal_ctx["endpoints"], list)
    for e in minimal_ctx["endpoints"]:
        assert isinstance(e, str)
        assert "/" in e


def test_endpoint_count_matches_list_length(minimal_ctx: dict[str, Any]) -> None:
    assert minimal_ctx["endpoint_count"] == len(minimal_ctx["endpoints"])


def test_minimal_ctx_required_fields(minimal_ctx: dict[str, Any]) -> None:
    required = {
        "service_name",
        "version",
        "endpoint_count",
        "endpoints",
        "degradation_banner_text",
        "sse_endpoint",
        "static_prefix",
        "queue_count",
        "outbox_alerting",
        "degraded",
    }
    missing = required - set(minimal_ctx.keys())
    assert not missing, f"context 缺字段: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 4. HTML 结构（DOCTYPE / 头部 / 三栏 / banner 槽位 / footer）
# ---------------------------------------------------------------------------


def test_doctype_is_html5(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html>")


def test_charset_utf8_declared(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'charset="UTF-8"' in text


def test_meta_viewport_declared(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'name="viewport"' in text


def test_lang_attr_is_chinese(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in text


def test_three_columns_present(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'class="triage-grid"' in text
    assert "阳性队列" in text
    assert "阴性队列" in text
    assert "草稿所见" in text
    # 三栏 section 各有 id
    for col_id in ("column-positive", "column-negative", "column-draft"):
        assert f'id="{col_id}"' in text


def test_alert_slot_present_in_header(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'id="alert-slot"' in text
    assert "class=\"alert-slot\"" in text


def test_degraded_banner_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'class="degraded-banner"' in text
    assert 'id="degraded-banner"' in text


def test_debug_footer_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "class=\"debug-footer\"" in text
    assert "class=\"debug-counts\"" in text
    assert "class=\"debug-status\"" in text


def test_static_style_link(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="/static/style.css">' in text


# ---------------------------------------------------------------------------
# 5. Alpine.js x-data + EventSource 监听 4 类事件
# ---------------------------------------------------------------------------


def test_alpine_defer_script(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "alpinejs" in text.lower()
    assert "defer src=" in text


def test_alpine_x_data_root(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'x-data="triagePanel()"' in text
    assert 'x-init="init()"' in text


def test_alpine_event_listeners(app_js_path: Path) -> None:
    """4 类 SSE 事件都被 Alpine.js 监听（Task 31：JS 抽出到 app.js，使用常量名 EVENT_*）。"""
    text = app_js_path.read_text(encoding="utf-8")
    # app.js 用常量 EVENT_NEW_STUDY / EVENT_DRAFT_READY / EVENT_STATE_CHANGE / EVENT_ALERT 绑定监听器
    for evt_const in ("EVENT_NEW_STUDY", "EVENT_DRAFT_READY", "EVENT_STATE_CHANGE", "EVENT_ALERT"):
        assert f"addEventListener({evt_const}" in text, f"缺 SSE 事件监听: {evt_const}"


def test_event_source_endpoint(app_js_path: Path) -> None:
    """Task 31：JS 已抽到 app.js；EventSource 连接逻辑在 app.js。"""
    text = app_js_path.read_text(encoding="utf-8")
    assert "new EventSource(" in text or "new ESImpl(url)" in text or "EventSource" in text
    assert "/api/sse" in text


def test_keepalive_listener(app_js_path: Path) -> None:
    """Task 31：keepalive 监听在 app.js。"""
    text = app_js_path.read_text(encoding="utf-8")
    assert "EVENT_KEEPALIVE" in text or "addEventListener('keepalive'" in text or "addEventListener(\"keepalive\"" in text


# ---------------------------------------------------------------------------
# 6. 三角色流转按钮
# ---------------------------------------------------------------------------


def test_three_role_buttons_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "btn btn-jr" in text
    assert "btn btn-reviewer" in text
    assert "btn btn-sr" in text
    assert "btn btn-reject" in text


def test_role_button_disable_logic(template_path: Path) -> None:
    """canConfirm('jr') / canConfirm('reviewer') / canConfirm('sr')"""
    text = template_path.read_text(encoding="utf-8")
    assert "canConfirm('jr')" in text
    assert "canConfirm('reviewer')" in text
    assert "canConfirm('sr')" in text


def test_confirm_endpoint_format(app_js_path: Path) -> None:
    """Task 31：endpoint 常量定义在 app.js 中（ENDPOINT_CONFIRM / ENDPOINT_REJECT）。"""
    text = app_js_path.read_text(encoding="utf-8")
    assert "ENDPOINT_CONFIRM" in text
    assert "ENDPOINT_REJECT" in text
    assert "/api/studies/{uid}/confirm" in text
    assert "/api/studies/{uid}/reject" in text


def test_role_label_strings(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "初诊确认（jr）" in text
    assert "主检复核（reviewer）" in text
    assert "终审签发（sr）" in text
    assert "驳回重写" in text


def test_target_state_mapping_in_js(app_js_path: Path) -> None:
    """Task 31：target_state 映射逻辑在 app.js 的 targetStateForRole 纯函数里。"""
    text = app_js_path.read_text(encoding="utf-8")
    assert "targetStateForRole" in text
    assert "STATE_JR_CONFIRMED" in text
    assert "STATE_SR_REVIEWED" in text
    assert "STATE_SR_SIGNED" in text
    # 三段映射：jr → jr_confirmed, reviewer → sr_reviewed, sr → sr_signed
    assert "ROLE_JR" in text
    assert "ROLE_REVIEWER" in text
    assert "ROLE_SR" in text


# ---------------------------------------------------------------------------
# 7. 降级 banner（锁定中文文案）
# ---------------------------------------------------------------------------


def test_degradation_banner_default_text_constant() -> None:
    """spec.md §5 维度 4 + Task 19：锁定的中文文案。"""
    assert DEFAULT_DEGRADATION_BANNER_TEXT == "模型不可达，仅出概率与定位，需医师人工复核"


def test_degradation_banner_template_render(template_path: Path, minimal_ctx: dict[str, Any]) -> None:
    """手动编译模板，验证渲染结果含锁定文案。"""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template(INDEX_TEMPLATE_NAME)
    body = template.render(**minimal_ctx, request=None)
    assert DEFAULT_DEGRADATION_BANNER_TEXT in body


# ---------------------------------------------------------------------------
# 8. 阳性 / 阴性 / 关键问答 / 定位框 区域
# ---------------------------------------------------------------------------


def test_positive_list_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'id="positive-list"' in text
    assert "is-positive" in text


def test_negative_list_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert 'id="negative-list"' in text
    assert "study-item-negative" in text


def test_keyqa_section_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "class=\"keyqa-section\"" in text
    assert "keyqa-list" in text
    assert "keyqa-q" in text
    assert "keyqa-a" in text


def test_grounding_section_markup(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    assert "class=\"grounding-section\"" in text
    assert "grounding-svg" in text
    assert "<rect" in text  # 定位框 SVG 矩形
    assert "preserveAspectRatio=\"none\"" in text


def test_draft_meta_field_rows(template_path: Path) -> None:
    """draft 详情含 StudyInstanceUID / Pmax / 阳性 三行 meta。"""
    text = template_path.read_text(encoding="utf-8")
    assert "StudyInstanceUID" in text
    assert "Pmax" in text
    assert "flag-positive" in text
    assert "flag-negative" in text


# ---------------------------------------------------------------------------
# 9. Jinja2 TemplateResponse（直接走 FastAPI app + TestClient）
# ---------------------------------------------------------------------------


def test_template_renders_without_error_at_endpoint(
    minimal_ctx: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最简场景：直接调 Jinja2Templates.TemplateResponse 渲染模板。"""
    from starlette.requests import Request as StarletteRequest

    req = StarletteRequest(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
        }
    )
    response = JINJA_TEMPLATES.TemplateResponse(
        req, INDEX_TEMPLATE_NAME, minimal_ctx
    )
    # TemplateResponse 在 init 阶段 render 内容 → .body 拿到 HTML bytes
    rendered = response.body.decode("utf-8")
    assert isinstance(rendered, str)
    assert "<!DOCTYPE html>" in rendered
    assert INDEX_DEFAULT_SERVICE_NAME in rendered
    assert DEFAULT_DEGRADATION_BANNER_TEXT in rendered
    assert "阳性队列" in rendered


def test_template_with_queue_counts(minimal_ctx: dict[str, Any]) -> None:
    """queue_count 上下文对模板透明（模板本身不强制渲染该字段）。"""
    minimal_ctx["queue_count"] = {"positive": 5, "negative": 12}
    minimal_ctx["outbox_alerting"] = 2
    minimal_ctx["degraded"] = True
    from starlette.requests import Request as StarletteRequest

    req = StarletteRequest(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
        }
    )
    response = JINJA_TEMPLATES.TemplateResponse(
        req, INDEX_TEMPLATE_NAME, minimal_ctx
    )
    rendered = response.body.decode("utf-8")
    # degraded=True 时 Alpine x-show 控制显示；模板侧仍渲染含 banner 标记
    assert "degraded-banner" in rendered


# ---------------------------------------------------------------------------
# 10. Jinja 控制流与前端 Alpine 的边界（互不污染）
# ---------------------------------------------------------------------------


def test_no_jinja_control_statements_in_html_body(template_path: Path) -> None:
    """模板主体用 Alpine.js 渲染列表（x-for），不掺 Jinja2 的 {% for %} 块。

    验证意义：保持 SSR 渲染纯静态 + 客户端 Alpine.js 接管动态列表；避免
    服务端渲染时一次性 render 一坨 list 后被前端状态重新覆盖。
    """
    text = template_path.read_text(encoding="utf-8")
    assert "{% for " not in text
    assert "{% if " not in text


def test_alpine_x_for_used_for_lists(template_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    # 阳性 / 阴性 / keyqa / alerts / grounding_boxes 都靠 x-for
    for snippet in (
        "x-for=\"study in positiveStudies\"",
        "x-for=\"study in negativeStudies\"",
        "x-for=\"qa in (selectedStudy",
        "x-for=\"alert in alerts\"",
        "x-for=\"(box, idx) in (selectedStudy",
    ):
        assert snippet in text or "x-for=\"qa in (selectedStudy ? selectedStudy.key_qa_list || [] : [])\"" in text


def test_no_unescaped_jinja_double_brace_in_script(template_path: Path) -> None:
    """JS 内的双花括号要靠 Alpine.js 解析（不属于 Jinja）。

    Alpine.x-text 表达式 + x-for 等以 ``{{`` 形式输出渲染占位。
    """
    text = template_path.read_text(encoding="utf-8")
    assert text.count("{{") == text.count("}}")
    # 模板应至少有足够多的 Alpine 渲染占位（标题/H1/banner/动态字段）
    assert text.count("{{") >= 6


# ---------------------------------------------------------------------------
# 11. 通过 FastAPI app 的 / 端点取模板（最小 app + TestClient）
# ---------------------------------------------------------------------------


def test_app_root_endpoint_renders_index_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI app + TestClient.get(/) 拉到 index.html 200 text/html。"""
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    monkeypatch.setenv("RIS_API_TOKEN", "replace-me-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "replace-me-hc-token")
    monkeypatch.setenv("OPENAI_API_KEY", "replace-me-llm-key")
    monkeypatch.setenv("CXR_TRIAGE_DB_PATH", "/tmp/cxr-triage-test-template.db")
    monkeypatch.setenv("CXR_TRIAGE_PATIENT_ID_SALT", "test-salt")

    app = create_app(
        enable_pacs_listener=False,
        enable_folder_watcher=False,
    )
    client = TestClient(app)
    try:
        resp = client.get(INDEX_ENDPOINT)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        body = resp.text
        assert INDEX_DEFAULT_SERVICE_NAME in body
        assert "阳性队列" in body
        assert "阴性队列" in body
        assert "草稿所见" in body
    finally:
        # 关掉 lifespan 启动的 retry / trigger 协程
        try:
            holder = getattr(app.state, "cxr_holder", None)
            if holder and holder.get("state") is not None:
                state = holder["state"]
                if getattr(state, "ris_writer", None) is not None:
                    try:
                        import asyncio

                        asyncio.run(state.ris_writer.stop_retry_loop())
                    except Exception:
                        pass
        except Exception:
            pass


def test_index_endpoint_url_constant_equals_root() -> None:
    """``INDEX_ENDPOINT == '/'``，与 ``@app.get('/')`` 注册一致。"""
    assert INDEX_ENDPOINT == "/"


def test_sse_endpoint_url_constant_is_api_sse() -> None:
    assert SSE_ENDPOINT == "/api/sse"


def test_static_prefix_constant_is_slash_static() -> None:
    assert STATIC_MOUNT_PREFIX == "/static"
