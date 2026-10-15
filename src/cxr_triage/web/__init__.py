"""web — FastAPI app + SSE + 浏览器三栏面板（Task 28-31 落点）。

子模块（Phase 4 已落 / 后续 task 落盘）：
- ``app``           FastAPI app + routers（/api/studies /confirm /reject /health
                    /sse /queue/positive|negative /outbox）
- ``sse``           SsePublisher 推送 4 类事件（new_study / draft_ready /
                    state_change / alert）—— Task 29 ✓
- ``templates``     Jinja2 三栏模板（阳性 / 阴性 / 草稿所见）—— Task 30 ✓
- ``static``        Alpine.js + CSS（草稿标识 / 阳性红标 / 降级 banner /
                    告警 banner）—— Task 27/31
"""
from __future__ import annotations

from pathlib import Path

# Web static 目录（banner CSS / JS / Alpine.js 模板等）
STATIC_DIR: Path = Path(__file__).parent / "static"

# Jinja2 templates 目录（浏览器三栏面板）
TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

# 默认模板名（Task 30：三栏面板模板）
INDEX_TEMPLATE_NAME: str = "index.html"

# 静态资源相对路径（FastAPI StaticFiles 挂载用）
STATIC_MOUNT_PREFIX: str = "/static"

# banner 资源（Task 27：告警 banner 红底白字）
ALERT_BANNER_CSS_FILENAME: str = "style.css"
ALERT_BANNER_JS_FILENAME: str = "alert_banner.js"

# SSE 路由端点（Task 29）
SSE_ENDPOINT: str = "/api/sse"

# 浏览器面板首页路径（Task 30）
INDEX_ENDPOINT: str = "/"

# 默认服务中文名（README 三秒说明一致）
INDEX_DEFAULT_SERVICE_NAME: str = "体检胸片所见草稿系统"

# 默认版本号（与 cxr_triage.__version__ 对齐）
INDEX_DEFAULT_VERSION: str = "0.1.0"

# 降级 banner 文案（Task 19/20 锁定，Task 30 渲染）
DEFAULT_DEGRADATION_BANNER_TEXT: str = (
    "模型不可达，仅出概率与定位，需医师人工复核"
)


__all__ = [
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "INDEX_TEMPLATE_NAME",
    "STATIC_MOUNT_PREFIX",
    "ALERT_BANNER_CSS_FILENAME",
    "ALERT_BANNER_JS_FILENAME",
    "SSE_ENDPOINT",
    "INDEX_ENDPOINT",
    "INDEX_DEFAULT_SERVICE_NAME",
    "INDEX_DEFAULT_VERSION",
    "DEFAULT_DEGRADATION_BANNER_TEXT",
]
