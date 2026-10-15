"""tests/test_wado.py — WADO-RS 异步取图客户端。

覆盖（CLAUDE.md Phase 4 验收 · Task 5）：

1. **happy path**：mock 200 + application/octet-stream → 字节流返回；
2. **Bearer Token** 注入（``PACS_WADO_TOKEN`` → Authorization header）；
3. **404 业务错误**（study 不存在）→ 抛 ``WadoFetchError`` 且**不重试**；
4. **5xx 重试**：503 → 重试 2 次后抛错；
5. **网络错误**（httpx.ConnectError）→ 重试 → 失败；
6. **content-type 校验失败**（text/html 返回）→ ``WadoFetchError``；
7. **空 study_uid** → ``WadoFetchError``；
8. **退避时间**：指数退避 ``1s * 2 ** attempt``；
9. **async context manager** ``async with WadoClient(cfg) as c`` 自动关闭；
10. **连接关闭幂等**：多次 close 不报错。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
import pytest_asyncio
import respx

from cxr_triage.config import CxrTriageConfig
from cxr_triage.pacs.wado import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    WadoClient,
    WadoFetchError,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def env(monkeypatch):
    """外发凭据；PACS_WADO_TOKEN 单独 env 注入 → Bearer。"""
    monkeypatch.setenv("PACS_AE_TITLE", "TEST_SCU")
    monkeypatch.setenv("PACS_HOST", "127.0.0.1")
    monkeypatch.setenv("PACS_PORT", "11112")
    monkeypatch.setenv("PACS_WADO_URL", "http://pacs.test:8080/wado-rs")
    monkeypatch.setenv("PACS_WADO_TOKEN", "real-wado-token-xyz")
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://hcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("OPENAI_MODEL", "medrax-llava-med-7b")
    return monkeypatch


@pytest.fixture
def cfg(env) -> CxrTriageConfig:
    return CxrTriageConfig()


@pytest.fixture
def sample_dicom_bytes() -> bytes:
    """最小可解析 DICOM 字节流（仅供 WADO 路径占位返回）。"""
    return b"DICM\x00\x01" + b"\x00" * 200


# 用稳定的 placeholder UIDs（不参与 URL 编码歧义）
STUDY_OK = "1.2.826.0.1.3680043.8.498.1.1"
STUDY_CTX = "1.2.826.0.1.3680043.8.498.1.2"
STUDY_MISSING = "1.2.826.0.1.3680043.8.498.1.404"
STUDY_DOWN = "1.2.826.0.1.3680043.8.498.1.503"
STUDY_TRANSIENT = "1.2.826.0.1.3680043.8.498.1.trn"
STUDY_UNREACHABLE = "1.2.826.0.1.3680043.8.498.1.net"
STUDY_HTML = "1.2.826.0.1.3680043.8.498.1.htm"
STUDY_BACKOFF = "1.2.826.0.1.3680043.8.498.1.bck"


# ======================================================================
# 1. happy path
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_happy_path(cfg, sample_dicom_bytes):
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(re.compile(r"/wado-rs/studies/")).respond(
            200,
            content=sample_dicom_bytes,
            headers={"content-type": "application/dicom"},
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            data = await c.fetch(STUDY_OK)
        assert data == sample_dicom_bytes
        assert route.called


@pytest.mark.asyncio
async def test_wado_fetch_accepts_octet_stream(cfg, sample_dicom_bytes):
    """部分 PACS 返回 ``application/octet-stream``，协议层容忍。"""
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        mock.get(re.compile(r"/wado-rs/studies/")).respond(
            200,
            content=sample_dicom_bytes,
            headers={"content-type": "application/octet-stream"},
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            data = await c.fetch(STUDY_OK)
        assert data == sample_dicom_bytes


# ======================================================================
# 2. Authorization 注入
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_injects_bearer_token(cfg, sample_dicom_bytes):
    """``PACS_WADO_TOKEN`` env → Authorization: Bearer ... 头。"""
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(re.compile(r"/wado-rs/studies/")).respond(
            200,
            content=sample_dicom_bytes,
            headers={"content-type": "application/dicom"},
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            await c.fetch(STUDY_OK)
        # respx 把请求都记录在 route.calls
        assert route.calls.call_count == 1
        request = route.calls.last.request
        assert request.headers.get("Authorization") == "Bearer real-wado-token-xyz"
        assert "application/dicom" in request.headers.get("Accept", "")


# ======================================================================
# 3. 4xx 业务错误（不重试）
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_404_does_not_retry(cfg):
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(re.compile(r"/wado-rs/studies/")).respond(
            404, json={"error": "study not found"}
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            with pytest.raises(WadoFetchError) as exc:
                await c.fetch(STUDY_MISSING)
        assert "404" in str(exc.value)
        # 404 不重试
        assert route.calls.call_count == 1


# ======================================================================
# 4. 5xx 重试 + 退避
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_retries_on_503(cfg, monkeypatch):
    """503 → 重试 ``max_retries=2`` 共 3 次尝试，全部 503 → 抛错。"""
    # 让退避几乎不耗时间，跑测试快
    monkeypatch.setattr(
        "cxr_triage.pacs.wado.RETRY_BASE_BACKOFF_SECONDS", 0.01
    )
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(re.compile(r"/wado-rs/studies/")).respond(
            503, json={"error": "down"}
        )
        async with WadoClient(
            cfg, timeout_seconds=5.0, max_retries=DEFAULT_MAX_RETRIES
        ) as c:
            with pytest.raises(WadoFetchError) as exc:
                await c.fetch(STUDY_DOWN)
    # 重试用尽共 ``max_retries + 1`` = 3 次
    assert route.calls.call_count == DEFAULT_MAX_RETRIES + 1
    assert "WADO 取图失败" in str(exc.value)


@pytest.mark.asyncio
async def test_wado_fetch_succeeds_after_retry(cfg, monkeypatch):
    """前 1 次 503 → 重试 → 第 2 次 200 → 成功。"""
    monkeypatch.setattr(
        "cxr_triage.pacs.wado.RETRY_BASE_BACKOFF_SECONDS", 0.01
    )
    responses = iter(
        [
            httpx.Response(503, json={"error": "transient"}),
            httpx.Response(
                200,
                content=b"DICM_OK",
                headers={"content-type": "application/dicom"},
            ),
        ]
    )
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(
            re.compile(r"/wado-rs/studies/")
        ).mock(side_effect=lambda req: next(responses))
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            data = await c.fetch(STUDY_TRANSIENT)
        assert data == b"DICM_OK"
        assert route.calls.call_count == 2


# ======================================================================
# 5. 网络错误（连接拒绝）
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_handles_connect_error(cfg, monkeypatch):
    """``httpx.ConnectError`` → 重试 → 重试用尽抛错。"""
    monkeypatch.setattr(
        "cxr_triage.pacs.wado.RETRY_BASE_BACKOFF_SECONDS", 0.01
    )
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        route = mock.get(re.compile(r"/wado-rs/studies/")).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            with pytest.raises(WadoFetchError):
                await c.fetch(STUDY_UNREACHABLE)
    assert route.calls.call_count == DEFAULT_MAX_RETRIES + 1


# ======================================================================
# 6. content-type 校验
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_rejects_html_response(cfg):
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        mock.get(re.compile(r"/wado-rs/studies/")).respond(
            200,
            text="<html>not dicom</html>",
            headers={"content-type": "text/html"},
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            with pytest.raises(WadoFetchError) as exc:
                await c.fetch(STUDY_HTML)
        assert "content-type" in str(exc.value)


# ======================================================================
# 7. 空 study_uid
# ======================================================================
@pytest.mark.asyncio
async def test_wado_fetch_rejects_empty_study_uid(cfg):
    async with WadoClient(cfg, timeout_seconds=5.0) as c:
        with pytest.raises(WadoFetchError) as exc:
            await c.fetch("")
        assert "study_uid 为空" in str(exc.value)


# ======================================================================
# 8. 退避时间（指数曲线）
# ======================================================================
@pytest.mark.asyncio
async def test_wado_backoff_exponential_curve(cfg, monkeypatch):
    """记录各次重试之间的间隔，验证指数退避曲线。"""
    timestamps: list[float] = []

    async def fake_sleep(d: float) -> None:
        timestamps.append(d)

    monkeypatch.setattr("cxr_triage.pacs.wado.RETRY_BASE_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr("cxr_triage.pacs.wado.asyncio.sleep", fake_sleep)
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        mock.get(re.compile(r"/wado-rs/studies/")).respond(503)
        async with WadoClient(
            cfg, timeout_seconds=5.0, max_retries=3
        ) as c:
            with pytest.raises(WadoFetchError):
                await c.fetch(STUDY_BACKOFF)
    # attempt 0 → 1s; attempt 1 → 2s; attempt 2 → 4s; attempt 3 → 不再 sleep
    assert timestamps == [1.0, 2.0, 4.0]


# ======================================================================
# 9. async context manager
# ======================================================================
@pytest.mark.asyncio
async def test_wado_client_context_manager(cfg, sample_dicom_bytes):
    with respx.mock(base_url="http://pacs.test:8080") as mock:
        mock.get(re.compile(r"/wado-rs/studies/")).respond(
            200, content=sample_dicom_bytes,
            headers={"content-type": "application/dicom"},
        )
        async with WadoClient(cfg, timeout_seconds=5.0) as c:
            assert c._client is not None
            data = await c.fetch(STUDY_CTX)
        # 退出 with → client 已关闭
        assert c._client is None
        assert data == sample_dicom_bytes


# ======================================================================
# 10. 关闭幂等
# ======================================================================
@pytest.mark.asyncio
async def test_wado_client_close_is_idempotent(cfg):
    c = WadoClient(cfg, timeout_seconds=5.0)
    await c.start()
    await c.close()
    await c.close()  # 二次关闭不应抛
    assert c._client is None
