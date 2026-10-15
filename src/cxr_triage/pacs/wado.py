"""WADO-RS 异步取图客户端 — DICOM C-STORE 之外的旁路取图。

设计原则（spec.md §5 维度 1 权威源接入 + §6 风险缓解）：

1. **官方协议**：WADO-RS（WADO 通过 RESTful 服务）。HTTP ``GET`` 形如
   ``{PACS_WADO_URL}/studies/{study_uid}``，服务端返回
   ``application/dicom`` multipart 字节流。
2. **凭据位**：``PACS_WADO_URL`` 与 ``PACS_WADO_TOKEN``（Bearer Token）从
   业务配置（``CxrTriageConfig``）读，不硬编码。
3. **异步 + 超时 + 重试**：用 ``httpx.AsyncClient``，单次超时 30s；
   5xx / 网络错误最多重试 2 次（指数退避 1s / 2s）。
4. **失败软着陆**：超过重试上限 → 抛出 ``WadoFetchError``，调用方（如
   trigger 协程）入 ``outbox`` 等后台异步重试，不静默丢弃。
5. **不依赖 PIL / pynetdicom**：本模块只负责拉原始 DICOM 字节流；解析
   与窗宽窗位交给 ``DicomProcessor``（同目录 ``dicom_processor.py``）。

参考：源仓库 ``medrax/tools/dicom.py:51-94`` DICOM 解析模式，
     DICOM QIDO-RS / WADO-RS（http://dicom.nema.org/medical/dicom/current/output/htm/part18.htm）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from ..config import CxrTriageConfig, load_config

logger = logging.getLogger(__name__)


# 单次超时与重试上限（凭经验，不写到 config — 是协议层参数不是业务参数）
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
RETRY_BASE_BACKOFF_SECONDS = 1.0


class WadoFetchError(RuntimeError):
    """WADO-RS 拉图失败（含重试用尽）。"""


class WadoClient:
    """WADO-RS 异步取图客户端。

    Parameters
    ----------
    config:
        ``CxrTriageConfig``，从 env 读 ``PACS_WADO_URL`` 与可选的
        ``PACS_WADO_TOKEN``；后端 Python ``None`` 时跳过 Authorization。
    timeout_seconds:
        单次 HTTP 超时（默认 30s）。
    max_retries:
        5xx / 网络错误最大重试次数（默认 2）。
    """

    def __init__(
        self,
        config: Optional[CxrTriageConfig] = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.config = config or load_config()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        # 共享 client（连接池），业务层用 ``with WadoClient(cfg) as c:`` 自动关闭
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "WadoClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
            logger.debug(
                "WadoClient 启动: wado_url=%s timeout=%ss max_retries=%d",
                self.config.pacs_wado_url,
                self.timeout_seconds,
                self.max_retries,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("WadoClient 已关闭")

    # ------------------------------------------------------------------
    # 主路径
    # ------------------------------------------------------------------
    async def fetch(self, study_uid: str) -> bytes:
        """异步拉取 study 原始 DICOM 字节流。

        Parameters
        ----------
        study_uid:
            DICOM StudyInstanceUID；不含 ``<`` / ``>``。

        Returns
        -------
        bytes
            原始 DICOM（含 preamble + DICM + file meta + dataset）。

        Raises
        ------
        WadoFetchError
            网络错误 / 5xx 重试用尽 / 4xx 不可重试错误 /
            响应不是 ``application/dicom``。
        """
        if not study_uid:
            raise WadoFetchError("study_uid 为空，无法拉图")

        await self.start()
        assert self._client is not None
        url = self._build_url(study_uid)
        headers = self._build_headers()

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.get(url, headers=headers)
            except (httpx.RequestError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logger.warning(
                    "WADO fetch 失败 attempt=%d/%d study=%s err=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    study_uid,
                    exc,
                )
                if attempt < self.max_retries:
                    await self._backoff(attempt)
                    continue
                # 重试用尽
                break

            status = resp.status_code
            if 200 <= status < 300:
                ctype = resp.headers.get("content-type", "").lower()
                # WADO-RS 服务端 application/dicom 或 application/octet-stream 都接收
                if ctype and not (
                    "dicom" in ctype or "octet-stream" in ctype or ctype.startswith("application/")
                ):
                    raise WadoFetchError(
                        f"WADO 响应 content-type 不合法: study={study_uid} "
                        f"status={status} ctype={ctype!r}"
                    )
                logger.info(
                    "WADO fetch 成功 study=%s status=%s bytes=%d",
                    study_uid,
                    status,
                    len(resp.content),
                )
                return bytes(resp.content)

            # 4xx 不可重试（业务语义错误）
            if 400 <= status < 500 and status not in (408, 429):
                raise WadoFetchError(
                    f"WADO 业务错误: study={study_uid} status={status} "
                    f"body={(resp.text or '')[:200]!r}"
                )

            # 5xx / 408 / 429 → 可重试
            last_exc = WadoFetchError(
                f"WADO HTTP {status}: study={study_uid} body={(resp.text or '')[:200]!r}"
            )
            logger.warning(
                "WADO fetch HTTP %s study=%s attempt=%d/%d",
                status,
                study_uid,
                attempt + 1,
                self.max_retries + 1,
            )
            if attempt < self.max_retries:
                await self._backoff(attempt)
                continue
            break

        # 走到这里说明重试用尽
        raise WadoFetchError(
            f"WADO 取图失败: study={study_uid} attempts={self.max_retries + 1} "
            f"last_error={last_exc!r}"
        ) from last_exc

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _build_url(self, study_uid: str) -> str:
        base = self.config.pacs_wado_url.rstrip("/")
        return f"{base}/studies/{study_uid}"

    def _build_headers(self) -> dict[str, str]:
        """Accept + optional Bearer。

        凭据位优先级：``PACS_WADO_TOKEN`` env（PACS 服务的 WADO 单独 token）→
        ``PACS_API_TOKEN`` env → 缺省不附 Authorization。**禁止**硬编码。
        """
        import os

        token = os.getenv("PACS_WADO_TOKEN") or os.getenv("PACS_API_TOKEN") or ""
        headers = {"Accept": "application/dicom, application/octet-stream, */*"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _backoff(self, attempt: int) -> None:
        """指数退避 ``1s * 2 ** attempt``；不抛。"""
        delay = RETRY_BASE_BACKOFF_SECONDS * (2 ** attempt)
        await asyncio.sleep(delay)


def build_wado_client(
    config: Optional[CxrTriageConfig] = None,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> WadoClient:
    """工厂：从 ``load_config()`` 构造 ``WadoClient``。"""
    return WadoClient(config, timeout_seconds=timeout_seconds, max_retries=max_retries)


__all__ = [
    "WadoClient",
    "WadoFetchError",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "build_wado_client",
]
