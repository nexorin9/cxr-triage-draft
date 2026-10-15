"""配置加载 — 所有凭据从环境变量读取；缺失关键项时 fail-fast。

设计原则（spec.md §5 维度 4 正式对接层 + CLAUDE.md 安全边界）：

1. **不硬编码任何凭据**：PACS / RIS / 体检主检 / ModelGateway 的 host/port/token/
   api_key 一律从环境变量读，由 pydantic-settings BaseSettings 校验。
2. **fail-fast**：关键外发凭据缺失时，启动即抛 ValidationError（不让服务
   半瘫地跑起来再发现写不出去）。
3. **可读即默认**：未填时给空字符串或本地占位（如 watch folder），但外发
   必填项不允许为空。
4. **统一入口**：业务层只 import ``CxrTriageConfig``，不再直接 os.getenv。

环境变量清单（与 ``.env.example`` 对齐）：

- ``PACS_AE_TITLE``            本地作为 SCU 监听的 AE 名
- ``PACS_HOST`` / ``PACS_PORT``  PACS 服务端 host:port
- ``PACS_WADO_URL``            WADO-RS 拉图端点
- ``RIS_API_URL``              RIS HTTP base URL
- ``RIS_API_TOKEN``            RIS Bearer Token
- ``HEALTHCHECK_API_URL``      体检主检 HTTP base URL
- ``HEALTHCHECK_API_TOKEN``    体检主检 Bearer Token
- ``OPENAI_BASE_URL``          OpenAI-compatible 模型网关端点
- ``OPENAI_API_KEY``           模型网关 API Key
- ``OPENAI_MODEL``             模型名（默认 medrax-llava-med-7b）
- ``POSITIVE_THRESHOLD``       阳性分诊阈值（默认 0.5）
- ``WATCH_FOLDER``             DICOM watch folder 兜底通道路径
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CxrTriageConfig(BaseSettings):
    """11 项凭据 + 业务阈值，统一从环境变量加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ====== PACS（DICOM C-STORE SCU 监听 + WADO-RS 拉图）======
    pacs_ae_title: str = Field(
        default="EXAMPLE_PACS_SCU",
        description="本地作为 SCU 监听的 Application Entity 名",
    )
    pacs_host: str = Field(default="127.0.0.1", description="PACS 服务端 host")
    pacs_port: int = Field(default=11112, description="PACS 服务端 port（DICOM 默认 104）")
    pacs_wado_url: str = Field(
        default="http://127.0.0.1:8080/wado-rs",
        description="WADO-RS 拉图端点",
    )

    # ====== RIS finding-draft 写回 ======
    ris_api_url: str = Field(
        default="http://127.0.0.1:8081",
        description="RIS HTTP base URL",
    )
    ris_api_token: str = Field(
        default="",
        description="RIS Bearer Token（必填，启动时校验）",
    )

    # ====== 体检主检 summary-ref 写回 ======
    healthcheck_api_url: str = Field(
        default="http://127.0.0.1:8082",
        description="体检主检 HTTP base URL",
    )
    healthcheck_api_token: str = Field(
        default="",
        description="体检主检 Bearer Token（必填，启动时校验）",
    )

    # ====== ModelGateway（OpenAI-compatible）======
    openai_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="OpenAI-compatible 端点 base URL",
    )
    openai_api_key: str = Field(
        default="",
        description="模型网关 API Key（必填，启动时校验）",
    )
    openai_model: str = Field(
        default="medrax-llava-med-7b",
        description="模型名（默认 medrax-llava-med-7b）",
    )

    # ====== 业务阈值与路径 ======
    positive_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="阳性分诊阈值：classifier max_prob ≥ 阈值 → positive_flag=true",
    )
    watch_folder: Path = Field(
        default=Path("/tmp/cxr-watch"),
        description="PACS C-STORE AE 协商失败时的 watch folder 兜底通道",
    )

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @field_validator("ris_api_token", "healthcheck_api_token", "openai_api_key")
    @classmethod
    def _require_credential(cls, v: str, info) -> str:
        """外发凭据位不允许为空。

        业务侧一旦启动就必须能真实外发；空字符串或占位字符串 ``"replace-me"``
        一律拒绝。开发环境若希望先跑本地 stub，请显式设 ``CXR_TRIAGE_ALLOW_PLACEHOLDER=1``。
        """
        import os

        allow_placeholder = os.getenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "0") == "1"
        stripped = (v or "").strip()
        if not stripped:
            raise ValueError(
                f"{info.field_name} 为必填外发凭据；从环境变量注入；"
                f"调试期可用 CXR_TRIAGE_ALLOW_PLACEHOLDER=1 跳过校验。"
            )
        if not allow_placeholder and stripped.lower().startswith("replace-me"):
            raise ValueError(
                f"{info.field_name} 仍为占位字符串（replace-me-*）；"
                f"请填入真实凭据或设 CXR_TRIAGE_ALLOW_PLACEHOLDER=1。"
            )
        return stripped

    @field_validator("pacs_port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"pacs_port 越界：{v} 不在 1–65535")
        return v

    # ------------------------------------------------------------------
    # 派生配置（不算凭据，只为业务层提供便捷入口）
    # ------------------------------------------------------------------
    @property
    def pacs_endpoint(self) -> str:
        """PACS host:port 字符串（不带协议）。"""
        return f"{self.pacs_host}:{self.pacs_port}"

    @property
    def ris_finding_draft_url(self) -> str:
        """RIS finding-draft 写回完整 URL。"""
        return f"{self.ris_api_url.rstrip('/')}/api/v1/studies/{{study_uid}}/finding-draft"

    @property
    def healthcheck_summary_ref_url(self) -> str:
        """体检主检 summary-ref 写回完整 URL 模板。"""
        return (
            f"{self.healthcheck_api_url.rstrip('/')}"
            f"/api/v1/healthcheck/{{visit_id}}/summary-ref"
        )

    @property
    def model_chat_completions_url(self) -> str:
        """OpenAI-compatible chat completions 端点。"""
        return f"{self.openai_base_url.rstrip('/')}/chat/completions"


def load_config(env_file: Optional[str | Path] = None) -> CxrTriageConfig:
    """统一入口：业务层只调它，不要直接 ``os.getenv``。

    Parameters
    ----------
    env_file:
        可选显式指定 .env 路径；默认走 pydantic-settings 的 ``.env`` 约定。
    """
    if env_file is not None:
        return CxrTriageConfig(_env_file=str(env_file))
    return CxrTriageConfig()


__all__ = ["CxrTriageConfig", "load_config"]
