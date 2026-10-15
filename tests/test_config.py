"""tests/test_config.py — 配置加载校验。

覆盖：
1. env 全配置 → 加载成功，字段值一致；
2. 缺外发凭据 → 抛 ValidationError（fail-fast）；
3. 占位字符串（replace-me-*） → 拒绝（除非显式 CXR_TRIAGE_ALLOW_PLACEHOLDER=1）；
4. POSITIVE_THRESHOLD 阈值越界 → 拒绝；
5. 派生属性（pacs_endpoint / ris_finding_draft_url / 等）拼装正确。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError


def _env(**kv):
    """临时把 kv 注入 os.environ；fixture 退出时还原。"""
    saved = {}
    for k, v in kv.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k, old in saved.items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def test_config_load_full_env(monkeypatch):
    """全 11 项凭据 + 阈值都在 → 加载成功 + 字段值一致。"""
    monkeypatch.setenv("PACS_AE_TITLE", "REAL_PACS_SCU")
    monkeypatch.setenv("PACS_HOST", "10.20.30.40")
    monkeypatch.setenv("PACS_PORT", "11112")
    monkeypatch.setenv("PACS_WADO_URL", "http://10.20.30.40/wado-rs")
    monkeypatch.setenv("RIS_API_URL", "http://ris.internal:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://hcheck.internal:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.internal:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("OPENAI_MODEL", "medrax-llava-med-7b")
    monkeypatch.setenv("POSITIVE_THRESHOLD", "0.65")
    monkeypatch.setenv("WATCH_FOLDER", "/var/spool/cxr-watch")

    from cxr_triage.config import CxrTriageConfig

    cfg = CxrTriageConfig()

    assert cfg.pacs_ae_title == "REAL_PACS_SCU"
    assert cfg.pacs_host == "10.20.30.40"
    assert cfg.pacs_port == 11112
    assert cfg.pacs_wado_url == "http://10.20.30.40/wado-rs"
    assert cfg.ris_api_url == "http://ris.internal:8081"
    assert cfg.ris_api_token == "real-ris-token-123"
    assert cfg.healthcheck_api_url == "http://hcheck.internal:8082"
    assert cfg.healthcheck_api_token == "real-hcheck-token-456"
    assert cfg.openai_base_url == "http://llm.internal:8000/v1"
    assert cfg.openai_api_key == "real-llm-key-789"
    assert cfg.openai_model == "medrax-llava-med-7b"
    assert cfg.positive_threshold == pytest.approx(0.65)
    assert cfg.watch_folder == Path("/var/spool/cxr-watch")

    # 派生属性
    assert cfg.pacs_endpoint == "10.20.30.40:11112"
    assert (
        cfg.ris_finding_draft_url
        == "http://ris.internal:8081/api/v1/studies/{study_uid}/finding-draft"
    )
    assert (
        cfg.healthcheck_summary_ref_url
        == "http://hcheck.internal:8082/api/v1/healthcheck/{visit_id}/summary-ref"
    )
    assert (
        cfg.model_chat_completions_url
        == "http://llm.internal:8000/v1/chat/completions"
    )


def test_config_missing_openai_api_key_fails(monkeypatch):
    """缺 OPENAI_API_KEY → fail-fast ValidationError。"""
    # 把所有可能的 OPENAI_API_KEY 来源清空，避免泄漏
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RIS_API_TOKEN", raising=False)
    monkeypatch.delenv("HEALTHCHECK_API_TOKEN", raising=False)
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    # 校验错误至少提到 OPENAI_API_KEY（或 openai_api_key）
    assert any(
        "openai_api_key" in str(err).lower() or "openai api key" in str(err).lower()
        for err in ei.value.errors()
    )


def test_config_missing_ris_token_fails(monkeypatch):
    """缺 RIS_API_TOKEN → fail-fast ValidationError。"""
    monkeypatch.delenv("RIS_API_TOKEN", raising=False)
    monkeypatch.delenv("HEALTHCHECK_API_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    assert any(
        "ris_api_token" in str(err).lower() or "ris api token" in str(err).lower()
        for err in ei.value.errors()
    )


def test_config_missing_healthcheck_token_fails(monkeypatch):
    """缺 HEALTHCHECK_API_TOKEN → fail-fast ValidationError。"""
    monkeypatch.delenv("RIS_API_TOKEN", raising=False)
    monkeypatch.delenv("HEALTHCHECK_API_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    assert any(
        "healthcheck_api_token" in str(err).lower()
        for err in ei.value.errors()
    )


def test_config_placeholder_token_rejected(monkeypatch):
    """占位字符串（replace-me-*）即使非空也拒绝。"""
    monkeypatch.setenv("RIS_API_TOKEN", "replace-me-with-real-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.delenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", raising=False)

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    assert any("ris_api_token" in str(err).lower() for err in ei.value.errors())


def test_config_placeholder_allowed_with_flag(monkeypatch):
    """CXR_TRIAGE_ALLOW_PLACEHOLDER=1 → 允许占位（开发跑 stub 用）。"""
    monkeypatch.setenv("RIS_API_TOKEN", "replace-me-with-real-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "replace-me-with-real-healthcheck-token")
    monkeypatch.setenv("OPENAI_API_KEY", "replace-me-with-real-api-key")
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")

    from cxr_triage.config import CxrTriageConfig

    cfg = CxrTriageConfig()
    assert cfg.ris_api_token.startswith("replace-me")


def test_config_positive_threshold_out_of_range(monkeypatch):
    """阈值越界 → ValidationError。"""
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("POSITIVE_THRESHOLD", "1.5")  # 越界

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    assert any("positive_threshold" in str(err).lower() for err in ei.value.errors())


def test_config_pacs_port_out_of_range(monkeypatch):
    """pacs_port 越界 → ValidationError。"""
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("PACS_PORT", "99999")

    from cxr_triage.config import CxrTriageConfig

    with pytest.raises(ValidationError) as ei:
        CxrTriageConfig()
    assert any("pacs_port" in str(err).lower() for err in ei.value.errors())


def test_config_defaults_match_env_example(monkeypatch):
    """未填任何环境变量 + CXR_TRIAGE_ALLOW_PLACEHOLDER=1 → 取默认值（与 .env.example 对齐）。"""
    monkeypatch.delenv("RIS_API_TOKEN", raising=False)
    monkeypatch.delenv("HEALTHCHECK_API_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PACS_AE_TITLE", raising=False)
    monkeypatch.delenv("PACS_HOST", raising=False)
    monkeypatch.delenv("PACS_PORT", raising=False)
    monkeypatch.delenv("PACS_WADO_URL", raising=False)
    monkeypatch.delenv("RIS_API_URL", raising=False)
    monkeypatch.delenv("HEALTHCHECK_API_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("POSITIVE_THRESHOLD", raising=False)
    monkeypatch.delenv("WATCH_FOLDER", raising=False)
    monkeypatch.setenv("RIS_API_TOKEN", "x")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    from cxr_triage.config import CxrTriageConfig

    cfg = CxrTriageConfig()
    assert cfg.pacs_ae_title == "EXAMPLE_PACS_SCU"
    assert cfg.pacs_host == "127.0.0.1"
    assert cfg.pacs_port == 11112
    assert cfg.openai_model == "medrax-llava-med-7b"
    assert cfg.positive_threshold == pytest.approx(0.5)
    assert cfg.watch_folder == Path("/tmp/cxr-watch")
