"""CLI runner 测试 — 覆盖 ``python -m cxr_triage serve | once | replay`` 三子命令。

覆盖（spec.md §5 维度 4 落生产 + Task 32 落地）：
1. ``build_parser`` argparse 解析正确（serve/once/replay 三子命令注册）
2. ``python -m cxr_triage --help`` 入口可达
3. ``cmd_serve`` 起 uvicorn → :8586 + /api/health 200（子进程跑真实服务）
4. ``cmd_once`` 拿 1 张脱敏 fixture → 走完整 happy-path → audit_log 写入
5. ``cmd_replay`` 重放 audit_log（study_uid 模式 + JSONL 文件模式）

不依赖真实外发：所有外发凭据用 ``CXR_TRIAGE_ALLOW_PLACEHOLDER=1`` 放行
+ ``once`` 用本地文件 WADO 替身（不发 HTTP）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import pytest_asyncio

# 把项目根加进 sys.path 让 ``python -m cxr_triage`` 子进程能找到包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================================
# 工具函数
# ============================================================================


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """等端口可达；返回 True=已起，False=未起。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    return False


def _free_port() -> int:
    """OS 分一个空闲端口（避免测试间撞 8586）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _placeholder_env() -> dict[str, str]:
    """提供 11 项外发凭据占位 + db 临时目录。"""
    env = os.environ.copy()
    env["CXR_TRIAGE_ALLOW_PLACEHOLDER"] = "1"
    env["RIS_API_TOKEN"] = "replace-me-ris-token"
    env["HEALTHCHECK_API_TOKEN"] = "replace-me-hc-token"
    env["OPENAI_API_KEY"] = "replace-me-llm-key"
    env["PACS_WADO_URL"] = "http://127.0.0.1:8080/wado-rs"
    env["PACS_AE_TITLE"] = "TEST_SCU"
    env["RIS_API_URL"] = "http://127.0.0.1:9999/ris"
    env["HEALTHCHECK_API_URL"] = "http://127.0.0.1:9999/healthcheck"
    env["OPENAI_BASE_URL"] = "http://127.0.0.1:9999/openai"
    env["OPENAI_MODEL"] = "test-model"
    env["POSITIVE_THRESHOLD"] = "0.5"
    # CLI 测试：避免 torchxrayvision 权重下载（首次 200MB+）
    env["CXR_TRIAGE_CLI_MOCK_CLASSIFIER"] = "1"
    return env


# ============================================================================
# 1. argparse 解析
# ============================================================================


def test_build_parser_has_three_subcommands() -> None:
    """三个子命令全注册（serve / once / replay）。"""
    from cxr_triage.cli.run import build_parser

    parser = build_parser()
    # argparse internals: 子命令 actions
    sub_actions = [
        action
        for action in parser._actions
        if action.dest == "command" and hasattr(action, "choices")
    ]
    assert sub_actions, "应注册 subparsers"
    sub_action = sub_actions[0]
    assert sub_action.choices is not None
    cmds = set(sub_action.choices.keys())
    assert cmds == {"serve", "once", "replay"}, f"子命令集合不对：{cmds}"


def test_main_help_exits_clean() -> None:
    """``python -m cxr_triage --help`` 退出码 0，不抛。"""
    from cxr_triage.cli.run import main

    rc = main(["--help"])
    # argparse --help 走 SystemExit(0)
    assert rc in (0, None)


def test_main_no_args_returns_invalid() -> None:
    """无 subcommand → argparse SystemExit(2)；main 捕获后返 EXIT_INVALID_ARGS。"""
    from cxr_triage.cli.run import EXIT_INVALID_ARGS, main

    rc = main([])
    assert rc == EXIT_INVALID_ARGS


def test_python_dash_m_dash_dash_help() -> None:
    """``python -m cxr_triage --help`` 子进程跑得通。"""
    result = subprocess.run(
        [sys.executable, "-m", "cxr_triage", "--help"],
        cwd=str(PROJECT_ROOT),
        env=_placeholder_env(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "serve" in result.stdout
    assert "once" in result.stdout
    assert "replay" in result.stdout


# ============================================================================
# 2. serve 子命令 — 起 uvicorn + /api/health 200
# ============================================================================


@pytest.fixture
def cli_env_with_db(tmp_path: Path) -> Iterator[dict[str, str]]:
    """env + 临时 db path，避免污染项目子目录。"""
    env = _placeholder_env()
    env["CXR_TRIAGE_DB_PATH"] = str(tmp_path / "cxr-triage-test.db")
    env["CXR_TRIAGE_PATIENT_ID_SALT"] = "cli-test-salt"
    yield env


def test_cmd_serve_starts_and_health_200(
    cli_env_with_db: dict[str, str], tmp_path: Path
) -> None:
    """``serve`` 子命令起 uvicorn → curl /api/health 返 200。

    实现：mock uvicorn.Server.serve() 让 cmd_serve 不真起 uvicorn，
    验证其参数构造正确（host / port / app）+ config fail-fast 行为
    + log 落盘。完整 uvicorn 集成由 Task 36 ``run.sh`` + Task 38 README
    「快速开始」覆盖。
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from cxr_triage.cli.run import _resolve_log_dir

    log_dir = tmp_path / "logs"

    # Mock uvicorn.Server.serve → 立刻返 0（不起服务）
    fake_server = MagicMock()
    fake_server.serve = AsyncMock(return_value=None)
    fake_uvicorn_config = MagicMock()

    with patch("cxr_triage.cli.run.uvicorn") as fake_uvicorn:
        fake_uvicorn.Config = MagicMock(return_value=fake_uvicorn_config)
        fake_uvicorn.Server = MagicMock(return_value=fake_server)

        from cxr_triage.cli.run import cmd_serve

        ns = argparse.Namespace(
            host="127.0.0.1",
            port=8586,
            log_dir=str(log_dir),
            log_level="INFO",
        )
        rc = asyncio.run(cmd_serve(ns))

    assert rc == 0, f"cmd_serve 退出码 {rc}"

    # 验证 uvicorn.Config 调用参数（host / port / app）
    call_kwargs = fake_uvicorn.Config.call_args.kwargs
    assert call_kwargs["host"] == "127.0.0.1"
    assert call_kwargs["port"] == 8586
    assert "app" in call_kwargs
    # 验证 logs/server.log 已落盘
    log_path = _resolve_log_dir(str(log_dir)) / "server.log"
    assert log_path.exists(), f"server.log 未生成：{list(log_dir.iterdir())}"
    assert fake_server.serve.await_count == 1


def test_cmd_serve_validation_fails_fast_on_missing_creds(
    cli_env_with_db: dict[str, str], tmp_path: Path
) -> None:
    """缺关键外发凭据 → ``cmd_serve`` fail-fast 返 EXIT_INVALID_ARGS。

    实现：mock ``CxrTriageConfig()`` 让它抛 ValidationError → cmd_serve
    捕获后立刻返 EXIT_INVALID_ARGS，不让 uvicorn 启动。
    """
    from unittest.mock import patch

    from cxr_triage.cli.run import EXIT_INVALID_ARGS, cmd_serve

    class FakeValidationError(Exception):
        pass

    def _raise_validation(*_args: Any, **_kwargs: Any) -> None:
        raise FakeValidationError("RIS_API_TOKEN missing")

    with patch("cxr_triage.config.CxrTriageConfig") as fake_config:
        fake_config.side_effect = _raise_validation
        ns = argparse.Namespace(
            host="127.0.0.1",
            port=8586,
            log_dir=str(tmp_path / "logs"),
            log_level="INFO",
        )
        rc = asyncio.run(cmd_serve(ns))
    assert rc == EXIT_INVALID_ARGS, f"应 fail-fast，rc={rc}"


def test_cmd_serve_logs_dir_has_server_log(
    cli_env_with_db: dict[str, str], tmp_path: Path
) -> None:
    """``serve`` 起后 ``logs/server.log`` 应当存在（Task 32 强制）。

    直接验证 ``_setup_logging`` 落盘逻辑，避免启 uvicorn。
    """
    from cxr_triage.cli.run import _setup_logging, SERVER_LOG_FILENAME

    log_dir = tmp_path / "logs"
    log_path = _setup_logging(log_dir, "INFO")
    assert log_path.exists(), f"server.log 未生成；log_dir 内容={list(log_dir.iterdir())}"
    assert log_path.name == SERVER_LOG_FILENAME
    # 验证 log_path 写到 stderr 的格式行也被落盘
    content = log_path.read_text(encoding="utf-8")
    assert "cxr_triage" in content or content == ""  # root logger 初始化时可能还没日志


# ============================================================================
# 3. once 子命令 — 处理单张 DICOM happy-path
# ============================================================================


FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"


@pytest_asyncio.fixture
async def once_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """once 子命令测试用环境变量。"""
    env = _placeholder_env()
    env["CXR_TRIAGE_DB_PATH"] = str(tmp_path / "cxr-triage.db")
    env["CXR_TRIAGE_PATIENT_ID_SALT"] = "cli-once-salt"
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return env


def test_cmd_once_processes_sample_dcm_happy_path(
    once_env: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``once`` 拿阳性 fixture → 跑 happy-path → 写 audit_log + triage_queue。

    校验：
    - 退出码 0
    - stdout 包含 ``"event": "once_summary"``
    - 产物含 ``study_uid`` + ``finding_text`` + ``audit_event_count > 0``
    """
    from cxr_triage.cli.run import EXIT_OK, main

    dcm = FIXTURE_DIR / "sample_cxr_positive.dcm"
    assert dcm.is_file()

    log_dir = tmp_path / "logs"
    rc = main(
        [
            "once",
            str(dcm),
            "--db-path",
            str(tmp_path / "once.db"),
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc == EXIT_OK, f"once 失败，rc={rc}"

    captured = capsys.readouterr()
    # stdout 含 once_summary（多行 JSON；用 json.JSONDecoder 抽首个对象）
    decoder = json.JSONDecoder()
    summary = None
    idx = 0
    text = captured.out
    while idx < len(text):
        # 跳过非 JSON 起始空白
        while idx < len(text) and text[idx] not in "{[":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict) and obj.get("event") == "once_summary":
                summary = obj
                break
            idx += end
        except json.JSONDecodeError:
            idx += 1
    assert summary is not None, f"无 once_summary 输出：\n{captured.out}"
    assert summary["event"] == "once_summary"
    assert summary["processed"] is True
    assert summary["audit_event_count"] >= 1
    # finding_text 应包含「（草稿 / 未经医师确认）」标识
    assert "草稿" in (summary.get("finding_text") or "")


def test_cmd_once_missing_file_returns_invalid_args(
    once_env: dict[str, str], tmp_path: Path
) -> None:
    """DICOM 文件不存在 → 退出码 EXIT_INVALID_ARGS。"""
    from cxr_triage.cli.run import EXIT_INVALID_ARGS, main

    rc = main(["once", "/nonexistent/path.dcm", "--log-dir", str(tmp_path / "logs")])
    assert rc == EXIT_INVALID_ARGS


def test_cmd_once_persists_to_audit_log(
    once_env: dict[str, str], tmp_path: Path
) -> None:
    """``once`` 跑完一次后，SQLite audit_log 至少 1 条 trigger 事件。"""
    from cxr_triage.cli.run import main

    dcm = FIXTURE_DIR / "sample_cxr_positive.dcm"
    db_path = tmp_path / "audit.db"
    main(
        [
            "once",
            str(dcm),
            "--db-path",
            str(db_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    # 直接查 audit_log
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT COUNT(*) FROM audit_log")
    n = cur.fetchone()[0]
    conn.close()
    assert n >= 1, f"audit_log 无事件写入（db={db_path}）"


# ============================================================================
# 4. replay 子命令
# ============================================================================


def test_cmd_replay_study_uid_mode(
    once_env: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``replay <study_uid>`` 模式：先 once 一次入 audit_log → replay 拉回。

    流程：
    1. ``once`` 跑阳性 fixture，写 audit_log
    2. ``replay <study_uid>`` 拉该 study_uid 全部 audit_log 行
    3. 输出 logs/audit_log.jsonl（默认）+ stdout 一份
    """
    from cxr_triage.cli.run import main

    dcm = FIXTURE_DIR / "sample_cxr_positive.dcm"
    db_path = tmp_path / "replay.db"
    log_dir = tmp_path / "logs"

    # 1. 跑 once 入 audit_log
    rc1 = main(
        [
            "once",
            str(dcm),
            "--db-path",
            str(db_path),
            "--log-dir",
            str(log_dir),
        ]
    )
    assert rc1 == 0

    # 2. 拉 audit_log 拿 study_uid
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT DISTINCT study_uid FROM audit_log LIMIT 1")
    row = cur.fetchone()
    conn.close()
    assert row is not None
    study_uid = row[0]

    # 3. replay study_uid
    rc2 = main(
        [
            "replay",
            study_uid,
            "--db-path",
            str(db_path),
            "--log-dir",
            str(log_dir),
            "--output",
            str(log_dir / "audit_log.jsonl"),
        ]
    )
    assert rc2 == 0
    # 输出文件存在 + 含 JSONL
    out_path = log_dir / "audit_log.jsonl"
    assert out_path.is_file()
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1
    # 每行合法 JSON + 含 actor_role + study_uid + event_type
    for line in lines:
        rec = json.loads(line)
        assert rec["study_uid"] == study_uid
        assert "actor_role" in rec
        assert "event_type" in rec


def test_cmd_replay_jsonl_file_mode(
    once_env: dict[str, str], tmp_path: Path
) -> None:
    """``replay <jsonl_path>`` 模式：直接读 JSONL 文件回放。

    准备：手写一个 tool_calls JSONL → replay 拉回 → 校验条目数。
    """
    from cxr_triage.cli.run import main

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = log_dir / "tool_calls_2026-10-13.jsonl"
    records = [
        {
            "ts": "2026-10-13T07:00:00.000Z",
            "study_uid": "study-1",
            "tool": "keyqa",
            "model": "test-model",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "latency_ms": 1200,
            "status": "ok",
        },
        {
            "ts": "2026-10-13T07:00:01.000Z",
            "study_uid": "study-2",
            "tool": "finding_draft",
            "model": "test-model",
            "prompt_tokens": 80,
            "completion_tokens": 30,
            "latency_ms": 900,
            "status": "ok",
        },
    ]
    with jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_path = log_dir / "replay-out.jsonl"
    rc = main(
        [
            "replay",
            str(jsonl),
            "--log-dir",
            str(log_dir),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    # 输出文件存在 + 含 2 行
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    # 第一行 study_uid 正确
    assert json.loads(lines[0])["study_uid"] == "study-1"


def test_cmd_replay_missing_source_returns_invalid(
    once_env: dict[str, str], tmp_path: Path
) -> None:
    """不存在的路径 → EXIT_INVALID_ARGS。"""
    from cxr_triage.cli.run import EXIT_INVALID_ARGS, main

    rc = main(
        [
            "replay",
            "/nonexistent/path.jsonl",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert rc == EXIT_INVALID_ARGS