"""cli — ``python -m cxr_triage`` 入口。

子模块：

- ``run``    argparse 解析 + serve / once / replay 三子命令；
            pyproject.toml ``[project.scripts]`` 入口 ``cxr-triage``
            与 ``python -m cxr_triage`` 都由 ``main()`` 暴露。

业务定位（CLAUDE.md「须至少有可重复的使用入口」+ Task 32 落地）：

- ``serve``  : 默认 :8586 起 FastAPI，浏览器访问 / 看三栏面板
- ``once``   : 处理单张 DICOM 跑 happy-path 不起服务，便于运维
              一次性脚本接入
- ``replay`` : 重放 ``audit_log`` + ``tool_calls_<date>.jsonl``，
              按 ``study_uid`` 或 limit 拉最近 N 条，便于飞检
              溯源与日常排障
"""
from __future__ import annotations

from .run import (
    DEFAULT_HOST,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    SERVER_LOG_FILENAME,
    build_parser,
    cmd_once,
    cmd_replay,
    cmd_serve,
    main,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_LEVEL",
    "SERVER_LOG_FILENAME",
    "build_parser",
    "cmd_serve",
    "cmd_once",
    "cmd_replay",
    "main",
]