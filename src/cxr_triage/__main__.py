"""``python -m cxr_triage`` 入口 — 把控制权交给 cli.run.main。

约定（pyproject.toml 已声明 ``[project.scripts] cxr-triage``）：
- ``cxr-triage <subcmd>``  →  CLI 脚本入口
- ``python -m cxr_triage`` →  本文件（等价于 ``cxr-triage``）
"""
from __future__ import annotations

import sys

from cxr_triage.cli.run import main

if __name__ == "__main__":
    raise SystemExit(main())