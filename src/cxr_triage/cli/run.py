"""CLI runner — ``python -m cxr_triage serve | once | replay``。

设计原则（CLAUDE.md 强制 + Task 32 落地）：

1. **可重复入口**：业务人员只需 ``python -m cxr_triage serve`` 即可拉起；
   ``once`` 与 ``replay`` 给运维 / 飞检一次性脚本接入。
2. **后台日志**：``logs/server.log`` + ``logs/tool_calls_<date>.jsonl`` +
   ``logs/audit_log.jsonl`` 三件套按 UTC 日期轮换；与 ``.gitignore`` 一致
   不入仓库。
3. **不硬编码凭据**：所有外发凭据仍走 ``CxrTriageConfig`` env 读；CLI
   启动时 fail-fast 缺失值。
4. **不耦合 web 层**：``once`` / ``replay`` 不拉 FastAPI；只调 core 模块
   （TriageTrigger / AuditLogger），便于 CI 集成。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import uvicorn


# ============================================================================
# 常量
# ============================================================================

#: 默认监听 host；用 0.0.0.0 让同网段浏览器可访问
DEFAULT_HOST: str = "0.0.0.0"

#: 默认端口；CLAUDE.md Phase 4 Task 32 落地「起 :8586」
DEFAULT_PORT: int = 8586

#: 默认日志目录（项目子目录内 logs/）
DEFAULT_LOG_DIR: Path = Path("./logs")

#: 默认日志级别
DEFAULT_LOG_LEVEL: str = "INFO"

#: 后台 server.log 文件名（按 UTC 日期轮换见 ``_setup_logging``）
SERVER_LOG_FILENAME: str = "server.log"

#: audit_log 重放 JSONL 输出文件（Task 32 ``replay`` 子命令）
REPLAY_LOG_FILENAME: str = "audit_log.jsonl"

#: 触发「once 处理后被跳过（已 processed）」的退出码
EXIT_OK: int = 0
EXIT_INVALID_ARGS: int = 2
EXIT_PROCESS_FAILED: int = 3
EXIT_NO_STUDY: int = 4

#: 默认 /api/health 探活超时（秒）
DEFAULT_HEALTH_PROBE_TIMEOUT: float = 2.0


logger = logging.getLogger("cxr_triage.cli")


# ============================================================================
# 日志 / 工具
# ============================================================================


def _setup_logging(log_dir: Path, log_level: str = DEFAULT_LOG_LEVEL) -> Path:
    """配置 stderr + ``log_dir/server.log`` 双路输出。

    返回 ``server.log`` 绝对路径（便于 progress.txt 记录）。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / SERVER_LOG_FILENAME

    # 重置 root logger 处理器，避免 uvicorn 默认配置叠加
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return log_path.resolve()


def _resolve_log_dir(raw: Optional[str]) -> Path:
    """解析日志目录；None / 空 → 默认 ``./logs``。"""
    return Path(raw).expanduser().resolve() if raw else DEFAULT_LOG_DIR.resolve()


def _resolve_db_path(raw: Optional[str]) -> Path:
    """解析数据库路径；None / 空 → 默认 ``./logs/cxr-triage.db``。"""
    if raw:
        return Path(raw).expanduser().resolve()
    return (DEFAULT_LOG_DIR / "cxr-triage.db").resolve()


# ============================================================================
# 子命令：serve
# ============================================================================


async def cmd_serve(args: argparse.Namespace) -> int:
    """启动 FastAPI 服务（默认 :8586）。

    行为：
    - 加载 ``CxrTriageConfig`` 验证凭据
    - ``logs/server.log`` 启动日志
    - uvicorn.Server.serve() 阻塞运行，SIGINT/SIGTERM 优雅退出
    """
    log_dir = _resolve_log_dir(args.log_dir)
    log_path = _setup_logging(log_dir, args.log_level)
    logger.info("启动 FastAPI on %s:%d，日志=%s", args.host, args.port, log_path)

    # 延迟 import 避免 cli 模块级副作用触发 create_app() 两次
    from cxr_triage.config import CxrTriageConfig
    from cxr_triage.web.app import app

    # fail-fast 检查（缺凭据立刻报错，不让服务半瘫启动）
    try:
        CxrTriageConfig()  # noqa: F841 — 仅触发 BaseSettings 校验
    except Exception as exc:  # noqa: BLE001
        logger.error("CxrTriageConfig 校验失败：%s", exc)
        print(
            json.dumps(
                {"event": "config_validation_failed", "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return EXIT_INVALID_ARGS

    config_obj = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_config=None,  # 我们自己管 logging
        access_log=True,
    )
    server = uvicorn.Server(config_obj)

    try:
        await server.serve()
    finally:
        logger.info("server stopped; log=%s", log_path)
    return EXIT_OK


# ============================================================================
# 子命令：once
# ============================================================================


async def cmd_once(args: argparse.Namespace) -> int:
    """处理单张 DICOM（happy-path），不起 HTTP 服务。

    流程：
    1. 校验 .dcm 存在
    2. 构造 Database / AuditLogger / TriageQueue / workflow / TriageTrigger
    3. 把 .dcm 写到 watch_folder，由 DicomFolderWatcher 一致性收尾
       （无需真起 SCU 监听），或者直接调 process_one
    4. 输出处理结果（finding_text + max_prob + positive_flag + 状态）

    简化：本命令采用「临时塞 incoming_studies → 调 TriageTrigger.process_one」
    路径，避免耦合 watch_folder 协程；CI 友好。

    Returns:
        0 = 已 processed；3 = 处理失败；4 = study 不在 incoming_studies
    """
    log_dir = _resolve_log_dir(args.log_dir)
    log_path = _setup_logging(log_dir, args.log_level)
    db_path = _resolve_db_path(args.db_path)
    logger.info("once: dcm=%s, db=%s, log=%s", args.dcm_path, db_path, log_path)

    dcm_path = Path(args.dcm_path).expanduser().resolve()
    if not dcm_path.is_file():
        logger.error("DICOM 文件不存在：%s", dcm_path)
        return EXIT_INVALID_ARGS

    # 延迟 import 避免循环依赖与 model 下载副作用
    from cxr_triage.config import CxrTriageConfig
    from cxr_triage.engine.triage_classifier import build_triage_classifier
    from cxr_triage.engine.triage_queue import TriageQueue
    from cxr_triage.engine.trigger import build_triage_trigger
    from cxr_triage.engine.finding_draft import build_finding_draft
    from cxr_triage.engine.image_preview import build_image_preview
    from cxr_triage.engine.positive_grounding import build_positive_grounding
    from cxr_triage.pacs.dicom_metadata import (
        parse_cxr_metadata,
        metadata_to_insert_row,
        DicomMetadataError,
    )
    from cxr_triage.persistence.migrations import init_db
    from cxr_triage.persistence.sqlite import Database
    from cxr_triage.pacs.dicom_processor import build_dicom_processor
    from cxr_triage.state.idempotency import StateIdempotencyStore
    from cxr_triage.writers.audit import build_audit_logger

    # 1. 配置 + 数据库 + 表
    config = CxrTriageConfig()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))
    await db.connect()
    await init_db(db)

    # 2. 工具注册表（仅本地确定路径，不走 LLM：classifier + finding_draft +
    #    grounding + image_preview；keyqa 跳过避免依赖模型网关）
    from cxr_triage.engine.workflow import ToolRegistry, CxrTriageWorkflow

    reg = ToolRegistry()
    # Mock 模式：env CXR_TRIAGE_CLI_MOCK_CLASSIFIER=1 → 用确定性 mock 模型，
    # 不下载 torchxrayvision DenseNet 权重（CI / 测试友好）
    use_mock_classifier = (
        os.environ.get("CXR_TRIAGE_CLI_MOCK_CLASSIFIER", "").strip().lower()
        in {"1", "true", "yes"}
    )
    if use_mock_classifier:
        from cxr_triage.engine.triage_classifier import (
            PATHOLOGY_NAMES as _PATH,
            POSITIVE_PROB_HEURISTIC as _THR,
        )

        class _MockCliModel:
            """CLI mock：固定概率（顶部病理 0.85 阳性）让 happy-path 跑通。"""

            def __init__(self) -> None:
                self.probs = [0.05] * len(_PATH)
                # 模拟 Pneumothorax 高概率（与阳性 fixture 对位）
                if "Pneumothorax" in _PATH:
                    self.probs[_PATH.index("Pneumothorax")] = 0.85

            def eval(self) -> "_MockCliModel":
                return self

            def __call__(self, x: Any) -> Any:
                import numpy as _np

                batch = x.shape[0] if hasattr(x, "shape") else 1
                arr = _np.zeros((batch, len(_PATH)), dtype=_np.float32)
                arr[:] = self.probs
                return arr

        classifier = build_triage_classifier(
            prob_threshold=_THR,
            device="cpu",
            model=_MockCliModel(),
            transform=lambda img: img,  # mock transform: passthrough
        )
    else:
        classifier = build_triage_classifier(
            prob_threshold=config.positive_threshold,
            device="cpu",
        )
    reg.register(classifier)
    reg.register(build_finding_draft())
    reg.register(build_positive_grounding())
    reg.register(build_image_preview())

    workflow = CxrTriageWorkflow(
        registry=reg,
        positive_threshold=config.positive_threshold,
    )

    audit = build_audit_logger(db=db, tool_calls_dir=str(log_dir / "tool_calls"))
    triage_queue = TriageQueue(db)
    idempotency = StateIdempotencyStore(db)

    # 3. 先读 DICOM bytes（后续 _LocalFileWadoClient 与 parse_cxr_metadata 都要用）
    with dcm_path.open("rb") as fh:
        dcm_bytes = fh.read()

    # 4. 本地文件 WADO 客户端（直接返回 dcm_bytes，不发 HTTP），
    #    让 TriageTrigger 不依赖真实 PACS 服务
    class _LocalFileWadoClient:
        """CLI ``once`` 专用 WADO 替身：从内存 dcm_bytes 直接返回。"""

        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.calls: int = 0

        async def fetch(self, study_uid: str) -> bytes:  # noqa: D401
            self.calls += 1
            return self._payload

        async def aclose(self) -> None:  # noqa: D401
            return None

        async def close(self) -> None:  # noqa: D401
            return None

    wado_client = _LocalFileWadoClient(dcm_bytes)

    trigger = build_triage_trigger(
        config=config,
        db=db,
        workflow=workflow,
        triage_queue=triage_queue,
        idempotency=idempotency,
        audit_logger=audit,
        wado_client=wado_client,  # type: ignore[arg-type]
        dicom_processor=build_dicom_processor(),
    )

    # 5. 解析 DICOM → 入 incoming_studies
    import pydicom
    from io import BytesIO

    try:
        ds = pydicom.dcmread(BytesIO(dcm_bytes), force=True)
        meta = parse_cxr_metadata(ds)
    except DicomMetadataError as exc:
        logger.error("DICOM metadata 解析失败：%s", exc)
        await db.close()
        return EXIT_INVALID_ARGS

    study_uid = args.study_uid or meta["study_uid"]
    # 把 meta dict 喂给 metadata_to_insert_row（保持与 PacsListener / Watcher 一致）
    row = metadata_to_insert_row(meta, source="cli_once")
    async with db.connection() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO incoming_studies "
            "(study_uid, patient_id_hash, accession_number, modality, status, source) "
            "VALUES (?, ?, ?, ?, 'pending', 'cli_once')",
            (row[0], row[1], row[2], row[3]),
        )
        await conn.commit()

    # 4. 跑 TriageTrigger.process_one
    processed = await trigger.process_one(study_uid)

    # 5. 输出结构化结果
    queue_row = await triage_queue.get_by_study(study_uid)
    audit_lines = await audit.list_audit_for_study(study_uid)
    summary = {
        "event": "once_summary",
        "study_uid": study_uid,
        "processed": processed,
        "finding_text": (queue_row or {}).get("finding_text"),
        "max_prob": (queue_row or {}).get("max_prob"),
        "positive_flag": (queue_row or {}).get("positive_flag"),
        "audit_event_count": len(audit_lines),
        "audit_events": [
            {
                "ts": e.get("ts"),
                "actor_role": e.get("actor_role"),
                "event_type": e.get("event_type"),
            }
            for e in audit_lines[-10:]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    await db.close()
    if not processed and not queue_row:
        return EXIT_NO_STUDY
    return EXIT_OK if processed else EXIT_PROCESS_FAILED


# ============================================================================
# 子命令：replay
# ============================================================================


async def cmd_replay(args: argparse.Namespace) -> int:
    """重放 ``audit_log`` + ``tool_calls_<date>.jsonl``。

    source 支持两种形态：

    1. ``study_uid``（非路径）→ 拉该 study 全部 audit_log 行，按 ts 升序
    2. ``logs/`` 目录 → 列目录里全部 ``tool_calls_*.jsonl`` + 汇总
       audit_log，按 ts 升序打印最近 ``--limit`` 条

    默认输出 JSON Lines 到 stdout；``--output`` 给文件路径则落盘
    （默认 ``logs/audit_log.jsonl``）。
    """
    log_dir = _resolve_log_dir(args.log_dir)
    _setup_logging(log_dir, args.log_level)

    source_str = args.source
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (log_dir / REPLAY_LOG_FILENAME)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from cxr_triage.persistence.sqlite import Database
    from cxr_triage.persistence.migrations import init_db
    from cxr_triage.writers.audit import AuditLogger

    db_path = _resolve_db_path(args.db_path)
    db = Database(str(db_path))
    audit: Optional[AuditLogger] = None
    try:
        await db.connect()
        # 让 init_db 容忍已存在表
        await init_db(db, include_idempotency=True, include_state_machine=True)
        audit = AuditLogger(db=db, tool_calls_dir=str(log_dir / "tool_calls"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "DB / AuditLogger 初始化失败（继续仅读 tool_calls JSONL）：%s", exc
        )

    events: list[dict[str, Any]] = []

    # Case 1: study_uid 形式
    is_path = "/" in source_str or source_str.endswith(".jsonl") or Path(source_str).exists()
    if not is_path and audit is not None:
        rows = await audit.list_audit_for_study(source_str)
        events.extend(rows)
    elif not is_path and audit is None:
        logger.error("study_uid 模式需要数据库；当前 db 不可用：%s", db_path)
        return EXIT_INVALID_ARGS
    else:
        # Case 2: 目录 / JSONL 文件
        source_path = Path(source_str).expanduser().resolve()
        if source_path.is_dir() and audit is not None:
            # 全部 audit_log 行
            async with db.connection() as conn:  # type: ignore[union-attr]
                cur = await conn.execute(
                    "SELECT actor_role, ts, study_uid, event_type, payload_json "
                    "FROM audit_log ORDER BY ts DESC LIMIT ?",
                    (args.limit,),
                )
                rows = await cur.fetchall()
                await cur.close()
            for row in rows:
                try:
                    payload_obj = json.loads(row["payload_json"])
                except (TypeError, ValueError):
                    payload_obj = row["payload_json"]
                events.append(
                    {
                        "ts": row["ts"],
                        "actor_role": row["actor_role"],
                        "study_uid": row["study_uid"],
                        "event_type": row["event_type"],
                        "payload": payload_obj,
                    }
                )
        elif source_path.is_file():
            # JSONL 文件
            from cxr_triage.writers.audit import AuditLogger as _AL

            records = _AL.read_tool_call_records(source_path)
            events.extend(records[-args.limit :])
        else:
            logger.error("source 路径不存在或不可解析：%s", source_str)
            return EXIT_INVALID_ARGS

    # 输出
    with output_path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
    logger.info("replay %d 条事件 → %s", len(events), output_path)

    # stdout 也打一份（限制条数便于人眼查看）
    for evt in events[-args.limit :]:
        print(json.dumps(evt, ensure_ascii=False))

    if audit is not None:
        await audit.close()
    await db.close()
    return EXIT_OK


# ============================================================================
# argparse
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器（默认 subparsers 必填）。"""
    parser = argparse.ArgumentParser(
        prog="cxr-triage",
        description="体检胸片所见草稿系统 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务（默认 :8586）")
    p_serve.add_argument("--host", default=DEFAULT_HOST, help="监听 host")
    p_serve.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="监听 port"
    )
    p_serve.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录"
    )
    p_serve.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, help="日志级别 (DEBUG/INFO/WARNING/ERROR)"
    )
    p_serve.set_defaults(func=cmd_serve)

    # ---- once ----
    p_once = sub.add_parser(
        "once", help="处理单张 DICOM 跑 happy-path（不起 HTTP 服务）"
    )
    p_once.add_argument(
        "dcm_path", help=".dcm 文件路径（必填；可以是任意 DICOM 文件）"
    )
    p_once.add_argument(
        "--study-uid",
        default=None,
        help="StudyInstanceUID（缺省从 DICOM meta 解析）",
    )
    p_once.add_argument(
        "--db-path", default=None, help="SQLite 数据库路径（默认 ./logs/cxr-triage.db）"
    )
    p_once.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录"
    )
    p_once.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, help="日志级别"
    )
    p_once.set_defaults(func=cmd_once)

    # ---- replay ----
    p_replay = sub.add_parser(
        "replay", help="重放 audit_log + tool_calls JSONL（study_uid 或目录）"
    )
    p_replay.add_argument(
        "source", help="study_uid 或 logs/ 目录或 tool_calls_*.jsonl 文件"
    )
    p_replay.add_argument(
        "--output", default=None, help="落盘路径（默认 logs/audit_log.jsonl）"
    )
    p_replay.add_argument(
        "--db-path", default=None, help="SQLite 数据库路径（默认 ./logs/cxr-triage.db）"
    )
    p_replay.add_argument(
        "--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录"
    )
    p_replay.add_argument(
        "--limit", type=int, default=20, help="最多输出 N 条事件"
    )
    p_replay.add_argument(
        "--log-level", default=DEFAULT_LOG_LEVEL, help="日志级别"
    )
    p_replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 总入口（同步包装 asyncio.run）。

    ``pyproject.toml`` ``[project.scripts] cxr-triage`` 与
    ``python -m cxr_triage`` 均由此函数驱动。
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse 在 --help / 错误时 SystemExit；转成退出码
        code = exc.code if isinstance(exc.code, int) else 2
        # --help 退出码 = 0；缺子命令 = 2
        return int(code)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_INVALID_ARGS
    try:
        return asyncio.run(func(args))
    except KeyboardInterrupt:
        logger.warning("用户 Ctrl-C 终止")
        return 130  # SIGINT convention


if __name__ == "__main__":
    raise SystemExit(main())