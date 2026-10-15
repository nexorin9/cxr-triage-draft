"""DICOM watch folder 兜底通道 — PACS C-STORE AE 协商失败时改走文件轮询。

设计原则（plan.md §6 PACS C-STORE AE 协商失败 缓解）：

1. **轮询而非网络监听**：当院内 PACS 与本系统的 C-STORE AE 协商失败（端口
   占用 / AE Title 冲突 / 网络策略阻断），把待入站 DICOM 文件放进
   ``watch_folder`` 目录（如 ``/tmp/cxr-watch``），本类按 5s 间隔轮询。
2. **入条统一**：仍走 ``parse_cxr_metadata`` → ``incoming_studies`` 插入，
   ``source='watch_folder'`` 标记渠道，便于审计追溯「这单是从文件来的」。
3. **处理后归档**：处理成功的 ``.dcm`` 移到 ``<watch_folder>/.processed/<ts>_<name>``
   避免重复消费；解析失败的 ``.dcm`` 移到 ``<watch_folder>/.errors/``，
   写 ``audit_log`` 后不静默吞数据。
4. **patient_id_hash 盐**：与 ``PacsListener`` 一致，从 env 注入。
5. **Modality 过滤复用**：CR / DX 之外的 modality 一律丢弃并写 audit。

参考：源仓库 ``medrax/tools/dicom.py`` DICOM 解析思路。
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pydicom

from ..config import CxrTriageConfig
from ..persistence.sqlite import Database
from .dicom_metadata import (
    DicomMetadataError,
    metadata_to_insert_row,
    parse_cxr_metadata,
)

logger = logging.getLogger(__name__)


# 默认轮询周期（5s）；可由构造参数覆盖。
DEFAULT_POLL_INTERVAL_SECONDS: float = 5.0


class FolderWatcherError(RuntimeError):
    """watch folder 启动 / 关闭 / 处理过程中出现的可恢复异常。"""


class DicomFolderWatcher:
    """DICOM watch folder 兜底轮询器。

    Parameters
    ----------
    config:
        业务配置（``watch_folder`` 路径由此读；测试可注入临时目录）。
    db:
        已 ``await db.connect()`` 打开的 ``Database`` 实例。
    poll_interval:
        轮询间隔（秒）；默认 5s；测试可注入更短值。
    audit_hook:
        可选回调 ``async def audit_hook(study_uid, event_type, payload)``；
        业务层（如 ``writers/audit.py``）可注入做模型调用记录。缺省时只
        写 ``audit_log`` 表。
    salt:
        PatientID hash 用的盐；缺省从 ``CXR_TRIAGE_PATIENT_ID_SALT`` 读。
    processed_subdir:
        已成功处理的 ``.dcm`` 移到该子目录（相对 watch_folder）；默认
        ``.processed``。
    error_subdir:
        解析失败的 ``.dcm`` 移到该子目录；默认 ``.errors``。
    """

    def __init__(
        self,
        config: CxrTriageConfig,
        db: Database,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        audit_hook: Optional[Callable[[str, str, dict], Awaitable[None]]] = None,
        salt: Optional[str] = None,
        processed_subdir: str = ".processed",
        error_subdir: str = ".errors",
    ) -> None:
        self.config = config
        self.db = db
        self.poll_interval = max(0.05, float(poll_interval))
        self._audit_hook = audit_hook
        self._salt = salt or os.getenv("CXR_TRIAGE_PATIENT_ID_SALT", "cxr-triage-salt")
        self._processed_subdir = processed_subdir
        self._error_subdir = error_subdir
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._stats = {
            "scanned": 0,           # 轮询过程中扫到的 .dcm 文件总数
            "ingested": 0,          # 成功入条 incoming_studies
            "filtered_modality": 0, # modality 非 CR/DX 被丢弃
            "errors": 0,            # 解析 / IO 失败
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """起后台协程轮询；幂等（重复调用直接返回）。"""
        if self._running:
            logger.warning("DicomFolderWatcher 已在运行，跳过重复 start()")
            return
        watch_dir = Path(self.config.watch_folder)
        watch_dir.mkdir(parents=True, exist_ok=True)
        # 提前建子目录，避免处理第一份文件时报 FileNotFoundError
        (watch_dir / self._processed_subdir).mkdir(parents=True, exist_ok=True)
        (watch_dir / self._error_subdir).mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(), name="cxr-triage-folder-watcher"
        )
        logger.info(
            "watch folder 兜底通道已启动: dir=%s interval=%.1fs",
            watch_dir,
            self.poll_interval,
        )

    async def stop(self) -> None:
        """优雅关闭：等当前轮询周期跑完，cancel 后台协程。"""
        if not self._running:
            return
        self._stop_event.set()
        if self._task is not None:
            try:
                # 给正在跑的一轮一点点时间优雅退出
                try:
                    await asyncio.wait_for(self._task, timeout=2.0)
                except asyncio.TimeoutError:
                    self._task.cancel()
                    try:
                        await self._task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
            finally:
                self._task = None
        self._running = False
        logger.info("watch folder 兜底通道已停止")

    # ------------------------------------------------------------------
    # 单轮轮询入口（同时供测试驱动）
    # ------------------------------------------------------------------
    async def poll_once(self) -> int:
        """跑一次扫描 + 处理。返回本轮成功入条数（用于测试）。"""
        watch_dir = Path(self.config.watch_folder)
        watch_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = watch_dir / self._processed_subdir
        error_dir = watch_dir / self._error_subdir
        processed_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        # 列出根目录下的 .dcm（不递归进入 .processed / .errors）
        try:
            entries = sorted(
                p
                for p in watch_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".dcm"
            )
        except FileNotFoundError:
            return 0
        ingested_count = 0
        for entry in entries:
            self._stats["scanned"] += 1
            try:
                moved_to_processed = await self._process_one(entry, processed_dir)
                if moved_to_processed:
                    ingested_count += 1
            except DicomMetadataError as exc:
                # 解析失败：移到 .errors/，写 audit
                self._stats["errors"] += 1
                await self._record_audit(
                    study_uid="",
                    event_type="watch_folder_metadata_error",
                    payload={
                        "file": str(entry),
                        "error": str(exc),
                    },
                )
                self._move_to(entry, error_dir)
            except Exception as exc:  # noqa: BLE001
                # 兜底：IO 错 / 数据库错 → 移到 .errors/，不阻断后续文件
                self._stats["errors"] += 1
                logger.exception("watch folder 处理失败: %s", entry)
                await self._record_audit(
                    study_uid="",
                    event_type="watch_folder_io_error",
                    payload={
                        "file": str(entry),
                        "error": str(exc),
                    },
                )
                try:
                    self._move_to(entry, error_dir)
                except Exception:  # noqa: BLE001
                    logger.exception("移到 .errors/ 也失败，保留原文件")
        return ingested_count

    async def _poll_loop(self) -> None:
        """后台轮询协程；遇到 stop_event 优雅退出。"""
        # 第一次延迟 poll_interval 让 FastAPI lifespan 把其它模块先就位
        try:
            while not self._stop_event.is_set():
                try:
                    await self.poll_once()
                except Exception:  # noqa: BLE001
                    logger.exception("watch folder 轮询一轮失败（不退出）")
                # 等 poll_interval 或 stop
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.poll_interval
                    )
                    # 收到 stop 信号，退出
                    return
                except asyncio.TimeoutError:
                    # 正常超时，继续下一轮
                    continue
        except asyncio.CancelledError:
            logger.info("watch folder 协程被取消")
            raise

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _process_one(self, entry: Path, processed_dir: Path) -> bool:
        """处理单个 ``.dcm``：读 metadata → 入条 → 移动文件。

        Returns
        -------
        True 表示成功入条（含已存在的 study_uid 跳过），文件已移到
        ``processed_dir``；False 表示被 modality 过滤丢弃（文件也移到
        ``processed_dir``，避免下一轮又扫到）。

        Raises
        ------
        DicomMetadataError 关键字段缺失时抛，由 ``poll_once`` 捕获并入
        ``error_dir``。
        """
        # 1. 用 pydicom 读 DICOM；FileDataset 构造方式与 PacsListener 的 _dispatch 一致
        ds = pydicom.dcmread(str(entry), force=True)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        meta = parse_cxr_metadata(ds, salt=self._salt)
        if meta is None:
            # Modality 过滤失败；写 audit，文件归档到 processed
            self._stats["filtered_modality"] += 1
            await self._record_audit(
                study_uid=getattr(ds, "StudyInstanceUID", "") or "",
                event_type="watch_folder_modality_filtered",
                payload={
                    "file": str(entry),
                    "modality": getattr(ds, "Modality", ""),
                },
            )
            self._move_to(entry, processed_dir)
            return False

        # 2. 入 incoming_studies（INSERT OR REPLACE → 同 StudyInstanceUID 幂等）
        await self._insert_incoming_studies(meta, source="watch_folder")
        self._stats["ingested"] += 1
        await self._record_audit(
            study_uid=meta["study_uid"],
            event_type="watch_folder_ingested",
            payload={
                "file": str(entry),
                "modality": meta["modality"],
                "patient_id_hash": meta.get("patient_id_hash", ""),
                "accession_number": meta.get("accession_number", ""),
            },
        )
        # 3. 处理成功后归档（带时间戳，避免同批次同名冲突）
        self._move_to(entry, processed_dir)
        return True

    async def _insert_incoming_studies(
        self, meta: dict[str, Any], *, source: str
    ) -> None:
        row = metadata_to_insert_row(meta, source=source)
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO incoming_studies (
                    study_uid, patient_id, patient_id_hash, accession_number,
                    modality, study_date, patient_sex, patient_age,
                    source, status, max_prob,
                    received_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ','now')
                )
                """,
                row,
            )
            await conn.commit()

    async def _record_audit(
        self, *, study_uid: str, event_type: str, payload: dict
    ) -> None:
        if self._audit_hook is not None:
            try:
                await self._audit_hook(study_uid, event_type, payload)
            except Exception:  # noqa: BLE001
                logger.exception("audit_hook 失败（不阻断 watch folder 入条）")
        try:
            async with self.db.connection() as conn:
                await conn.execute(
                    "INSERT INTO audit_log (actor_role, study_uid, event_type, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    ("system", study_uid, event_type, _json.dumps(payload, ensure_ascii=False)),
                )
                await conn.commit()
        except Exception:  # noqa: BLE001
            logger.exception("audit_log 写入失败（不阻断 watch folder 入条）")

    def _move_to(self, src: Path, dest_dir: Path) -> None:
        """把 ``src`` 移到 ``dest_dir``；同名冲突加 ISO 时间戳后缀。"""
        if not src.exists():
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / src.name
        if target.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            target = dest_dir / f"{src.stem}_{ts}{src.suffix}"
        try:
            shutil.move(str(src), str(target))
        except Exception:  # noqa: BLE001
            # 退路：rename / copy
            try:
                src.rename(target)
            except Exception:  # noqa: BLE001
                shutil.copy2(str(src), str(target))
                src.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 调试 / 健康检查
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def build_folder_watcher(
    db: Database,
    *,
    config: Optional[CxrTriageConfig] = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    audit_hook: Optional[Callable[[str, str, dict], Awaitable[None]]] = None,
) -> DicomFolderWatcher:
    """工厂函数：从 ``load_config()`` 构造 ``DicomFolderWatcher``。"""
    from ..config import load_config  # 局部导入避免循环

    cfg = config or load_config()
    return DicomFolderWatcher(
        cfg,
        db,
        poll_interval=poll_interval,
        audit_hook=audit_hook,
    )


__all__ = [
    "DicomFolderWatcher",
    "FolderWatcherError",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "build_folder_watcher",
]
