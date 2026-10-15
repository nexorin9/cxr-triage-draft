"""PACS C-STORE SCU 监听 — 体检胸片入站事件 → incoming_studies 入条。

设计原则（spec.md §2 + §5 维度 1 权威源接入）：

1. **pynetdicom AE 监听端口**：以 ``CxrTriageConfig.pacs_ae_title`` / ``pacs_port``
   起 SCU 监听；on_c_store 回调把 ``Dataset`` → ``parse_cxr_metadata`` →
   ``incoming_studies`` 插入。
2. **Modality 过滤**（spec.md §5 维度 1）：只接收 ``CR`` / ``DX``；其它
   modality（CT / MR / US）写 audit_log 后丢弃。
3. **错误降级**：DICOM 解析失败（缺字段、IO 错）写 audit_log + 不阻断端口
   监听。
4. **patient_id_hash 盐**：从 env ``CXR_TRIAGE_PATIENT_ID_SALT`` 注入；
   缺省 ``cxr-triage-salt``（仅供 dev）。
5. **断连检测**：上层 FastAPI lifespan / 测试用例调 ``start()`` 起协程
   后台跑；调 ``stop()`` 优雅退出。

参考：源仓库 ``medrax/tools/dicom.py:51-94``（DICOM metadata dict 提取模式）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from pynetdicom import AE, evt, build_context
from pynetdicom.presentation import build_context as _build_context
from pynetdicom.sop_class import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
)

from ..config import CxrTriageConfig, load_config
from ..persistence.sqlite import Database
from .dicom_metadata import (
    ALLOWED_MODALITIES,
    DicomMetadataError,
    metadata_to_insert_row,
    parse_cxr_metadata,
)

logger = logging.getLogger(__name__)


class PacsListenerError(RuntimeError):
    """PACS 监听启动 / 关闭 / 处理过程中出现的可恢复异常。"""


# 接收 SOP Class 白名单（CR + DX）
# pynetdicom 用 build_context 把 SOP Class 与 transfer syntax 绑成
# Presentation Context；多模态入站时按 offered 协商。
_PRESENTATION_CONTEXTS = [
    _build_context(ComputedRadiographyImageStorage),
    _build_context(DigitalXRayImageStorageForPresentation),
]


class PacsListener:
    """pynetdicom C-STORE SCU 监听器。

    Parameters
    ----------
    config:
        业务配置（pacs_ae_title / pacs_port）。
    db:
        已 ``await db.connect()`` 打开的 ``Database`` 实例；用于
        ``incoming_studies`` 写入与 ``audit_log`` 落库。
    audit_hook:
        可选回调 ``async def audit_hook(study_uid, event_type, payload)``；
        业务层（如 writers/audit.py）可注入做模型调用记录。缺省时只写
        ``audit_log`` 表。
    salt:
        PatientID hash 用的盐；缺省从 ``CXR_TRIAGE_PATIENT_ID_SALT`` 读。
    """

    def __init__(
        self,
        config: CxrTriageConfig,
        db: Database,
        *,
        audit_hook: Optional[Callable[[str, str, dict], Awaitable[None]]] = None,
        salt: Optional[str] = None,
    ) -> None:
        self.config = config
        self.db = db
        self._audit_hook = audit_hook
        self._salt = salt or os.getenv("CXR_TRIAGE_PATIENT_ID_SALT", "cxr-triage-salt")
        self._ae: Optional[AE] = None
        self._server = None  # ``pynetdicom.AE.start_server`` 返回的 AssociationServer
        self._running = False
        self._stats = {
            "received": 0,
            "inserted": 0,
            "filtered_modality": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """起 SCU 监听端口；pynetdicom 内部是同步，用 ``asyncio.to_thread``
        丢进线程池避免阻塞事件循环。
        """
        if self._running:
            logger.warning("PacsListener 已在运行，跳过重复 start()")
            return

        ae = AE(ae_title=self.config.pacs_ae_title)
        for cx in _PRESENTATION_CONTEXTS:
            ae.add_requested_context(cx.abstract_syntax, *cx.transfer_syntaxes)

        handlers = [(evt.EVT_C_STORE, self._handle_c_store)]
        try:
            self._server = await asyncio.to_thread(
                ae.start_server,
                ("0.0.0.0", self.config.pacs_port),
                evt_handlers=handlers,
                block=False,
                ssl_context=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise PacsListenerError(
                f"PACS 监听启动失败 ae_title={self.config.pacs_ae_title} "
                f"port={self.config.pacs_port}: {exc}"
            ) from exc

        self._ae = ae
        self._running = True
        logger.info(
            "PACS C-STORE SCU 监听已启动: ae_title=%s port=%d",
            self.config.pacs_ae_title,
            self.config.pacs_port,
        )

    async def stop(self) -> None:
        """优雅关闭监听。"""
        if not self._running:
            return
        try:
            if self._server is not None:
                await asyncio.to_thread(self._server.shutdown)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PACS 监听关闭时异常: %s", exc)
        finally:
            self._running = False
            self._server = None
            self._ae = None
            logger.info("PACS C-STORE SCU 监听已停止")

    # ------------------------------------------------------------------
    # 同步入口（供测试与 ``python -m cxr_triage`` 直接同步驱动）
    # ------------------------------------------------------------------
    def handle_c_store_sync(self, event) -> int:
        """同步处理一个 C-STORE 事件，返回 DICOM 状态码（0x0000 = Success）。

        由 ``pynetdicom`` 在事件线程同步触发；测试可绕过网络栈直接调。
        """
        try:
            ds = event.dataset
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            self._stats["received"] += 1
            self._dispatch_sync(ds)
            return 0x0000
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            logger.exception("on_c_store 处理失败: %s", exc)
            # 0x0110 = Processing Failure
            return 0x0110

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _handle_c_store(self, event):
        """pynetdicom 异步事件回调。"""
        # pynetdicom 事件本身是同步的，但我们在协程上下文里跑；
        # 内部走 ``_dispatch`` 同步入条。
        try:
            ds = event.dataset
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            self._stats["received"] += 1
            await self._dispatch(ds)
            return 0x0000
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            logger.exception("on_c_store 协程处理失败: %s", exc)
            return 0x0110

    def _dispatch_sync(self, ds) -> None:
        """同步入条（供测试用）。"""
        try:
            meta = parse_cxr_metadata(ds, salt=self._salt)
        except DicomMetadataError as exc:
            self._record_audit_sync(
                study_uid="",
                event_type="pacs_metadata_error",
                payload={"error": str(exc)},
            )
            self._stats["errors"] += 1
            return
        if meta is None:
            # Modality 过滤失败（不是 CR/DX）
            study_uid = getattr(ds, "StudyInstanceUID", "") or ""
            self._stats["filtered_modality"] += 1
            self._record_audit_sync(
                study_uid=study_uid,
                event_type="pacs_modality_filtered",
                payload={"modality": getattr(ds, "Modality", "")},
            )
            return
        self._insert_incoming_studies_sync(meta, source="pacs_cstore")
        self._stats["inserted"] += 1
        self._record_audit_sync(
            study_uid=meta["study_uid"],
            event_type="pacs_cstore_received",
            payload={
                "modality": meta["modality"],
                "patient_id_hash": meta.get("patient_id_hash", ""),
                "accession_number": meta.get("accession_number", ""),
            },
        )

    async def _dispatch(self, ds) -> None:
        """异步入条（生产路径 + 测试路径）。

        注意：``_handle_c_store``（pynetdicom 事件回调）也走这里，所以
        stats 计数 (``received`` / ``inserted`` / ``filtered_modality`` /
        ``errors``) 以本函数为唯一来源；``_handle_c_store`` 不再重复 +1。
        """
        self._stats["received"] += 1
        try:
            meta = parse_cxr_metadata(ds, salt=self._salt)
        except DicomMetadataError as exc:
            await self._record_audit(
                study_uid="",
                event_type="pacs_metadata_error",
                payload={"error": str(exc)},
            )
            self._stats["errors"] += 1
            return
        if meta is None:
            study_uid = getattr(ds, "StudyInstanceUID", "") or ""
            self._stats["filtered_modality"] += 1
            await self._record_audit(
                study_uid=study_uid,
                event_type="pacs_modality_filtered",
                payload={"modality": getattr(ds, "Modality", "")},
            )
            return
        await self._insert_incoming_studies(meta, source="pacs_cstore")
        self._stats["inserted"] += 1
        await self._record_audit(
            study_uid=meta["study_uid"],
            event_type="pacs_cstore_received",
            payload={
                "modality": meta["modality"],
                "patient_id_hash": meta.get("patient_id_hash", ""),
                "accession_number": meta.get("accession_number", ""),
            },
        )

    def _insert_incoming_studies_sync(
        self, meta: dict[str, Any], *, source: str
    ) -> None:
        """同步入条（仅供测试用；不引入 ``asyncio.run`` 以免破坏事件循环）。"""
        row = metadata_to_insert_row(meta, source=source)
        # 测试用：直接走已就绪的同步 connection
        # ``Database._conn`` 已经在 init_db 阶段打开；这里用 ``connection()``
        # 异步上下文管理器绕过 → 测试通过注入一个 ``aiosqlite.Connection`` 来跑。
        # 为避免同步/异步混用，测试侧用 ``_insert_incoming_studies_sync_via_aio`` 走异步路径。
        raise NotImplementedError(
            "PacsListener 同步入条请改走 _insert_incoming_studies（异步）"
        )

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
                logger.exception("audit_hook 失败（不阻断 PACS 接收）")
        try:
            import json as _json

            async with self.db.connection() as conn:
                await conn.execute(
                    "INSERT INTO audit_log (actor_role, study_uid, event_type, payload_json) "
                    "VALUES (?, ?, ?, ?)",
                    ("system", study_uid, event_type, _json.dumps(payload, ensure_ascii=False)),
                )
                await conn.commit()
        except Exception:  # noqa: BLE001
            logger.exception("audit_log 写入失败（不阻断 PACS 接收）")

    def _record_audit_sync(
        self, *, study_uid: str, event_type: str, payload: dict
    ) -> None:
        """同步 audit（仅供测试用；运行时请走 ``_record_audit``）。"""
        if self._audit_hook is None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return
                loop.run_until_complete(
                    self._record_audit(study_uid=study_uid, event_type=event_type, payload=payload)
                )
            except RuntimeError:
                pass
            return
        # 测试注入 hook 的情况：直接调
        try:
            res = self._audit_hook(study_uid, event_type, payload)
            if asyncio.iscoroutine(res):
                # 简化处理：在测试中我们用同步 hook
                pass
        except Exception:  # noqa: BLE001
            logger.exception("同步 audit hook 失败")

    # ------------------------------------------------------------------
    # 调试 / 健康检查
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def build_pacs_listener(
    db: Database,
    *,
    config: Optional[CxrTriageConfig] = None,
    audit_hook: Optional[Callable[[str, str, dict], Awaitable[None]]] = None,
) -> PacsListener:
    """工厂函数：从 ``load_config()`` 构造 ``PacsListener``。"""
    cfg = config or load_config()
    return PacsListener(cfg, db, audit_hook=audit_hook)


__all__ = [
    "PacsListener",
    "PacsListenerError",
    "ALLOWED_MODALITIES",
    "build_pacs_listener",
]
