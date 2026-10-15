"""pacs — PACS DICOM 适配层（C-STORE SCU 监听 + WADO-RS 取图 + 窗宽窗位）。

公开入口（按 task 顺序渐进加入）：

- ``parse_cxr_metadata``     DICOM Dataset → dict（StudyInstanceUID/PatientID/...）
- ``PacsListener``            pynetdicom SCU 监听器（on_c_store 回调）
- ``PacsListenerError``       启动 / 协商 / 解析异常
- ``ALLOWED_MODALITIES``      接收 modality 白名单（CR / DX）
- ``DicomMetadataError``      关键字段缺失异常
- ``WadoClient``              WADO-RS 异步取图客户端（T5）
- ``WadoFetchError``          WADO 取图失败异常（T5）
- ``DicomProcessor``          调窗 + PNG 编码处理器（T5）
- ``DicomProcessError``       调窗 / 编码失败异常（T5）
- ``DicomProcessResult``      调窗产物（png_bytes + metadata）（T5）
- ``DicomFolderWatcher``      PACS 协商失败时的文件轮询兜底（T6）
- ``FolderWatcherError``      watch folder 启动 / 关闭异常（T6）

后续 task 会加入：triage_queue / state_machine 等业务模块。
"""
from __future__ import annotations

from .dicom_metadata import (
    ALLOWED_MODALITIES,
    DicomMetadataError,
    metadata_to_insert_row,
    parse_cxr_metadata,
    patient_id_hash,
)
from .dicom_processor import (
    DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH,
    DicomProcessError,
    DicomProcessResult,
    DicomProcessor,
    build_dicom_processor,
)
from .folder_watcher import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DicomFolderWatcher,
    FolderWatcherError,
    build_folder_watcher,
)
from .listener import PacsListener, PacsListenerError
from .wado import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    WadoClient,
    WadoFetchError,
    build_wado_client,
)

__all__ = [
    # metadata
    "ALLOWED_MODALITIES",
    "DicomMetadataError",
    "parse_cxr_metadata",
    "patient_id_hash",
    "metadata_to_insert_row",
    # listener
    "PacsListener",
    "PacsListenerError",
    # wado (T5)
    "WadoClient",
    "WadoFetchError",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "build_wado_client",
    # dicom processor (T5)
    "DicomProcessor",
    "DicomProcessError",
    "DicomProcessResult",
    "DEFAULT_WINDOW_WIDTH",
    "DEFAULT_WINDOW_LEVEL",
    "build_dicom_processor",
    # folder watcher (T6)
    "DicomFolderWatcher",
    "FolderWatcherError",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "build_folder_watcher",
]
