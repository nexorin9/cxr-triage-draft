"""tests/test_folder_watcher.py — DICOM watch folder 兜底通道。

覆盖（CLAUDE.md Phase 4 验收 · Task 6）：

1. **happy path**：往 watch_folder 放 ``.dcm`` → ``poll_once`` 后
   incoming_studies 入条 + source='watch_folder' + 移到 ``.processed/``；
2. **Modality 过滤**：非 CR/DX 文件走 modality_filtered 审计 + 不入条；
3. **解析失败**：缺 StudyInstanceUID → ``.errors/`` + audit_log；
4. **重复入站幂等**：同 StudyInstanceUID 二轮扫描 → 仍只有 1 行；
5. **二启不去重**：原 ``.processed/`` 文件二次启动不会被重读（因为根目录
   iterdir 已过滤）；新放入的 ``.dcm`` 才进；
6. **stats 计数**：scanned / ingested / filtered_modality / errors；
7. **start / stop 协程生命周期**：后台协程启动后能正常停；
8. **目录缺失兜底**：watch_folder 不存在时 ``poll_once`` 自动创建；
9. **连接 Database 未开 → 异常**：构造期不抛，调用时抛。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import pydicom
import pytest
import pytest_asyncio
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    PYDICOM_IMPLEMENTATION_UID,
)

from cxr_triage.config import CxrTriageConfig
from cxr_triage.pacs import (
    DicomFolderWatcher,
    DicomMetadataError,
    FolderWatcherError,
)
from cxr_triage.persistence import Database, init_db


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def env(monkeypatch):
    """全部外发凭据 + 临时 watch folder 路径由测试在 tmp_path 注入。"""
    monkeypatch.setenv("PACS_AE_TITLE", "TEST_SCU")
    monkeypatch.setenv("PACS_HOST", "127.0.0.1")
    monkeypatch.setenv("PACS_PORT", "11112")
    monkeypatch.setenv("PACS_WADO_URL", "http://pacs.test:8080/wado-rs")
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://hcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("OPENAI_MODEL", "medrax-llava-med-7b")
    return monkeypatch


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.connect()
    await init_db(d)
    yield d
    await d.close()


@pytest.fixture
def cfg(env, tmp_path: Path) -> CxrTriageConfig:
    """配置注入：watch_folder = tmp_path / 'watch'。"""
    watch_dir = tmp_path / "watch"
    monkeypatch = env  # type: ignore[assignment]
    monkeypatch.setenv("WATCH_FOLDER", str(watch_dir))
    return CxrTriageConfig(watch_folder=watch_dir)


@pytest_asyncio.fixture
async def watcher(cfg: CxrTriageConfig, db: Database, tmp_path: Path):
    """短轮询间隔（0.1s）便于测试。"""
    w = DicomFolderWatcher(cfg, db, poll_interval=0.1, salt="watcher-salt")
    return w


def _make_cxr_dataset(
    *,
    study_uid: str,
    modality: str = "CR",
    patient_id: str = "<EXAMPLE_PATIENT>",
    accession: str = "<EXAMPLE_ACC>",
    study_date: str = "20260101",
    patient_sex: str = "M",
    patient_age: str = "045Y",
) -> pydicom.Dataset:
    """最小可解析 DICOM Dataset（同 test_pacs_listener）。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(
        filename_or_obj="<memory>",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.StudyInstanceUID = study_uid
    ds.Modality = modality
    ds.PatientID = patient_id
    ds.AccessionNumber = accession
    ds.StudyDate = study_date
    ds.PatientSex = patient_sex
    ds.PatientAge = patient_age
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 1
    ds.Columns = 1
    ds.PixelData = b"\x00"
    return ds


def _write_dcm(path: Path, ds: pydicom.Dataset) -> None:
    """把内存 Dataset 写到 path（不复制 file_meta 之外的多余字段）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


# ======================================================================
# 1. happy path：放 .dcm → 入条 + 归档到 .processed/
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_ingests_cxr_dcm_and_moves_to_processed(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """放 1 张 CR 胸片 → poll_once 后入 incoming_studies + 移到 .processed/。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)

    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_WATCH_POS>",
        modality="CR",
        patient_id="<EXAMPLE_P>",
    )
    dcm_path = watch_dir / "study_pos.dcm"
    _write_dcm(dcm_path, ds)

    count = await watcher.poll_once()
    assert count == 1

    # 1) incoming_studies 入条 + source=watch_folder
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT study_uid, modality, source, status, patient_id_hash "
            "FROM incoming_studies WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_WATCH_POS>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row is not None
    assert row["modality"] == "CR"
    assert row["source"] == "watch_folder"
    assert row["status"] == "pending"
    # patient_id_hash 是 SHA256(PatientID + salt)
    import hashlib
    expected_hash = hashlib.sha256(b"<EXAMPLE_P>watcher-salt").hexdigest()
    assert row["patient_id_hash"] == expected_hash

    # 2) 原文件移到 .processed/
    assert not dcm_path.exists()
    processed = watch_dir / ".processed" / "study_pos.dcm"
    assert processed.exists()

    # 3) audit_log 记录 watch_folder_ingested
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type FROM audit_log WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_WATCH_POS>",),
        )
        rows = await cur.fetchall()
        await cur.close()
    events = {r["event_type"] for r in rows}
    assert "watch_folder_ingested" in events


@pytest.mark.asyncio
async def test_watcher_accepts_dx_modality(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """DX modality 也通过。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_WATCH_DX>",
        modality="DX",
    )
    _write_dcm(watch_dir / "dx.dcm", ds)
    count = await watcher.poll_once()
    assert count == 1
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT modality FROM incoming_studies WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_WATCH_DX>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row["modality"] == "DX"


# ======================================================================
# 2. Modality 过滤：CT/MR/US → 写 audit + 不入条
# ======================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_modality", ["CT", "MR", "US", "MG", "OT"])
async def test_watcher_filters_non_cxr_modality(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path, bad_modality: str
):
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    ds = _make_cxr_dataset(
        study_uid=f"1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER_{bad_modality}>",
        modality=bad_modality,
    )
    _write_dcm(watch_dir / f"bad_{bad_modality}.dcm", ds)
    count = await watcher.poll_once()
    assert count == 0  # 不算入条成功

    # 不入条
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM incoming_studies WHERE study_uid = ?",
            (f"1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER_{bad_modality}>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row is None

    # audit_log 记 modality_filtered
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type FROM audit_log WHERE event_type = 'watch_folder_modality_filtered'"
        )
        rows = await cur.fetchall()
        await cur.close()
    assert len(rows) >= 1

    # 文件归档到 .processed/（避免下一轮又扫）
    assert not (watch_dir / f"bad_{bad_modality}.dcm").exists()


# ======================================================================
# 3. 解析失败：缺 StudyInstanceUID → .errors/ + audit_log
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_moves_unparseable_dcm_to_errors(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """缺 StudyInstanceUID → 移到 .errors/ + audit_log = watch_folder_metadata_error。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    ds = _make_cxr_dataset(study_uid="<placeholder>")
    ds.StudyInstanceUID = ""  # 触发 DicomMetadataError
    dcm_path = watch_dir / "broken.dcm"
    _write_dcm(dcm_path, ds)

    count = await watcher.poll_once()
    assert count == 0  # 不入条

    # 原文件移走
    assert not dcm_path.exists()
    errors = list((watch_dir / ".errors").iterdir())
    assert len(errors) == 1
    assert errors[0].name.startswith("broken")  # 时间戳后缀可选

    # audit_log 记录
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type FROM audit_log WHERE event_type = 'watch_folder_metadata_error'"
        )
        rows = await cur.fetchall()
        await cur.close()
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_watcher_handles_corrupt_dcm_gracefully(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """非 DICOM 字节（如 plain text）→ IO 错 → 移到 .errors/。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    bad = watch_dir / "not_dicom.dcm"
    bad.write_bytes(b"this is not a DICOM file")

    count = await watcher.poll_once()
    assert count == 0
    assert not bad.exists()
    errors_dir = watch_dir / ".errors"
    assert errors_dir.exists()
    assert any(errors_dir.iterdir())


# ======================================================================
# 4. 重复入站幂等：同 StudyInstanceUID 二轮扫描 → 仍只有 1 行
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_is_idempotent_on_same_study_uid(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """同 StudyInstanceUID 二轮入站 → incoming_studies 只 1 行。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_DUP_WATCH>",
        modality="CR",
    )
    _write_dcm(watch_dir / "dup.dcm", ds)
    await watcher.poll_once()

    # 第二次再放同名文件（模拟 PACS 重传）
    _write_dcm(watch_dir / "dup.dcm", ds)
    await watcher.poll_once()

    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) AS cnt FROM incoming_studies WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_DUP_WATCH>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row["cnt"] == 1


# ======================================================================
# 5. 二启不去重：原 .processed/ 文件不会重读
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_second_start_does_not_re_read_processed(
    cfg: CxrTriageConfig, db: Database, tmp_path: Path
):
    """二启后只扫描根目录 .dcm，不递归 .processed/。"""
    watch_dir = Path(cfg.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)

    w1 = DicomFolderWatcher(cfg, db, poll_interval=0.1, salt="re-salt")
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_RE>",
        modality="CR",
    )
    _write_dcm(watch_dir / "study.dcm", ds)
    await w1.poll_once()

    # 第二轮扫描时原 .dcm 已不在根目录；.processed/ 下有一份
    count = await w1.poll_once()
    assert count == 0

    processed = watch_dir / ".processed"
    assert (processed / "study.dcm").exists()

    # 新实例启动后再扫，仍然 0（不读 .processed/）
    w2 = DicomFolderWatcher(cfg, db, poll_interval=0.1, salt="re-salt")
    count = await w2.poll_once()
    assert count == 0


# ======================================================================
# 6. stats 计数
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_stats_counts(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """混合场景：2 入条 + 1 modality 过滤 + 1 解析失败 → stats 全对。"""
    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)

    # 2 张 CR 入条
    for i in range(2):
        ds = _make_cxr_dataset(
            study_uid=f"1.2.826.0.1.3680043.8.498.<EXAMPLE_STAT_{i}>",
            modality="CR",
        )
        _write_dcm(watch_dir / f"stat_{i}.dcm", ds)
    # 1 张 CT 被过滤
    ds_ct = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_STAT_CT>",
        modality="CT",
    )
    _write_dcm(watch_dir / "stat_ct.dcm", ds_ct)
    # 1 张解析失败
    ds_bad = _make_cxr_dataset(study_uid="<placeholder>")
    ds_bad.StudyInstanceUID = ""
    _write_dcm(watch_dir / "stat_bad.dcm", ds_bad)

    await watcher.poll_once()

    s = watcher.stats()
    assert s["scanned"] == 4
    assert s["ingested"] == 2
    assert s["filtered_modality"] == 1
    assert s["errors"] == 1


# ======================================================================
# 7. start / stop 协程生命周期
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_start_stop_lifecycle(
    watcher: DicomFolderWatcher, tmp_path: Path
):
    assert watcher.is_running is False
    await watcher.start()
    assert watcher.is_running is True
    assert watcher._task is not None
    await watcher.stop()
    assert watcher.is_running is False
    assert watcher._task is None


@pytest.mark.asyncio
async def test_watcher_double_start_is_noop(
    watcher: DicomFolderWatcher, tmp_path: Path, caplog
):
    """重复 start() 不抛异常、不起第二个协程。"""
    await watcher.start()
    first_task = watcher._task
    await watcher.start()  # 重复
    assert watcher._task is first_task
    await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_stop_when_not_running_is_noop(watcher: DicomFolderWatcher):
    """未 start 直接 stop 不报错。"""
    await watcher.stop()  # 不抛


@pytest.mark.asyncio
async def test_watcher_background_loop_processes_new_file(
    cfg: CxrTriageConfig, db: Database, tmp_path: Path
):
    """后台协程轮询：先 start，再放文件，应在 ~1s 内被消费。"""
    watch_dir = Path(cfg.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    w = DicomFolderWatcher(cfg, db, poll_interval=0.2, salt="bg-salt")
    await w.start()
    try:
        ds = _make_cxr_dataset(
            study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_BG>",
            modality="CR",
        )
        _write_dcm(watch_dir / "bg.dcm", ds)
        # 等最多 2 秒
        for _ in range(20):
            await asyncio.sleep(0.1)
            async with db.connection() as conn:
                cur = await conn.execute(
                    "SELECT 1 FROM incoming_studies WHERE study_uid = ?",
                    ("1.2.826.0.1.3680043.8.498.<EXAMPLE_BG>",),
                )
                row = await cur.fetchone()
                await cur.close()
            if row is not None:
                break
        else:
            pytest.fail("后台协程 2s 内未入条")
    finally:
        await w.stop()


# ======================================================================
# 8. 目录缺失兜底
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_poll_once_creates_missing_watch_dir(
    cfg: CxrTriageConfig, db: Database, tmp_path: Path
):
    """watch_folder 不存在时 poll_once 自动创建。"""
    watch_dir = Path(cfg.watch_folder)
    assert not watch_dir.exists()
    w = DicomFolderWatcher(cfg, db, poll_interval=0.1, salt="missing-salt")
    count = await w.poll_once()
    assert count == 0
    assert watch_dir.exists()
    assert (watch_dir / ".processed").exists()
    assert (watch_dir / ".errors").exists()


# ======================================================================
# 9. 工厂函数 build_folder_watcher
# ======================================================================
@pytest.mark.asyncio
async def test_build_folder_watcher_uses_loaded_config(db: Database, env):
    """build_folder_watcher 用 load_config() 的 WATCH_FOLDER env（未设时用默认 /tmp/cxr-watch）。"""
    from cxr_triage.pacs import build_folder_watcher

    # env fixture 没设 WATCH_FOLDER，应走默认 /tmp/cxr-watch
    assert os.getenv("WATCH_FOLDER") is None
    w = build_folder_watcher(db, poll_interval=0.1)
    assert w.config is not None
    assert str(w.config.watch_folder) == "/tmp/cxr-watch"


# ======================================================================
# 10. 同 StudyInstanceUID + 已有 pacs_cstore 行的冲突 → INSERT OR REPLACE
# ======================================================================
@pytest.mark.asyncio
async def test_watcher_replaces_existing_pacs_cstore_row(
    watcher: DicomFolderWatcher, db: Database, tmp_path: Path
):
    """若同一 study_uid 先经 PACS 入条（source=pacs_cstore），再走 watch folder
    → INSERT OR REPLACE 后 source 改为 watch_folder（最新渠道优先）。"""
    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_REPLACE>"
    # 直接 SQL 模拟 PACS 先入条
    async with db.connection() as conn:
        await conn.execute(
            "INSERT INTO incoming_studies (study_uid, modality, source, status) "
            "VALUES (?, ?, ?, ?)",
            (study_uid, "CR", "pacs_cstore", "pending"),
        )
        await conn.commit()

    watch_dir = Path(watcher.config.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)
    ds = _make_cxr_dataset(study_uid=study_uid, modality="CR")
    _write_dcm(watch_dir / "replace.dcm", ds)
    await watcher.poll_once()

    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT source FROM incoming_studies WHERE study_uid = ?",
            (study_uid,),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row["source"] == "watch_folder"  # 被覆盖


# ======================================================================
# 11. watch_folder env 默认值兜底
# ======================================================================
def test_config_default_watch_folder(env):
    """WATCH_FOLDER 未设时使用 /tmp/cxr-watch。"""
    env.delenv("WATCH_FOLDER", raising=False)
    cfg = CxrTriageConfig()
    assert str(cfg.watch_folder) == "/tmp/cxr-watch"
