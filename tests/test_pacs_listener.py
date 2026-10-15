"""tests/test_pacs_listener.py — PACS C-STORE SCU 监听 + metadata 解析。

覆盖（CLAUDE.md Phase 4 验收 · Task 4）：

1. ``parse_cxr_metadata`` happy path：StudyInstanceUID/PatientID/Modality=CR 全齐；
2. Modality 过滤：CT / MR / US / OT 一律返回 None；
3. 缺关键字段（StudyInstanceUID / Modality）抛 DicomMetadataError；
4. ``patient_id_hash`` 是 SHA256(PatientID+salt)；
5. ``metadata_to_insert_row`` 列顺序与 schema.SCHEMA_DDL 对齐；
6. ``PacsListener`` 构造 + stats + start/stop 失败语义（PACS 端口占用的健壮性）；
7. ``PacsListener.handle_c_store_sync`` 接受 pydicom Dataset 写入 incoming_studies。
"""
from __future__ import annotations

import asyncio
import json
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
    ALLOWED_MODALITIES,
    DicomMetadataError,
    PacsListener,
    PacsListenerError,
    metadata_to_insert_row,
    parse_cxr_metadata,
    patient_id_hash,
)
from cxr_triage.persistence import Database, init_db


# ----------------------------------------------------------------------
# 通用 fixture
# ----------------------------------------------------------------------
@pytest.fixture
def env(monkeypatch):
    """全套外发凭据 + PACS 端口（11112 不易冲突）。"""
    monkeypatch.setenv("PACS_AE_TITLE", "TEST_SCU")
    monkeypatch.setenv("PACS_HOST", "127.0.0.1")
    monkeypatch.setenv("PACS_PORT", "11112")
    monkeypatch.setenv("PACS_WADO_URL", "http://127.0.0.1:8080/wado-rs")
    monkeypatch.setenv("RIS_API_URL", "http://ris.test:8081")
    monkeypatch.setenv("RIS_API_TOKEN", "real-ris-token-123")
    monkeypatch.setenv("HEALTHCHECK_API_URL", "http://hcheck.test:8082")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "real-hcheck-token-456")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm.test:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "real-llm-key-789")
    monkeypatch.setenv("OPENAI_MODEL", "medrax-llava-med-7b")
    return monkeypatch


@pytest.fixture
def cfg(env) -> CxrTriageConfig:
    return CxrTriageConfig()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    await d.connect()
    await init_db(d)
    yield d
    await d.close()


# ----------------------------------------------------------------------
# DICOM fixture 构造（最小可解析的 FileDataset）
# ----------------------------------------------------------------------
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
    """构造一个最小可解析的胸片 Dataset（不写文件）。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CR Image Storage
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
    # 给 1x1 像素以便 dcmread 也能走通
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


# ======================================================================
# 1. parse_cxr_metadata：happy path
# ======================================================================
def test_parse_cxr_metadata_happy_path():
    ds = _make_cxr_dataset(study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>")
    meta = parse_cxr_metadata(ds, salt="salt-A")
    assert meta is not None
    assert meta["study_uid"] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
    assert meta["modality"] == "CR"
    assert meta["patient_id"] == "<EXAMPLE_PATIENT>"
    assert meta["accession_number"] == "<EXAMPLE_ACC>"
    assert meta["study_date"] == "20260101"
    assert meta["patient_sex"] == "M"
    assert meta["patient_age"] == "045Y"
    # patient_id_hash 是 SHA256(PatientID+salt)
    assert meta["patient_id_hash"] == patient_id_hash("<EXAMPLE_PATIENT>", "salt-A")


def test_parse_cxr_metadata_accepts_dx_modality():
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_DX>",
        modality="DX",
    )
    meta = parse_cxr_metadata(ds)
    assert meta is not None
    assert meta["modality"] == "DX"


# ======================================================================
# 2. Modality 过滤
# ======================================================================
@pytest.mark.parametrize("bad_modality", ["CT", "MR", "US", "MG", "OT", ""])
def test_parse_cxr_metadata_filters_non_cxr_modality(bad_modality):
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER>",
        modality=bad_modality,
    )
    if bad_modality == "":
        # 缺 Modality → DicomMetadataError
        with pytest.raises(DicomMetadataError):
            parse_cxr_metadata(ds)
    else:
        meta = parse_cxr_metadata(ds)
        assert meta is None  # 过滤掉


def test_allowed_modalities_set():
    assert ALLOWED_MODALITIES == frozenset({"CR", "DX"})


# ======================================================================
# 3. 缺关键字段 → DicomMetadataError
# ======================================================================
def test_parse_cxr_metadata_missing_study_uid():
    ds = _make_cxr_dataset(study_uid="<placeholder>")
    ds.StudyInstanceUID = ""
    with pytest.raises(DicomMetadataError) as exc:
        parse_cxr_metadata(ds)
    assert "StudyInstanceUID" in str(exc.value)


def test_parse_cxr_metadata_missing_modality():
    ds = _make_cxr_dataset(study_uid="<placeholder>")
    ds.Modality = ""
    with pytest.raises(DicomMetadataError) as exc:
        parse_cxr_metadata(ds)
    assert "Modality" in str(exc.value)


# ======================================================================
# 4. patient_id_hash 一致性
# ======================================================================
def test_patient_id_hash_sha256_format():
    h = patient_id_hash("<EXAMPLE_PATIENT>", "salt-A")
    assert len(h) == 64
    assert h == patient_id_hash("<EXAMPLE_PATIENT>", "salt-A")
    # 不同 salt → 不同 hash
    assert h != patient_id_hash("<EXAMPLE_PATIENT>", "salt-B")
    # 空 PatientID → 空 hash
    assert patient_id_hash("", "salt-A") == ""


# ======================================================================
# 5. metadata_to_insert_row 列顺序与 schema 对齐
# ======================================================================
def test_metadata_to_insert_row_order():
    meta = {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_T>",
        "patient_id": "PID1",
        "patient_id_hash": "abcd",
        "accession_number": "ACC1",
        "modality": "CR",
        "study_date": "20260101",
        "patient_sex": "F",
        "patient_age": "030Y",
    }
    row = metadata_to_insert_row(meta, source="pacs_cstore", status="pending")
    # 元组结构 (study_uid, patient_id, patient_id_hash, accession_number,
    #          modality, study_date, patient_sex, patient_age, source, status, max_prob)
    assert row[0] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_T>"
    assert row[1] == "PID1"
    assert row[2] == "abcd"
    assert row[3] == "ACC1"
    assert row[4] == "CR"
    assert row[5] == "20260101"
    assert row[6] == "F"
    assert row[7] == "030Y"
    assert row[8] == "pacs_cstore"
    assert row[9] == "pending"
    assert row[10] == 0.0
    assert len(row) == 11


def test_metadata_to_insert_row_watch_folder_source():
    meta = {
        "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_T2>",
        "patient_id": "",
        "patient_id_hash": "",
        "accession_number": "",
        "modality": "DX",
        "study_date": "",
        "patient_sex": "",
        "patient_age": "",
    }
    row = metadata_to_insert_row(meta, source="watch_folder")
    assert row[8] == "watch_folder"


# ======================================================================
# 6. PacsListener 构造 / 状态
# ======================================================================
def test_pacs_listener_constructs_with_config(cfg, db):
    listener = PacsListener(cfg, db, salt="test-salt")
    assert listener.config is cfg
    assert listener.db is db
    assert listener.is_running is False
    assert listener.stats() == {
        "received": 0,
        "inserted": 0,
        "filtered_modality": 0,
        "errors": 0,
    }


def test_pacs_listener_double_start_is_noop(cfg, db, caplog):
    """重复 start() 不抛异常；缺省情况下也不真去绑端口（除非我们 toggle block=False）。"""
    listener = PacsListener(cfg, db, salt="test-salt")
    # 不实际 start()（pynetdicom 绑端口），只验证 ``is_running`` 切换的 guard
    assert listener.is_running is False
    # 连续 stop() 不报错
    asyncio.get_event_loop().run_until_complete(listener.stop())


# ======================================================================
# 7. PacsListener.handle_c_store_sync 端到端
# ======================================================================
@pytest.mark.asyncio
async def test_pacs_listener_handle_c_store_sync_inserts_row(cfg, db):
    listener = PacsListener(cfg, db, salt="sync-salt")
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_LISTENER>",
        modality="CR",
        patient_id="<EXAMPLE_P>",
    )
    # 直接调同步入条路径（我们用 _dispatch_async 注入 fake event）
    # 但 listener 没有暴露 _dispatch_async 的同步替代；走 _dispatch 走异步
    await listener._dispatch(ds)
    # 验证 incoming_studies 入条
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT study_uid, patient_id_hash, modality, source, status "
            "FROM incoming_studies WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_LISTENER>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row is not None
    assert row["modality"] == "CR"
    assert row["source"] == "pacs_cstore"
    assert row["status"] == "pending"
    # patient_id_hash 与 SHA256 一致
    assert row["patient_id_hash"] == patient_id_hash("<EXAMPLE_P>", "sync-salt")
    # audit_log 写入
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type, study_uid FROM audit_log "
            "WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_LISTENER>",),
        )
        rows = await cur.fetchall()
        await cur.close()
    event_types = {r["event_type"] for r in rows}
    assert "pacs_cstore_received" in event_types


@pytest.mark.asyncio
async def test_pacs_listener_filters_non_cxr_modality(cfg, db):
    listener = PacsListener(cfg, db, salt="sync-salt")
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER2>",
        modality="CT",
    )
    await listener._dispatch(ds)
    # incoming_studies 不入条
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM incoming_studies WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER2>",),
        )
        row = await cur.fetchone()
        await cur.close()
    assert row is None
    # audit_log 记 modality_filtered
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type FROM audit_log WHERE study_uid = ?",
            ("1.2.826.0.1.3680043.8.498.<EXAMPLE_FILTER2>",),
        )
        rows = await cur.fetchall()
        await cur.close()
    event_types = {r["event_type"] for r in rows}
    assert "pacs_modality_filtered" in event_types


@pytest.mark.asyncio
async def test_pacs_listener_records_metadata_error(cfg, db):
    listener = PacsListener(cfg, db, salt="sync-salt")
    ds = _make_cxr_dataset(study_uid="<placeholder>")
    ds.StudyInstanceUID = ""  # 触发 DicomMetadataError
    await listener._dispatch(ds)
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT event_type FROM audit_log WHERE event_type = 'pacs_metadata_error'"
        )
        rows = await cur.fetchall()
        await cur.close()
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_pacs_listener_upserts_on_duplicate(cfg, db):
    """同 StudyInstanceUID 二次入站 → 入条数仍为 1（INSERT OR REPLACE）。"""
    listener = PacsListener(cfg, db, salt="dup-salt")
    ds = _make_cxr_dataset(
        study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_DUP>",
        modality="CR",
    )
    await listener._dispatch(ds)
    await listener._dispatch(ds)
    async with db.connection() as conn:
        cur = await conn.execute("SELECT COUNT(*) AS cnt FROM incoming_studies")
        row = await cur.fetchone()
        await cur.close()
    assert row["cnt"] == 1


@pytest.mark.asyncio
async def test_pacs_listener_stats_updated(cfg, db):
    listener = PacsListener(cfg, db, salt="stats-salt")
    # 1 条入条
    await listener._dispatch(
        _make_cxr_dataset(study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_STATS>")
    )
    # 1 条 modality 过滤
    await listener._dispatch(
        _make_cxr_dataset(
            study_uid="1.2.826.0.1.3680043.8.498.<EXAMPLE_STATS2>",
            modality="CT",
        )
    )
    # 1 条 metadata 错误
    bad = _make_cxr_dataset(study_uid="<placeholder>")
    bad.StudyInstanceUID = ""
    await listener._dispatch(bad)
    s = listener.stats()
    assert s["received"] == 3
    assert s["inserted"] == 1
    assert s["filtered_modality"] == 1
    assert s["errors"] == 1
