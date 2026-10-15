"""tests/test_trigger.py — Task 18 TriageTrigger 单测。

目标（spec.md §5 维度 1 + plan.md §5 深度 (a)）：

1. **happy-path（mock 全套）**：
   - 拉 1 条 status=pending 行 → StateGraph 跑出阳性草稿 → 入
     triage_queue + 推 SSE + 标 processed。
2. **幂等命中**：draft_idempotency 表已有 (uid, tool_version) →
   不重跑 StateGraph → 直接入队 + 推 SSE + 标 processed +
   audit ``trigger.idempotent_hit``。
3. **失败路径**：
   - WadoFetchError → failed_retry + audit ``trigger.wado_failed``；
   - DicomProcessError → failed_retry + audit
     ``trigger.dicom_process_failed``；
   - WorkflowError → failed_retry + audit ``trigger.workflow_failed``；
   - enqueue 异常 → failed_retry + audit ``trigger.exception``；
4. **降级路径**：StateGraph 跑出 degraded=True → 仍入队 +
   audit ``trigger.processed_degraded`` + SSE payload 含 degradation_reasons。
5. **process_pending 批量**：3 条 pending 一次性跑完，1 条异常不影响其它。
6. **run_pending_loop 协程**：start() → 跑一轮 → stop() 优雅退出；
   start 重复幂等；未 start 直接 stop 幂等。
7. **入参校验**：config 类型错 / poll_interval 越界 / pending_batch_size
   非正整数 / 缺 wado_client → TriageTriggerError。
8. **factory**：build_triage_trigger / build_triage_trigger_from_env。
9. **stats**：scanned / idempotent_hits / processed / degraded / wado_failed
   / dicom_process_failed / workflow_failed / exceptions 全对上。
10. **audit 不阻断**：audit_logger=None 时不抛；audit_logger 抛异常不阻断主路径。
11. **SSE 推送**：sse_publish 接受正确参数；缺省 / 异常 均不阻断。
12. **incoming_studies.status 推进**：pending → processed / failed_retry
    （不删除行，便于飞检回放）。
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cxr_triage.config import CxrTriageConfig
from cxr_triage.engine import (
    EVENT_TRIGGER_DICOM_PROCESS_FAILED,
    EVENT_TRIGGER_IDEMPOTENT_HIT,
    EVENT_TRIGGER_PROCESSED,
    EVENT_TRIGGER_PROCESSED_DEGRADED,
    EVENT_TRIGGER_STARTED,
    EVENT_TRIGGER_WADO_FAILED,
    EVENT_TRIGGER_WORKFLOW_FAILED,
    STATUS_FAILED_RETRY,
    STATUS_PENDING,
    STATUS_PROCESSED,
    TriageTrigger,
    TriageTriggerError,
    build_triage_trigger,
    build_triage_trigger_from_env,
)
from cxr_triage.engine.workflow import CxrTriageWorkflow, ToolRegistry, WorkflowError
from cxr_triage.engine.triage_queue import TriageQueue
from cxr_triage.pacs.dicom_processor import DicomProcessError, DicomProcessResult
from cxr_triage.pacs.wado import WadoFetchError
from cxr_triage.persistence.idempotency import IDEMPOTENCY_TTL_SECONDS
from cxr_triage.persistence.migrations import init_db
from cxr_triage.persistence.sqlite import Database
from cxr_triage.state.idempotency import StateIdempotencyStore
from cxr_triage.writers.audit import AuditLogger


# ---------------------------------------------------------------------------
# Fixtures + 工具
# ---------------------------------------------------------------------------


VALID_DCM_BYTES = b"DICM" + b"\x00" * 240  # 占位
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 240  # PNG magic + 占位
)


def _make_png_bytes() -> bytes:
    """合法 PNG 字节流（最小合法 IHDR + IDAT + IEND 块）。"""
    import io

    from PIL import Image

    img = Image.new("L", (8, 8), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_config(**overrides) -> CxrTriageConfig:
    """构造最小合法 CxrTriageConfig（不走 env 验证；用 model_construct 跳过）。"""
    base = {
        "pacs_ae_title": "TEST",
        "pacs_host": "127.0.0.1",
        "pacs_port": 11112,
        "pacs_wado_url": "http://test.local/wado",
        "pacs_wado_token": "test-wado-token",
        "ris_api_url": "http://test.local/ris",
        "ris_api_token": "test-ris-token",
        "healthcheck_api_url": "http://test.local/healthcheck",
        "healthcheck_api_token": "test-hc-token",
        "openai_api_key": "test-openai-key",
        "openai_base_url": "http://test.local/openai",
        "openai_model": "test-model",
        "watch_folder": "/tmp/cxr-watch-test",
        "positive_threshold": 0.5,
    }
    base.update(overrides)
    return CxrTriageConfig.model_construct(**base)


class MockTool:
    """duck-typed 工具 stub（与 test_state_workflow 一致）。"""

    def __init__(self, name: str, outputs: dict | None = None):
        self.name = name
        self._outputs = dict(outputs or {})

    async def run(self, state: dict) -> dict:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self._outputs.items()}


def _build_workflow(classifier_outputs: dict | None = None) -> CxrTriageWorkflow:
    """构造一个含 5 件 mock 工具的 workflow。"""
    if classifier_outputs is None:
        classifier_outputs = {
            "classifier_probs": [
                {"name": "Pneumothorax", "prob": 0.85, "present": True},
            ],
            "classifier_max_prob": 0.85,
        }
    reg = ToolRegistry()
    reg.register(MockTool("classifier", outputs=classifier_outputs))
    reg.register(MockTool("keyqa", outputs={
        "key_qa_list": [
            {"q": "心影", "a": "心影未见明显增大"},
        ],
    }))
    reg.register(MockTool("finding_draft", outputs={
        "finding_text": "右下肺野见少量气胸征象（草稿 / 未经医师确认），余肺野清晰。",
        "finding_struct": {
            "pathologies": [{"name": "Pneumothorax", "prob": 0.85, "present": True}],
            "key_qa": [],
            "grounding_boxes": [],
        },
    }))
    reg.register(MockTool("positive_grounding", outputs={
        "grounding_boxes": [
            {"x": 0.12, "y": 0.18, "w": 0.15, "h": 0.10, "label": "右下肺野", "prob": 0.81},
        ],
    }))
    reg.register(MockTool("image_preview", outputs={
        "png_preview_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    }))
    wf = CxrTriageWorkflow(reg, positive_threshold=0.5)
    wf.build_graph()
    return wf


def _build_degraded_workflow() -> CxrTriageWorkflow:
    """构造降级 workflow（classifier 抛异常）。"""

    class FailingClassifier(MockTool):
        def __init__(self):
            super().__init__("classifier")

        async def run(self, state: dict) -> dict:
            raise RuntimeError("classifier_unreachable_mock")

    reg = ToolRegistry()
    reg.register(FailingClassifier())
    reg.register(MockTool("keyqa"))
    reg.register(MockTool("finding_draft"))
    reg.register(MockTool("positive_grounding"))
    reg.register(MockTool("image_preview", outputs={"png_preview_b64": "x"}))
    wf = CxrTriageWorkflow(reg, positive_threshold=0.5)
    wf.build_graph()
    return wf


async def _seed_incoming(
    db: Database, study_uid: str, **overrides
) -> None:
    """往 incoming_studies 插一条 status=pending 行。"""
    row = {
        "study_uid": study_uid,
        "patient_id": overrides.get("patient_id", "<EXAMPLE_PATIENT>"),
        "patient_id_hash": overrides.get(
            "patient_id_hash", "deadbeef" * 8
        ),
        "accession_number": overrides.get("accession_number", "<EXAMPLE_ACC>"),
        "modality": overrides.get("modality", "CR"),
        "study_date": overrides.get("study_date", "2026-10-13"),
        "patient_sex": overrides.get("patient_sex", "F"),
        "patient_age": overrides.get("patient_age", "052Y"),
        "source": overrides.get("source", "pacs_cstore"),
        "status": STATUS_PENDING,
        "max_prob": 0.0,
    }
    async with db.connection() as conn:
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
            (
                row["study_uid"],
                row["patient_id"],
                row["patient_id_hash"],
                row["accession_number"],
                row["modality"],
                row["study_date"],
                row["patient_sex"],
                row["patient_age"],
                row["source"],
                row["status"],
                row["max_prob"],
            ),
        )
        await conn.commit()


async def _get_incoming_status(db: Database, study_uid: str) -> str | None:
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT status FROM incoming_studies WHERE study_uid=?", (study_uid,)
        )
        row = await cur.fetchone()
        await cur.close()
    return row["status"] if row else None


class FakeWado:
    """WadoClient 替身：可配置返 dcm_bytes / 抛 WadoFetchError / 抛其他异常。"""

    def __init__(self, dcm_bytes: bytes = VALID_DCM_BYTES, raise_exc: Exception | None = None):
        self._dcm_bytes = dcm_bytes
        self._raise = raise_exc
        self.calls: list[str] = []

    async def fetch(self, study_uid: str) -> bytes:
        self.calls.append(study_uid)
        if self._raise is not None:
            raise self._raise
        return self._dcm_bytes

    async def close(self) -> None:
        pass


class FakeDicomProcessor:
    """DicomProcessor 替身：可配置返 png_bytes / 抛 DicomProcessError。"""

    def __init__(
        self,
        png_bytes: bytes | None = None,
        dcm_metadata: dict | None = None,
        raise_exc: Exception | None = None,
    ):
        self._png_bytes = png_bytes if png_bytes is not None else _make_png_bytes()
        self._dcm_metadata = dcm_metadata or {
            "rows": 256,
            "columns": 256,
            "bits_stored": 8,
            "window_width": 400.0,
            "window_level": 40.0,
        }
        self._raise = raise_exc
        self.calls: list[bytes] = []

    def process(self, dcm_bytes: bytes) -> DicomProcessResult:
        self.calls.append(dcm_bytes)
        if self._raise is not None:
            raise self._raise
        return DicomProcessResult(
            png_bytes=self._png_bytes,
            png_width=8,
            png_height=8,
            metadata=dict(self._dcm_metadata),
        )


@pytest.fixture
async def db_setup(tmp_path):
    """起一个 tmp_path 的 Database + 4 表 + 幂等表 + 状态机表。"""
    db = Database(tmp_path / "test-trigger.db")
    await db.connect()
    await init_db(db)
    yield db
    await db.close()


@pytest.fixture
async def db_and_components(db_setup):
    """db + workflow + triage_queue + idempotency 一套基础组件。"""
    config = _make_config()
    wf = _build_workflow()
    queue = TriageQueue(db_setup)
    idem = StateIdempotencyStore(db_setup, ttl_seconds=IDEMPOTENCY_TTL_SECONDS)
    yield {
        "db": db_setup,
        "config": config,
        "workflow": wf,
        "queue": queue,
        "idempotency": idem,
    }


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


class TestTriggerConstructor:
    def test_invalid_config_type_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config="not-a-config",
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
            )

    def test_invalid_db_type_raises(self):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db="not-a-db",
                workflow=_build_workflow(),
                triage_queue=MagicMock(),
                idempotency=MagicMock(),
            )

    def test_invalid_workflow_type_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow="not-a-workflow",
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
            )

    def test_invalid_triage_queue_type_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue="not-a-queue",
                idempotency=StateIdempotencyStore(db_setup),
            )

    def test_invalid_idempotency_type_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency="not-an-idempotency",
            )

    def test_invalid_poll_interval_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                poll_interval_seconds=0.0,
            )
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                poll_interval_seconds=1000.0,
            )

    def test_invalid_pending_batch_size_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                pending_batch_size=0,
            )
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                pending_batch_size="not-int",
            )

    def test_invalid_sse_publish_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                sse_publish="not-callable",
            )

    def test_invalid_audit_logger_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                audit_logger="not-an-audit",
            )

    def test_invalid_wado_client_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                wado_client="not-a-wado",
            )

    def test_invalid_dicom_processor_raises(self, db_setup):
        with pytest.raises(TriageTriggerError):
            TriageTrigger(
                config=_make_config(),
                db=db_setup,
                workflow=_build_workflow(),
                triage_queue=TriageQueue(db_setup),
                idempotency=StateIdempotencyStore(db_setup),
                dicom_processor="not-a-dp",
            )

    def test_default_construction(self, db_and_components):
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        assert trigger.poll_interval_seconds == 5.0
        assert trigger.pending_batch_size == 50
        assert not trigger.is_running
        stats = trigger.stats()
        assert stats["scanned"] == 0
        assert stats["processed"] == 0


# ---------------------------------------------------------------------------
# happy-path
# ---------------------------------------------------------------------------


class TestTriggerHappyPath:
    async def test_process_one_happy_path(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>")
        sse_calls: list[tuple[str, dict]] = []

        async def sse(study_uid, payload):
            sse_calls.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=sse,
        )

        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>")
        assert ok is True

        # 1) status → processed
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
        ) == STATUS_PROCESSED

        # 2) triage_queue 入条
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 1
        assert pending[0]["study_uid"] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
        assert pending[0]["positive_flag"] == 1
        assert "气胸征象" in pending[0]["finding_text"]

        # 3) idempotency 缓存
        cached = await db_and_components["idempotency"].lookup(
            "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
        )
        assert cached is not None
        assert cached["classifier_max_prob"] == 0.85
        assert "tools_executed" in cached

        # 4) SSE 推送
        assert len(sse_calls) == 1
        assert sse_calls[0][0] == "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>"
        assert sse_calls[0][1]["max_prob"] == 0.85
        assert sse_calls[0][1]["reused_from_idempotency"] is False

        # 5) stats
        stats = trigger.stats()
        assert stats["processed"] == 1
        assert stats["idempotent_hits"] == 0

    async def test_process_one_negative_path(self, db_and_components):
        """阴性：max_prob=0.10 < 阈值 → tools_to_execute=[image_preview] only。"""
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>")
        # 重置 workflow 用 negative classifier
        wf_neg = _build_workflow(classifier_outputs={
            "classifier_probs": [
                {"name": "Pneumothorax", "prob": 0.10, "present": False},
            ],
            "classifier_max_prob": 0.10,
        })
        sse_calls: list = []

        async def sse(study_uid, payload):
            sse_calls.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=wf_neg,
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=sse,
        )

        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_NEG>")
        assert ok is True
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 1
        assert pending[0]["positive_flag"] == 0
        assert pending[0]["max_prob"] == pytest.approx(0.10)
        # 阴性也要进队（spec.md §2：阴性也入列；浏览器第二栏）
        assert sse_calls[0][1]["positive_flag"] is False

    async def test_process_one_degraded_path(self, db_and_components):
        """降级：classifier 抛异常 → degraded=True → 仍入队 + degraded audit。"""
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_DEG>")
        wf_deg = _build_degraded_workflow()
        sse_calls: list = []

        async def sse(study_uid, payload):
            sse_calls.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=wf_deg,
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=sse,
        )

        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_DEG>")
        assert ok is True
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_DEG>"
        ) == STATUS_PROCESSED
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 1
        # degraded=True → 入队时 max_prob=0.0
        assert pending[0]["max_prob"] == pytest.approx(0.0)
        # SSE payload 含 degradation_reasons
        assert sse_calls[0][1]["degraded"] is True
        assert any("classifier" in r for r in sse_calls[0][1]["degradation_reasons"])
        # stats: degraded=1
        assert trigger.stats()["degraded"] == 1

    async def test_process_one_idempotent_hit(self, db_and_components):
        """幂等命中：先 cache_draft → process_one 命中 → 不重跑 StateGraph。"""
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_IDEM>"
        await _seed_incoming(db_and_components["db"], uid)

        # 预填 idempotency
        cached = {
            "study_uid": uid,
            "classifier_probs": [{"name": "Pneumothorax", "prob": 0.88, "present": True}],
            "classifier_max_prob": 0.88,
            "positive_flag": True,
            "finding_text": "右下肺野见少量气胸征象（草稿），余肺野清晰。",
            "tool_version": "0.1.0",
        }
        await db_and_components["idempotency"].cache_draft(uid, cached)

        sse_calls: list = []

        async def sse(study_uid, payload):
            sse_calls.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=sse,
        )

        ok = await trigger.process_one(uid)
        assert ok is True
        # status → processed
        assert await _get_incoming_status(db_and_components["db"], uid) == STATUS_PROCESSED
        # 入队
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 1
        assert pending[0]["max_prob"] == pytest.approx(0.88)
        # SSE payload 含 reused_from_idempotency=True
        assert sse_calls[0][1]["reused_from_idempotency"] is True
        # stats: idempotent_hits=1
        assert trigger.stats()["idempotent_hits"] == 1

    async def test_process_one_study_not_in_incoming_returns_false(self, db_and_components):
        """study 不在 incoming_studies → 返回 False。"""
        sse_calls: list = []

        async def sse(study_uid, payload):
            sse_calls.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=sse,
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_MISSING>")
        assert ok is False
        assert len(sse_calls) == 0

    async def test_process_one_invalid_study_uid_raises(self, db_and_components):
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        with pytest.raises(TriageTriggerError):
            await trigger.process_one("")
        with pytest.raises(TriageTriggerError):
            await trigger.process_one(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 失败路径
# ---------------------------------------------------------------------------


class TestTriggerFailures:
    async def test_wado_failure_marks_failed_retry(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_WADO>")
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(raise_exc=WadoFetchError("503 mock")),
            dicom_processor=FakeDicomProcessor(),
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_WADO>")
        assert ok is False
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_WADO>"
        ) == STATUS_FAILED_RETRY
        # triage_queue 不入条
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 0
        # stats: wado_failed=1
        assert trigger.stats()["wado_failed"] == 1

    async def test_dicom_process_failure_marks_failed_retry(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_DP>")
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(
                raise_exc=DicomProcessError("no pixel data")
            ),
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_DP>")
        assert ok is False
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_DP>"
        ) == STATUS_FAILED_RETRY
        assert trigger.stats()["dicom_process_failed"] == 1

    async def test_workflow_error_marks_failed_retry(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_WF>")

        class FailingWorkflow:
            async def run(self, study_uid, png_bytes, metadata):
                raise WorkflowError("bad state")

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=FailingWorkflow(),  # type: ignore[arg-type]
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_WF>")
        assert ok is False
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_WF>"
        ) == STATUS_FAILED_RETRY
        assert trigger.stats()["workflow_failed"] == 1

    async def test_enqueue_failure_marks_failed_retry(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_ENQ>")

        class FailingQueue:
            async def enqueue(self, *args, **kwargs):
                raise RuntimeError("enqueue boom")

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=FailingQueue(),  # type: ignore[arg-type]
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_ENQ>")
        assert ok is False
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_ENQ>"
        ) == STATUS_FAILED_RETRY
        assert trigger.stats()["exceptions"] == 1

    async def test_idempotency_cache_failure_does_not_block(self, db_and_components):
        """idempotency.cache_draft 异常 → 不阻断主路径；仍入队 + 标 processed。"""
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_IC>")

        class FailingIdempotency:
            async def lookup(self, *a, **kw):
                return None

            async def cache_draft(self, *a, **kw):
                raise RuntimeError("cache_draft boom")

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=FailingIdempotency(),  # type: ignore[arg-type]
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
        )
        ok = await trigger.process_one("1.2.826.0.1.3680043.8.498.<EXAMPLE_IC>")
        assert ok is True  # cache 失败不阻断
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_IC>"
        ) == STATUS_PROCESSED
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 1
        # exceptions 计数从 cache 失败记一次
        assert trigger.stats()["exceptions"] == 1


# ---------------------------------------------------------------------------
# process_pending 批量
# ---------------------------------------------------------------------------


class TestTriggerProcessPending:
    async def test_process_pending_batch(self, db_and_components):
        # 3 条 pending + 1 条已 processed
        for i in range(3):
            await _seed_incoming(
                db_and_components["db"],
                f"1.2.826.0.1.3680043.8.498.<EXAMPLE_BATCH_{i}>",
            )
        # 第 4 条已 processed → 不应被消费
        await _seed_incoming(
            db_and_components["db"],
            "1.2.826.0.1.3680043.8.498.<EXAMPLE_DONE>",
        )
        async with db_and_components["db"].connection() as conn:
            await conn.execute(
                "UPDATE incoming_studies SET status=? WHERE study_uid=?",
                (STATUS_PROCESSED, "1.2.826.0.1.3680043.8.498.<EXAMPLE_DONE>"),
            )
            await conn.commit()

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
        )
        n = await trigger.process_pending()
        assert n == 3
        pending = await db_and_components["queue"].list_pending()
        assert len(pending) == 3
        assert trigger.stats()["scanned"] == 3
        assert trigger.stats()["processed"] == 3

    async def test_process_pending_one_failure_does_not_block_others(
        self, db_and_components
    ):
        """3 条 pending：第 2 条 wado 失败 → 另 2 条仍 processed。"""
        for i in range(3):
            await _seed_incoming(
                db_and_components["db"],
                f"1.2.826.0.1.3680043.8.498.<EXAMPLE_FAIL_{i}>",
            )
        # 单独造一个会失败的 wado（按 study_uid 区分）
        class SelectiveWado:
            def __init__(self):
                self.calls = []

            async def fetch(self, study_uid):
                self.calls.append(study_uid)
                if "FAIL_1" in study_uid:
                    raise WadoFetchError("503 selective mock")
                return VALID_DCM_BYTES

            async def close(self):
                pass

        wado = SelectiveWado()
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=wado,
            dicom_processor=FakeDicomProcessor(),
        )
        n = await trigger.process_pending()
        assert n == 2
        assert trigger.stats()["wado_failed"] == 1
        assert trigger.stats()["processed"] == 2
        # FAIL_1 → failed_retry
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_FAIL_1>"
        ) == STATUS_FAILED_RETRY
        # FAIL_0 / FAIL_2 → processed
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_FAIL_0>"
        ) == STATUS_PROCESSED
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_FAIL_2>"
        ) == STATUS_PROCESSED

    async def test_process_pending_empty(self, db_and_components):
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        n = await trigger.process_pending()
        assert n == 0


# ---------------------------------------------------------------------------
# run_pending_loop 协程
# ---------------------------------------------------------------------------


class TestTriggerLoop:
    async def test_start_stop_loop(self, db_and_components):
        await _seed_incoming(db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_LOOP>")
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            poll_interval_seconds=0.2,
        )
        await trigger.start()
        assert trigger.is_running
        # 等 0.6s 让 run_pending_loop 跑 2-3 轮
        await asyncio.sleep(0.6)
        await trigger.stop()
        assert not trigger.is_running
        # 已被消费
        assert await _get_incoming_status(
            db_and_components["db"], "1.2.826.0.1.3680043.8.498.<EXAMPLE_LOOP>"
        ) == STATUS_PROCESSED

    async def test_start_idempotent(self, db_and_components):
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            poll_interval_seconds=0.5,
        )
        await trigger.start()
        await trigger.start()  # 不应抛
        await trigger.stop()

    async def test_stop_without_start_is_idempotent(self, db_and_components):
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        await trigger.stop()  # 不应抛
        await trigger.stop()


# ---------------------------------------------------------------------------
# 审计 + SSE 推送
# ---------------------------------------------------------------------------


class TestTriggerAuditAndSse:
    async def test_audit_logged_on_happy_path(self, db_and_components):
        audit = AuditLogger(db_path=db_and_components["db"].db_path)
        await audit.ensure_db()
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_AUDIT>"
        await _seed_incoming(db_and_components["db"], uid)
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            audit_logger=audit,
        )
        await trigger.process_one(uid)
        # 查 audit_log
        async with db_and_components["db"].connection() as conn:
            cur = await conn.execute(
                "SELECT event_type FROM audit_log WHERE study_uid=? ORDER BY id",
                (uid,),
            )
            rows = await cur.fetchall()
            await cur.close()
        events = [r["event_type"] for r in rows]
        assert EVENT_TRIGGER_STARTED in events
        assert EVENT_TRIGGER_PROCESSED in events

    async def test_audit_logged_on_idempotent_hit(self, db_and_components):
        audit = AuditLogger(db_path=db_and_components["db"].db_path)
        await audit.ensure_db()
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_AUDIT_IDEM>"
        await _seed_incoming(db_and_components["db"], uid)
        await db_and_components["idempotency"].cache_draft(
            uid,
            {
                "study_uid": uid,
                "classifier_max_prob": 0.80,
                "positive_flag": True,
                "finding_text": "test draft",
                "tool_version": "0.1.0",
            },
        )
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            audit_logger=audit,
        )
        await trigger.process_one(uid)
        async with db_and_components["db"].connection() as conn:
            cur = await conn.execute(
                "SELECT event_type FROM audit_log WHERE study_uid=? AND event_type=?",
                (uid, EVENT_TRIGGER_IDEMPOTENT_HIT),
            )
            rows = await cur.fetchall()
            await cur.close()
        assert len(rows) == 1

    async def test_audit_logged_on_failure(self, db_and_components):
        audit = AuditLogger(db_path=db_and_components["db"].db_path)
        await audit.ensure_db()
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_AUDIT_FAIL>"
        await _seed_incoming(db_and_components["db"], uid)
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(raise_exc=WadoFetchError("503")),
            dicom_processor=FakeDicomProcessor(),
            audit_logger=audit,
        )
        await trigger.process_one(uid)
        async with db_and_components["db"].connection() as conn:
            cur = await conn.execute(
                "SELECT event_type FROM audit_log WHERE study_uid=? AND event_type=?",
                (uid, EVENT_TRIGGER_WADO_FAILED),
            )
            rows = await cur.fetchall()
            await cur.close()
        assert len(rows) == 1

    async def test_no_audit_logger_is_ok(self, db_and_components):
        """audit_logger=None 时不抛。"""
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_NOAUDIT>"
        await _seed_incoming(db_and_components["db"], uid)
        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            audit_logger=None,
        )
        ok = await trigger.process_one(uid)
        assert ok is True

    async def test_sse_callback_receives_payload(self, db_and_components):
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_SSE>"
        await _seed_incoming(db_and_components["db"], uid)
        captured: list = []

        async def cb(study_uid, payload):
            captured.append((study_uid, payload))

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=cb,
        )
        await trigger.process_one(uid)
        assert len(captured) == 1
        # payload 必含字段
        payload = captured[0][1]
        assert payload["study_uid"] == uid
        assert payload["max_prob"] == 0.85
        assert payload["positive_flag"] is True
        assert "finding_text" in payload
        assert "finding_struct" in payload
        assert "grounding_boxes" in payload
        assert "tools_executed" in payload

    async def test_sse_callback_exception_does_not_block(self, db_and_components):
        uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_SSEFAIL>"
        await _seed_incoming(db_and_components["db"], uid)

        async def cb(study_uid, payload):
            raise RuntimeError("SSE boom")

        trigger = TriageTrigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=cb,
        )
        ok = await trigger.process_one(uid)
        assert ok is True  # SSE 失败不阻断
        assert await _get_incoming_status(
            db_and_components["db"], uid
        ) == STATUS_PROCESSED


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


class TestTriggerFactories:
    def test_build_triage_trigger_returns_instance(self, db_and_components):
        trigger = build_triage_trigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        assert isinstance(trigger, TriageTrigger)

    def test_build_triage_trigger_with_all_args(self, db_and_components):
        audit = AuditLogger(db_path=db_and_components["db"].db_path)

        async def cb(uid, payload):
            pass

        trigger = build_triage_trigger(
            config=db_and_components["config"],
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
            audit_logger=audit,
            wado_client=FakeWado(),
            dicom_processor=FakeDicomProcessor(),
            sse_publish=cb,
            poll_interval_seconds=2.0,
            pending_batch_size=10,
        )
        assert trigger.poll_interval_seconds == 2.0
        assert trigger.pending_batch_size == 10

    def test_build_triage_trigger_from_env_uses_load_config(
        self, db_and_components, monkeypatch
    ):
        monkeypatch.setenv("PACS_AE_TITLE", "TEST")
        monkeypatch.setenv("PACS_HOST", "127.0.0.1")
        monkeypatch.setenv("PACS_PORT", "11112")
        monkeypatch.setenv("PACS_WADO_URL", "http://test.local/wado")
        monkeypatch.setenv("RIS_API_URL", "http://test.local/ris")
        monkeypatch.setenv("RIS_API_TOKEN", "test-ris")
        monkeypatch.setenv("HEALTHCHECK_API_URL", "http://test.local/hc")
        monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "test-hc")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://test.local/oai")
        monkeypatch.setenv("OPENAI_MODEL", "test-model")
        trigger = build_triage_trigger_from_env(
            db=db_and_components["db"],
            workflow=db_and_components["workflow"],
            triage_queue=db_and_components["queue"],
            idempotency=db_and_components["idempotency"],
        )
        assert trigger.config.pacs_ae_title == "TEST"
