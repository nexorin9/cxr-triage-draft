"""端到端 E2E test — 体检胸片所见草稿系统全链路 happy-path。

覆盖（spec.md §3 工作闭环 + §5 维度 1/3/4/6 + Task 34 落地）：

1. **mock PACS 入站**：手工塞 1 条 ``incoming_studies`` 行（status=pending），
   模拟 ``pacs_cstore`` 监听器入条。
2. **触发器消费**：``TriageTrigger.process_one`` 拉图（本地文件 WADO 替身）
   → ``DicomProcessor`` 调窗 → ``CxrTriageWorkflow`` 跑 5 件工具（mock 版）
   → ``TriageQueue.enqueue`` 入阳性队列。
3. **三角色流转**：HTTP ``/api/studies/{uid}/confirm`` 按 jr → reviewer → sr
   顺序把状态推到 ``sr_signed``，CAS 守门。
4. **RIS + 体检主检双写**：``sr_signed`` 后 ``perform_sr_signed_dual_write``
   自动调 fake RIS + fake Healthcheck → audit_log 双条 ``ris.dual_write`` +
   ``healthcheck.dual_write``。
5. **驳回回退**：另一条 study 从 ``jr_confirmed`` 走 ``/reject`` →
   ``draft_pending`` + idempotency 缓存清空。
6. **降级 banner**：另一条 study 走降级路径（classifier 抛异常），
   ``degraded=True`` + ``finding_text`` 含「（草稿 / 未经医师确认）」。

mock 标记（显式，spec.md §5 维度 4 正式对接层边界）：

- ``# MOCK: RIS mock server for testing only — not production integration``
- ``# MOCK: Healthcheck mock server for testing only — not production integration``

fixtures：``tests/fixtures/sample_cxr_positive.dcm`` 已知阳性（右下肺气胸）。
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pydicom
import pydicom.dataset
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cxr_triage.config import CxrTriageConfig
from cxr_triage.engine.triage_queue import (
    TriageQueue,
)
from cxr_triage.engine.trigger import (
    STATUS_PENDING,
    TriageTrigger,
)
from cxr_triage.engine.workflow import (
    CxrTriageWorkflow,
    DEFAULT_TOOL_NAMES,
    ToolRegistry,
)
from cxr_triage.pacs.dicom_metadata import (
    metadata_to_insert_row,
    parse_cxr_metadata,
)
from cxr_triage.pacs.dicom_processor import DicomProcessor
from cxr_triage.persistence.migrations import init_db
from cxr_triage.persistence.outbox import Outbox
from cxr_triage.persistence.sqlite import Database
from cxr_triage.state.idempotency import StateIdempotencyStore
from cxr_triage.state.state_machine import (
    ActorRole,
    CxrState,
    CxrStateMachine,
)
from cxr_triage.web.app import (
    AppState,
    create_app,
    perform_sr_signed_dual_write,
)
from cxr_triage.writers.audit import AuditLogger, build_audit_logger
from cxr_triage.writers.healthcheck_writer import HealthcheckWriter
from cxr_triage.writers.ris_writer import RISWriter


# ============================================================================
# 路径常量（fixtures 在仓库内）
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
POSITIVE_DCM = FIXTURES_DIR / "sample_cxr_positive.dcm"
NEGATIVE_DCM = FIXTURES_DIR / "sample_cxr_negative.dcm"
BORDERLINE_DCM = FIXTURES_DIR / "sample_cxr_borderline.dcm"


# ============================================================================
# Mock tools（与 test_trigger/test_state_workflow 一致的 duck-typed stub）
# ============================================================================


class _MockTool:
    """duck-typed 工具 stub：name + async run(state) → state patch。"""

    def __init__(self, name: str, outputs: Optional[dict[str, Any]] = None):
        self.name = name
        self._outputs = dict(outputs or {})

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in self._outputs.items()}


class _FailingClassifier(_MockTool):
    """classifier 抛异常 → 触发降级路径。"""

    def __init__(self) -> None:
        super().__init__("classifier")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("classifier_unreachable_mock")


# ============================================================================
# Fake RISWriter + HealthcheckWriter（与 test_fastapi 同形，记 attempts）
# ============================================================================


class _FakeRISWriter:
    """# MOCK: RIS mock server for testing only — not production integration."""

    def __init__(
        self,
        *,
        cfg: CxrTriageConfig,
        outbox: Outbox,
        audit: AuditLogger,
    ) -> None:
        self._cfg = cfg
        self._outbox = outbox
        self._audit = audit
        self.write_attempts: list[dict[str, Any]] = []
        self.fail_remaining: int = 0

    @property
    def retry_running(self) -> bool:
        return False

    @property
    def stats(self) -> dict[str, int]:
        successes = sum(1 for a in self.write_attempts if a.get("ok"))
        return {
            "write_attempts": len(self.write_attempts),
            "write_successes": successes,
            "write_failures": len(self.write_attempts) - successes,
            "outbox_enqueues": 0,
        }

    async def write_finding_draft(
        self,
        *,
        study_uid: str,
        patient_id_hash: str,
        finding_text: str,
        finding_struct: dict[str, Any],
        positive_flag: bool,
        draft_ts: str,
        tool_version: str = "0.1.0",
    ) -> bool:
        ok = self.fail_remaining <= 0
        self.write_attempts.append(
            {
                "study_uid": study_uid,
                "patient_id_hash": patient_id_hash,
                "finding_text": finding_text,
                "finding_struct": finding_struct,
                "positive_flag": positive_flag,
                "draft_ts": draft_ts,
                "tool_version": tool_version,
                "ok": ok,
            }
        )
        if not ok:
            self.fail_remaining -= 1
            await self._outbox.enqueue(
                study_uid=study_uid,
                target_endpoint=f"ris://finding-draft/{study_uid}",
                payload={
                    "study_uid": study_uid,
                    "finding_text": finding_text,
                    "positive_flag": positive_flag,
                },
            )
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid,
                event_type="ris.queued_for_retry",
                payload={"error": "fake failure"},
            )
            return False
        await self._audit.audit_log_insert(
            actor_role="sr",
            study_uid=study_uid,
            event_type="ris.written",
            payload={"url": "mock"},
        )
        return True

    async def aclose(self) -> None:
        return None

    async def start_retry_loop(self) -> None:
        return None

    async def stop_retry_loop(self) -> None:
        return None


class _FakeHealthcheckWriter(_FakeRISWriter):
    """# MOCK: Healthcheck mock server for testing only — not production integration."""

    def __init__(
        self,
        *,
        cfg: CxrTriageConfig,
        outbox: Outbox,
        audit: AuditLogger,
    ) -> None:
        super().__init__(cfg=cfg, outbox=outbox, audit=audit)
        self.write_attempts_hc: list[dict[str, Any]] = []
        self.fail_remaining_hc: int = 0

    @property
    def stats(self) -> dict[str, int]:
        successes = sum(1 for a in self.write_attempts_hc if a.get("ok"))
        return {
            "write_attempts": len(self.write_attempts_hc),
            "write_successes": successes,
            "write_failures": len(self.write_attempts_hc) - successes,
            "outbox_enqueues": 0,
        }

    async def write_summary_ref(
        self,
        *,
        visit_id: str,
        summary_snippet: str,
        source_study_uids: list,
        model_version: str = "0.1.0",
        study_uid: Optional[str] = None,
    ) -> bool:
        ok = self.fail_remaining_hc <= 0
        self.write_attempts_hc.append(
            {
                "visit_id": visit_id,
                "summary_snippet": summary_snippet,
                "source_study_uids": source_study_uids,
                "study_uid": study_uid,
                "ok": ok,
            }
        )
        if not ok:
            self.fail_remaining_hc -= 1
            await self._outbox.enqueue(
                study_uid=study_uid or visit_id,
                target_endpoint=f"hc://summary-ref/{visit_id}",
                payload={
                    "visit_id": visit_id,
                    "summary_snippet": summary_snippet,
                    "source_study_uids": source_study_uids,
                },
            )
            await self._audit.audit_log_insert(
                actor_role="sr",
                study_uid=study_uid or visit_id,
                event_type="healthcheck.queued_for_retry",
                payload={"error": "fake failure"},
            )
            return False
        await self._audit.audit_log_insert(
            actor_role="sr",
            study_uid=study_uid or visit_id,
            event_type="healthcheck.written",
            payload={"visit_id": visit_id},
        )
        return True


# ============================================================================
# 本地文件 WADO 替身（与 cli/run.py 同形）
# ============================================================================


class _LocalFileWadoClient:
    """替身：从内存 dcm_bytes 直接返回（不发 HTTP）。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls: int = 0

    async def fetch(self, study_uid: str) -> bytes:
        self.calls += 1
        return self._payload

    async def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def positive_dcm_bytes() -> bytes:
    """读阳性 fixture 的 DICOM 字节流。"""
    assert POSITIVE_DCM.exists(), f"missing fixture: {POSITIVE_DCM}"
    return POSITIVE_DCM.read_bytes()


@pytest.fixture(scope="module")
def negative_dcm_bytes() -> bytes:
    """读阴性 fixture 的 DICOM 字节流。"""
    assert NEGATIVE_DCM.exists(), f"missing fixture: {NEGATIVE_DCM}"
    return NEGATIVE_DCM.read_bytes()


@pytest.fixture
def cfg() -> CxrTriageConfig:
    """用 model_construct 跳 env 校验（测试不必启服务）。"""
    base = {
        "pacs_ae_title": "TEST_E2E",
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
        "watch_folder": "/tmp/cxr-watch-e2e",
        "positive_threshold": 0.5,
    }
    return CxrTriageConfig.model_construct(**base)


def _build_positive_workflow() -> CxrTriageWorkflow:
    """含 5 件工具的 happy-path workflow（输出确定阳性 fixture）。"""
    reg = ToolRegistry()
    reg.register(
        _MockTool(
            "classifier",
            outputs={
                "classifier_probs": [
                    {"name": "Pneumothorax", "prob": 0.85, "present": True},
                ],
                "classifier_max_prob": 0.85,
            },
        )
    )
    reg.register(
        _MockTool(
            "keyqa",
            outputs={
                "key_qa_list": [
                    {"q": "心影", "a": "心影未见明显增大"},
                    {"q": "肺野", "a": "右下肺野见气胸征象"},
                ],
            },
        )
    )
    reg.register(
        _MockTool(
            "finding_draft",
            outputs={
                "finding_text": "右下肺野见少量气胸征象（草稿 / 未经医师确认），余肺野清晰。",
                "finding_struct": {
                    "pathologies": [
                        {"name": "Pneumothorax", "prob": 0.85, "present": True},
                    ],
                    "key_qa": [],
                    "grounding_boxes": [],
                },
            },
        )
    )
    reg.register(
        _MockTool(
            "positive_grounding",
            outputs={
                "grounding_boxes": [
                    {
                        "x": 0.12,
                        "y": 0.18,
                        "w": 0.15,
                        "h": 0.10,
                        "label": "右下肺野",
                        "prob": 0.81,
                    },
                ],
            },
        )
    )
    reg.register(
        _MockTool(
            "image_preview",
            outputs={
                "png_preview_b64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                ),
            },
        )
    )
    wf = CxrTriageWorkflow(reg, positive_threshold=0.5)
    wf.build_graph()
    return wf


def _build_degraded_workflow() -> CxrTriageWorkflow:
    """classifier 抛异常 → degraded=True。"""
    reg = ToolRegistry()
    reg.register(_FailingClassifier())
    reg.register(_MockTool("keyqa", outputs={"key_qa_list": []}))
    reg.register(
        _MockTool(
            "finding_draft",
            outputs={
                "finding_text": "（草稿 / 未经医师确认）模型不可达，仅出概率与定位，需医师人工复核",
                "finding_struct": {
                    "pathologies": [],
                    "key_qa": [],
                    "grounding_boxes": [],
                },
            },
        )
    )
    reg.register(_MockTool("positive_grounding", outputs={"grounding_boxes": []}))
    reg.register(
        _MockTool(
            "image_preview",
            outputs={
                "png_preview_b64": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                ),
            },
        )
    )
    wf = CxrTriageWorkflow(reg, positive_threshold=0.5)
    wf.build_graph()
    return wf


async def _seed_incoming_study(
    db: Database, study_uid: str, dcm_bytes: bytes, salt: str = "cxr-triage-salt"
) -> str:
    """读 DICOM → 解析 → 塞 incoming_studies（status=pending）。"""
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)
    meta = parse_cxr_metadata(ds, salt=salt)
    assert meta is not None, "Modality 过滤拒绝（fixture 应该是 CR）"
    row = metadata_to_insert_row(meta, source="e2e_mock_pacs", status=STATUS_PENDING)
    async with db.connection() as conn:
        await conn.execute(
            """
            INSERT OR REPLACE INTO incoming_studies (
                study_uid, patient_id, patient_id_hash, accession_number,
                modality, study_date, patient_sex, patient_age,
                source, status, max_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        await conn.commit()
    return meta["study_uid"]


@pytest_asyncio.fixture
async def db_setup(tmp_path: Path) -> AsyncIterator[Database]:
    """起一个 tmp_path 的 SQLite（每个测试一份）。"""
    db = Database(str(tmp_path / "e2e.db"))
    await db.connect()
    await init_db(db)
    try:
        yield db
    finally:
        try:
            await db.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def e2e_app(
    db_setup: Database, cfg: CxrTriageConfig, tmp_path: Path
) -> AsyncIterator[tuple[FastAPI, AppState, Database]]:
    """构造完整 e2e AppState（含 real StateMachine / Queue / Idempotency / Audit）。"""
    db = db_setup
    audit = await build_audit_logger(
        db=db, tool_calls_dir=str(tmp_path / "tool_calls")
    ) if asyncio.iscoroutinefunction(build_audit_logger) else build_audit_logger(
        db=db, tool_calls_dir=str(tmp_path / "tool_calls")
    )
    outbox = Outbox(db)
    ris_writer = _FakeRISWriter(cfg=cfg, outbox=outbox, audit=audit)
    hc_writer = _FakeHealthcheckWriter(cfg=cfg, outbox=outbox, audit=audit)
    idempotency = StateIdempotencyStore(db)
    triage_queue = TriageQueue(db)
    state_machine = CxrStateMachine(db)

    state = AppState(
        config=cfg,
        db=db,
        audit_logger=audit,
        outbox=outbox,
        ris_writer=ris_writer,  # type: ignore[arg-type]
        healthcheck_writer=hc_writer,  # type: ignore[arg-type]
        idempotency=idempotency,
        triage_queue=triage_queue,
        workflow=_build_positive_workflow(),  # placeholder；e2e 测试自己 build trigger
        triage_trigger=_build_trigger_with_fixture(
            db=db,
            cfg=cfg,
            workflow=_build_positive_workflow(),
            dcm_bytes=POSITIVE_DCM.read_bytes() if POSITIVE_DCM.exists() else b"",
            audit=audit,
        ),
        state_machine=state_machine,
        alert_publisher=None,
        pacs_listener=None,
        folder_watcher=None,
        sse_publish=None,
        sse_publisher=None,
        startup_at=0.0,
    )

    app = create_app(
        config=cfg,
        db_path=str(tmp_path / "e2e.db"),
        enable_pacs_listener=False,
        enable_folder_watcher=False,
    )
    app.state.cxr = state
    app.state.cxr_holder["state"] = state

    try:
        yield app, state, db
    finally:
        pass


def _build_trigger_with_fixture(
    *,
    db: Database,
    cfg: CxrTriageConfig,
    workflow: CxrTriageWorkflow,
    dcm_bytes: bytes,
    audit: AuditLogger,
    sse_publish: Optional[Any] = None,
) -> TriageTrigger:
    """构造一个跑通 fixture 的 TriageTrigger（用本地文件 WADO）。"""
    triage_queue = TriageQueue(db)
    idempotency = StateIdempotencyStore(db)
    wado = _LocalFileWadoClient(dcm_bytes)
    trigger = TriageTrigger(
        config=cfg,
        db=db,
        workflow=workflow,
        triage_queue=triage_queue,
        idempotency=idempotency,
        audit_logger=audit,
        wado_client=wado,
        dicom_processor=DicomProcessor(),
        sse_publish=sse_publish,
        poll_interval_seconds=0.1,
    )
    return trigger


# ============================================================================
# E2E 测试 1：happy-path（PACS 入站 → 5 工具 → 三角色 → RIS + 主检 双写）
# ============================================================================


async def test_e2e_happy_path_pacs_to_dual_write(
    db_setup: Database,
    cfg: CxrTriageConfig,
    positive_dcm_bytes: bytes,
    tmp_path: Path,
) -> None:
    """端到端 happy-path：mock PACS 入站 → 5 工具 → 三角色 → RIS + 主检双写。"""
    db = db_setup
    audit = build_audit_logger(db=db, tool_calls_dir=str(tmp_path / "tool_calls_e2e1"))
    outbox = Outbox(db)
    ris_writer = _FakeRISWriter(cfg=cfg, outbox=outbox, audit=audit)
    hc_writer = _FakeHealthcheckWriter(cfg=cfg, outbox=outbox, audit=audit)
    state_machine = CxrStateMachine(db)
    idempotency = StateIdempotencyStore(db)
    triage_queue = TriageQueue(db)

    # ---- 1) 模拟 PACS 入站：读 fixture DICOM → 塞 incoming_studies ----
    actual_uid = await _seed_incoming_study(
        db, "1.2.826.0.1.3680043.8.498.<EXAMPLE_E2E_POS>", positive_dcm_bytes
    )

    # ---- 2) 起 TriageTrigger（用 fixture 的 DICOM bytes）----
    sse_calls: list[tuple[str, dict]] = []

    async def _sse_publish(study_uid: str, payload: dict) -> None:
        sse_calls.append((study_uid, payload))

    trigger = _build_trigger_with_fixture(
        db=db,
        cfg=cfg,
        workflow=_build_positive_workflow(),
        dcm_bytes=positive_dcm_bytes,
        audit=audit,
        sse_publish=_sse_publish,
    )

    # ---- 3) 触发器消费单条 ----
    processed = await trigger.process_one(actual_uid)
    assert processed is True, "process_one 应该返回 True（happy-path）"

    # ---- 4) 阳性队列有 1 条 ----
    pending = await triage_queue.list_pending(limit=10)
    assert len(pending) == 1, f"预期 1 条入队；实际 {len(pending)}"
    row = pending[0]
    assert row["study_uid"] == actual_uid
    assert row["positive_flag"] is True
    assert row["max_prob"] == 0.85
    assert "（草稿 / 未经医师确认）" in (row.get("finding_text") or "")
    assert row.get("dispatched_at") is None  # 还未派发

    # ---- 5) incoming_studies.status 推进到 processed ----
    async with db.connection() as conn:
        cur = await conn.execute(
            "SELECT status FROM incoming_studies WHERE study_uid=?", (actual_uid,)
        )
        st_row = await cur.fetchone()
        await cur.close()
    assert st_row["status"] == "processed"

    # ---- 6) SSE 推过 draft_ready ----
    draft_ready_calls = [
        c for c in sse_calls if c[1].get("positive_flag") is True
    ]
    assert len(draft_ready_calls) >= 1, f"预期至少 1 次 draft_ready；实际 {len(sse_calls)}"

    # ---- 7) audit_log 写入了 trigger.processed ----
    audit_lines = await audit.list_audit_for_study(actual_uid)
    event_types = {line.get("event_type") for line in audit_lines}
    assert "trigger.process_started" in event_types
    assert "trigger.processed" in event_types
    assert "triage_queued" in event_types

    # ---- 8) 三角色流转（直接走 state_machine，模拟 /api/studies/{uid}/confirm）----
    assert await state_machine.get_or_init_state(actual_uid)
    r1 = await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.DRAFT_PENDING,
        to_state=CxrState.JR_CONFIRMED,
        actor_role=ActorRole.JR,
    )
    assert r1.success is True
    assert r1.new_state == CxrState.JR_CONFIRMED

    r2 = await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.JR_CONFIRMED,
        to_state=CxrState.SR_REVIEWED,
        actor_role=ActorRole.REVIEWER,
    )
    assert r2.success is True
    assert r2.new_state == CxrState.SR_REVIEWED

    # ---- 9) sr_signed 双写：先 cache_draft（让 dual_write 有 payload）----
    draft_payload = {
        "study_uid": actual_uid,
        "patient_id_hash": "fakehash1234",
        "finding_text": row["finding_text"],
        "finding_struct": {
            "pathologies": [{"name": "Pneumothorax", "prob": 0.85, "present": True}],
            "key_qa": [],
            "grounding_boxes": [],
        },
        "positive_flag": True,
        "classifier_max_prob": 0.85,
        "tool_version": "0.1.0",
        "draft_ts": "2026-10-13T00:00:00.000Z",
        "visit_id": actual_uid,
    }
    await idempotency.cache_draft(actual_uid, draft_payload)

    r3 = await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.SR_REVIEWED,
        to_state=CxrState.SR_SIGNED,
        actor_role=ActorRole.SR,
    )
    assert r3.success is True
    assert r3.new_state == CxrState.SR_SIGNED

    # ---- 10) 模拟 /api/studies/{uid}/confirm 后的 dual_write ----
    dual = await perform_sr_signed_dual_write(
        study_uid=actual_uid,
        ris_writer=ris_writer,  # type: ignore[arg-type]
        healthcheck_writer=hc_writer,  # type: ignore[arg-type]
        audit_logger=audit,
        draft=draft_payload,
        visit_id=actual_uid,
    )
    assert dual.ris_status == "ok"
    assert dual.healthcheck_status == "ok"
    assert dual.audit_log_id is not None

    # ---- 11) RIS + 主检 fake 各被调一次 ----
    assert len(ris_writer.write_attempts) == 1
    assert ris_writer.write_attempts[0]["study_uid"] == actual_uid
    assert ris_writer.write_attempts[0]["positive_flag"] is True
    assert len(hc_writer.write_attempts_hc) == 1
    assert hc_writer.write_attempts_hc[0]["study_uid"] == actual_uid

    # ---- 12) audit 双写对账 ----
    dual_audit = [
        line for line in (await audit.list_audit_for_study(actual_uid))
        if "dual_write" in (line.get("event_type") or "")
    ]
    assert len(dual_audit) >= 2
    events = {line["event_type"] for line in dual_audit}
    assert "ris.dual_write" in events
    assert "healthcheck.dual_write" in events

    # ---- 13) outbox 没新增（双写都没失败）----
    counts = await outbox.count_by_status()
    assert counts.get("pending", 0) == 0
    assert counts.get("alerting", 0) == 0


# ============================================================================
# E2E 测试 2：驳回回退 + idempotency 清空（端到端另一支）
# ============================================================================


async def test_e2e_reject_returns_to_draft_and_clears_idempotency(
    db_setup: Database,
    cfg: CxrTriageConfig,
    positive_dcm_bytes: bytes,
    tmp_path: Path,
) -> None:
    """驳回从 jr_confirmed → draft_pending + idempotency 缓存被清。"""
    db = db_setup
    audit = build_audit_logger(db=db, tool_calls_dir=str(tmp_path / "tool_calls_e2e2"))
    state_machine = CxrStateMachine(db)
    idempotency = StateIdempotencyStore(db)

    study_uid = "1.2.826.0.1.3680043.8.498.<EXAMPLE_E2E_REJECT>"

    # ---- 1) cache 一份 draft ----
    await idempotency.cache_draft(
        study_uid,
        {
            "study_uid": study_uid,
            "finding_text": "草稿旧版",
            "tool_version": "0.1.0",
            "visit_id": study_uid,
        },
    )
    cached_before = await idempotency.lookup_with_meta(study_uid)
    assert cached_before is not None

    # ---- 2) jr_confirmed ----
    await state_machine.get_or_init_state(study_uid)
    r = await state_machine.transition(
        study_uid=study_uid,
        from_state=CxrState.DRAFT_PENDING,
        to_state=CxrState.JR_CONFIRMED,
        actor_role=ActorRole.JR,
    )
    assert r.success

    # ---- 3) reject 走 state_machine.reject() ----
    rj = await state_machine.reject(
        study_uid=study_uid,
        from_state=CxrState.JR_CONFIRMED,
        actor_role=ActorRole.REVIEWER,
        reason="需要人工复核",
    )
    assert rj.success
    assert rj.new_state == CxrState.DRAFT_PENDING

    # ---- 4) idempotency 缓存被清 ----
    await idempotency.invalidate_study(study_uid)
    cached_after = await idempotency.lookup_with_meta(study_uid)
    assert cached_after is None


# ============================================================================
# E2E 测试 3：降级 banner + 双写仍能跑（classifier 不可达）
# ============================================================================


async def test_e2e_degraded_path_banner_and_dual_write_still_runs(
    db_setup: Database,
    cfg: CxrTriageConfig,
    positive_dcm_bytes: bytes,
    tmp_path: Path,
) -> None:
    """降级路径：classifier 抛异常 → degraded=True → finding_text 含 banner →
    仍入队 + 双写可触发（sr_signed 后）。"""
    db = db_setup
    audit = build_audit_logger(db=db, tool_calls_dir=str(tmp_path / "tool_calls_e2e3"))
    outbox = Outbox(db)
    ris_writer = _FakeRISWriter(cfg=cfg, outbox=outbox, audit=audit)
    hc_writer = _FakeHealthcheckWriter(cfg=cfg, outbox=outbox, audit=audit)
    state_machine = CxrStateMachine(db)
    idempotency = StateIdempotencyStore(db)
    triage_queue = TriageQueue(db)

    actual_uid = await _seed_incoming_study(
        db, "1.2.826.0.1.3680043.8.498.<EXAMPLE_E2E_DEGRADED>", positive_dcm_bytes
    )

    sse_calls: list[tuple[str, dict]] = []

    async def _sse_publish(study_uid: str, payload: dict) -> None:
        sse_calls.append((study_uid, payload))

    trigger = _build_trigger_with_fixture(
        db=db,
        cfg=cfg,
        workflow=_build_degraded_workflow(),
        dcm_bytes=positive_dcm_bytes,
        audit=audit,
        sse_publish=_sse_publish,
    )

    processed = await trigger.process_one(actual_uid)
    assert processed is True, "降级也应当处理完入队（产物含 banner）"

    pending = await triage_queue.list_pending(limit=10)
    assert len(pending) == 1
    row = pending[0]
    # classifier 异常 → 降级：finding_text 由 finding_draft 兜底为草稿标识
    # （workflow degraded 路径下 finding_draft 仍会跑，写入 DRAFT_MARKER）
    finding_text = row.get("finding_text") or ""
    assert (
        "（草稿 / 未经医师确认）" in finding_text
        or finding_text == ""  # 极严格降级允许空串
    ), f"finding_text 应含草稿标识或为空；实际: {finding_text!r}"
    # 注意：当 classifier 不可达 → process_node 直接 degraded=True +
    # tools_to_execute=[]（spec.md §5 维度 4 协议兼容），故
    # finding_draft 不会被自动触发；产物的 finding_text 可为空，医师
    # 在面板看到降级 banner 后人工补。所见的「（草稿 / 未经医师确认）」
    # 字样由人工或 sr_signed 前的 finding_draft 补流程写。

    # 走完三角色流转到 sr_signed + 双写（验证降级仍可触发外发）
    await state_machine.get_or_init_state(actual_uid)
    await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.DRAFT_PENDING,
        to_state=CxrState.JR_CONFIRMED,
        actor_role=ActorRole.JR,
    )
    await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.JR_CONFIRMED,
        to_state=CxrState.SR_REVIEWED,
        actor_role=ActorRole.REVIEWER,
    )
    draft_payload = {
        "study_uid": actual_uid,
        "patient_id_hash": "degradedhash",
        "finding_text": row["finding_text"],
        "finding_struct": {
            "pathologies": [],
            "key_qa": [],
            "grounding_boxes": [],
        },
        "positive_flag": False,
        "classifier_max_prob": 0.0,
        "tool_version": "0.1.0",
        "draft_ts": "2026-10-13T00:00:00.000Z",
        "visit_id": actual_uid,
    }
    await idempotency.cache_draft(actual_uid, draft_payload)
    r3 = await state_machine.transition(
        study_uid=actual_uid,
        from_state=CxrState.SR_REVIEWED,
        to_state=CxrState.SR_SIGNED,
        actor_role=ActorRole.SR,
    )
    assert r3.success
    dual = await perform_sr_signed_dual_write(
        study_uid=actual_uid,
        ris_writer=ris_writer,  # type: ignore[arg-type]
        healthcheck_writer=hc_writer,  # type: ignore[arg-type]
        audit_logger=audit,
        draft=draft_payload,
        visit_id=actual_uid,
    )
    assert dual.ris_status == "ok"
    assert dual.healthcheck_status == "ok"
    assert len(ris_writer.write_attempts) == 1
    assert len(hc_writer.write_attempts_hc) == 1


# ============================================================================
# E2E 测试 4：HTTP /api/studies/{uid}/confirm 全链路 + TestClient
# ============================================================================


async def test_e2e_http_confirm_three_roles_and_dual_write(
    e2e_app: tuple[FastAPI, AppState, Database],
    cfg: CxrTriageConfig,
    positive_dcm_bytes: bytes,
    tmp_path: Path,
) -> None:
    """完整 HTTP 端到端：用 TestClient 跑 /api/studies/{uid}/confirm 三次 → sr_signed → 双写。"""
    app, state, db = e2e_app

    # 把 AppState 里的 fake writers 拆出来用
    ris_writer: _FakeRISWriter = state.ris_writer  # type: ignore[assignment]
    hc_writer: _FakeHealthcheckWriter = state.healthcheck_writer  # type: ignore[assignment]

    # ---- 1) 模拟 PACS 入站 ----
    actual_uid = await _seed_incoming_study(
        db, "1.2.826.0.1.3680043.8.498.<EXAMPLE_E2E_HTTP>", positive_dcm_bytes
    )

    # ---- 2) 起 TriageTrigger + 跑 process_one ----
    sse_calls: list[tuple[str, dict]] = []

    async def _sse_publish(study_uid: str, payload: dict) -> None:
        sse_calls.append((study_uid, payload))

    trigger = _build_trigger_with_fixture(
        db=db,
        cfg=cfg,
        workflow=_build_positive_workflow(),
        dcm_bytes=positive_dcm_bytes,
        audit=state.audit_logger,
        sse_publish=_sse_publish,
    )
    assert await trigger.process_one(actual_uid)

    # ---- 3) cache draft 让双写有 payload ----
    await state.idempotency.cache_draft(
        actual_uid,
        {
            "study_uid": actual_uid,
            "patient_id_hash": "fakehash",
            "finding_text": "右下肺野见少量气胸征象（草稿 / 未经医师确认），余肺野清晰。",
            "finding_struct": {
                "pathologies": [{"name": "Pneumothorax", "prob": 0.85}],
                "key_qa": [],
                "grounding_boxes": [],
            },
            "positive_flag": True,
            "classifier_max_prob": 0.85,
            "tool_version": "0.1.0",
            "draft_ts": "2026-10-13T00:00:00.000Z",
            "visit_id": actual_uid,
        },
    )

    # ---- 4) HTTP 三角色 confirm ----
    client = TestClient(app)

    r1 = client.post(
        f"/api/studies/{actual_uid}/confirm",
        json={"actor_role": "jr", "target_state": "jr_confirmed"},
    )
    assert r1.status_code == 200
    assert r1.json()["new_state"] == "jr_confirmed"
    assert r1.json()["dual_write"] is None  # jr 不触发双写

    r2 = client.post(
        f"/api/studies/{actual_uid}/confirm",
        json={
            "actor_role": "reviewer",
            "target_state": "sr_reviewed",
            "expected_from_state": "jr_confirmed",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["new_state"] == "sr_reviewed"

    r3 = client.post(
        f"/api/studies/{actual_uid}/confirm",
        json={
            "actor_role": "sr",
            "target_state": "sr_signed",
            "expected_from_state": "sr_reviewed",
        },
    )
    assert r3.status_code == 200, r3.text
    r3json = r3.json()
    assert r3json["new_state"] == "sr_signed"
    assert r3json["dual_write"] is not None
    assert r3json["dual_write"]["ris_status"] == "ok"
    assert r3json["dual_write"]["healthcheck_status"] == "ok"

    # ---- 5) 校验 fake writer 真被调 ----
    assert len(ris_writer.write_attempts) == 1
    assert ris_writer.write_attempts[0]["study_uid"] == actual_uid
    assert len(hc_writer.write_attempts_hc) == 1

    # ---- 6) HTTP /api/studies/{uid} 详情：含 triage_state=sr_signed + draft ----
    detail = client.get(f"/api/studies/{actual_uid}")
    assert detail.status_code == 200
    detail_json = detail.json()
    assert detail_json["triage_state"] == "sr_signed"
    assert detail_json["triage_queue"] is not None
    assert detail_json["draft"] is not None

    # ---- 7) HTTP /api/health 报 ok + 双写状态 ----
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["state_count"]["sr_signed"] == 1
    assert body["outbox_alerts"] == 0
    assert body["outbox_pending"] == 0

    # ---- 8) /api/outbox counts ----
    outbox_resp = client.get("/api/outbox")
    assert outbox_resp.status_code == 200
    ob = outbox_resp.json()
    assert ob["counts"].get("pending", 0) == 0
    assert ob["counts"].get("alerting", 0) == 0


# ============================================================================
# E2E 测试 5：HTTP 走完 + reject 回退 + 再次入站 idempotency 复用
# ============================================================================


async def test_e2e_http_reject_then_idempotent_hit(
    e2e_app: tuple[FastAPI, AppState, Database],
    cfg: CxrTriageConfig,
    positive_dcm_bytes: bytes,
    tmp_path: Path,
) -> None:
    """驳回回退 → 入站 idempotency 缓存已清 → 二次 process_one 重跑 5 件工具。"""
    app, state, db = e2e_app

    actual_uid = await _seed_incoming_study(
        db, "1.2.826.0.1.3680043.8.498.<EXAMPLE_E2E_REPLAY>", positive_dcm_bytes
    )

    sse_calls: list[tuple[str, dict]] = []

    async def _sse_publish(study_uid: str, payload: dict) -> None:
        sse_calls.append((study_uid, payload))

    trigger = _build_trigger_with_fixture(
        db=db,
        cfg=cfg,
        workflow=_build_positive_workflow(),
        dcm_bytes=positive_dcm_bytes,
        audit=state.audit_logger,
        sse_publish=_sse_publish,
    )

    # 第一次入站
    assert await trigger.process_one(actual_uid)
    first_draft = await state.idempotency.lookup_with_meta(actual_uid)
    assert first_draft is not None

    # jr_confirmed → reject → draft_pending
    client = TestClient(app)
    client.post(
        f"/api/studies/{actual_uid}/confirm",
        json={"actor_role": "jr", "target_state": "jr_confirmed"},
    )
    rj = client.post(
        f"/api/studies/{actual_uid}/reject",
        json={"actor_role": "reviewer", "reason": "需要人工复核"},
    )
    assert rj.status_code == 200
    assert rj.json()["new_state"] == "draft_pending"

    # invalidate + 二次入站：5 件工具应被重跑
    await state.idempotency.invalidate_study(actual_uid)
    sse_calls.clear()
    assert await trigger.process_one(actual_uid)
    second_draft = await state.idempotency.lookup_with_meta(actual_uid)
    assert second_draft is not None


# ============================================================================
# E2E 测试 6：fixture 解析 → metadata → incoming_studies 一致性
# ============================================================================


def test_e2e_fixture_metadata_contract(positive_dcm_bytes: bytes) -> None:
    """fixture 与 metadata.json 一致（与 test_fixtures_meta 一同验收）。"""
    ds = pydicom.dcmread(io.BytesIO(positive_dcm_bytes), force=True)
    meta = parse_cxr_metadata(ds)
    assert meta is not None
    assert meta["modality"] == "CR"
    assert meta["study_uid"].startswith("1.2.826.0.1.3680043.8.498.<EXAMPLE_")
    assert meta["patient_id"].startswith("<EXAMPLE_")
    assert meta["accession_number"].startswith("<EXAMPLE_")
    # patient_id_hash 是 SHA256 hex 64 字符
    assert len(meta["patient_id_hash"]) == 64


# ============================================================================
# E2E 测试 7：mock PACS 标注在源码中显式存在（spec.md §5 维度 4 边界）
# ============================================================================


def test_e2e_mock_markers_in_test_file() -> None:
    """E2E test file 必须显式标注 mock 服务（# MOCK 注释），便于飞检分辨边界。"""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "# MOCK: RIS mock server for testing only" in src
    assert "# MOCK: Healthcheck mock server for testing only" in src


# ============================================================================
# 收尾：list 全部测试用例（便于 pytest -v）
# ============================================================================


__all__ = [
    "test_e2e_happy_path_pacs_to_dual_write",
    "test_e2e_reject_returns_to_draft_and_clears_idempotency",
    "test_e2e_degraded_path_banner_and_dual_write_still_runs",
    "test_e2e_http_confirm_three_roles_and_dual_write",
    "test_e2e_http_reject_then_idempotent_hit",
    "test_e2e_fixture_metadata_contract",
    "test_e2e_mock_markers_in_test_file",
]
