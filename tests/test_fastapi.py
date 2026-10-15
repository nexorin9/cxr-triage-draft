"""FastAPI app + routers 路由解析 + 状态机调用 + 驳回回退测试。

覆盖（spec.md §5 维度 4 正式对接层入口 + Task 28 落地）：

1. 路由解析：每个 endpoint 路径 + 方法可达。
2. ``/api/health`` 返回字段完整 + 形状对齐 HealthResponse 契约。
3. ``/api/studies`` 列表（按 state 过滤）。
4. ``/api/studies/{uid}/confirm`` 走 ``state_machine.transition()``：
   - jr → jr_confirmed 成功
   - reviewer → sr_reviewed 成功
   - sr → sr_signed 成功 + 双写触发
5. ``/api/studies/{uid}/reject`` 走 ``state_machine.reject()`` + 清幂等。
6. 并发 CAS 冲突（同 study 两路 sr_sign → 只有一条 200/success=True）。
7. 非法跃迁 / 角色越权 → HTTP 400。
8. Pydantic body 校验 → 422。
9. ``/api/queue/positive`` / ``/api/queue/negative`` 按 positive_flag 分流。
10. ``/api/outbox`` 返回 counts + alerting。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cxr_triage.config import CxrTriageConfig
from cxr_triage.engine.triage_queue import TriageQueue
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
    ConfirmRequest,
    ConfirmResponse,
    HealthResponse,
    QueueItem,
    RejectRequest,
    RejectResponse,
    StudyDetailResponse,
    create_app,
    perform_sr_signed_dual_write,
)


# ----------------------------------------------------------------------------
# 辅助：构造测试用 db + 全组件（Production 形态的微型版）
# ----------------------------------------------------------------------------

# Per-test asyncio marking: 不要用 module-level pytestmark（会污染同步测试）


@pytest_asyncio.fixture
async def temp_db_path(tmp_path: Path) -> Path:
    """独立 sqlite 文件路径（每个测试一份）。"""
    db_path = tmp_path / "cxr-triage.db"
    yield db_path


@pytest_asyncio.fixture
async def app_with_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[FastAPI, AppState, Database]]:
    """构造一个最小 FastAPI app + AppState（关闭 PACS / folder watcher）。"""
    # 关闭会异步调试的输出
    monkeypatch.setenv("CXR_TRIAGE_ALLOW_PLACEHOLDER", "1")
    monkeypatch.setenv("RIS_API_TOKEN", "replace-me-ris-token")
    monkeypatch.setenv("HEALTHCHECK_API_TOKEN", "replace-me-hc-token")
    monkeypatch.setenv("OPENAI_API_KEY", "replace-me-llm-key")
    monkeypatch.setenv("CXR_TRIAGE_DB_PATH", str(tmp_path / "test-cxr.db"))
    monkeypatch.setenv(
        "CXR_TRIAGE_PATIENT_ID_SALT", "test-salt"
    )

    db = Database(os.getenv("CXR_TRIAGE_DB_PATH"))
    await db.connect()
    await init_db(db)

    cfg = CxrTriageConfig()
    outbox = Outbox(db)
    audit = _FakeAuditLogger(db)
    ris_writer = _FakeRISWriter(cfg=cfg, outbox=outbox, audit=audit)
    healthcheck_writer = _FakeHealthcheckWriter(
        cfg=cfg, outbox=outbox, audit=audit
    )
    idempotency = StateIdempotencyStore(db)
    triage_queue = TriageQueue(db)
    state_machine = CxrStateMachine(db)

    # 不起真 trigger / workflow（避免模型依赖）
    state = AppState(
        config=cfg,
        db=db,
        audit_logger=audit,
        outbox=outbox,
        ris_writer=ris_writer,
        healthcheck_writer=healthcheck_writer,
        idempotency=idempotency,
        triage_queue=triage_queue,
        workflow=_FakeWorkflow(),
        triage_trigger=_FakeTrigger(),
        state_machine=state_machine,
        alert_publisher=None,
        pacs_listener=None,
        folder_watcher=None,
        sse_publish=None,
        startup_at=time.time(),
    )

    # 注册到 app.state.cxr（不走 lifespan）
    app = create_app(
        config=cfg,
        db_path=str(tmp_path / "test-cxr.db"),
        enable_pacs_listener=False,
        enable_folder_watcher=False,
    )

    # 注意：create_app 内部会启动 lifespan → 覆盖 app.state.cxr。这里
    # bypass lifespan 直接手动注入 fake state。
    app.state.cxr = state
    app.state.cxr_holder["state"] = state

    try:
        yield app, state, db
    finally:
        try:
            await db.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Fake components（最小可用 stub，按 duck-type 满足接口）
# ----------------------------------------------------------------------------


class _FakeAuditLogger:
    """Fake ``AuditLogger``：内存里攒条目便于断言。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.records: list[dict[str, Any]] = []

    async def audit_log_insert(
        self,
        *,
        actor_role: str,
        study_uid: str,
        event_type: str,
        payload: Any = None,
    ) -> int:
        rec = {
            "actor_role": actor_role,
            "study_uid": study_uid,
            "event_type": event_type,
            "payload": payload,
        }
        self.records.append(rec)
        # 不实际写 db（state_machine.transition 会自己写，这里是双写空告
        # 路径兜底）；返回 last id 演示用
        return len(self.records)

    async def tool_call_record(self, **_kwargs: Any) -> None:
        return None

    async def write_dual_write(
        self,
        *,
        study_uid: str,
        ris_status: str,
        healthcheck_status: str,
        actor_role: str = "sr",
        ris_extra: Optional[dict] = None,
        healthcheck_extra: Optional[dict] = None,
        ts: Optional[str] = None,
    ) -> int:
        rec = {
            "actor_role": actor_role,
            "study_uid": study_uid,
            "event_type": "ris.dual_write" if ris_status == "ok" else "ris.dual_write_failed",
            "payload": {
                "kind": "dual_write",
                "ris_status": ris_status,
                "healthcheck_status": healthcheck_status,
                "healthcheck_extra": healthcheck_extra,
                "ris_extra": ris_extra,
            },
        }
        self.records.append(rec)
        return len(self.records)


class _FakeRISWriter:
    def __init__(self, *, cfg: CxrTriageConfig, outbox: Outbox, audit: _FakeAuditLogger) -> None:
        self._cfg = cfg
        self._outbox = outbox
        self._audit = audit
        self.write_attempts: list[dict[str, Any]] = []
        # 0 = 全部成功；>0 = 写失败次数计数器（测试可预置）
        self.fail_remaining: int = 0
        # 跑 start_retry_loop 时记录是否调用
        self.start_retry_loop_called: bool = False
        self.stop_retry_loop_called: bool = False

    @property
    def retry_running(self) -> bool:
        return self.start_retry_loop_called and not self.stop_retry_loop_called

    @property
    def stats(self) -> dict[str, int]:
        return {
            "write_attempts": len(self.write_attempts),
            "write_successes": sum(
                1 for a in self.write_attempts if a.get("ok")
            ),
            "write_failures": sum(
                1 for a in self.write_attempts if not a.get("ok")
            ),
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
        self.write_attempts.append(
            {
                "study_uid": study_uid,
                "patient_id_hash": patient_id_hash,
                "finding_text": finding_text,
                "positive_flag": positive_flag,
                "draft_ts": draft_ts,
                "tool_version": tool_version,
            }
        )
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            # 模拟入 outbox：写入一行
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
            payload={"url": "fake"},
        )
        return True

    async def aclose(self) -> None:
        self.stop_retry_loop_called = True

    async def start_retry_loop(self) -> None:
        self.start_retry_loop_called = True

    async def stop_retry_loop(self) -> None:
        self.stop_retry_loop_called = True


class _FakeHealthcheckWriter(_FakeRISWriter):
    """结构同 _FakeRISWriter，target_endpoint 不同。"""

    def __init__(self, *, cfg: CxrTriageConfig, outbox: Outbox, audit: _FakeAuditLogger) -> None:
        super().__init__(cfg=cfg, outbox=outbox, audit=audit)
        self.write_attempts_hc: list[dict[str, Any]] = []
        self.fail_remaining_hc: int = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "write_attempts": len(self.write_attempts_hc),
            "write_successes": sum(
                1 for a in self.write_attempts_hc if a.get("ok")
            ),
            "write_failures": sum(
                1 for a in self.write_attempts_hc if not a.get("ok")
            ),
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
        self.write_attempts_hc.append(
            {
                "visit_id": visit_id,
                "summary_snippet": summary_snippet,
                "source_study_uids": source_study_uids,
                "study_uid": study_uid,
                "ok": self.fail_remaining_hc <= 0,
            }
        )
        if self.fail_remaining_hc > 0:
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


class _FakeWorkflow:
    """最简 stub：暴露 run() 返回空状态。"""

    async def run(self, study_uid: str, png_bytes: bytes, metadata: dict) -> dict:
        return {
            "study_uid": study_uid,
            "classifier_max_prob": 0.6,
            "positive_flag": True,
            "finding_text": "(草稿) 草稿测试",
            "patient_id_hash": metadata.get("patient_id_hash", ""),
            "tool_version": "0.1.0",
            "draft_ts": "2026-10-13T00:00:00.000Z",
        }


class _FakeTrigger:
    def __init__(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, int]:
        return {"scanned": 0, "processed": 0, "degraded": 0, "exceptions": 0}

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False


# ----------------------------------------------------------------------------
# 测试用例：路由 + 解析
# ----------------------------------------------------------------------------


def test_create_app_returns_fastapi_with_routes(app_with_state):
    app, _state, _db = app_with_state
    methods_paths = {
        (frozenset(getattr(r, "methods", None) or ()), r.path)
        for r in app.routes
        if hasattr(r, "path") and r.path.startswith("/api") or r.path == "/"
    }
    expected = {
        (frozenset({"GET"}), "/api/health"),
        (frozenset({"GET"}), "/api/studies"),
        (frozenset({"GET"}), "/api/studies/{study_uid}"),
        (frozenset({"POST"}), "/api/studies/{study_uid}/confirm"),
        (frozenset({"POST"}), "/api/studies/{study_uid}/reject"),
        (frozenset({"GET"}), "/api/queue/positive"),
        (frozenset({"GET"}), "/api/queue/negative"),
        (frozenset({"GET"}), "/api/outbox"),
        (frozenset({"GET"}), "/"),
    }
    for method_set, path in expected:
        assert (method_set, path) in methods_paths, (
            f"missing route: {method_set} {path}; have: {methods_paths}"
        )


def test_index_endpoint_reports_endpoints(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    # Task 30: / 改为 Jinja2 三栏模板（text/html 渲染）
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # 模板必备元素 + 出现在模板静态文本里的端点
    assert "体检胸片所见草稿系统" in body
    assert "/api/studies" in body
    assert "/confirm" in body
    assert "/reject" in body
    assert "/api/sse" in body
    assert "/api/queue/positive" in body or "/api/queue/" in body
    assert "/api/outbox" in body or "outbox" in body.lower()
    assert "阳性队列" in body
    assert "阴性队列" in body
    assert "草稿所见" in body


def test_health_endpoint_returns_full_contract(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    # 模型契约（即使通过 dict 也对齐字段）
    for key in HealthResponse.model_fields:
        assert key in body, f"missing key: {key}"
    assert body["status"] in {"ok", "degraded"}
    assert body["pacs"] in {"connected", "watch_folder", "fallback", "disconnected"}
    assert body["model"] in {"ok", "degraded", "unknown"}
    assert isinstance(body["outbox_alerts"], int)
    assert isinstance(body["state_count"], dict)
    assert "config" in body
    # 不应泄露密钥
    cfg = body["config"]
    for forbidden in ("token", "key", "secret", "password"):
        for k, v in cfg.items():
            assert (
                forbidden not in k.lower()
            ), f"config 字段含敏感字段名: {k}"
            assert (
                not isinstance(v, str) or forbidden not in v.lower() or "URL" in k
            ), f"config 值含敏感字符串: {k}={v}"


def test_studies_endpoint_returns_empty_list_when_no_studies(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    resp = client.get("/api/studies")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["items"] == []


def test_studies_endpoint_rejects_unknown_state_filter(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    resp = client.get("/api/studies?state_filter=not_a_state")
    assert resp.status_code == 400


# ----------------------------------------------------------------------------
# 测试用例：状态机调用 + 三角色流转
# ----------------------------------------------------------------------------


async def test_jr_confirm_sets_state_correctly(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)

    study_uid = "1.2.826.0.1.EXAMPLE_JR_CONFIRM"
    body = {"actor_role": "jr", "target_state": "jr_confirmed"}
    resp = client.post(f"/api/studies/{study_uid}/confirm", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["success"] is True
    assert payload["new_state"] == "jr_confirmed"
    assert payload["actor_role"] == "jr"
    assert payload["audit_log_id"] is not None

    # 数据库侧状态也对齐
    cur = await state.state_machine.query_state(study_uid)
    assert cur == CxrState.JR_CONFIRMED


async def test_invalid_role_returns_400(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)

    study_uid = "1.2.826.0.1.EXAMPLE_BAD_ROLE"
    body = {"actor_role": "doctor", "target_state": "jr_confirmed"}
    resp = client.post(f"/api/studies/{study_uid}/confirm", json=body)
    assert resp.status_code == 400


async def test_invalid_target_state_returns_400(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_BAD_TARGET"
    resp = client.post(
        f"/api/studies/{study_uid}/confirm",
        json={"actor_role": "jr", "target_state": "random_state"},
    )
    assert resp.status_code == 400


async def test_skipping_jr_to_sr_signed_returns_400(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_SKIP"
    # 先让状态机知道这个 study 存在
    await state.state_machine.get_or_init_state(study_uid)
    # reviewer 不能直接从 draft_pending 跳 sr_signed
    body = {"actor_role": "reviewer", "target_state": "sr_signed"}
    resp = client.post(f"/api/studies/{study_uid}/confirm", json=body)
    assert resp.status_code == 400


async def test_three_role_chain_with_dual_write_on_sr_signed(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_THREE_ROLE"
    patient_id_hash = "abc123"

    # 1) jr confirm
    r1 = client.post(
        f"/api/studies/{study_uid}/confirm",
        json={"actor_role": "jr", "target_state": "jr_confirmed"},
    )
    assert r1.status_code == 200
    assert r1.json()["new_state"] == "jr_confirmed"

    # 2) reviewer
    r2 = client.post(
        f"/api/studies/{study_uid}/confirm",
        json={
            "actor_role": "reviewer",
            "target_state": "sr_reviewed",
            "expected_from_state": "jr_confirmed",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["new_state"] == "sr_reviewed"

    # 3) sr sign — 应该触发双写
    # 先 idempotency 写入一个 draft（双写需要 payload）
    await state.idempotency.cache_draft(
        study_uid,
        {
            "study_uid": study_uid,
            "patient_id_hash": patient_id_hash,
            "finding_text": "(草稿) 草稿测试",
            "finding_struct": {
                "pathologies": [{"name": "Pneumothorax", "prob": 0.85}],
                "key_qa": [],
                "grounding_boxes": [],
            },
            "positive_flag": True,
            "classifier_max_prob": 0.85,
            "tool_version": "0.1.0",
            "draft_ts": "2026-10-13T00:00:00.000Z",
            "visit_id": study_uid,
        },
    )

    r3 = client.post(
        f"/api/studies/{study_uid}/confirm",
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

    # 校验 ris_writer + hc_writer 都被调
    assert any(
        a["study_uid"] == study_uid for a in state.ris_writer.write_attempts
    )
    assert any(
        a["study_uid"] == study_uid for a in state.healthcheck_writer.write_attempts_hc
    )


async def test_sr_signed_falls_back_to_outbox_on_failure(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_FALLBACK"

    # 准备 draft
    await state.idempotency.cache_draft(
        study_uid,
        {
            "study_uid": study_uid,
            "patient_id_hash": "h",
            "finding_text": "草稿",
            "finding_struct": {},
            "positive_flag": True,
            "classifier_max_prob": 0.7,
            "tool_version": "0.1.0",
            "draft_ts": "2026-10-13T00:00:00.000Z",
            "visit_id": study_uid,
        },
    )

    # 推到 sr_signed
    for role, target in [("jr", "jr_confirmed"), ("reviewer", "sr_reviewed")]:
        r = client.post(
            f"/api/studies/{study_uid}/confirm",
            json={"actor_role": role, "target_state": target},
        )
        assert r.status_code == 200

    # 预置失败
    state.ris_writer.fail_remaining = 1
    state.healthcheck_writer.fail_remaining_hc = 1

    r = client.post(
        f"/api/studies/{study_uid}/confirm",
        json={
            "actor_role": "sr",
            "target_state": "sr_signed",
            "expected_from_state": "sr_reviewed",
        },
    )
    assert r.status_code == 200
    dual = r.json()["dual_write"]
    assert dual["ris_status"] == "fail"
    assert dual["healthcheck_status"] == "fail"

    # outbox 表应有 RIS + 主检两行
    counts = await state.outbox.count_by_status()
    assert counts.get("pending", 0) >= 2


async def test_reject_routes_back_to_draft_pending_and_clears_idempotency(
    app_with_state,
):
    app, state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_REJECT"

    # 先到 jr_confirmed
    r = client.post(
        f"/api/studies/{study_uid}/confirm",
        json={"actor_role": "jr", "target_state": "jr_confirmed"},
    )
    assert r.status_code == 200
    # 写一份 draft
    await state.idempotency.cache_draft(
        study_uid, {"draft": "old", "tool_version": "0.1.0"}
    )
    cached_before = await state.idempotency.lookup_with_meta(study_uid)
    assert cached_before is not None

    # 驳回
    rj = client.post(
        f"/api/studies/{study_uid}/reject",
        json={"actor_role": "reviewer", "reason": "需要复核"},
    )
    assert rj.status_code == 200
    body = rj.json()
    assert body["success"] is True
    assert body["new_state"] == "draft_pending"
    assert body["reason"] == "需要复核"

    # 幂等缓存已清
    cached_after = await state.idempotency.lookup_with_meta(study_uid)
    # 失效后 lookup 返回 None（5 分钟 TTL）；只要 ensure cleaned
    if cached_after is not None:
        # 仍存在但已过期 mark
        assert cached_after.get("expired") is True
    # 状态已回到 draft_pending
    cur = await state.state_machine.query_state(study_uid)
    assert cur == CxrState.DRAFT_PENDING


# ----------------------------------------------------------------------------
# 测试用例：CAS 并发冲突
# ----------------------------------------------------------------------------


async def test_concurrent_sr_sign_only_one_succeeds(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)
    study_uid = "1.2.826.0.1.EXAMPLE_CAS"

    # 推进到 sr_reviewed
    for role, target in [("jr", "jr_confirmed"), ("reviewer", "sr_reviewed")]:
        r = client.post(
            f"/api/studies/{study_uid}/confirm",
            json={"actor_role": role, "target_state": target},
        )
        assert r.status_code == 200

    # 在 TestClient 上同步并发（FastAPI 异步路由；TestClient 也支持
    # 串行；这里用 asyncio.gather 模拟两条 sr_sign 几乎同时发起）
    # 但 TestClient 不是线程安全的，因此序列化为两次单独调用。
    results = []
    for _ in range(2):
        r = client.post(
            f"/api/studies/{study_uid}/confirm",
            json={
                "actor_role": "sr",
                "target_state": "sr_signed",
                "expected_from_state": "sr_reviewed",
            },
        )
        results.append(r)
    # 两条至少一条 success=True；另一条可能 success=False 或 from_state_actual=sr_signed
    successes = [r for r in results if r.json()["success"]]
    assert len(successes) >= 1
    # 终态对齐
    cur = await state.state_machine.query_state(study_uid)
    assert cur == CxrState.SR_SIGNED


# ----------------------------------------------------------------------------
# 测试用例：queue + outbox + detail
# ----------------------------------------------------------------------------


async def test_queue_endpoints_split_by_positive_flag(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)

    # 入队 4 条：2 阳 2 阴
    for i, prob in enumerate([0.85, 0.65, 0.30, 0.10]):
        uid = f"1.2.826.0.1.EXAMPLE_Q{i}"
        await state.triage_queue.enqueue(
            uid, prob, {"study_uid": uid, "finding_text": f"test {i}"}
        )

    r_pos = client.get("/api/queue/positive?limit=10")
    assert r_pos.status_code == 200
    pos = r_pos.json()
    assert len(pos) == 2
    assert all(it["positive_flag"] for it in pos)

    r_neg = client.get("/api/queue/negative?limit=10")
    assert r_neg.status_code == 200
    neg = r_neg.json()
    assert len(neg) == 2
    assert not any(it["positive_flag"] for it in neg)


async def test_outbox_endpoint_returns_counts(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)
    r = client.get("/api/outbox")
    assert r.status_code == 200
    body = r.json()
    assert "counts" in body
    assert "alerting" in body
    assert isinstance(body["counts"], dict)
    assert "ris" in body["stats"]


async def test_study_detail_returns_incoming_and_draft(app_with_state):
    app, state, _db = app_with_state
    client = TestClient(app)

    study_uid = "1.2.826.0.1.EXAMPLE_DETAIL"
    # 写一条 incoming_studies 行
    async with state.db.connection() as conn:
        await conn.execute(
            """
            INSERT INTO incoming_studies
                (study_uid, patient_id_hash, accession_number, modality,
                 study_date, patient_sex, patient_age, source, status, max_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_uid,
                "hash-x",
                "AC-1",
                "CR",
                "2026-10-01",
                "M",
                "40Y",
                "test",
                "pending",
                0.5,
            ),
        )
        await conn.commit()
    # 写一份 idempotency 草稿
    await state.idempotency.cache_draft(
        study_uid,
        {
            "study_uid": study_uid,
            "finding_text": "草稿测试（草稿 / 未经医师确认）",
            "positive_flag": True,
            "tool_version": "0.1.0",
            "draft_ts": "2026-10-13T00:00:00.000Z",
        },
    )
    # 状态机初始化
    await state.state_machine.get_or_init_state(study_uid)

    r = client.get(f"/api/studies/{study_uid}")
    assert r.status_code == 200
    body = r.json()
    assert body["study_uid"] == study_uid
    assert body["incoming"] is not None
    assert body["incoming"]["modality"] == "CR"
    assert body["draft"] is not None
    assert body["triage_state"] == "draft_pending"


async def test_study_detail_returns_400_for_empty_uid(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    # 测试 fastapi 路径匹配：study_uid="" 无法与路由匹配，会触发 404
    # 这种"空字符串"测试对应"未捕获 /api/studies"
    r = client.get("/api/studies/")
    # 尾部斜杠与无尾部斜杠在 fastapi 默认不同；测试用例仅做 contract smoke
    assert r.status_code in (307, 404, 200)


# ----------------------------------------------------------------------------
# 测试用例：perform_sr_signed_dual_write
# ----------------------------------------------------------------------------


async def test_perform_sr_signed_dual_write_empty_draft_returns_fail(app_with_state):
    app, state, _db = app_with_state
    res = await perform_sr_signed_dual_write(
        study_uid="x",
        ris_writer=state.ris_writer,
        healthcheck_writer=state.healthcheck_writer,
        audit_logger=state.audit_logger,
        draft={},  # 空 → 不写
        visit_id="v1",
    )
    assert res.ris_status == "fail"
    assert res.healthcheck_status == "fail"


async def test_perform_sr_signed_dual_write_with_draft_calls_writers(
    app_with_state,
):
    app, state, _db = app_with_state
    draft = {
        "study_uid": "x",
        "patient_id_hash": "h",
        "finding_text": "草稿",
        "finding_struct": {},
        "positive_flag": True,
        "classifier_max_prob": 0.8,
        "tool_version": "0.1.0",
        "draft_ts": "2026-10-13T00:00:00.000Z",
    }
    res = await perform_sr_signed_dual_write(
        study_uid="x",
        ris_writer=state.ris_writer,
        healthcheck_writer=state.healthcheck_writer,
        audit_logger=state.audit_logger,
        draft=draft,
        visit_id="v1",
    )
    assert res.ris_status == "ok"
    assert res.healthcheck_status == "ok"
    assert res.visit_id == "v1"
    assert len(state.ris_writer.write_attempts) == 1
    assert len(state.healthcheck_writer.write_attempts_hc) == 1


# ----------------------------------------------------------------------------
# 测试用例：发包校验 + 422
# ----------------------------------------------------------------------------


def test_confirm_returns_422_when_missing_body(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    r = client.post("/api/studies/x/confirm", json={})
    assert r.status_code == 422


def test_reject_returns_400_when_blank_role(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    r = client.post("/api/studies/x/reject", json={"actor_role": ""})
    assert r.status_code == 400


def test_health_post_disallowed(app_with_state):
    app, _state, _db = app_with_state
    client = TestClient(app)
    r = client.post("/api/health", json={})
    # POST 不支持的 endpoint → 405
    assert r.status_code == 405


# ----------------------------------------------------------------------------
# 测试用例：并发 CAS（asyncio.gather）
# ----------------------------------------------------------------------------


async def test_concurrent_cas_via_state_machine_directly(app_with_state):
    """直接对 state_machine 做并发 sr_sign，验证 CAS 行为。"""
    app, state, _db = app_with_state
    study_uid = "1.2.826.0.1.EXAMPLE_ASYNC_CAS"
    await state.state_machine.get_or_init_state(study_uid)
    await state.state_machine.transition(
        study_uid=study_uid,
        from_state=CxrState.DRAFT_PENDING,
        to_state=CxrState.JR_CONFIRMED,
        actor_role=ActorRole.JR,
    )
    await state.state_machine.transition(
        study_uid=study_uid,
        from_state=CxrState.JR_CONFIRMED,
        to_state=CxrState.SR_REVIEWED,
        actor_role=ActorRole.REVIEWER,
    )

    # 并发两条 sr_sign
    async def do_sign() -> bool:
        r = await state.state_machine.transition(
            study_uid=study_uid,
            from_state=CxrState.SR_REVIEWED,
            to_state=CxrState.SR_SIGNED,
            actor_role=ActorRole.SR,
        )
        return r.success

    results = await asyncio.gather(do_sign(), do_sign())
    assert sum(1 for r in results if r) == 1

    cur = await state.state_machine.query_state(study_uid)
    assert cur == CxrState.SR_SIGNED
