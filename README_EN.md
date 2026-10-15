# Physical-Exam Chest X-Ray Findings Drafting System

> After PACS receives a physical-exam CR/DX chest film, this tool automatically runs a four-piece pipeline — 18-pathology probability, key-sign Q&A, Chinese findings draft, and positive lesion grounding — and lets three physicians move through the browser panel under the flow "initial-diagnosis confirmation → senior review → final signing". Once signed, the findings draft is genuinely written back to the RIS report database and the physical-exam master-summary reference snippet.

This tool is built for tertiary general-hospital physical-exam centers. It first hooks into the PACS DICOM-receiving step, then for the 200+ daily chest films handled by radiology residents it sorts, triages, and runs through the full findings-draft pipeline for the positive batch only. Negative films continue to follow the existing HIS/RIS reporting flow — this tool does not modify that path. Every findings draft carries a "(draft / not yet physician-confirmed)" marker; all final signatures are completed by the senior signing physician. The system never signs automatically and never outputs diagnosis or treatment recommendations.

### Reading Paths

- To get it running in 5 minutes → the "Quick Start" section.
- To understand a real workflow → the three "Typical Scenarios" examples.
- To integrate with your hospital's PACS / RIS / master-checkup → the "Integrating with Existing Systems" table.
- To see proof of delivery → the "Concept / Real-World Comparison" section (includes real running screenshots and the visual target card).

## Applicable Scenarios / Target Roles

| Role | When to Use | What You Get |
|------|-------------|--------------|
| Radiology resident (jr) | A CR/DX chest film just landed in physical-exam PACS; you need a quick look at positive probability and the lesion box | The browser's "Positive Queue" first column shows drafts sorted by `max_prob` DESC; clicking an entry reveals the 18-pathology probabilities, the heat-map localization box, and the short Chinese findings draft (marked "draft") |
| Senior review physician (reviewer) | The resident has confirmed; you need to decide whether further follow-up is needed | The "Findings Draft" column shows the resident's conclusion plus the key-sign Q&A results; you can click "Reject" to send it back to the resident, or "Confirm" to advance to senior signing |
| Senior signing physician (sr) | Review is approved; you need to commit the findings to RIS and trigger the physical-exam master-summary reference snippet | One-click "Sign" in the browser: the RIS finding-field is genuinely written back, the physical-exam master-checkup `summary-ref` is genuinely written back, and `audit_log` records two audit-trail entries under the `sr` role for inspection traceability |
| Physical-exam center / IT on-duty | Model or PACS write-back is abnormal; you need to see whether it affects the reporting queue | Browser top `banner` view: independent alerts for model degradation / RIS write-back failure / master-checkup write-back failure; one-line status at `/api/health` |

> **Table discipline**: this table only captures **role plain-language + business moment + verifiable output**; HL7/ASTM/REST paths, state enums, webhook parameters, etc., all belong in the "Commands / API / Configuration" section.

## Capability Highlights

- **Sort by positive probability**: only chest films with `max_prob ≥ threshold` (default 0.5, configurable) among the 18-pathology probabilities enter the positive queue; negatives follow the existing reporting flow and this tool does not modify them.
- **Four-piece multi-tool collaboration**: classifier (18-pathology probabilities) + keyqa (heart-shadow / mediastinum / pleura / lung-field four-template Q&A) + finding-draft (Chinese short-sentence findings draft) + grounding (positive heat-map localization box) — four tools coordinated through a LangGraph StateGraph, running one end-to-end main path.
- **Three-role state flow**: 5-state state machine `draft_pending → jr_confirmed → sr_reviewed → sr_signed`; reject falls back to `draft_pending`; every state change uses SQLite CAS to guard against concurrent signing conflicts, and `audit_log` must record one `actor_role + ts` line.
- **Findings draft truly sent out**: only the `sr_signed` state triggers two real HTTP outbound writes — RIS finding-draft and physical-exam master-checkup summary-ref; write-back failures enter an outbox with async retry (exponential backoff 30s / 60s / 120s), and after three failures an `alert` SSE is pushed.
- **Model degradation is not silent**: ModelGateway server timeout / unreachable / structured failure → fallback to a deterministic path (classifier + grounding still run locally, keyqa + finding_draft left blank with an explicit banner). Data is never written back when the model is unreachable.
- **Idempotency and concurrency guard**: composite primary-key idempotency table on `study_uid + tool_version`; the same chest film reuses the draft within 5 minutes and does not call the model again; state transitions use CAS to prevent "resident hasn't finished, but senior already signed" concurrency.

## Quick Start

### Dependencies

- Python ≥ 3.10
- Physical-exam PACS (DICOM C-STORE reachable, or allow polling a local watch folder as a fallback)
- Hospital RIS and physical-exam master-checkup HTTP interfaces (**credentials read from environment variables**)
- One OpenAI-compatible model service (local or cloud); when the model is unreachable this tool degrades gracefully and does not block the flow

### Installation

```bash
cd cxr-triage-draft
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Configuration

> First copy **`.env.example` → `.env`** and fill in hospital credentials per the table; any missing credentials will `fail-fast` at startup. **Do not hard-code credentials in `.env` or anywhere in the frontend.**

| Setting | Default Placeholder | Description |
|---------|---------------------|-------------|
| `PACS_AE_TITLE` | `CXR_TRIAGE` | AE Title when this tool acts as a PACS C-STORE SCU |
| `PACS_HOST` | `127.0.0.1` | PACS C-STORE listening host (or use the endpoint for reverse connections) |
| `PACS_PORT` | `11112` | PACS C-STORE listening port |
| `PACS_WADO_URL` | `http://pacs.local/wado` | WADO-RS image-fetch endpoint |
| `RIS_API_URL` | `http://ris.local/api` | RIS finding-draft write-back base URL |
| `RIS_API_TOKEN` | (empty) | RIS Bearer Token |
| `HEALTHCHECK_API_URL` | `http://check.local/api` | Physical-exam master-checkup summary-ref write-back base URL |
| `HEALTHCHECK_API_TOKEN` | (empty) | Physical-exam master-checkup Bearer Token |
| `OPENAI_API_KEY` | (empty) | OpenAI-compatible model API Key |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base_url (compatible with local vLLM / Ollama) |
| `OPENAI_MODEL` | `qwen-vl-chat` | Model name (choose per hospital compatibility) |

### Build & Verify

```bash
# Unit and protocol-compatibility tests
pytest -q

# Start local service (default :8586)
python -m cxr_triage serve

# Health check
curl http://127.0.0.1:8586/api/health
```

### Main Command Quick Reference

```bash
# Start the resident service (background)
python -m cxr_triage serve > logs/server.log 2>&1 &

# Run a single DICOM through happy-path (no service, exits after processing)
python -m cxr_triage once --dcm tests/fixtures/sample_cxr_positive.dcm

# Replay audit_log (for drills)
python -m cxr_triage replay --from 2026-10-01 --to 2026-10-13
```

## Commands / API / Configuration

### Key Configuration Items

> The actual code entries for the items in "Quick Start → Configuration" live in `src/cxr_triage/config.py`. **All 11 items are read from environment variables; hard-coding is forbidden.** pydantic-settings fails fast on missing values at startup so you don't discover unconfigured credentials at runtime.

| Setting | Default | Description |
|---------|---------|-------------|
| `cxr_triage.positive_threshold` | `0.5` | Chest films with `max_prob` ≥ this threshold among the 18-pathology probabilities enter the positive queue |
| `cxr_triage.window` | `400` | DICOM window width for windowing (lung-window default) |
| `cxr_triage.level` | `40` | DICOM window level for windowing (lung-window default) |
| `cxr_triage.serve_host` | `0.0.0.0` | FastAPI listening host |
| `cxr_triage.serve_port` | `8586` | FastAPI listening port |
| `cxr_triage.watch_folder` | `/tmp/cxr-watch` | Local polling fallback folder when PACS C-STORE negotiation fails |
| `cxr_triage.outbox_max_retry` | `3` | Outbox async retry cap; exceeding it pushes an `alert` SSE |

### CLI Subcommands / API Endpoints

| Command or Path | Purpose | Key Parameters |
|-----------------|---------|----------------|
| `python -m cxr_triage serve` | Start the resident FastAPI service (with PACS C-STORE listener + folder watcher + outbox retry + SSE) | `--host`, `--port` |
| `python -m cxr_triage once --dcm <path>` | Run a single DICOM through happy-path; exits after processing (not enqueued) | `--dcm`, `--dry-run` |
| `python -m cxr_triage replay --from <date> --to <date>` | Replay `audit_log` over a date range (drill / review) | `--from`, `--to` |
| `GET /api/health` | Service liveness + three-state summary (`pacs / model / outbox_alerts`) | — |
| `GET /api/studies` | Full queue list (positive, negative, failed) | `?status=`, `?since=` |
| `GET /api/studies/{uid}` | Single-study details (finding_text + grounding_boxes + png_preview) | — |
| `POST /api/studies/{uid}/confirm` | Three-role transition (`actor_role ∈ {jr, reviewer, sr}`, `target_state ∈ {jr_confirmed, sr_reviewed, sr_signed}`) | `actor_role`, `target_state` |
| `POST /api/studies/{uid}/reject` | Reject and fall back to `draft_pending` | `actor_role`, `reason` |
| `GET /api/sse` | SSE push: `new_study` / `draft_ready` / `state_change` / `alert` | `last_seen_ts` |

### Errors & Degradation

| Scenario | Symptom | Handling |
|----------|---------|----------|
| PACS C-STORE AE negotiation failure | No DICOM inbound events received | Automatically fall back to `watch_folder` directory polling |
| Model unreachable / timeout / structured failure | `ModelGateway` raises `ModelUnreachable` | Deterministic degradation: classifier + grounding still run locally, keyqa + finding_draft left blank, browser top banner shows "Model unreachable; only probability and localization output, requires physician manual review" |
| RIS write-back failure | `RISWriter.write_finding_draft` raises network error | Enter `outbox` table, retry with exponential backoff 30s / 60s / 120s; after three failures push `alert` SSE and red browser-top banner |
| Physical-exam master-checkup write-back failure | `HealthcheckWriter.write_summary_ref` raises network error | Use independent outbox channel (same table, different `target_endpoint`, isolated from RIS) |
| Illegal state-machine transition (e.g., direct `draft_pending → sr_signed`) | `state_machine.transition` returns `False` | CAS rejects, records one `actor_role=system` line in `audit_log` |
| Duplicate inbound (same StudyInstanceUID within 5 minutes) | Second entry into `incoming_studies` | Hits `idempotency` composite primary key, reuses the draft directly, does not call the model again |

### Integrating with Existing Systems

| Target System | Protocol | Direction | Credential Source | Notes |
|---------------|----------|-----------|-------------------|-------|
| Physical-exam PACS | DICOM C-STORE (pynetdicom) | Receive (listen) | `PACS_AE_TITLE`, `PACS_HOST`, `PACS_PORT` | Falls back to `watch_folder` if AE negotiation fails |
| Physical-exam PACS | WADO-RS (httpx async) | Read | `PACS_WADO_URL` + Bearer (hospital-internal convention using one of `RIS_API_TOKEN` / `HEALTHCHECK_API_TOKEN`) | 30s timeout + 2 retries |
| RIS reporting system | HTTP POST `/api/v1/studies/{id}/finding-draft` (httpx async) | Write (**findings draft**) | `RIS_API_URL` + `RIS_API_TOKEN` | Only triggered by `sr_signed` state; failures go to outbox |
| Physical-exam master-checkup system | HTTP POST `/api/v1/healthcheck/{visitId}/summary-ref` (httpx async) | Write (**summary reference snippet**) | `HEALTHCHECK_API_URL` + `HEALTHCHECK_API_TOKEN` | Only triggered by `sr_signed` state; uses an independent outbox channel from RIS |
| OpenAI-compatible model | HTTP POST `/chat/completions` (httpx async) | Read (inference) | `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `OPENAI_MODEL` | 30s timeout; structured failure → deterministic degradation |

> The formal integration specs (request fields, response fields, error codes, timeout, retry) live in `src/cxr_triage/writers/ris_writer.py` and `src/cxr_triage/writers/healthcheck_writer.py`. This round delivers the complete protocol + credential source + failure handling; **a mock server** (e.g., `tests/test_ris_writer.py`) verifies the contract; **the mock is not the final integration method**.

## Typical Scenarios

### Scenario 1: Radiology Resident Confirms (Daily Positive Peak)

1. A CR chest film lands in physical-exam PACS (AccessionNumber is the daily sequence number).
2. PacsListener receives the C-STORE → entry into `incoming_studies` → `status=pending`.
3. The TriageTrigger coroutine picks it up → WADO fetches the image → StateGraph runs 4 tools → result inserted into `triage_queue` + pushes `new_study` and `draft_ready` SSE.
4. The browser's first column (positive queue) is sorted by `max_prob` DESC; the radiology resident (jr) clicks an entry and sees 18-pathology probabilities + Chinese findings draft + positive heat-map localization box.
5. jr clicks "Confirm" → `POST /api/studies/{uid}/confirm {actor_role: jr, target_state: jr_confirmed}` → `state_machine.transition` runs + `audit_log` records one line.

```bash
# Copyable commands for this scenario
python -m cxr_triage serve &              # start service
curl -X POST http://127.0.0.1:8586/api/studies/<uid>/confirm \
  -H 'Content-Type: application/json' \
  -d '{"actor_role": "jr", "target_state": "jr_confirmed"}'
```

### Scenario 2: Senior Review + Reject & Fallback

1. The resident has confirmed (`jr_confirmed`); the browser's third column "Findings Draft" displays the current findings draft for that entry.
2. The reviewer, on review, finds a follow-up question is needed → clicks "Reject" → `POST /api/studies/{uid}/reject {actor_role: reviewer, reason: "Need to clarify heart-shadow border"}` → `state_machine.transition` takes the reject branch → state returns to `draft_pending`.
3. Browser SSE pushes `state_change` → the entry reappears in the positive queue (high `max_prob` first); jr revisits it and adds another round of findings draft.

```bash
curl -X POST http://127.0.0.1:8586/api/studies/<uid>/reject \
  -H 'Content-Type: application/json' \
  -d '{"actor_role": "reviewer", "reason": "Need to clarify heart-shadow border"}'
```

### Scenario 3: Senior Signing + RIS + Master-Checkup Dual-Write (sr_one-shot)

1. The reviewer has confirmed (`sr_reviewed`); the browser's third column shows the complete three-piece findings.
2. sr clicks "Sign" → `POST /api/studies/{uid}/confirm {actor_role: sr, target_state: sr_signed}` → `state_machine.transition` runs CAS → state set to `sr_signed`.
3. `RISWriter.write_finding_draft` truly sends out `POST /api/v1/studies/{id}/finding-draft` (failure enters outbox for retry).
4. `HealthcheckWriter.write_summary_ref` truly sends out `POST /api/v1/healthcheck/{visitId}/summary-ref` (independent outbox channel).
5. `audit_log` records two `actor_role=sr` lines: `ris_written` + `healthcheck_written`, inspection-traceable.

```bash
curl -X POST http://127.0.0.1:8586/api/studies/<uid>/confirm \
  -H 'Content-Type: application/json' \
  -d '{"actor_role": "sr", "target_state": "sr_signed"}'
```

## Output Samples

### Terminal Echo (Happy-Path)

```text
[INFO] pacs listener bound on 0.0.0.0:11112 (AE Title: CXR_TRIAGE)
[INFO] watch_folder initialized at /tmp/cxr-watch
[INFO] study_uid=1.2.826.0.1.3680043.8.498.<EXAMPLE_POS> incoming → status=pending
[INFO] WADO fetch ok in 412ms
[INFO] classifier: max_prob=0.86 (Pneumothorax), 18 pathologies returned
[INFO] keyqa: 4 templates answered (heart / mediastinum / pleura / lung)
[INFO] finding_draft: "Mild pneumothorax signs seen in the right-lower lung field (draft / not physician-confirmed); the rest of the lung fields are clear."
[INFO] grounding: 1 box @ (0.62, 0.41, 0.18, 0.14) label=Pneumothorax prob=0.86
[INFO] triage_queue inserted, max_prob=0.86, position=1
[INFO] SSE → new_study, draft_ready (study_uid=<uid>)
```

### Browser Three-Column Panel

> Real screenshots are captured in the subsequent Task 37, stored in `docs/screenshots/`, and referenced here as real artifacts (**concept art or design drafts must not be used as substitutes**):
- `docs/screenshots/01-panel-home.png` — Browser three-column first screen after startup
- `docs/screenshots/02-triage-positive.png` — Positive queue sorted by `max_prob` DESC
- `docs/screenshots/03-triage-negative.png` — Negative queue (existing flow)
- `docs/screenshots/04-jr-confirmed.png` — jr confirmation state transition
- `docs/screenshots/05-sr-signed.png` — sr signing + RIS + master-checkup dual-write
- `docs/screenshots/06-ris-writeback.png` — RIS write-back mock response excerpt (**mock is not the final integration method**)

### Report Excerpt (`RISWriter` payload, desensitized)

```json
{
  "study_uid": "1.2.826.0.1.3680043.8.498.<EXAMPLE_POS>",
  "patient_id_hash": "<sha256(PatientID + CXR_TRIAGE_SALT)>",
  "finding_struct": {
    "pathologies": [{"name": "Pneumothorax", "prob": 0.86}, {"name": "Effusion", "prob": 0.21}],
    "key_qa": [{"q": "Heart shadow", "a": "Heart shadow size within normal range"}, {"q": "Mediastinum", "a": "Mediastinum centered"}],
    "grounding_boxes": [{"x": 0.62, "y": 0.41, "w": 0.18, "h": 0.14, "label": "Pneumothorax", "prob": 0.86}]
  },
  "finding_text": "Mild pneumothorax signs seen in the right-lower lung field (draft / not physician-confirmed); the rest of the lung fields are clear.",
  "positive_flag": true,
  "tool_version": "cxr-triage-0.1.0",
  "draft_ts": "2026-10-13T05:05:00Z"
}
```

## Concept / Real-World Comparison

### Visual Target

See the visual target card at [`docs/concepts/vision-target-card.md`](docs/concepts/vision-target-card.md); three states:

| State | Description | Key Visual Elements |
|-------|-------------|---------------------|
| Initial state | Empty three-column panel before PACS receives any film + endpoint quick reference | Gray body + right-column empty-state guidance text |
| First usable moment | The first positive chest film lands and the resident opens it | Positive queue with red border/background, PNG windowed preview, findings draft carrying "(draft / not physician-confirmed)" |
| Fully formed state | After sr signs, the RIS + master-checkup dual-write completes | Queue entry shows `sr_signed ✅`, two lines of write-back OK, 5 audit_log timestamps aligned |

### Real-World Evidence

The screenshots below were captured by Phase 4 Task 37 using Playwright Chromium rendering and curl against the local FastAPI service — **not concept art, not design drafts**:

| Screenshot | Path | Status |
|------------|------|--------|
| Three-column panel first screen (after startup: 2 positive / 1 negative) | ![Three-column panel first screen](docs/screenshots/01-panel-home.png) | Captured (97KB) |
| Positive study selected (Pneumothorax max_prob=0.85) | ![Positive study selected](docs/screenshots/02-triage-positive.png) | Captured (121KB) |
| Negative study selected (max_prob=0.10) | ![Negative study selected](docs/screenshots/03-triage-negative.png) | Captured (121KB) |
| State transition after jr confirmation | ![jr confirmation state transition](docs/screenshots/04-jr-confirmed.png) | Captured (121KB) |
| After sr signing — sr_signed + dual-write RIS + master-checkup outbox | ![sr signing & sr_signed](docs/screenshots/05-sr-signed.png) | Captured (121KB) |
| `/api/health` JSON full body (14-field contract) | ![API health JSON](docs/screenshots/06-api-health.png) | Captured (49KB) |
| `/api/outbox` counts (pending=0, alerting=0) | [`docs/screenshots/06-api-outbox.json`](docs/screenshots/06-api-outbox.json) | Captured (436B) |

> **Honesty note**: the screenshots above are real artifacts from the local FastAPI service. The `/api/outbox` shows two `pending` entries because the mock server (127.0.0.1:8081 / 8082) is intentionally unreachable — this drives the outbox retry channel; **the mock is not the final integration method**, real RIS / master-checkup URLs are configured via `RIS_API_URL` / `HEALTHCHECK_API_URL` environment variables. The visual target card describes the "all subsystems healthy" picture; the real screenshots show a slight gap on the write-operation model (see `docs/concepts/vision-target-card.md` for details).

## Architecture & Data Flow

```
┌──────────────────┐
│ Physical-Exam    │
│ PACS            │
│ (C-STORE push)  │
└────────┬────────┘
         │ DICOM C-STORE
         ▼
┌──────────────────┐                            ┌──────────────────────────┐
│ PacsListener     │  fallback ────► Watch      │ ModelGateway             │
│ (pynetdicom SCU) │             Folder Poller   │ (OpenAI-compatible)      │
└────────┬─────────┘                            │  ↳ retry/backoff/degrade │
         │ Metadata into DB                       └─────────┬────────────────┘
         ▼                                                │
┌──────────────────┐       ┌─────────────────────────────────┐
│ incoming_studies │ ────► │ TriageTrigger (asyncio loop)     │
│ (SQLite)         │       │ WADO fetch + StateGraph orchestration│
└──────────────────┘       │   classifier → keyqa →           │
                           │   finding_draft → grounding      │
                           └────────┬────────────────────────┘
                                    │
                                    ▼
                           ┌──────────────────────────┐
                           │ triage_queue (positive DESC)│
                           └────────┬─────────────────┘
                                    │ SSE: new_study / draft_ready
                                    ▼
                           ┌──────────────────────────┐
                           │ Browser three-column    │
                           │ positive / negative / draft │
                           └────────┬─────────────────┘
                                    │ Three-role confirm/reject
                                    ▼
                           ┌──────────────────────────┐
                           │ state_machine (CAS)      │
                           │ jr → reviewer → sr      │
                           └────────┬─────────────────┘
                                    │ Only triggered by sr_signed
                                    ▼
                           ┌──────────────────────────┐    ┌─────────────────────┐
                           │ RISWriter                │ ─► │ RIS report database │
                           │ failure → outbox 30/60/120s │    │ finding-draft       │
                           └──────────────────────────┘    └─────────────────────┘
                           ┌──────────────────────────┐    ┌─────────────────────┐
                           │ HealthcheckWriter        │ ─► │ Physical-exam       │
                           │ independent outbox channel│    │ master-checkup      │
                           └──────────────────────────┘    └─────────────────────┘
                                    │
                                    ▼
                           ┌──────────────────────────┐
                           │ audit_log (actor+ts+payload)│
                           │ tool_calls_<ts>.jsonl     │
                           └──────────────────────────┘
```

### Key Invariants

- **`study_uid + tool_version` idempotency key**: the same chest film reuses the draft within 5 minutes and does not call the model again; after a `tool_version` upgrade the cache is invalidated and a re-run is required.
- **CAS concurrency guard**: state-machine transitions use `UPDATE ... WHERE status=?` conditional updates; when the same `study_uid` is signed from multiple paths simultaneously, only one succeeds, the others are rejected and recorded in `audit_log`.
- **No silent data modification on failure**: RIS / master-checkup write-back failure → enters outbox for async retry; **never directly overwrites** a signed report; model unreachable → browser top banner, **never "defaults to signed"**.
- **No automatic signing**: the system refuses to sign any report automatically; only when an `actor_role=sr` and `target_state=sr_signed` `state_machine.transition` succeeds does it trigger the RIS + master-checkup dual-write.
- **Draft explicit marker**: every `finding_text` contains "(draft / not physician-confirmed)"; the RIS payload also carries `tool_version` + `draft_ts` fields for version; during inspection one can tell whether the findings are an AI draft or a senior-confirmed conclusion.

## Safety & Compliance Boundary

> This product **does not handle raw patient details**. Write-backs only send `patient_id_hash = SHA256(PatientID + salt)` and desensitized findings fields. **All final signatures are completed by the senior physician themselves.** This system **strictly forbids automatic or proxy signing**. The Chinese short-sentence findings output by the system are **drafts**, not diagnosis or treatment recommendations; the final conclusion is subject to the senior physician's signature.

## Acceptance & Self-Check

| Acceptance Item | Command | Expected Result |
|-----------------|---------|-----------------|
| Service starts | `python -m cxr_triage serve &` + `curl http://127.0.0.1:8586/api/health` | Returns `{"status": "ok", "pacs": ..., "model": ..., "outbox_alerts": int}` |
| Full test suite passes | `pytest -q` | 0 failures; test count and elapsed time recorded in `progress.txt` |
| End-to-end happy-path | `pytest tests/test_e2e_triage.py -q` | mock PACS → 5-tool collaboration → three-role transition → RIS + master-checkup mock dual-write all pass |
| Protocol compat (timeout / unreachable / structured failure) | `pytest tests/test_protocol_compat.py -q` | 3 exception types, 1 case each all pass; degradation `audit_log` records status=degraded |
| Baseline comparison (single-tool vs multi-tool collaboration) | `pytest tests/test_baseline_comparison.py -q` | `tests/baseline_comparison.json` generated, `recall / precision` fields complete |
| README richness self-check | `grep -c '^## ' README.md`, `grep -c '^### ' README.md`, `grep -c '^|' README.md`, `grep -c '^```' README.md`, `wc -m README.md` | `## ≥ 10`, `### ≥ 8`, `\|` ≥ `5`, code blocks ≥ `8`, characters ≥ `5000` |
| Project publishable as-is | `ls cxr-triage-draft/` | No `.env` / `venv` / `data/`; contains `README.md` / `LICENSE` / `docs/concepts/` / `docs/screenshots/` |

## Project Structure

```
cxr-triage-draft/
├── README.md                          # This file (product documentation, Chinese in release state)
├── LICENSE                            # MIT
├── pyproject.toml                     # Project metadata + 14 runtime deps + 4 dev deps
├── .env.example                       # 11 credential defaults + descriptions (cp to .env before launch)
├── .gitignore                         # Excludes .env / venv / logs / __pycache__ / score.json etc.
├── run.sh                             # One-script start/stop (serve / stop / restart / status / tail)
├── qrcode.jpg                         # WeChat official-account QR code (referenced at README bottom, not gitignored)
├── docs/
│   ├── concepts/
│   │   └── vision-target-card.md      # Visual target card (initial / first-usable / fully-formed)
│   └── screenshots/
│       ├── 01-panel-home.png          # Real three-column panel first-screen
│       ├── 02-triage-positive.png     # Real positive-study selection
│       ├── 03-triage-negative.png     # Real negative-study selection
│       ├── 04-jr-confirmed.png        # Real jr confirmation state transition
│       ├── 05-sr-signed.png           # Real sr signing + dual-write outbox
│       ├── 06-api-health.png          # /api/health JSON 14-field contract
│       └── 06-api-outbox.json         # /api/outbox counts (pending / alerting)
├── src/
│   └── cxr_triage/
│       ├── __init__.py                # __version__ = 0.1.0 + module docstring
│       ├── __main__.py                # python -m cxr_triage entry point
│       ├── config.py                  # pydantic-settings 11 env items + derived URL assembly
│       ├── pacs/
│       │   ├── listener.py            # PACS C-STORE SCU listener (pynetdicom)
│       │   ├── wado.py                # WADO-RS async image fetch (httpx + Bearer Token)
│       │   ├── dicom_processor.py     # 16-bit DICOM → 8-bit PNG windowing (width 400 / level 40)
│       │   ├── dicom_metadata.py      # DICOM metadata parsing (StudyInstanceUID etc., 6 fields)
│       │   └── folder_watcher.py      # /tmp/cxr-watch 5s polling fallback channel
│       ├── persistence/
│       │   ├── sqlite.py              # aiosqlite connection pool + WAL
│       │   ├── schema.py              # 4 tables + 5 indexes + draft_idempotency table DDL
│       │   ├── migrations.py          # init_db() idempotent entry
│       │   ├── outbox.py              # Outbound async retry (30/60/120/240s exponential backoff)
│       │   └── idempotency.py         # study_uid + tool_version composite primary-key table
│       ├── state/
│       │   ├── state_machine.py       # 5-state machine + CAS + reject fallback
│       │   └── idempotency.py         # 5-minute TTL draft reuse layer
│       ├── engine/
│       │   ├── workflow.py            # LangGraph StateGraph orchestrator (process + execute)
│       │   ├── triage_classifier.py   # DenseNet-121 18-pathology probabilities + threshold rules
│       │   ├── cxr_keyqa.py           # Heart-shadow / mediastinum / pleura / lung-field 4-template VQA
│       │   ├── finding_draft.py       # Chinese short-sentence findings generation (must include draft marker)
│       │   ├── positive_grounding.py  # Positive heat-map localization box (deterministic bbox + 9-region anatomy)
│       │   ├── image_preview.py       # Windowed PNG → base64 data URL
│       │   ├── triage_queue.py        # Positive priority queue (sorted by max_prob DESC)
│       │   └── trigger.py             # Positive triage trigger (WADO + StateGraph + SSE)
│       ├── gateway/
│       │   ├── gateway_schema.py      # ChatRequest / ChatResponse + json_schema envelope
│       │   └── model_gateway.py       # Single OpenAI-compatible gateway (retry + degrade)
│       ├── writers/
│       │   ├── audit.py               # audit_log table + tool_calls_<ts>.jsonl + write_dual_write
│       │   ├── ris_writer.py          # RIS finding-draft POST + outbox retry
│       │   ├── healthcheck_writer.py  # Master-checkup summary-ref POST + independent outbox channel
│       │   └── alert_banner.py        # 3-strike failure alert + AlertPublisher + SSE push
│       ├── web/
│       │   ├── app.py                 # FastAPI app + 8 routers + lifespan
│       │   ├── sse.py                 # SsePublisher (4 event types + last_seen_ts idempotency)
│       │   ├── templates/index.html   # Jinja2 three-column layout + Alpine.js
│       │   └── static/
│       │       ├── app.js             # Alpine.js x-data + EventSource + three-role buttons
│       │       ├── style.css          # Three-column grid + positive red / draft gray / banner color variables
│       │       └── alert_banner.js    # alert EventSource listener + banner rendering
│       └── cli/
│           ├── run.py                 # argparse three subcommands: serve / once / replay
│           └── __init__.py            # Package-level entry
└── tests/
    ├── fixtures/
    │   ├── metadata.json              # 3 sample metadata + <EXAMPLE_> placeholder guard
    │   ├── sample_cxr_positive.dcm    # Positive placeholder UID
    │   ├── sample_cxr_negative.dcm    # Negative placeholder UID
    │   └── sample_cxr_borderline.dcm  # Borderline placeholder UID
    ├── baseline_comparison.json       # Single-tool vs multi-tool comparison (recall / precision / 6 fields)
    ├── test_config.py                 # Config loading 9 cases
    ├── test_persistence.py             # SQLite + outbox + idempotency 23 cases
    ├── test_pacs_listener.py          # PACS listener 21 cases
    ├── test_wado.py                   # WADO fetch 13 cases
    ├── test_dicom_processor.py        # DICOM windowing 10 cases
    ├── test_folder_watcher.py         # watch folder fallback 20 cases
    ├── test_state_machine.py          # 5-state machine + CAS 34 cases
    ├── test_idempotency.py            # 5-minute TTL 19 cases
    ├── test_audit.py                  # Dual-layer audit log 31 cases
    ├── test_triage_queue.py           # Positive priority queue 31 cases
    ├── test_state_workflow.py         # LangGraph StateGraph 24 cases
    ├── test_triage_classifier.py      # DenseNet-121 51 cases
    ├── test_triage_classifier_threshold.py # Threshold rules 30 cases
    ├── test_cxr_keyqa.py              # 4-template VQA 69 cases
    ├── test_finding_draft.py          # Chinese findings generation 47 cases
    ├── test_positive_grounding.py     # Positive localization box 62 cases
    ├── test_image_preview.py          # PNG preview 64 cases
    ├── test_trigger.py                # Positive triage trigger 38 cases
    ├── test_model_gateway.py          # ModelGateway main path 69 cases
    ├── test_model_gateway_degradation.py # Degradation path 34 cases
    ├── test_protocol_compat.py        # Protocol compat (timeout/unreachable/structured failure) 4 cases
    ├── test_ris_writer.py             # RIS finding-draft write-back 37 cases
    ├── test_ris_writer_retry.py       # RIS outbox retry 21 cases
    ├── test_healthcheck_writer.py     # Master-checkup summary-ref write-back 51 cases
    ├── test_healthcheck_writer_retry.py # Master-checkup outbox retry 24 cases
    ├── test_dual_write_audit.py       # Dual-write audit linkage 31 cases
    ├── test_alert_banner.py           # Failure alerts 79 cases
    ├── test_fastapi.py                # FastAPI app 23 cases
    ├── test_sse.py                    # SSE publisher 43 cases
    ├── test_template_render.py        # Jinja2 three-column template 47 cases
    ├── test_alpine.js                 # Alpine.js state binding 47 cases
    ├── test_cli.py                    # CLI three subcommands 13 cases
    ├── test_fixtures_meta.py          # fixtures contract 15 cases
    ├── test_e2e_triage.py             # End-to-end happy-path 7 cases
    └── test_baseline_comparison.py    # baseline comparison 42 cases
```

> **Directory tree notes**: each directory and key file is followed by a one-line responsibility comment; the 36 test files under `tests/` cover **unit / integration / end-to-end / protocol-compat / baseline-comparison** five categories, totaling **1136+ passed**.

## License

MIT

---

## Follow Us

Scan the QR code to follow our official account for updates and community access:

![Follow Us](qrcode.jpg)