# Incremental CCTV Log Analytics

A hybrid real-time microservices and Dynamic RAG platform connecting live CCTV vehicle detection (YOLO + ByteTrack), cloud message transport (AWS S3 + SQS), durable worker ingestion, deterministic traffic analytics, MiniLM semantic embeddings, Gemini grounding, and a Streamlit dashboard.

---

## Component Verification Status Matrix

| Component | Status | Verification Details |
|---|---|---|
| **Live Event Adapter (`adapter.py`)** | **Verified** | Converts live YOLO/ByteTrack S3 envelopes into RAG events; preserves `vehicle_subtype` (car, truck, bus, motorcycle), `direction` (positive, negative), and line IDs. |
| **Worker Bridge (`live-cctv-cloud/worker.py`)** | **Verified** | SQS message consumption, S3 retrieval, durable idempotency, visibility heartbeat, DLQ forwarding, and poison message quarantine. |
| **SQLite Storage & Rollups** | **Verified** | ACID transactions, advisory file locking, 1-hour rollup summaries, dynamic updates, and crash recovery. |
| **Directional & Subtype Analytics (`analytics.py`)** | **Verified** | Exact counts by subtype, direction, subtype × direction cross-tabulation, and directional difference delta. Strictly deterministic. |
| **Natural Language Routing (`service.py`, `query_parser.py`)** | **Verified** | Routes analytical queries directly to deterministic engine (bypassing LLM). Ambiguous queries request clarification. |
| **MiniLM 384-d Embeddings (`embeddings.py`)** | **Verified** | `sentence-transformers/all-MiniLM-L6-v2` verified with **real local model execution**; generates 384-d normalized embeddings, semantic ranking, and cosine similarity scoring. |
| **Gemini Grounding Adapter (`rag_generate.py`)** | **Verified (Mocks/Contract)** | Google GenAI SDK integration with evidence pruning, grounding system prompt, and sanitized error handling. *Live API call pending operator credential configuration.* |
| **Controlled AWS S3/SQS Transport** | **Verified** | Live S3 object upload to `cctv/test-batches/` and SQS dispatch/deletion against `traffic-events-queue` verified end-to-end. |
| **HLS CCTV Stream Reachability** | **Verified** | Caltrans District 11 HLS stream (`https://wzmedia.dot.ca.gov/D11/C009_NB_5_JNO_Old_Town.stream/playlist.m3u8`) verified live and responding with HTTP 200. |
| **Live Camera Capture Execution (`vehicle_counting.py`)** | **Blocked on Operator Approval** | Stream is live, but `ultralytics` is not installed by default in the host environment. Proposed command prepared below. |
| **Production AWS & Supabase Cutover** | **Pending Operator Action** | Schema DDL ready in `database/schema.sql`. Production SQS queue (`cctv-processing-queue`) untouched to guarantee zero production disruption. |

---

## Quick Start

### 1. Environment & Dependencies

Requirements: Python 3.11+ (tested on Python 3.14).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-embeddings.txt
python -m pip install -e .
Copy-Item .env.example .env
```

### 2. Run Complete Regression Test Suite

```powershell
py -3.14 -m unittest discover -s tests -v
```
*(73 tests passing across adapters, analytics, worker, Gemini grounding, MiniLM embeddings, and Streamlit AppTest; 1 test skipped for optional PostgreSQL parser).*

### 3. Start Streamlit Dashboard

```powershell
py -3.14 -m streamlit run app.py
```

---

## Environment Configuration

Configuration is loaded from `.env` or system environment variables:

```bash
# Storage configuration
STORAGE_BACKEND=sqlite
SQLITE_PATH=cctv_rag.sqlite3
TIME_ZONE=UTC
SUMMARY_WINDOW_MINUTES=60

# MiniLM Embedding configuration
EMBEDDING_MODE=minilm                     # 'disabled' or 'minilm'
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Gemini Generative AI configuration (Optional)
GEMINI_API_KEY=                           # Your Google AI Studio API key
GEMINI_MODEL=gemini-2.5-flash             # gemini-2.5-flash, gemini-2.0-flash, or gemini-1.5-flash

# AWS Pipeline (Worker & Producer)
AWS_REGION=ap-south-1
S3_BUCKET_NAME=generalized-etl-engine-2026-0001
CCTV_S3_PREFIX=cctv/live-batches/
CCTV_QUEUE_URL=https://sqs.ap-south-1.amazonaws.com/292981482205/cctv-processing-queue

# Remote PostgreSQL / Supabase (Optional)
DATABASE_URL=postgresql://user:pass@host:5432/dbname?sslmode=require
```

> **Security Note**: Never commit API keys, AWS credentials, or database passwords to version control. All exception handlers and loggers sanitize error messages to prevent credential leakage.

---

## AI Services Architecture

### 1. Deterministic Analytics Engine First
The system strictly enforces that numerical totals, vehicle subtype counts, direction comparisons, and date rankings are calculated deterministically from verified structured database records. The LLM is **never** invoked for numerical aggregations, preventing hallucinations.

### 2. MiniLM 384-Dimensional Semantic Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- Weights Cache: Cached under `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`.
- Offline Graceful Fallback: If `sentence-transformers` is absent or the model fails to load, `Runtime` automatically falls back to metadata-only retrieval without crashing.
- Storage Persistence: SQLite and PostgreSQL store the normalized vector, model ID, and `embedding_content_hash`.

### 3. Grounded Gemini Contextual Explanations
- SDK: Google GenAI official client (`from google import genai`).
- Evidence Pruning: Sends only structured summary intervals and verified facts to the model. Internal hash digests and raw event arrays are excluded.
- Grounding Instruction: The system prompt instructs Gemini to explain only what the retrieved evidence directly supports, to distinguish zero events from missing data, and never to fabricate vehicle observations.
- Fault Tolerance: Network timeouts, rate limits (HTTP 429), or empty responses trigger an explicit deterministic fallback with a `service_warning`.

---

## Controlled CCTV Live Camera Test Procedure

To execute live video capture and event generation on the Caltrans HLS stream:

### Step 1: Install Detection Prerequisites
```powershell
py -3.14 -m pip install ultralytics
```

### Step 2: Run Controlled Capture (30-second capture test)
```powershell
cd live-cctv-cloud
py -3.14 vehicle_counting.py
```
*(Press `q` after 30 seconds to terminate. Live line crossings will be appended to `live-cctv-cloud/output/cctv_events_v2.jsonl` and batched into `live-cctv-cloud/output/batches/`).*

### Step 3: Run Producer Against Isolated Test Queue
```powershell
$env:CCTV_LOG_FILE="output/cctv_events_v2.jsonl"
$env:CCTV_S3_PREFIX="cctv/test-batches/"
$env:CCTV_QUEUE_URL="https://sqs.ap-south-1.amazonaws.com/292981482205/traffic-events-queue"
py -3.14 live_s3_sqs_producer_v2.py
```

### Step 4: Run Worker in Single-Batch Isolated Test Mode
```powershell
$env:CCTV_QUEUE_URL="https://sqs.ap-south-1.amazonaws.com/292981482205/traffic-events-queue"
$env:SQLITE_PATH="cctv_live_test.sqlite3"
$env:WORKER_MAX_BATCHES="1"
py -3.14 worker.py
```

### Step 5: Verify in Dashboard
```powershell
$env:SQLITE_PATH="cctv_live_test.sqlite3"
py -3.14 -m streamlit run app.py
```
