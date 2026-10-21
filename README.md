# Incremental CCTV Log Analytics

A standalone Python component for **processed records**, not video capture. SQLite development mode works without credentials, a model download, or an LLM. Sample records are synthetic. No real CCTV camera, S3/SQS transport, Supabase deployment, or Gemini endpoint was connected during development.

## Quick start

Use Python 3.11+ (verified here with Python 3.12). Open a terminal in this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
python -m cctv_rag ingest sample_data/batch_001.json --batch-id batch_001
python -m cctv_rag ingest sample_data/batch_002.json --batch-id batch_002
python -m cctv_rag ask "How many vehicle crossings did CAM_01 record on October 9, 2026?"
python -m streamlit run app.py --server.address 127.0.0.1
```

On Linux/macOS, activate with `source .venv/bin/activate` and copy the environment example with `cp`. The component is a local development UI; add authentication and deployment controls before exposing it remotely.

```powershell
python -m cctv_rag ask "How many vehicle crossings did CAM_01 record across both days?" --start 2026-10-09T00:00:00Z --end 2026-10-11T00:00:00Z
python -m cctv_rag ask "How many vehicle crossings were recorded by all cameras on October 10, 2026?"
python -m cctv_rag ask "Summarize the observations for CAM_01 on October 10, 2026."
python -m cctv_rag status
python -m pytest tests -q
```

The SQL parser test requires optional `pip install pglast`; otherwise that test skips. All other tests use isolated memory/temporary SQLite databases and fake embeddings. The UI test skips if Streamlit is absent. No test connects to production storage.

## Architecture and files

`schemas.py` validates the upstream event contract. `ingestion.py` commits structured events idempotently and tracks retryable work. `preprocess.py` normalizes timestamps and hashes content. `summarizer.py` maintains current deterministic camera/window/source summaries. `embeddings.py` defines the replaceable embedding interface and MiniLM adapter. `storage.py` provides SQLite; `database.py` provides PostgreSQL; `database/schema.sql` creates isolated tables and pgvector indexes. `vector_store.py` provides an explicitly local cosine scan, not a PostgreSQL emulation.

`analytics.py` calculates exact recorded counts independently of retrieval. `query_parser.py` resolves supported questions and rejects ambiguous ranges or unsupported numerical modifiers. `retriever.py` filters current evidence. `service.py` routes requests. `rag_generate.py` supplies deterministic answers and a Gemini adapter. `config.py` and `runtime.py` wire providers. `app.py` contains UI only; `__main__.py` is a CLI. `tests/` contains regression tests; `sample_data/` contains the two synthetic batches.

## Data contract

Every event requires:

| Field | Meaning |
|---|---|
| `event_id` | Globally stable upstream identity, 1–200 characters |
| `batch_id` | Stable original batch identity; must match the function argument |
| `camera_id` | Stable case-sensitive identifier; `CAM_01` naming supports direct question parsing |
| `timestamp_utc` | ISO-8601 with timezone; normalized to UTC. Naive timestamps are rejected |
| `object_type` | `vehicle`, `pedestrian`, `bicycle`, or `other` |
| `track_id` | Track identifier; required for a line crossing, optional for a detection |
| `confidence` | Finite numeric value from 0 to 1 |
| `event_type` | `detection` or `line_crossing` |
| `crossing_count` | Exactly 0 for detection, exactly 1 for line crossing |
| `source_type` | `synthetic` or `live` |
| `synthetic` | Boolean consistent with `source_type` |
| `source_ref` | Nonempty provenance reference, e.g. S3 object key |
| `revision` | Positive integer, defaults to 1; increase for a deliberate correction |

Unknown fields and malformed records are rejected. Ingestion time is stored separately from event time. Detections represent frame observations and may repeat the same tracked object; pedestrian detection counts are **not unique people**. Vehicle counts include only vehicle line-crossing events. Upstream must emit one stable ID per actual crossing; this component cannot detect two distinct IDs referring to the same physical crossing.

Identical event content is unchanged. A correction requires a larger revision and the same camera, original batch, track ID, and synthetic/live provenance. Conflicting same/older revisions or changed identity are rejected atomically. Timestamp, confidence, object/event classification, and source reference can be corrected with a higher revision. Resend corrections using their original batch ID. Changing a timestamp refreshes both its old and new windows. IDs reused across batches are conflicts.

## Ingestion, retries, and consistency

```python
from cctv_rag.runtime import Runtime
from cctv_rag.ingestion import ingest_batch

runtime = Runtime()
try:
    result = ingest_batch(records=processed_records,
                          batch_id=stable_batch_id,
                          ingestor=runtime.ingestor)
finally:
    runtime.close()
```

Each response includes `status`, `attempts`, timestamps, per-record new/unchanged/modified outcomes, pending windows, and safe errors. Validation errors include field paths without echoing raw input. Empty batches and duplicate IDs inside a batch fail. Validation/structured-storage failures roll back the entire event transaction. Provider and driver exception strings are not logged or persisted.

The structured transaction also marks affected old summaries dirty. Those summaries immediately become ineligible for retrieval. Each summary and its embedding then commits atomically. An embedding failure leaves structured records queryable, retains pending windows, and returns `partially_failed`. Retry the same stable batch to finish pending work without duplicating counts. A new failed window has no retrievable summary until it succeeds. A database outage that prevents recording status raises an exception; the caller must retry and must not acknowledge success. A process crash can leave `processing`; replay resumes persisted pending work.

`succeeded` means all required work under the selected configuration completed. With `EMBEDDING_MODE=disabled`, metadata summaries are supported and explicitly marked `vector_status=disabled`; this does not claim vector availability. With MiniLM selected, embedding failure prevents success. Supported status vocabulary also includes `pending`; this synchronous implementation starts attempts as `processing` rather than adding an asynchronous queue.

Writers are serialized by a local file lock (SQLite) or session advisory lock (PostgreSQL), with transactions protecting each commit. Lock/storage acquisition failures propagate for retry. Use a Runtime per worker/connection, not a shared connection across threads. This intentionally favors correctness over high ingestion throughput.

Summary IDs hash camera, UTC window start, and source type. Synthetic and live windows remain separate. Versions start at 1 and increase when the canonical summary content changes. Content includes verified facts, event IDs/references, and a hash of source records. Only the current version is stored; there is no revision audit archive. Unchanged batches do not refresh summaries or regenerate embeddings. Window duration is recorded in the database and cannot be silently changed: use a new database or implement a reviewed migration.

## Time, coverage, analytics, and routing

Storage and output intervals use UTC and **[start, end)** boundaries. Date-only questions use `TIME_ZONE` (UTC by default); explicit timestamp filters must carry a timezone. Two dates describe an inclusive range of calendar days. Daily comparisons preserve the supplied date order. Hourly rankings and grouped day/hour keys are UTC. Sub-day clock wording, relative dates, unspecified dates, and unsupported numerical modifiers request clarification. “Both days” needs explicit filters; no chat history is silently assumed. Arbitrary camera names can be supplied through the explicit camera filter.

Supported exact metrics: vehicle crossings, pedestrian detections, and valid source records. Functions support all-camera totals, camera/object grouping, daily comparisons, and hourly rankings. Results report filters, source types, contributing/available records, interval, coverage, and availability. These aggregates operate on structured records, never on the number of retrieved summaries.

Coverage remains **unknown** because this contract does not yet accept trustworthy upstream availability intervals. No records yields `value=null` / `unavailable`, not zero. When usable records exist but none match a metric, zero means **zero among the available records**, explicitly `partial`; it is never presented as zero total traffic. Complete hour/day traffic cannot be established from event timestamps alone. Empty observed-group lists do not imply absent groups had zero activity.

Retrieval uses overlapping windows, camera/source filters, current/non-dirty state, optional cosine threshold, and a limit. Every context answer preserves full summary intervals; a summary overlapping a sub-hour request is not an exact count for that sub-hour. Without embeddings, selection is deterministic metadata-only and does not claim semantic relevance. With a model, only matching model/content-hash vectors are eligible. Exact numerical answers are always deterministic even if retrieval or Gemini fails. Contextual Gemini responses use strict grounding instructions; as with any LLM, these instructions are not a formal guarantee of factuality. Verified evidence is displayed separately.

## MiniLM

```powershell
python -m pip install -r requirements-embeddings.txt
# In .env: EMBEDDING_MODE=minilm
python -m cctv_rag reindex
```

The actual adapter loads `sentence-transformers/all-MiniLM-L6-v2`, verifies 384 output dimensions, normalizes vectors, and rejects invalid/zero vectors. First use needs a model download or an existing local model cache. The model identifier and embedding content hash are persisted. `reindex` is explicit maintenance for enabling/changing providers; ordinary ingestion never rebuilds the whole vector database. Restart the process when changing a cached model. Model revisions and a deployment lockfile should be pinned when qualifying a production environment. The real model was not downloaded/executed here; tests use clearly named fake vectors and verify the adapter’s dimension guard.

## Supabase / pgvector setup

1. Create a **dedicated test project/database** first. Run `database/schema.sql` in its SQL editor with an operator role permitted to create extensions and schema objects. The script is additive and scoped to `cctv_rag`; it never truncates existing tables.
2. Create a dedicated server login. Grant USAGE on `cctv_rag` and SELECT/INSERT/UPDATE/DELETE on its tables. Grant USAGE on the schema containing the vector type if your installation requires it. Do not grant schema ownership or expose this connection in browser code. Keep the schema outside exposed PostgREST schemas.
3. Set `STORAGE_BACKEND=postgres` and `DATABASE_URL` to the provider’s direct or **session-mode** pooled connection, with TLS (`sslmode=require`; prefer `verify-full` with trusted CA configuration). This adapter uses session advisory locks, so transaction-mode pooling is unsuitable. Permit outbound network access to the configured database endpoint and port.
4. Set `EMBEDDING_MODE=minilm` for vector retrieval, then ingest the fixtures and repeat the tests manually in the dedicated test database. The adapter uses parameterized SQL, native timestamptz columns, `vector(384)`, current-hash checks, and an HNSW cosine index.

`SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` are documented placeholders, unused by this direct-SQL adapter. No Supabase service-role key is needed. Database credentials remain server-side. Schema syntax was parsed locally; actual extension permissions, index behavior, access grants, transaction semantics, and remote connections remain unverified. Local tests never use `DATABASE_URL` to mutate remote data. The CLI/UI intentionally follow configured storage; check configuration before manual ingestion.

For larger datasets, move Python analytics into optimized SQL aggregates and benchmark filtered pgvector recall/query plans. HNSW filtering can return fewer neighbors than requested; tune index/search settings against your workload.

## Gemini setup

Set both `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` to a model supported by your account. Leave both empty for deterministic local mode. The adapter uses `google-genai` and passes only the question, verified analytics, and selected evidence. Review whether your data may be sent to that provider before configuring it. Exact count answers bypass the LLM. Provider errors yield a safe warning and deterministic evidence, never a fabricated successful Gemini response. API connectivity and live model behavior were not tested without credentials.

## Synthetic fixtures and verification

Batch 1: five CAM_01 vehicle crossings on October 9, 2026, plus one pedestrian and one vehicle detection. Batch 2: six CAM_01 and two CAM_02 crossings on October 10, plus one pedestrian and one vehicle detection. Each event has a unique stable ID and is explicitly synthetic. All fixture records fall in the 10:00–11:00 UTC window on their day. There are 17 records and 13 crossings across both batches, with 11 crossings for CAM_01.

Executed verification: **29 Pytest tests passed**, including the Streamlit interaction test and optional PostgreSQL syntax parser. Tests cover A–H: 5 initially, 11 across both days, 8 across all cameras on October 10; repeated ingestion leaves counts/versions/embedding calls unchanged; correct contextual evidence; affected-window updates; safe failure/retry; and unavailable missing data. Additional tests cover boundaries, timezone offsets, out-of-order records, provenance filters, persisted SQLite state, invalid schemas, configuration, parameterization, and LLM fallback. See `VERIFICATION.md` for commands and limitations.

## Live CCTV / AWS handoff

The upstream component owns capture, OpenCV/model inference, tracking, crossing-line definitions, Docker workers, S3, and SQS. Generate stable crossing IDs from a camera + tracking-session + track + line/direction + crossing-sequence identity (or persist a UUID when first emitted); track IDs alone are insufficient across restarts. Derive batch IDs from a stable message/source object identity, never from retry time. Send timezone-bearing UTC timestamps, the original source object key, and truthful live/synthetic flags.

A future authenticated API or SQS consumer should validate caller identity and message size, deserialize records, call `ingest_batch`, acknowledge only `succeeded`, and retry `partially_failed` / transient exceptions with backoff. Route permanent contract conflicts to a dead-letter/manual review path. Preserve original event/batch IDs and revision numbers on retries. After restart, replay unacknowledged batches. New records automatically refresh only their affected windows through this function.

**Still to build/verify:** the authenticated transport/queue bridge; SQS acknowledgment and dead-letter policy; trusted coverage metadata; real capture-to-query end-to-end tests; Supabase/Gemini/model qualification; deployment authentication, capacity limits, observability, and backup/recovery. Uploading JSON to S3 alone does not update this component or Supabase. This project is an offline-verified integration component, not a live-integrated deployment.

API implementation references: [Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html), [SentenceTransformer model API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), and [Gemini generate content](https://ai.google.dev/api/generate-content).
