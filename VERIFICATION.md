# Verification record — 2026-10-15

The starting directory contained only empty `work/` and `outputs/` folders. `git status --short` reported that it was not a Git repository. There were no application files or parent AGENTS.md instructions to preserve. Python was not on PATH; tests used the bundled Python executable at `C:\Users\Gopica\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.

Dependencies were installed into workspace-only `work/test_deps` after network permission was granted. Optional PostgreSQL syntax tooling was installed into `work/sql_deps`. No credentials were configured or read from a secret file. These temporary packages are not part of the deliverable.

| Stage | Files / implemented work | Commands and observed result | Limitations / next step at that stage |
|---|---|---|---|
| 1 | `schemas.py`, `preprocess.py`, `storage.py`, foundation tests | `python -m unittest discover -s outputs/cctv_rag/tests -v`: 2 passed | No Pytest initially; proceed to fixtures and ingestion |
| 2 | `ingestion.py`, `analytics.py`, synthetic JSON fixtures, analytics tests | Same unittest command: 5 passed; totals 5, 11, 8, repeat 11 | Summary processor not yet wired; proceed to incremental summaries |
| 3 | `summarizer.py`, `embeddings.py`, `retriever.py`, `vector_store.py`, fake provider and update tests | Same unittest command: 9 passed | Real MiniLM not downloaded; proceed to remote adapter and routing |
| 4 | `database.py`, `database/schema.sql` | PostgreSQL parser accepted all 14 SQL statements; parameterization/filter unit test passed in final suite | No database endpoint/credentials; SQL execution and pgvector remain unverified |
| 5 | `query_parser.py`, `service.py`, `rag_generate.py`, query/grounding tests | Same unittest command: 15 passed | No Gemini credentials; deterministic fallback verified; proceed to UI |
| 6 | `config.py`, `runtime.py`, `app.py`, CLI, package/environment manifests, UI test | First full Pytest run: 24 passed, including Streamlit AppTest ingestion and Ask interaction | No remote deployment/browser visual inspection; proceed to final review |
| 7 | Edge/adapter tests, contract/security fixes, README and verification report | Final Pytest suite: 29 passed, 0 failed, 0 skipped. Wheel built and installed; installed CLI returned 5 for Batch 1 | External services and end-to-end live path still require qualification |

Final test invocation in this workspace:

```powershell
$env:PYTHONPATH = (Resolve-Path work/test_deps).Path + ';' + (Resolve-Path outputs).Path + ';' + (Resolve-Path work/sql_deps).Path
& 'C:\Users\Gopica\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest outputs/cctv_rag/tests -q
```

Standard installed-project equivalent: `python -m pytest tests -q`. The tests never use a configured production database. Streamlit AppTest uses a temporary SQLite file; core tests use isolated in-memory stores, except the persistence test which uses a temporary directory. Tests subclass unittest and are also executable with the built-in runner. Optional SQL-parser/UI tests explicitly skip if their dependencies are absent.

Additional executed checks:

- Parsed `database/schema.sql` with pglast's PostgreSQL parser: 14 valid statements. This checks grammar, not permissions, extension installation, query plans, or runtime behavior.
- `python -m pip wheel --no-deps --no-build-isolation ./outputs/cctv_rag --wheel-dir work/wheels`: built a wheel successfully.
- Installed that wheel into isolated `work/installed`, ingested Batch 1 through its CLI, then asked the required question: `succeeded`, exact value 5, source synthetic, coverage unknown, status partial.
- Reviewed output files and performed a credential-pattern scan. No real credentials or `.env` file are included. Git diff is unavailable because there was no Git repository; all deliverable files are new.

Verified synthetic expectations:

| Scenario | Observed result |
|---|---|
| A: CAM_01, October 9 | 5 |
| B: CAM_01, October 9–10 | 11 |
| C: All cameras, October 10 | 8 |
| D: Repeat Batch 2 | 11; no extra events, summary versions, or embedding generation |
| E: Context query | Current CAM_01 October 10 summary, deterministic evidence-only answer |
| F: Correction/new window | Changed hash/version; only affected windows refreshed |
| G: Failure and retry | Safe visible failed/partially_failed state; no duplicate records; retry succeeds |
| H: Missing camera/date | Unavailable/null, never an invented zero |

Known limitations are documented in README: no trusted coverage contract, no audit-history table, deliberately serialized ingestion, in-process Python aggregation, limited deterministic question grammar, and no authenticated transport or production deployment controls. The real MiniLM model, Supabase/pgvector deployment, Gemini API, and CCTV/AWS capture-to-query path were not verified. Fake embeddings are test-only and never selected by the UI.
