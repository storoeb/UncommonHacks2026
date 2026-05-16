# ProphetHacks Agent — Forecasting Track

<!-- TODO: final product name -->

> We close the gap Prof. Xu highlights in PM-RANK 0.3.1: **Brier is an absolute metric; averaged return (AVER) is relative**—so a forecaster can look “good” on Brier and still lose on the payoff students actually care about. Our stack turns that tension into a design problem: calibrate on realized markets, then learn *where* to shade toward or away from the posted market.

## What this is

Naïve LLM submissions match the consensus price when it is safe, which yields accidental Brier improvement but **~0 AVER**; deviating without structure burns Brier and still loses AVER. Neither regime reliably wins under the hackathon utility.

Our entry is a **market-aware meta-calibrator**: a three-model Wafer ensemble produces raw beliefs `p_llm`; Snowflake stores **resolved** Kalshi history with **AI_EMBED** vectors for similarity retrieval; a Snowflake **AutoML classification** model (planned; see below) maps meta-features to calibrated probabilities; a second **AutoML regression** (planned) learns a per-market shading policy between meta-prediction and `q_kalshi`. The live **FastAPI** agent speaks OpenAI-compatible `/v1/chat/completions` so Prophet Arena can call us without custom glue.

## Quick start

```bash
unzip prophet_agent_uchicago_hack.zip
cd prophet_agent_uchicago_hack
cp .env.example .env   # judge fills in Snowflake + Wafer credentials
./run.sh               # pip install -r requirements.txt; uvicorn on :8000
```

Smoke the agent (from repo root, with the server running):

```bash
curl -sS -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "@tests/fixtures/sample_event.json"
```

Request and response conventions: **`docs/sdk_notes.md`**.

**Reality check (May 16, 2026):** The orchestrated **calibrator** and **α-policy** stages in `pipeline.py` are **pass-through stubs** until Phase 4/6 training notebooks land and inference is wired to `SNOWFLAKE.ML.*` models (or the sklearn fallback in `calibrator.py`). The ensemble, parser, retrieval, logging path, and HTTP surface are real; AutoML training is the remaining centerpiece.

### Environment variables (`.env`)

| Variable | Purpose |
|----------|---------|
| `SNOWFLAKE_ACCOUNT` | Account locator |
| `SNOWFLAKE_USER` / `SNOWFLAKE_PASSWORD` or PAT | Auth |
| `SNOWFLAKE_WAREHOUSE` | e.g. `COMPUTE_WH` |
| `SNOWFLAKE_DATABASE` | Training DB name (bootstrap creates if missing) |
| `SNOWFLAKE_SCHEMA` | Schema for tables/views |
| `SNOWFLAKE_ROLE` | Role with `CORTEX_USER` / ML privileges as needed |
| `WAFER_API_KEY` | Wafer.ai key |
| `WAFER_BASE_URL` | Default `https://pass.wafer.ai/v1` |
| `OPENAI_API_KEY` | Optional fallback if you wire a non-Wafer path |

`scripts/check_snowflake_connection.py` is useful before bootstrap to validate PAT/password/MFA and discover writable schemas.

### Why this should win (judge lens)

- **Metric literacy:** We anchor the build in [Prof. Xu’s PM-RANK 0.3.1 writeup](https://ai-prophet.github.io/pm_ranking/blogpost/ranking_llm_250727.html)—Brier vs AVER is not a footnote; it is the product spec.
- **Labeled training path:** Resolved markets → features → **AutoML** (planned) is the honest way to claim “calibration” without hand-waving.
- **Platform coherency:** Snowflake holds history, embeddings, neighbor search, model artifacts (future), and the Streamlit demo—five surfaces, one narrative.
- **Operational:** OpenAI-compatible agent + `pytest` + fixture **`sample_event.json`** = low friction for technical reviewers.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            JUDGE / PROPHET ARENA             │
                    │  POST /v1/chat/completions  (10-min window) │
                    └─────────────────────┬───────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────┐
                    │   OUR AGENT (FastAPI, OpenAI-compatible)     │
                    │                                              │
                    │   1. Parse event (question, outcomes, q_ik)  │
                    │   2. Fan out to Wafer ensemble (parallel)    │
                    │   3. Embed question, retrieve neighbors      │
                    │   4. Build feature vector                    │
                    │   5. Score with meta-calibrator              │
                    │   6. Apply confidence shading vs q_kalshi    │
                    │   7. Return JSON probabilities in response   │
                    └────┬────────────┬──────────────────┬─────────┘
                         │            │                  │
                         ▼            ▼                  ▼
                  ┌──────────┐ ┌──────────────┐ ┌─────────────────┐
                  │  Wafer   │ │   Snowflake  │ │ Cortex ML model │
                  │ GLM-5.1  │ │ Cortex Search│ │ (loaded from    │
                  │ Qwen-397B│ │ + AI_EMBED   │ │  Snowflake)     │
                  │ Qwen-35B │ │ HISTORICAL   │ │                 │
                  └──────────┘ │ _MARKETS tbl │ └─────────────────┘
                               └──────────────┘
                                       ▲
                                       │ (built once, offline)
                               ┌───────┴────────┐
                               │ Kalshi history │
                               │   importer     │
                               └────────────────┘

Demo path:
   Streamlit-in-Snowflake ──► same Snowflake tables ──► visual story
```

**Components (C1–C9):**

1. **C1 — Snowflake schema (`sql/ddl.sql`)** — tables for historical markets, agent predictions, backfill features, and training views once AutoML is enabled.
2. **C2 — Kalshi historical importer (`scripts/import_kalshi_history.py`)** — paginated public API ingest with `MERGE` into `HISTORICAL_MARKETS`.
3. **C3 — Embedding pipeline (`scripts/embed_questions.py`)** — batched `SNOWFLAKE.CORTEX.EMBED_TEXT_768` updates for retrieval.
4. **C4 — Wafer ensemble (`src/prophet_agent/llm/wafer.py`)** — parallel async multi-model probabilities plus disagreement variance.
5. **C5 — Meta-calibrator (AutoML #1, `SNOWFLAKE.ML.CLASSIFICATION`)** — maps `(p_ensemble, q_kalshi, base_rate, category, disagreement)` → calibrated `p`. **Stub in live pipeline until Phase 4 notebook + `calibrator.py` Snowflake path are fully exercised;** see `src/prophet_agent/calibrator.py` for the intended interface and sklearn-shadow hook.
6. **C6 — Agent endpoint (`src/prophet_agent/server.py`)** — OpenAI-style `/v1/chat/completions`, `/v1/models`, `/healthz`.
7. **C7 — α-policy (AutoML #2, `SNOWFLAKE.ML.REGRESSION`)** — learns per-market blend weight between meta-`p` and `q_kalshi`. **Stub until Phase 6.**
8. **C8 — Streamlit demo (`streamlit_app/app.py`)** — four-tab dashboard; deployable in Snowsight (see `streamlit_app/README.md`). Pareto tab still uses placeholder eval points until Phase 4/6 numbers land.
9. **C9 — Packaging (`run.sh`, this README, `docs/sdk_notes.md`)** — unzip-and-run path for judges.

## The eight beats (the pitch in writing)

**1. We read your scoring doc.**  
Prof. Xu’s PM-RANK 0.3.1 post is explicit: *“the Brier score is an **absolute metric**, while the averaged return score is a **relative metric**.”* That single sentence is why naive “just output a number” systems miscarry the leaderboard psychology we are graded on. We quote it because our architecture is built to reason about **both**—not to hack one scalar.

**2. Naïve LLM submissions optimize Brier accidentally and AVER never.**  
Copying the market is safe for Brier and starves AVER. Wild guesses outside the market destroy Brier and still rarely beat utility. Our design assumes both failure modes are the default unless you add **structure**: labeled history, calibration, and a learned tradeoff.

**3. We built institutional memory.**  
Snowflake holds resolved markets, embeddings, and (after backfill) ensemble and base-rate features. At inference we retrieve neighbors with `VECTOR_COSINE_SIMILARITY`, aggregate realized outcomes into a base rate, and surface neighbor diagnostics in `prophet_debug` for demos. The “memory” is queryable: same tables feed the agent, the Streamlit “Institutional Memory” tab, and the training views—no hidden local cache that invalidates the story.

**4. Ensemble beliefs, not single-model opinions.**  
We fan out to **GLM-5.1**, **Qwen3.5-397B**, and **Qwen3.6-35B** through Wafer. Disagreement variance is a first-class feature for both calibration and (eventually) α policy. Parallel async calls mean wall time tracks the **slowest** model, not the sum—important inside Arena’s fixed clock.

**5. The model is trained on the only data with labels: resolved markets.**  
The meta-calibrator is meant to train on rows from resolved Kalshi history—`META_TRAIN` ties ensemble outputs, prices, neighbor base rates, and binary outcomes. **Today’s shipped inference still uses stubs;** completing `notebooks/04_train_automl_calibrator.py` and swapping the pipeline hook is what turns this beat from story into measured Brier on holdout. If you only judge the pre-train zip, insist on opening `prophet_debug`: you will still see **real** `p_ensemble`, **real** neighbors when Snowflake is populated, and **honest** `p_meta`/`alpha` once stubs are replaced.

**6. We don't just have a Brier↔AVER dial — we learn a policy for where to turn it.**  
The α-blend between calibrated beliefs and posted prices is the operational knob. The plan is a **Snowflake AutoML regressor** on per-market features with target α* from offline search—**not yet wired in this snapshot.** Until then, treat shading as **architected but not trained**—the blend formula and feature hooks are why Beat 6 survives code review even before the Pareto slide is final.

**7. Snowflake is the institutional memory AND the AutoML brain.**  
This is the side-prize thesis: **AI_EMBED** + vector retrieval + **`SNOWFLAKE.ML.CLASSIFICATION`** + **`SNOWFLAKE.ML.REGRESSION`** + **Streamlit-in-Snowflake**, all on the same platform. Two AutoML surfaces doubles the “first-party Snowflake” story; we are explicit that training notebooks are the gating artifact. Cortex Search indexing is optional at our scale; brute-force cosine over thousands of rows is already native SQL—another honest Snowflake detail.

**8. Live demo.**  
`./run.sh` proves the harness integration; **Streamlit-in-Snowflake** (`streamlit_app/app.py`, deploy notes in `streamlit_app/README.md`) is the visual—ensemble rows, neighbors, stubs vs future calibrator, and the Pareto story once eval numbers exist.

## How to reproduce the eval

1. `python scripts/bootstrap_snowflake.py` — idempotent DB/schema + `sql/ddl.sql` (requires `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA` in `.env`).
2. `python scripts/import_kalshi_history.py --limit 1500` — optional flags: `--throttle`, `--commit-every`, `--event-pages`, `--market-pages`, `--strategy {per-event,bulk}`.
3. `python scripts/backfill_candlesticks.py --limit 1000` — optional: `--throttle`, `--commit-every`, `--dry-run`.
4. `python scripts/embed_questions.py` — optional: `--batch-size`, `--max-batches` (defaults: 500 / 50).
5. `python scripts/backfill_features.py --limit 1000 --yes` — optional: `--concurrency`, `--dry-run`; `--yes` skips the cost confirmation prompt.
6. **When present:** run `notebooks/04_train_automl_calibrator.py` — trains the calibrator (`SNOWFLAKE.ML.CLASSIFICATION` or sklearn shadow per script), evaluates Brier vs baselines, and enables non-stub inference when integrated with `pipeline.py`. **This script is part of the Phase 4 deliverable; it may not ship in every early zip.**
7. **When present:** run `notebooks/06_train_alpha_policy.py` — builds `ALPHA_TRAIN`, trains `SNOWFLAKE.ML.REGRESSION`, Pareto plot. **Phase 6 deliverable; may be absent pre-submission cutoff.**
8. `./run.sh` — agent on port **8000**.

**Operational note:** Until candlestick backfill and AutoML training are complete end-to-end, treat `q_kalshi` features and calibration metrics as **work-in-progress**—the code paths exist so judges can run the agent and inspect `prophet_debug` today.

**Troubleshooting**

- **Import / API errors:** ensure `requirements.txt` installed in the same interpreter `uvicorn` uses (see `run.sh`).
- **Snowflake auth:** PAT + role must see `CORTEX.*` embed function used in `embed_questions.py`; trial regions vary—run `check_snowflake_connection.py` first.
- **Wafer rate limits during backfill:** lower `--concurrency` in `backfill_features.py`; the importer already supports `--throttle`.
- **Empty retrieval:** `HISTORICAL_MARKETS` may be empty or embeddings NULL—complete steps 2–4 before expecting meaningful neighbors.

## Repo layout

```
├── README.md
├── PITCH.md
├── requirements.txt
├── run.sh
├── conftest.py
├── .env.example
├── sql/
│   └── ddl.sql
├── docs/
│   └── sdk_notes.md
├── src/prophet_agent/
│   ├── __init__.py
│   ├── server.py
│   ├── pipeline.py
│   ├── parser.py
│   ├── snowflake_client.py
│   ├── calibrator.py
│   ├── llm/
│   │   └── wafer.py
│   └── retrieval/
│       ├── __init__.py
│       └── base_rate.py
├── scripts/
│   ├── bootstrap_snowflake.py
│   ├── check_snowflake_connection.py
│   ├── discover_writable_schemas.py
│   ├── import_kalshi_history.py
│   ├── backfill_candlesticks.py
│   ├── backfill_features.py
│   ├── embed_questions.py
│   ├── probe_kalshi.py
│   ├── probe_kalshi_events.py
│   ├── probe_kalshi_candlesticks.py
│   └── probe_wafer.py
├── notebooks/
│   └── 03_retrieval_eda.py
├── streamlit_app/
│   ├── app.py
│   ├── environment.yml
│   └── README.md
└── tests/
    ├── fixtures/
    │   └── sample_event.json
    ├── test_parser.py
    ├── test_retrieval.py
    ├── test_server.py
    ├── test_wafer_parse.py
    └── test_embed_questions.py
```

## Command reference (`scripts/`)

| Script | Role |
|--------|------|
| `bootstrap_snowflake.py` | Create DB/schema; apply `sql/ddl.sql` statement-by-statement. |
| `check_snowflake_connection.py` | PAT/password/MFA sanity; discovery hints for roles/warehouses. |
| `discover_writable_schemas.py` | Helper for finding writable schema targets on unfamiliar accounts. |
| `import_kalshi_history.py` | Paginated Kalshi public API → `MERGE` into `HISTORICAL_MARKETS`. |
| `backfill_candlesticks.py` | Fills price fields from `/candlesticks` where available. |
| `embed_questions.py` | Batched `EMBED_TEXT_768` for rows missing vectors. |
| `backfill_features.py` | Wafer ensemble + base-rate backfill into feature tables (cost prompt unless `--yes`). |
| `probe_kalshi*.py` / `probe_wafer.py` | Engineering probes—safe to ignore for judging. |

## Tests

From the repo root:

```bash
pytest
```

Coverage today: parser heuristics, Wafer JSON extraction, retrieval aggregation, `embed_questions` CLI args, server integration (including the sample fixture request path).

## Snowflake side prize

We lean into **five distinct Snowflake surfaces** in one story: **`HISTORICAL_MARKETS` + AI_EMBED** for institutional memory, **`VECTOR_COSINE_SIMILARITY`** for retrieval, **`SNOWFLAKE.ML.CLASSIFICATION`** for calibration (planned), **`SNOWFLAKE.ML.REGRESSION`** for the α-policy (planned), and **Streamlit-in-Snowflake** as the judging visual over the same tables. That is not “Snowflake as dumb storage”—it is Snowflake as memory, feature store, model host, and demo shell for forecasting accuracy under **Brier + AVER**.

## Streamlit demo

- **App entry:** `streamlit_app/app.py`
- **Deploy / local run:** follow **`streamlit_app/README.md`** (Snowsight Streamlit + `streamlit run` path).

Pre-submission smoke (adapted from internal checklist): fresh venv → `pip install -r requirements.txt` → confirm Snowflake + Wafer env vars → `./run.sh` → `curl` sample fixture → optional `pytest`. Streamlit: open SiS URL or `streamlit run streamlit_app/app.py` and confirm all four tabs render (empty-state OK). If `AGENT_PREDICTIONS` logging is enabled and Snowflake reachable, confirm a row appears for the sample call—logging is best-effort and must not take down the server on failure.

## See also

- [PM-RANK 0.3.1 — *Skills of large language models in quantitative finance*](https://ai-prophet.github.io/pm_ranking/blogpost/ranking_llm_250727.html) — Brier vs AVER, the **`q_ik`** baseline, and CRRA utility context for the hackathon track.
- [`docs/sdk_notes.md`](docs/sdk_notes.md) — HTTP contract for `/v1/chat/completions`.
- Internal engineering plan (optional in judge zip): `whimsical-kindling-hamster.md`.

## License + credits

Licensed under the MIT License — see `LICENSE` if present in your zip, or treat this as MIT for hackathon judging unless the team specifies otherwise.

**Team:** ProphetHacks 2026 — University of Chicago *(update with final roster)*
