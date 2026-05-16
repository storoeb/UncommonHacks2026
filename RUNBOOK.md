# Runbook — ProphetHacks 2026

Sequential steps to go from a fresh clone to a running, trained, evaluated agent.
Every command runs from the repo root. All scripts are idempotent — safe to re-run.

---

## Step 0 — Install dependencies

```powershell
pip install -r requirements.txt
pip install ai-prophet   # for the eval step at the end
```

---

## Step 1 — Verify Snowflake credentials

```powershell
python scripts/check_snowflake_connection.py
```

Should print your account, region, user, role, warehouse, database, schema.
Fix `.env` if anything is wrong before continuing.

---

## Step 2 — Bootstrap Snowflake schema (one-time)

Creates the database + schema if missing, then applies `sql/ddl.sql`
(all five tables: `HISTORICAL_MARKETS`, `ENSEMBLE_BACKFILL`,
`BASE_RATE_BACKFILL`, `META_PREDICTIONS_BACKFILL`, `AGENT_PREDICTIONS`).

```powershell
python scripts/bootstrap_snowflake.py
```

---

## Step 3 — Ingest resolved Kalshi markets

Paginates Kalshi's public API and MERGEs resolved binary markets into
`HISTORICAL_MARKETS`. ~20 min for 1500 markets.

```powershell
python scripts/import_kalshi_history.py --limit 1500
```

Check what landed:

```powershell
python scripts/check_data.py
```

---

## Step 4 — Backfill real Kalshi prices (parallel with Step 5)

Fetches open + 24h-pre prices from Kalshi's candlestick endpoint and
updates `q_kalshi_at_open` / `q_kalshi_at_24h_pre` in `HISTORICAL_MARKETS`.
~15 min. Note: ~55% of markets (mostly Elections) have no candlestick data
available from the API — those rows keep `[0.5, 0.5]` placeholders.

```powershell
python scripts/backfill_candlesticks.py --limit 1500
```

---

## Step 5 — Embed question text (parallel with Step 4)

Runs `SNOWFLAKE.CORTEX.EMBED_TEXT_768` on every row where
`question_embedding IS NULL`. Runs entirely inside Snowflake. ~5 min.

```powershell
python scripts/embed_questions.py
```

---

## Step 6 — Backfill ensemble + retrieval features

For each historical market, calls Wafer (single cheap model) to get
"as-if live" ensemble predictions, and runs Snowflake vector retrieval
(self-excluded). Writes to `ENSEMBLE_BACKFILL` + `BASE_RATE_BACKFILL`.

Cost: ~$0.00038 per market. 670 markets ≈ $0.25.

**Do not kill this run** — it uses a single Snowflake connection and
kills will roll back uncommitted rows. Let it finish.

```powershell
python scripts/backfill_features.py --limit 670 --concurrency 50 --yes
```

Confirm rows landed:

```powershell
python scripts/check_data.py
# ensemble backfill rows should be ~670+
```

---

## Step 7 — Train the AutoML calibrator

Creates `META_TRAIN` view (joins backfill tables), attempts
`SNOWFLAKE.ML.CLASSIFICATION META_CALIBRATOR`, always trains sklearn
`GradientBoostingClassifier` as fallback. Writes calibrated predictions
into `META_PREDICTIONS_BACKFILL`. Saves `notebooks/_calibrator.joblib`
and `notebooks/_calibrator_last_metrics.json`.

```powershell
python notebooks/04_train_automl_calibrator.py
```

The holdout Brier table printed at the end is your calibrator eval.

---

## Step 8 — Train the α-policy + generate Pareto plot

Creates `ALPHA_TRAIN` view, computes `alpha_star` per market (line-search),
attempts `SNOWFLAKE.ML.REGRESSION ALPHA_POLICY`, always trains sklearn
`GradientBoostingRegressor` as fallback. Saves `notebooks/_alpha_policy.joblib`,
`notebooks/_alpha_global.json`, `notebooks/_pareto.png`,
and `notebooks/_pareto_metrics.json` (read by the Streamlit demo).

```powershell
python notebooks/06_train_alpha_policy.py
```

The Brier + AVER summary printed at the end is your pitch table.

---

## Step 9 — Start the agent

```powershell
$env:PYTHONPATH = "src"
uvicorn prophet_agent.server:app --host 0.0.0.0 --port 8000
```

Or via the shell script (handles venv activation):

```bash
./run.sh
```

Healthcheck:

```powershell
Invoke-RestMethod http://localhost:8000/healthz
```

---

## Step 10 — Smoke test

```powershell
$body = Get-Content tests/fixtures/sample_event.json -Raw
$r = Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/chat/completions `
     -ContentType "application/json" -Body $body
$r.choices[0].message.content
```

Should return something like:

```
Based on ensemble forecast over 3 models (...) the calibrated probabilities over outcomes [Yes, No] are:
{"probabilities": [0.48, 0.52]}
```

Check `$r.prophet_debug.neighbor_count` is ~15 and `$r.prophet_debug.elapsed_s` is under 60.

---

## Step 11 — Evaluate against resolved questions

Fetches the 26 resolved sample questions from the Prophet Arena dataset
registry and scores the running agent on each one.

```powershell
prophet forecast retrieve --dataset sample-resolved --include-resolved -o resolved.json
python scripts/evaluate_local.py resolved.json --timeout 300 --concurrency 7
```

The final two lines give you `Mean Brier` and `Mean AVER` — paste these
into `PITCH.md` to replace the `{CALIB_BRIER}` and `{PARETO_*}` tokens.

---

## Step 12 — Commit and push

```powershell
git add .
git commit -m "ProphetHacks 2026: trained calibrator + alpha-policy, eval results"
git push
```

---

## Re-training after more backfill data

If you run more backfill (steps 4–6 are idempotent, just re-run with
the same or higher `--limit`), retrain by repeating steps 7–8 and
restarting the agent. The new `.joblib` files are picked up on restart.

---

## Quick reference — check data state at any time

```powershell
python scripts/check_data.py
```

| Column | What to look for |
|---|---|
| Embedded | Should equal Rows (all 1548) after Step 5 |
| RealPrices | ~670 max (Elections have no candlestick data) |
| ensemble backfill rows | Target ~670+ after Step 6 |
| meta_predictions rows | Populated by Step 7 |
