# ProphetHacks 2026 — Demo Dashboard

Four-tab Streamlit app over the same Snowflake tables the forecasting
agent reads from. Designed to deploy to **Streamlit-in-Snowflake (SiS)**
for the pitch, with a local-dev path for iteration.

## Tabs

1. **Live Forecast** — calls the Wafer ensemble live, shows per-model
   probabilities, ensemble mean + variance, retrieved historical
   neighbors, final calibrated prediction, alpha, and a bet decision.
2. **Brier vs AVER Pareto** — the metric tradeoff plot quoting Prof. Xu's
   PM-RANK 0.3.1 blogpost. Placeholder points today; real eval data when
   Phases 4 & 6 land.
3. **By-Category Performance** — per-category Brier of agent vs Kalshi
   baseline, plus historical-market counts from `HISTORICAL_MARKETS`.
4. **Institutional Memory** — type any question, get the 10 most similar
   resolved markets via `AI_EMBED` + `VECTOR_COSINE_SIMILARITY`. The
   Snowflake side-prize visual.

Every tab handles empty-state gracefully — no stack traces in front of
the judges if a table is empty or a dependency hasn't landed yet.

## Local dev

From the repo root (one level above this folder):

```bash
# .env at the project root must have SNOWFLAKE_* + WAFER_API_KEY filled in.
pip install -r requirements.txt   # streamlit comes via requirements
streamlit run streamlit_app/app.py
```

That's the whole story for local iteration. The app autoloads
`../.env` via `prophet_agent.snowflake_client` and
`prophet_agent.llm.wafer`. The Wafer ensemble call runs out-of-process
against `pass.wafer.ai`.

## Deploy to Streamlit-in-Snowflake

In Snowsight:

1. Switch role to one with `CREATE STREAMLIT` on `PROPHET_HACK.PUBLIC`
   (e.g. `ACCOUNTADMIN`) and warehouse `COMPUTE_WH`.
2. **Projects** → **Streamlit** → **+ Streamlit App**.
3. Set the database to `PROPHET_HACK` and schema to `PUBLIC`.
4. In the file editor:
   - Replace the default `streamlit_app.py` with the contents of `app.py`.
   - Add a sibling `environment.yml` and paste in this folder's
     `environment.yml`. (SiS pulls these from Snowflake's curated
     `snowflake` Anaconda channel.)
5. Save and **Run**. The app will use the active Snowsight session role
   — no `.env` needed.

### SiS notes

- The Wafer ensemble call requires outbound HTTPS to `pass.wafer.ai`.
  In SiS that means an **External Access Integration** + a Network Rule
  for `pass.wafer.ai`, attached to the Streamlit app. Without it, tab 1
  will show an empty-state error and tabs 2–4 still work normally — the
  app degrades gracefully.
- `httpx` is imported lazily by `prophet_agent.llm.wafer`. If it's not
  in the SiS environment, tab 1 will surface an import error message
  rather than crashing.
- All Snowflake queries use parameterized binds; no user input is
  f-stringed into SQL.

## Files

- `app.py` — the dashboard (4 tabs)
- `environment.yml` — SiS Anaconda manifest (`snowflake` channel)
- `README.md` — this file
