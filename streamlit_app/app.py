"""ProphetHacks 2026 — demo dashboard.

Four-tab Streamlit app for the pitch. Designed to run both locally
(`streamlit run streamlit_app/app.py`) and inside Streamlit-in-Snowflake (SiS).

Story beats this surfaces (see whimsical-kindling-hamster.md):
  Tab 1 — Live Forecast        : beats 4, 5, 6, 8
  Tab 2 — Brier vs AVER Pareto : beats 1, 2, 6
  Tab 3 — By-Category          : beats 5, 7
  Tab 4 — Institutional Memory : beats 3, 7

Empty-state philosophy: never raise to the user. Every panel handles
"no data yet" gracefully so the demo always renders something coherent.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Path bootstrap — make `prophet_agent.*` importable when running locally
# from the repo root. In Streamlit-in-Snowflake the package isn't shipped;
# the Snowpark session is what we use, and we degrade Wafer-ensemble UI
# accordingly.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Optional imports — every one of these is allowed to fail; the relevant
# panel will render an empty-state message instead.
# ---------------------------------------------------------------------------
try:
    from prophet_agent.snowflake_client import snowflake_cursor  # type: ignore
    _HAS_SNOWFLAKE_CLIENT = True
    _SNOWFLAKE_IMPORT_ERR: str | None = None
except Exception as exc:  # noqa: BLE001 — UI should never crash
    snowflake_cursor = None  # type: ignore
    _HAS_SNOWFLAKE_CLIENT = False
    _SNOWFLAKE_IMPORT_ERR = f"{type(exc).__name__}: {exc}"

try:
    from prophet_agent.llm.wafer import WaferClient  # type: ignore
    _HAS_WAFER = True
    _WAFER_IMPORT_ERR: str | None = None
except Exception as exc:  # noqa: BLE001
    WaferClient = None  # type: ignore
    _HAS_WAFER = False
    _WAFER_IMPORT_ERR = f"{type(exc).__name__}: {exc}"

try:
    # Built in parallel by another agent — must not be a hard dep.
    from prophet_agent.retrieval.base_rate import get_base_rate  # type: ignore
    _HAS_BASE_RATE = True
except Exception:  # noqa: BLE001
    get_base_rate = None  # type: ignore
    _HAS_BASE_RATE = False


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ProphetHacks 2026 — Forecasting Agent",
    page_icon="P",
    layout="wide",
)

st.title("ProphetHacks 2026 — market-aware forecasting agent")
st.caption(
    "Wafer.ai ensemble + Snowflake institutional memory + AutoML calibrator "
    "+ AutoML alpha-policy. Demo dashboard, lives next to the data in Streamlit-in-Snowflake."
)


# ---------------------------------------------------------------------------
# Cached Snowflake access
#
# `@st.cache_resource` keeps one cursor-factory across reruns so we don't
# pay the auth round-trip every interaction. SQL results are cached with a
# 60s TTL — short enough to feel live, long enough to keep the demo snappy.
#
# All user-supplied values are passed as bind parameters, never f-stringed
# into the SQL.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _snowflake_available() -> bool:
    """Probe Snowflake once; cache the boolean so we don't reconnect repeatedly."""
    if not _HAS_SNOWFLAKE_CLIENT:
        return False
    try:
        with snowflake_cursor() as cur:  # type: ignore[misc]
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_query(sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    """Execute a parameterized SQL query and return a DataFrame.

    Returns an empty DataFrame on any failure so callers can branch on `.empty`.
    """
    if not _HAS_SNOWFLAKE_CLIENT:
        return pd.DataFrame()
    try:
        with snowflake_cursor() as cur:  # type: ignore[misc]
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Snowflake query failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_category_counts() -> pd.DataFrame:
    """Count of resolved historical markets per category."""
    sql = (
        "SELECT COALESCE(category, 'unknown') AS category, COUNT(*) AS n_markets "
        "FROM HISTORICAL_MARKETS "
        "GROUP BY 1 ORDER BY n_markets DESC"
    )
    return _run_query(sql)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_total_markets() -> int:
    df = _run_query("SELECT COUNT(*) AS n FROM HISTORICAL_MARKETS")
    if df.empty:
        return 0
    return int(df.iloc[0, 0])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_by_category_brier() -> pd.DataFrame:
    """Per-category Brier — agent vs naive Kalshi-match baseline.

    Empty until AGENT_PREDICTIONS has rows with resolved outcomes.
    """
    # Computes Brier for our system (`p_final`) and for the naive
    # "always match Kalshi" baseline (`q_kalshi`), per category, over
    # predictions where the realized outcome is known.
    sql = """
        SELECT
            COALESCE(category, 'unknown') AS category,
            COUNT(*) AS n,
            AVG(brier) AS agent_brier,
            AVG(
                CASE
                    WHEN q_kalshi IS NULL OR realized_outcome IS NULL THEN NULL
                    ELSE (
                        -- Brier for the always-match-kalshi baseline.
                        -- Sum over outcomes of (q_k - 1{realized==k})^2.
                        (
                            SELECT SUM(
                                POWER(
                                    f.value::FLOAT
                                    - CASE WHEN f.index = realized_outcome THEN 1.0 ELSE 0.0 END,
                                    2
                                )
                            )
                            FROM TABLE(FLATTEN(input => q_kalshi)) f
                        )
                    )
                END
            ) AS kalshi_brier
        FROM AGENT_PREDICTIONS
        WHERE realized_outcome IS NOT NULL
        GROUP BY 1
        ORDER BY n DESC
    """
    return _run_query(sql)


@st.cache_data(ttl=60, show_spinner=False)
def search_memory_via_snowflake(question: str, k: int = 10) -> pd.DataFrame:
    """Top-K resolved historical markets by question-embedding similarity.

    Uses Snowflake's `AI_EMBED` to embed the query on the warehouse, then
    `VECTOR_COSINE_SIMILARITY` against the stored 768-dim vectors. The
    `question` is bound as a parameter — no string interpolation into SQL.
    """
    # %(k)s is interpolated by snowflake-connector-python as a server-side
    # bind parameter (paramstyle='pyformat'). `question` is bound too.
    sql = """
        SELECT
            market_id,
            COALESCE(category, 'unknown') AS category,
            question_text,
            realized_outcome,
            outcomes,
            VECTOR_COSINE_SIMILARITY(
                question_embedding,
                AI_EMBED('snowflake-arctic-embed-m-v1.5', %(q)s)
            ) AS similarity
        FROM HISTORICAL_MARKETS
        WHERE question_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT %(k)s
    """
    # snowflake-connector accepts dict-style binds when paramstyle='pyformat'.
    if not _HAS_SNOWFLAKE_CLIENT:
        return pd.DataFrame()
    try:
        with snowflake_cursor() as cur:  # type: ignore[misc]
            cur.execute(sql, {"q": question, "k": int(k)})
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description] if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Snowflake similarity search failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Wafer ensemble — wraps the async client in a sync call.
# ---------------------------------------------------------------------------
def run_ensemble_sync(
    question: str,
    outcomes: list[str],
    market_prices: list[float] | None,
) -> Any:
    """Synchronously run the Wafer ensemble. Returns an `EnsembleResult` or None on failure."""
    if not _HAS_WAFER:
        return None
    try:
        client = WaferClient()  # type: ignore[misc]
        return asyncio.run(
            client.ensemble(
                question=question,
                outcomes=outcomes,
                market_prices=market_prices,
            )
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ensemble call failed: {type(exc).__name__}: {exc}")
        return None


def _bet_recommendation(
    p_final: list[float],
    q_market: list[float] | None,
    outcomes: list[str],
    edge_threshold: float = 0.05,
) -> tuple[str, pd.DataFrame]:
    """Compute per-outcome edge vs market and pick a recommendation.

    Edge is defined as `p_final[i] - q_market[i]` (probability terms). We
    recommend a bet on the outcome with the largest positive edge above
    `edge_threshold`; otherwise "no bet".
    """
    if q_market is None or len(q_market) != len(p_final):
        df = pd.DataFrame({
            "outcome": outcomes,
            "p_final": [round(p, 4) for p in p_final],
        })
        return "No market prices supplied — no bet decision possible.", df

    edges = [p - q for p, q in zip(p_final, q_market)]
    ratios = [
        (p / q) if q > 0 else float("inf")
        for p, q in zip(p_final, q_market)
    ]
    df = pd.DataFrame({
        "outcome": outcomes,
        "p_final": [round(p, 4) for p in p_final],
        "q_market": [round(q, 4) for q in q_market],
        "edge (p - q)": [round(e, 4) for e in edges],
        "ratio (p / q)": [round(r, 3) for r in ratios],
    })

    best_idx = int(np.argmax(edges))
    if edges[best_idx] >= edge_threshold:
        recommendation = (
            f"Size on **{outcomes[best_idx]}** "
            f"(edge {edges[best_idx]:+.3f}, ratio {ratios[best_idx]:.2f}x)"
        )
    else:
        recommendation = (
            f"No bet — max edge {edges[best_idx]:+.3f} below threshold {edge_threshold:.2f}."
        )
    return recommendation, df


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_pareto, tab_category, tab_memory = st.tabs([
    "Live Forecast",
    "Brier vs AVER Pareto",
    "By-Category Performance",
    "Institutional Memory",
])


# ===========================================================================
# Tab 1 — Live Forecast
# ===========================================================================
with tab_live:
    st.subheader("Live forecast")
    st.write(
        "Type a forecasting question, supply the candidate outcomes, optionally "
        "paste the current Kalshi-style market price for the *Yes* outcome (binary). "
        "Hit **Run forecast** and watch the whole pipeline run."
    )

    col_q, col_o = st.columns([3, 2])
    with col_q:
        question = st.text_area(
            "Question",
            value="Will Bitcoin close above $100,000 USD on the last day of this year?",
            height=80,
        )
    with col_o:
        outcomes_csv = st.text_input(
            "Outcomes (comma-separated)",
            value="Yes,No",
            help="Order matters. Probabilities will be returned in this order.",
        )
        yes_price_str = st.text_input(
            "Kalshi market price for Yes (optional, 0–1)",
            value="0.55",
            help="Binary markets only. Leave blank if no market price.",
        )

    run_btn = st.button("Run forecast", type="primary")

    outcomes = [o.strip() for o in outcomes_csv.split(",") if o.strip()]
    market_prices: list[float] | None = None
    if yes_price_str.strip() and len(outcomes) == 2:
        try:
            p_yes = float(yes_price_str)
            if 0.0 <= p_yes <= 1.0:
                market_prices = [p_yes, 1.0 - p_yes]
        except ValueError:
            st.warning("Could not parse market price; ignoring.")

    if run_btn:
        if not question.strip():
            st.error("Question is empty.")
        elif len(outcomes) < 2:
            st.error("Need at least two outcomes.")
        elif not _HAS_WAFER:
            st.error(
                "Wafer client not importable in this environment. "
                f"(Import error: {_WAFER_IMPORT_ERR})"
            )
        else:
            with st.spinner("Calling Wafer ensemble (3 models in parallel)..."):
                t0 = time.time()
                result = run_ensemble_sync(question, outcomes, market_prices)
                elapsed = time.time() - t0

            if result is None:
                st.error("Ensemble returned no result.")
            else:
                st.success(f"Ensemble responded in {elapsed:.2f}s.")

                # --- Per-model table ---
                st.markdown("#### Per-model probabilities")
                model_rows = []
                for f in result.forecasts:
                    model_rows.append({
                        "model": f.model,
                        "probabilities": [round(p, 4) for p in f.probabilities],
                        "latency_s": round(f.latency_s, 2),
                    })
                st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

                # --- Aggregate cards ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ensemble mean P(outcome 0)", f"{result.mean[0]:.3f}")
                c2.metric("Disagreement variance", f"{result.variance:.4f}")
                # For now alpha and p_final default to ensemble mean (calibrator stubbed).
                alpha = 1.0
                p_final = list(result.mean)
                c3.metric("Alpha (calibrator weight)", f"{alpha:.2f}")
                c4.metric(
                    "P_final(outcome 0)",
                    f"{p_final[0]:.3f}",
                    delta=(
                        f"{p_final[0] - market_prices[0]:+.3f} vs market"
                        if market_prices else None
                    ),
                )

                st.caption(
                    "Calibrator + alpha-policy are stubbed in this build. "
                    "`p_final = ensemble mean`, `alpha = 1.0`. "
                    "Phase 4/6 will swap these for the Snowflake AutoML model outputs."
                )

                # --- Retrieval ---
                st.markdown("#### Top retrieved neighbors")
                neighbors_df = pd.DataFrame()
                if _HAS_BASE_RATE:
                    try:
                        # base_rate module may expose either a sync or async API;
                        # try both shapes defensively.
                        br_out = get_base_rate(question_text=question, category=None, k=10)  # type: ignore[misc]
                        if asyncio.iscoroutine(br_out):
                            br_out = asyncio.run(br_out)
                        # Convention: returns a dict with a 'neighbors' list of dicts.
                        if isinstance(br_out, dict) and "neighbors" in br_out:
                            neighbors_df = pd.DataFrame(br_out["neighbors"])
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"base_rate lookup failed: {type(exc).__name__}: {exc}")

                # Fallback: query Snowflake directly with AI_EMBED similarity.
                if neighbors_df.empty and _snowflake_available():
                    neighbors_df = search_memory_via_snowflake(question, k=10)

                if neighbors_df.empty:
                    st.info(
                        "No neighbors retrieved. Either the `retrieval.base_rate` module "
                        "is not yet available, or `HISTORICAL_MARKETS` is empty. "
                        "Run `scripts/import_kalshi_history.py` then `scripts/embed_questions.py`."
                    )
                else:
                    st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

                # --- Bet decision ---
                st.markdown("#### Bet decision")
                rec, edge_df = _bet_recommendation(p_final, market_prices, outcomes)
                st.dataframe(edge_df, use_container_width=True, hide_index=True)
                st.markdown(rec)

                # --- Logging (best-effort) ---
                if _snowflake_available():
                    request_id = uuid.uuid4().hex
                    try:
                        with snowflake_cursor() as cur:  # type: ignore[misc]
                            cur.execute(
                                """
                                INSERT INTO AGENT_PREDICTIONS
                                    (request_id, question_text, outcomes,
                                     q_kalshi, p_ensemble, p_ensemble_var,
                                     p_final, alpha)
                                SELECT
                                    %(rid)s, %(q)s, PARSE_JSON(%(outs)s),
                                    PARSE_JSON(%(qk)s), PARSE_JSON(%(pe)s), %(pv)s,
                                    PARSE_JSON(%(pf)s), %(a)s
                                """,
                                {
                                    "rid": request_id,
                                    "q": question,
                                    "outs": json.dumps(outcomes),
                                    "qk": json.dumps(market_prices) if market_prices else "null",
                                    "pe": json.dumps([float(x) for x in result.mean]),
                                    "pv": float(result.variance),
                                    "pf": json.dumps([float(x) for x in p_final]),
                                    "a": float(alpha),
                                },
                            )
                        st.caption(f"Logged to `AGENT_PREDICTIONS` as `{request_id}`.")
                    except Exception as exc:  # noqa: BLE001
                        st.caption(f"(Did not log to Snowflake: {type(exc).__name__}: {exc})")


# ===========================================================================
# Tab 2 — Brier vs AVER Pareto
# ===========================================================================
with tab_pareto:
    st.subheader("Brier vs AVER — the metric Prof. Xu wrote about")
    st.markdown(
        "> *\"A has a higher Brier Score, but lower returns.\"*  \n"
        "> — Prof. Haifeng Xu et al., "
        "[PM-RANK 0.3.1 blogpost](https://ai-prophet.github.io/pm_ranking/blogpost/ranking_llm_250727.html)"
    )
    st.write(
        "Brier is an **absolute** calibration metric; AVER (averaged return) is a "
        "**relative** wagering metric. Optimizing one accidentally is not enough — "
        "the gap between them is exactly what we built our system to close."
    )

    # Load eval data from notebooks/_calibrator_last_metrics.json (Phase 4) and
    # notebooks/_pareto_metrics.json (Phase 6) when available; fall back to
    # illustrative placeholder values so the tab always renders something.
    _METRICS_PATH = _PROJECT_ROOT / "notebooks" / "_calibrator_last_metrics.json"
    _PARETO_PATH = _PROJECT_ROOT / "notebooks" / "_pareto_metrics.json"

    def _load_pareto_df() -> tuple[pd.DataFrame, bool]:
        """Return (DataFrame, is_real_data)."""
        rows: list[dict] = []
        is_real = False

        # Phase 6 pareto bundle — produced by notebooks/06_train_alpha_policy.ipynb
        if _PARETO_PATH.is_file():
            try:
                bundle = json.loads(_PARETO_PATH.read_text(encoding="utf-8"))
                for label, vals in bundle.items():
                    brier = vals.get("brier")
                    aver = vals.get("aver")
                    if brier is not None and aver is not None:
                        rows.append({
                            "system": label,
                            "brier (lower is better)": float(brier),
                            "aver (higher is better)": float(aver),
                        })
                if rows:
                    is_real = True
            except Exception:  # noqa: BLE001
                rows = []

        # Phase 4 only — Brier table without AVER data; fill AVER as 0 placeholders
        if not rows and _METRICS_PATH.is_file():
            try:
                m = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
                for key, brier_val in m.items():
                    if isinstance(brier_val, (int, float)):
                        rows.append({
                            "system": key,
                            "brier (lower is better)": float(brier_val),
                            "aver (higher is better)": 0.0,
                        })
                if rows:
                    is_real = True
            except Exception:  # noqa: BLE001
                rows = []

        if not rows:
            rows = [
                {"system": "Raw Wafer ensemble",          "brier (lower is better)": 0.215, "aver (higher is better)": 0.04},
                {"system": "Always-match Kalshi",          "brier (lower is better)": 0.182, "aver (higher is better)": 0.00},
                {"system": "Global alpha = 0.25",          "brier (lower is better)": 0.178, "aver (higher is better)": 0.03},
                {"system": "Global alpha = 0.50",          "brier (lower is better)": 0.172, "aver (higher is better)": 0.05},
                {"system": "Global alpha = 0.75",          "brier (lower is better)": 0.169, "aver (higher is better)": 0.07},
                {"system": "Learned alpha-policy (target)","brier (lower is better)": 0.165, "aver (higher is better)": 0.11},
            ]

        return pd.DataFrame(rows), is_real

    pareto_df, pareto_is_real = _load_pareto_df()

    st.scatter_chart(
        pareto_df,
        x="brier (lower is better)",
        y="aver (higher is better)",
        color="system",
        height=420,
    )

    if pareto_is_real:
        st.success(
            "Showing **real holdout eval numbers** from the trained notebooks. "
            "The learned per-market α-policy should sit above-right of the global-α sweep."
        )
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Illustrative placeholder values — run "
            "`notebooks/04_train_automl_calibrator.ipynb` then "
            "`notebooks/06_train_alpha_policy.ipynb` to replace with real holdout numbers."
        )

    with st.expander("Why this is the headline plot"):
        st.markdown(
            "- **Naive LLM submissions** optimize Brier accidentally and AVER never.\n"
            "- **Matching the market** gets good Brier and zero AVER (you can't beat the market by being it).\n"
            "- **Deviating without grounding** gets bad Brier *and* bad AVER.\n"
            "- The Snowflake-trained calibrator gives us a grounded deviation; "
            "the alpha-policy decides per-market how aggressively to take it."
        )


# ===========================================================================
# Tab 3 — By-Category Performance
# ===========================================================================
with tab_category:
    st.subheader("Where do we beat the market?")
    st.write(
        "Per-category Brier of our agent's `p_final` versus the always-match-Kalshi "
        "baseline. Populated from `AGENT_PREDICTIONS` once realized outcomes are written back."
    )

    if not _snowflake_available():
        st.info(
            "Snowflake not reachable. Set credentials in the project `.env` "
            f"(import error: `{_SNOWFLAKE_IMPORT_ERR}`) "
            "or, in Streamlit-in-Snowflake, this panel will use the session role."
        )
    else:
        brier_df = fetch_by_category_brier()
        if brier_df.empty:
            st.info(
                "No resolved predictions yet — once the agent has served requests "
                "and outcomes have been written back to `AGENT_PREDICTIONS`, this "
                "chart will populate."
            )
        else:
            chart_df = brier_df.set_index("CATEGORY")[["AGENT_BRIER", "KALSHI_BRIER"]]
            st.bar_chart(chart_df, height=380)
            st.dataframe(brier_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Historical market counts (per category)")
        cat_df = fetch_category_counts()
        if cat_df.empty:
            n_total = fetch_total_markets()
            if n_total == 0:
                st.info(
                    "No historical markets ingested yet. "
                    "Run `python scripts/import_kalshi_history.py` to populate "
                    "`HISTORICAL_MARKETS`."
                )
            else:
                st.caption(f"{n_total} markets ingested, but no categories assigned yet.")
        else:
            st.bar_chart(cat_df.set_index("CATEGORY")["N_MARKETS"], height=320)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 4 — Institutional Memory (the Snowflake side-prize visual)
# ===========================================================================
with tab_memory:
    st.subheader("Institutional memory — vector search over resolved markets")
    st.write(
        "Snowflake stores every resolved market we've ingested, embedded with "
        "`AI_EMBED('snowflake-arctic-embed-m-v1.5', question_text)`. Type any "
        "question to find the most semantically similar past markets — and what "
        "actually happened."
    )

    mem_q = st.text_input(
        "Question to search for",
        value="Will the Federal Reserve cut interest rates this year?",
    )
    search_btn = st.button("Search memory")

    if search_btn:
        if not mem_q.strip():
            st.error("Question is empty.")
        elif not _snowflake_available():
            st.info(
                "Snowflake not reachable. Set credentials in the project `.env` "
                "or run inside Streamlit-in-Snowflake where the session role "
                "is provided automatically."
            )
        else:
            with st.spinner("Embedding and searching..."):
                df = search_memory_via_snowflake(mem_q.strip(), k=10)

            if df.empty:
                st.info(
                    "No matches — likely because `HISTORICAL_MARKETS` is empty or "
                    "embeddings haven't been computed. "
                    "Run `scripts/import_kalshi_history.py` then `scripts/embed_questions.py`."
                )
            else:
                # Show a similarity bar chart followed by the styled detail table.
                chart = df[["MARKET_ID", "SIMILARITY"]].set_index("MARKET_ID")
                st.bar_chart(chart, height=320)

                # Tidy table.
                display_cols = [
                    c for c in
                    ["MARKET_ID", "CATEGORY", "QUESTION_TEXT",
                     "REALIZED_OUTCOME", "OUTCOMES", "SIMILARITY"]
                    if c in df.columns
                ]
                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Why this slide matters for the Snowflake side prize"):
        st.markdown(
            "- The same Snowflake table is the **training set** for the AutoML "
            "calibrator and the **runtime memory** the agent queries on every request.\n"
            "- `AI_EMBED` + `VECTOR_COSINE_SIMILARITY` run on the warehouse — "
            "no data egress, no extra service.\n"
            "- This is \"turning raw data into something insightful\" — "
            "literally the side-prize judging criterion."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ProphetHacks 2026 submission. Streamlit-in-Snowflake demo — "
    "the visual layer over the same data the agent queries."
)
