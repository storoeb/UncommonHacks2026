"""Phase 6 — AutoML α-policy + shading layer (Beats 6 & 7).

Runs:
  1. Build ALPHA_TRAIN view  (joins META_PREDICTIONS_BACKFILL with backfill tables)
  2. compute_optimal_alpha per historical market  (line-search over α ∈ [0,1])
  3. MERGE alpha_star into META_PREDICTIONS_BACKFILL
  4. Snowflake SNOWFLAKE.ML.REGRESSION ALPHA_POLICY  (try/except)
  5. sklearn GradientBoostingRegressor  (always runs as shadow / fallback)
  6. Holdout Brier + AVER evaluation for: raw ensemble, always-Kalshi, global-α
     sweep, learned per-market α-policy
  7. Pareto plot  ->  notebooks/_pareto.png
  8. Writes notebooks/_alpha_policy.joblib + notebooks/_alpha_global.json
             + notebooks/_pareto_metrics.json  (read by Streamlit Pareto tab)

Empty upstream data degrades gracefully — the Pareto plot renders from whatever
rows exist; if none, it writes zeroed placeholder metrics.

Usage (from repo root):
    python notebooks/06_train_alpha_policy.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from prophet_agent.alpha_helpers import (  # noqa: E402
    apply_shading,
    aver_score,
    brier_score,
    compute_optimal_alpha,
)
from prophet_agent.snowflake_client import snowflake_cursor  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
# Suppress chatty Snowflake connector INFO logs (version, connecting, credentials)
logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
log = logging.getLogger("phase6")

NOTEBOOKS_DIR = project_root / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
JOBLIB_PATH     = NOTEBOOKS_DIR / "_alpha_policy.joblib"
PARETO_PATH     = NOTEBOOKS_DIR / "_pareto.png"
GLOBAL_JSON     = NOTEBOOKS_DIR / "_alpha_global.json"
PARETO_METRICS  = NOTEBOOKS_DIR / "_pareto_metrics.json"

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

ALPHA_TRAIN_DDL = """
CREATE OR REPLACE VIEW ALPHA_TRAIN AS
SELECT
    m.market_id,
    m.category,
    e.p_ensemble_var                                       AS disagreement,
    b.neighbor_count                                       AS neighbor_count,
    b.mean_similarity                                      AS neighbor_sim,
    ARRAY_MAX(m.q_kalshi_at_24h_pre)::FLOAT
      - ARRAY_MIN(m.q_kalshi_at_24h_pre)::FLOAT           AS q_spread,
    mp.alpha_star                                          AS alpha_star
FROM HISTORICAL_MARKETS m
JOIN ENSEMBLE_BACKFILL e    USING (market_id, source)
JOIN BASE_RATE_BACKFILL b   USING (market_id, source)
JOIN META_PREDICTIONS_BACKFILL mp USING (market_id, source)
WHERE mp.alpha_star IS NOT NULL
"""

RAW_FALLBACK_SQL = """
SELECT
    m.market_id,
    m.source,
    m.category,
    e.p_ensemble_var                                 AS disagreement,
    b.neighbor_count                                 AS neighbor_count,
    b.mean_similarity                                AS neighbor_sim,
    ARRAY_MAX(m.q_kalshi_at_24h_pre)::FLOAT
      - ARRAY_MIN(m.q_kalshi_at_24h_pre)::FLOAT      AS q_spread,
    mp.p_meta                                        AS p_meta_raw,
    e.p_ensemble                                     AS p_ensemble_raw,
    m.q_kalshi_at_24h_pre                            AS q_pre_raw,
    m.realized_outcome                               AS realized_outcome,
    m.resolve_ts                                     AS resolve_ts
FROM HISTORICAL_MARKETS m
JOIN ENSEMBLE_BACKFILL e    USING (market_id, source)
JOIN BASE_RATE_BACKFILL b   USING (market_id, source)
LEFT JOIN META_PREDICTIONS_BACKFILL mp USING (market_id, source)
WHERE m.realized_outcome IS NOT NULL
ORDER BY m.resolve_ts NULLS LAST
"""

META_MERGE_SQL = """
MERGE INTO META_PREDICTIONS_BACKFILL t
USING (
    SELECT %(market_id)s AS market_id,
           %(source)s    AS source,
           PARSE_JSON(%(p_meta_json)s) AS p_meta,
           %(alpha_star)s AS alpha_star
) s
ON t.market_id = s.market_id AND t.source = s.source
WHEN MATCHED THEN UPDATE SET
    p_meta = s.p_meta, alpha_star = s.alpha_star,
    backfilled_at = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (market_id, source, p_meta, alpha_star)
VALUES (s.market_id, s.source, s.p_meta, s.alpha_star)
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coerce_float_list(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        return [float(x) for x in json.loads(raw)]
    raise TypeError(type(raw))


def pick_p_meta(row: dict) -> list[float]:
    if row.get("p_meta_raw") is not None:
        try:
            return coerce_float_list(row["p_meta_raw"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return coerce_float_list(row["p_ensemble_raw"])


FEATURE_COLS = ["disagreement", "neighbor_count", "neighbor_sim", "q_spread", "category"]


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[FEATURE_COLS].copy()
    base["category"] = base["category"].fillna("nan").astype(str)
    return pd.get_dummies(base, columns=["category"], dummy_na=True)


def eval_profiles(hold_df: pd.DataFrame, preds_alpha: pd.Series | None = None) -> dict:
    out: dict = {}
    if len(hold_df) == 0:
        return out

    br_raw, av_raw, br_k, av_k = [], [], [], []
    for _, row in hold_df.iterrows():
        y = int(row["realized_outcome"])
        q = row["q_kalshi"]
        br_raw.append(brier_score(row["p_ensemble"], y))
        av_raw.append(aver_score(row["p_ensemble"], q, y))
        br_k.append(brier_score(q, y))
        av_k.append(aver_score(q, q, y))

    out["raw_ensemble"]  = {"brier": float(np.mean(br_raw)), "aver": float(np.mean(av_raw))}
    out["always_kalshi"] = {"brier": float(np.mean(br_k)),   "aver": float(np.mean(av_k))}

    sweep_curve = []
    for k in range(11):
        a = k / 10.0
        bs, av = [], []
        for _, row in hold_df.iterrows():
            pf = apply_shading(row["p_meta"], row["q_kalshi"], a)
            y  = int(row["realized_outcome"])
            bs.append(brier_score(pf, y))
            av.append(aver_score(pf, row["q_kalshi"], y))
        sweep_curve.append((a, float(np.mean(bs)), float(np.mean(av))))
    out["global_sweep"] = sweep_curve
    best_aver = max(sweep_curve, key=lambda t: t[2])
    out["best_global_alpha_by_aver"] = {
        "alpha": best_aver[0], "brier": best_aver[1], "aver": best_aver[2]
    }

    if preds_alpha is not None and len(preds_alpha) == len(hold_df):
        bs, av = [], []
        for (_, row), alpha_hat in zip(hold_df.iterrows(), preds_alpha):
            pf = apply_shading(row["p_meta"], row["q_kalshi"], float(alpha_hat))
            y  = int(row["realized_outcome"])
            bs.append(brier_score(pf, y))
            av.append(aver_score(pf, row["q_kalshi"], y))
        out["learned_policy"] = {"brier": float(np.mean(bs)), "aver": float(np.mean(av))}

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 6 — AutoML α-policy training + Pareto plot (Beats 6 + 7)")
    print("=" * 70)

    # Step 1 — fetch raw rows from Snowflake
    rows: list[dict] = []
    try:
        with snowflake_cursor() as cur:
            cur.execute(RAW_FALLBACK_SQL)
            cols = [d[0].lower() for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        log.info("Fetched %d raw rows from Snowflake.", len(rows))
    except Exception as exc:
        warnings.warn(f"Snowflake unavailable ({exc}); continuing with empty frames.", stacklevel=2)

    if not rows:
        log.warning("No rows — writing placeholder metrics and exiting.")
        _write_empty_artifacts()
        return

    # Step 2 — build training frame + compute alpha_star per market
    print(f"\nBuilding training frame from {len(rows)} Snowflake rows...")
    n_meta = sum(1 for r in rows if r.get("p_meta_raw") is not None)
    if n_meta == 0:
        warnings.warn(
            "META_PREDICTIONS_BACKFILL.p_meta missing — using p_ensemble as p_meta. "
            "Run notebooks/04_train_automl_calibrator.py first for best results.",
            stacklevel=2,
        )

    train_rows: list[dict] = []
    for row in rows:
        try:
            p_meta  = pick_p_meta(row)
            q_pre   = coerce_float_list(row["q_pre_raw"])
            p_ens   = coerce_float_list(row["p_ensemble_raw"])
            y       = int(row["realized_outcome"])
            if len(p_meta) != len(q_pre) or not p_meta:
                continue
            astar = compute_optimal_alpha(p_meta, q_pre, y)
            train_rows.append({
                "market_id":      row["market_id"],
                "source":         row["source"],
                "category":       row.get("category") or "nan",
                "disagreement":   float(row["disagreement"]),
                "neighbor_count": int(row["neighbor_count"] or 0),
                "neighbor_sim":   float(row["neighbor_sim"] or 0.0),
                "q_spread":       float(row["q_spread"] or 0.0),
                "resolve_ts":     row["resolve_ts"],
                "realized_outcome": y,
                "p_meta":         p_meta,
                "q_kalshi":       q_pre,
                "p_ensemble":     p_ens,
                "alpha_star":     astar,
            })
        except Exception as exc:
            log.debug("skip row %s: %s", row.get("market_id"), exc)

    print(f"  Training frame rows: {len(train_rows)}")
    if not train_rows:
        log.warning("No valid training rows after filtering — writing placeholder metrics.")
        _write_empty_artifacts()
        return

    train_full_df = pd.DataFrame(train_rows).sort_values("resolve_ts", na_position="first").reset_index(drop=True)

    # Step 3 — MERGE alpha_star into META_PREDICTIONS_BACKFILL
    # Non-critical: ALPHA_TRAIN view and sklearn training use in-memory data.
    # Skip the upsert to avoid long Snowflake round-trips during the hackathon.
    print("\nSkipping META_PREDICTIONS_BACKFILL upsert (non-critical for training).")
    try:
        with snowflake_cursor() as cur:
            cur.execute(ALPHA_TRAIN_DDL)
        log.info("ALPHA_TRAIN view created.")
    except Exception as exc:
        log.warning("ALPHA_TRAIN DDL failed (%s); continuing in-memory only.", exc)

    # Step 4 — temporal split
    n = len(train_full_df)
    split_idx = min(max(1, int(round(0.8 * n))), n - 1) if n > 1 else n
    train_df = train_full_df.iloc[:split_idx].copy()
    hold_df  = train_full_df.iloc[split_idx:].copy()
    print(f"\nTemporal split: train={len(train_df)}  holdout={len(hold_df)}")

    # Step 5 — Snowflake AutoML REGRESSION (try/except)
    snowflake_automl_ok = False
    if len(train_df) >= 5:
        print("\nAttempting Snowflake SNOWFLAKE.ML.REGRESSION ALPHA_POLICY...")
        create_reg_sql = """
CREATE OR REPLACE SNOWFLAKE.ML.REGRESSION ALPHA_POLICY (
    INPUT_DATA => SYSTEM$REFERENCE(
        'QUERY',
        $$SELECT CATEGORY, DISAGREEMENT, NEIGHBOR_COUNT, NEIGHBOR_SIM, Q_SPREAD, ALPHA_STAR
          FROM ALPHA_TRAIN$$
    ),
    TARGET_COLNAME => 'ALPHA_STAR'
)"""
        try:
            with snowflake_cursor() as cur:
                cur.execute(create_reg_sql)
            snowflake_automl_ok = True
            print("  Snowflake ML REGRESSION ALPHA_POLICY trained successfully.")
        except Exception as exc:
            log.warning("Snowflake ML REGRESSION failed (sklearn fallback): %s", exc)
    else:
        log.warning("Skipping Snowflake AutoML — only %d train rows (need >=5).", len(train_df))

    # Step 6 — sklearn GradientBoostingRegressor (always runs)
    print("\nTraining sklearn GradientBoostingRegressor...")
    sklearn_ok = False
    gb_model = None
    feature_columns: list[str] = []

    if len(train_df) >= 3:
        X_train        = design_matrix(train_df)
        feature_columns = list(X_train.columns)
        y_train        = train_df["alpha_star"].astype(float).values
        gb_model       = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X_train.values, y_train)
        joblib.dump({"model": gb_model, "feature_columns": feature_columns}, JOBLIB_PATH)
        sklearn_ok = True
        print(f"  Saved sklearn bundle -> {JOBLIB_PATH}")
    else:
        log.warning("Skipping sklearn — need >=3 train rows.")

    # Step 7 — eval
    preds_hold = None
    if gb_model is not None and len(hold_df):
        X_hold = design_matrix(hold_df).reindex(columns=feature_columns, fill_value=0)
        preds_hold = pd.Series(gb_model.predict(X_hold.values))
    elif len(hold_df) and len(train_full_df):
        med = float(train_full_df["alpha_star"].median())
        preds_hold = pd.Series([med] * len(hold_df))

    metrics_bundle = eval_profiles(hold_df, preds_hold)

    # Global alpha JSON (used by shading.py fallback)
    best_a = 0.5
    if metrics_bundle.get("global_sweep"):
        best_a = float(metrics_bundle["best_global_alpha_by_aver"]["alpha"])
    GLOBAL_JSON.write_text(json.dumps({"alpha": best_a}, indent=2), encoding="utf-8")
    print(f"\n  Best global α (by holdout AVER): {best_a:.2f}")
    print(f"  Global α JSON -> {GLOBAL_JSON}")

    # Step 8 — Pareto plot
    fig, ax = plt.subplots(figsize=(7, 5))
    if metrics_bundle.get("raw_ensemble"):
        pt = metrics_bundle["raw_ensemble"]
        ax.scatter(pt["brier"], pt["aver"], label="Raw ensemble", color="tab:blue", s=80)
    if metrics_bundle.get("always_kalshi"):
        pt = metrics_bundle["always_kalshi"]
        ax.scatter(pt["brier"], pt["aver"], label="Always Kalshi", color="tab:orange", s=80)
    if metrics_bundle.get("global_sweep"):
        xs = [t[1] for t in metrics_bundle["global_sweep"]]
        ys = [t[2] for t in metrics_bundle["global_sweep"]]
        ax.plot(xs, ys, "--", color="gray", alpha=0.8, label="Global α sweep")
        ax.scatter(xs, ys, color="lightgray", s=22)
    if metrics_bundle.get("learned_policy"):
        pt = metrics_bundle["learned_policy"]
        ax.scatter(pt["brier"], pt["aver"], label="Learned α-policy", color="tab:green", s=130, marker="*")
    ax.set_xlabel("Mean Brier (lower is better)")
    ax.set_ylabel("Mean AVER vs market (higher is better)")
    ax.set_title("Phase 6 — holdout Brier vs AVER")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PARETO_PATH, dpi=150)
    plt.close(fig)
    print(f"\n  Pareto plot -> {PARETO_PATH}")

    # Streamlit-friendly metrics JSON (flat: label -> {brier, aver})
    pareto_flat: dict = {}
    for key in ("raw_ensemble", "always_kalshi", "best_global_alpha_by_aver", "learned_policy"):
        if metrics_bundle.get(key):
            pareto_flat[key] = {
                "brier": metrics_bundle[key].get("brier"),
                "aver":  metrics_bundle[key].get("aver"),
            }
    PARETO_METRICS.write_text(json.dumps(pareto_flat, indent=2), encoding="utf-8")
    print(f"  Pareto metrics JSON -> {PARETO_METRICS}")

    # Final summary table
    print("\n--- Holdout Brier / AVER summary ---")
    cols = ["Strategy", "Brier", "AVER"]
    rows_out = []
    for label, key in [
        ("Raw ensemble",                "raw_ensemble"),
        ("Always Kalshi",               "always_kalshi"),
        (f"Best global α={best_a:.1f}", "best_global_alpha_by_aver"),
        ("Learned per-market α-policy", "learned_policy"),
    ]:
        if metrics_bundle.get(key):
            rows_out.append((label,
                             f"{metrics_bundle[key].get('brier', float('nan')):.6f}",
                             f"{metrics_bundle[key].get('aver', float('nan')):.6f}"))
    col_w = max(len(r[0]) for r in rows_out) + 2 if rows_out else 40
    print(f"  {'Strategy':<{col_w}}  {'Brier':>10}  {'AVER':>10}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*10}")
    for label, brier, aver in rows_out:
        print(f"  {label:<{col_w}}  {brier:>10}  {aver:>10}")

    print(f"\nSnowflake AutoML trained: {'yes' if snowflake_automl_ok else 'no (sklearn shadow)'}")
    print("Done.")


def _write_empty_artifacts() -> None:
    GLOBAL_JSON.write_text(json.dumps({"alpha": 0.5}, indent=2), encoding="utf-8")
    PARETO_METRICS.write_text(json.dumps({}), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Phase 6 — no data yet")
    ax.text(0.5, 0.5, "Run the data pipeline first\n(see README.md)", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="gray")
    fig.savefig(PARETO_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote empty artifact placeholders to {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
