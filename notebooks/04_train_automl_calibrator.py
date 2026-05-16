"""Phase 4 — Train Snowflake AutoML meta-calibrator (Beats 5 & 7).

Runs:
  1. CREATE OR REPLACE VIEW META_TRAIN  (temporal join of backfill tables)
  2. Temporal 80/20 train/holdout split by resolve_ts
  3. Snowflake SNOWFLAKE.ML.CLASSIFICATION META_CALIBRATOR  (try/except)
  4. sklearn GradientBoostingClassifier  (always runs as shadow / fallback)
  5. Brier leaderboard on holdout
  6. MERGE all rows into META_PREDICTIONS_BACKFILL  (feeds Phase 6)
  7. Writes notebooks/_calibrator.joblib + notebooks/_calibrator_last_metrics.json

Empty upstream data (ENSEMBLE_BACKFILL empty) triggers a synthetic path so
you can inspect the pipeline skeleton without live Snowflake data.

Usage (from repo root):
    python notebooks/04_train_automl_calibrator.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from prophet_agent.calibrator import (  # noqa: E402
    CALIBRATOR_CATEGORY_ORDER,
    build_calibrator_feature_rows,
    calibrator_feature_column_names,
)
from prophet_agent.snowflake_client import snowflake_cursor  # noqa: E402

NOTEBOOKS_DIR = project_root / "notebooks"
JOBLIB_PATH = NOTEBOOKS_DIR / "_calibrator.joblib"
METRICS_JSON = NOTEBOOKS_DIR / "_calibrator_last_metrics.json"
FEATURE_COLUMNS = calibrator_feature_column_names()

NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

META_TRAIN_SQL = """
CREATE OR REPLACE VIEW META_TRAIN AS
SELECT
    m.market_id,
    m.source,
    m.resolve_ts,
    m.category                               AS CATEGORY,
    m.q_kalshi_at_24h_pre[f.index]::FLOAT   AS Q,
    e.p_ensemble[f.index]::FLOAT             AS P_LLM,
    e.p_ensemble_var                         AS DISAGREEMENT,
    b.base_rate[f.index]::FLOAT              AS BASE_RATE,
    f.index::INTEGER                         AS OUTCOME_IDX,
    CASE WHEN m.realized_outcome = f.index THEN 1 ELSE 0 END::INTEGER AS Y
FROM HISTORICAL_MARKETS m
JOIN ENSEMBLE_BACKFILL e   USING (market_id, source)
JOIN BASE_RATE_BACKFILL b  USING (market_id, source),
    LATERAL FLATTEN(input => m.outcomes) f
WHERE m.realized_outcome IS NOT NULL
"""

SCORE_ROWS_SQL = """
CREATE OR REPLACE VIEW META_SCORE_ROWS AS
SELECT
    m.market_id,
    m.source,
    m.category                               AS CATEGORY,
    m.q_kalshi_at_24h_pre[f.index]::FLOAT   AS Q,
    e.p_ensemble[f.index]::FLOAT             AS P_LLM,
    e.p_ensemble_var                         AS DISAGREEMENT,
    b.base_rate[f.index]::FLOAT              AS BASE_RATE,
    f.index::INTEGER                         AS OUTCOME_IDX
FROM HISTORICAL_MARKETS m
JOIN ENSEMBLE_BACKFILL e   USING (market_id, source)
JOIN BASE_RATE_BACKFILL b  USING (market_id, source),
    LATERAL FLATTEN(input => m.outcomes) f
"""

MERGE_META_SQL = """
MERGE INTO META_PREDICTIONS_BACKFILL t
 USING (SELECT %(market_id)s m_id, %(source)s src, PARSE_JSON(%(p_meta)s) pj) s
 ON t.market_id = s.m_id AND t.source = s.src
 WHEN MATCHED THEN UPDATE SET p_meta = s.pj, backfilled_at = CURRENT_TIMESTAMP()
 WHEN NOT MATCHED THEN INSERT (market_id, source, p_meta) VALUES (s.m_id, s.src, s.pj)
"""

# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    if s <= 1e-12:
        n = len(xs)
        return [1.0 / n] * n if n else []
    return [e / s for e in ex]


def _sify(x: object) -> str:
    return str(x)


def market_train_keys(ms: pd.DataFrame, frac: float = 0.8) -> set[tuple[str, str]]:
    u = ms.drop_duplicates(["market_id", "source"]).sort_values(
        ["resolve_ts", "market_id", "source"], kind="stable"
    )
    if u.empty:
        return set()
    kt = max(1, int(math.floor(len(u) * frac)))
    head = u.iloc[:kt]
    return {(_sify(a), _sify(b)) for a, b in zip(head.market_id.tolist(), head.source.tolist())}


def is_train_series(df: pd.DataFrame, tr: set[tuple[str, str]]) -> pd.Series:
    keys = [_sify(a) + "||" + _sify(b) for a, b in zip(df.market_id.tolist(), df.source.tolist())]
    tr_s = {_sify(a) + "||" + _sify(b) for a, b in tr}
    return pd.Series([k in tr_s for k in keys], index=df.index)


def xy_design(df_piece: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[list[float]] = []
    ys: list[int] = []
    grp = sorted(
        df_piece.groupby(["market_id", "source"]),
        key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
    )
    for (_, _), g0 in grp:
        g = g0.sort_values("outcome_idx")
        rows, _ = build_calibrator_feature_rows(
            p_ensemble=g["p_llm"].astype(float).tolist(),
            q_kalshi=g["q"].astype(float).tolist(),
            base_rate=g["base_rate"].astype(float).tolist(),
            category=str(g["category"].iloc[0]),
            disagreement=float(g["disagreement"].iloc[0]),
        )
        Xs.extend(rows)
        ys.extend(g["y"].astype(int).tolist())
    return np.asarray(Xs, float), np.asarray(ys, int)


def slices_hold(df_hold: pd.DataFrame) -> dict[tuple[str, str], slice]:
    out: dict[tuple[str, str], slice] = {}
    i = 0
    grp = sorted(
        df_hold.groupby(["market_id", "source"]),
        key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
    )
    for (mid, src), gg in grp:
        k = gg.shape[0]
        out[(_sify(mid), _sify(src))] = slice(i, i + k)
        i += k
    return out


def winner_index(df_hold: pd.DataFrame) -> dict[tuple[str, str], int]:
    w: dict[tuple[str, str], int] = {}
    for (mid, src), gg in df_hold.groupby(["market_id", "source"], sort=False):
        ys = gg.sort_values("outcome_idx")["y"].astype(int).to_numpy()
        w[(_sify(mid), _sify(src))] = int(np.argmax(ys))
    return w


def vec_per_market(df_hold: pd.DataFrame, column: str) -> dict[tuple[str, str], np.ndarray]:
    dd: dict[tuple[str, str], np.ndarray] = {}
    grp = sorted(
        df_hold.groupby(["market_id", "source"]),
        key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
    )
    for (mid, src), gg in grp:
        g = gg.sort_values("outcome_idx")
        dd[(_sify(mid), _sify(src))] = np.asarray(softmax(g[column].astype(float).tolist()))
    return dd


def from_positives(
    raw: np.ndarray,
    smp: dict[tuple[str, str], slice],
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for k, sl in smp.items():
        chunk = raw[sl] if raw.size else np.asarray([])
        out[k] = np.asarray(softmax(list(chunk)) if chunk.size else [])
    return out


def pooled_brier(
    probs: dict[tuple[str, str], np.ndarray],
    win: dict[tuple[str, str], int],
) -> float:
    acc: list[float] = []
    for k, pv in probs.items():
        if pv.size == 0:
            continue
        oh = np.zeros_like(pv)
        wi = win.get(k, -1)
        if 0 <= wi < len(pv):
            oh[wi] = 1.0
        acc.append(float(np.mean((pv - oh) ** 2)))
    return float(np.mean(acc)) if acc else float("nan")


def parse_variant(v: object) -> float:
    blob = v
    if hasattr(blob, "as_dict"):
        try:
            blob = blob.as_dict()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            blob = {}
    if isinstance(blob, (bytes, str)):
        try:
            blob = json.loads(blob)
        except json.JSONDecodeError:
            blob = {}
    if not isinstance(blob, dict):
        return float("nan")
    inner = blob.get("probability") if isinstance(blob.get("probability"), dict) else blob
    if not isinstance(inner, dict):
        return float("nan")
    try:
        return float(inner.get("1"))  # type: ignore[arg-type]
    except (TypeError, ValueError, KeyError):
        pass
    vals = list(inner.values())
    try:
        return float(vals[-1]) if vals else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Synthetic fallback dataset
# ---------------------------------------------------------------------------

def synth_df() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n_m, nk = 50, 3
    rows: list[dict] = []
    for mi in range(n_m):
        mid = f"synthetic-{mi:05d}"
        cat = rng.choice(list(CALIBRATOR_CATEGORY_ORDER))
        ts = pd.Timestamp("2020-01-01") + pd.Timedelta(days=mi)
        w = int(rng.integers(0, nk))
        disc = float(rng.uniform(0.004, 0.08))
        for j in range(nk):
            rows.append(
                dict(
                    market_id=mid,
                    source="synthetic",
                    resolve_ts=ts,
                    category=cat,
                    q=float(rng.random()),
                    p_llm=float(rng.beta(2, 2)),
                    disagreement=disc,
                    base_rate=float(rng.beta(2, 2)),
                    outcome_idx=j,
                    y=int(j == w),
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 4 — AutoML meta-calibrator training (Beats 5 + 7)")
    print("=" * 70)

    snowflake_automl_ok = False
    snow_note = ""
    last_pointer = str(JOBLIB_PATH.resolve())
    synthetic = False
    count_meta = -1

    # Step 1 — create META_TRAIN view
    try:
        with snowflake_cursor() as cur:
            cur.execute(META_TRAIN_SQL)
            cur.execute("SELECT COUNT(*) FROM META_TRAIN")
            count_meta = cur.fetchone()[0]
    except Exception as e:  # noqa: BLE001
        synthetic = True
        snow_note = str(e)
        count_meta = 0

    print(f"META_TRAIN row count: {count_meta}")

    if count_meta == 0 and not synthetic:
        warnings.warn(
            "META_TRAIN empty — run scripts/backfill_features.py first.",
            RuntimeWarning,
            stacklevel=2,
        )
        synthetic = True

    # Step 2 — pull into pandas (or synthesize)
    pull = pd.DataFrame()
    if not synthetic:
        try:
            with snowflake_cursor() as cur:
                cur.execute(
                    """
                    SELECT MARKET_ID, SOURCE, RESOLVE_TS,
                           CATEGORY, Q, P_LLM,
                           DISAGREEMENT, BASE_RATE,
                           OUTCOME_IDX, Y
                      FROM META_TRAIN
                    """
                )
                cols = [d[0].lower() for d in cur.description]
                pull = pd.DataFrame(cur.fetchall(), columns=cols)
        except Exception as e:  # noqa: BLE001
            snow_note += f" | pull fail: {e}"
            synthetic = True

    if synthetic or pull.empty:
        warnings.warn("Using synthetic META_TRAIN data frame.", RuntimeWarning, stacklevel=2)
        full = synth_df()
        synthetic = True
    else:
        full = pull

    msk = market_train_keys(full[["market_id", "source", "resolve_ts"]])
    train_df = full.loc[is_train_series(full, msk)].reset_index(drop=True)
    hold_df = full.loc[~is_train_series(full, msk)].reset_index(drop=True)
    print(f"Temporal split: train={len(train_df)}  holdout={len(hold_df)}")

    # Create split views in Snowflake when not synthetic
    if not synthetic:
        ks = [{"market_id": a, "source": b} for a, b in sorted(msk)] if msk else []
        try:
            with snowflake_cursor() as cur:
                cur.execute(
                    "CREATE OR REPLACE TEMPORARY TABLE PHASE4_META_KEYS "
                    "(MARKET_ID STRING, SOURCE STRING)"
                )
                if ks:
                    cur.executemany(
                        "INSERT INTO PHASE4_META_KEYS (MARKET_ID, SOURCE) "
                        "VALUES (%(market_id)s, %(source)s)",
                        ks,
                    )
                cur.execute(
                    """
                    CREATE OR REPLACE VIEW META_TRAIN_TRAIN AS
                    SELECT mt.* FROM META_TRAIN mt
                    JOIN PHASE4_META_KEYS kk USING (market_id, source)
                    """
                )
                cur.execute(
                    """
                    CREATE OR REPLACE VIEW META_TRAIN_HOLDOUT_FEATS AS
                    SELECT CATEGORY, Q, P_LLM, DISAGREEMENT, BASE_RATE
                      FROM META_TRAIN mt
                     WHERE NOT EXISTS (
                           SELECT 1 FROM PHASE4_META_KEYS kk
                            WHERE kk.market_id = mt.market_id
                              AND kk.source = mt.source)
                    """
                )
                cur.execute(SCORE_ROWS_SQL)
        except Exception as e:  # noqa: BLE001
            snow_note += f" | split DDL: {e}"

    # Step 3 — sklearn GradientBoostingClassifier (always runs)
    print("\nTraining sklearn GradientBoostingClassifier...")
    X_train, y_train = xy_design(train_df)
    slmap = slices_hold(hold_df)
    wmap = winner_index(hold_df)
    X_hold, _ = xy_design(hold_df)

    model = GradientBoostingClassifier(random_state=0)
    if len(X_train):
        model.fit(X_train, y_train)
    else:
        z = np.zeros((2, len(FEATURE_COLUMNS)))
        model.fit(z, np.array([0, 1]))

    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, JOBLIB_PATH)
    print(f"  Saved sklearn bundle -> {JOBLIB_PATH}")

    prob_pos_hold = (
        model.predict_proba(X_hold)[:, 1]
        if X_hold.shape[0]
        else np.asarray([], dtype=float)
    )
    sk_map = from_positives(prob_pos_hold, slmap) if prob_pos_hold.size else {}

    b_llm = pooled_brier(vec_per_market(hold_df, "p_llm"), wmap) if len(hold_df) else float("nan")
    b_k   = pooled_brier(vec_per_market(hold_df, "q"), wmap)     if len(hold_df) else float("nan")
    b_sk  = pooled_brier(sk_map, wmap)                           if sk_map       else float("nan")

    # Step 4 — Snowflake AutoML (try/except)
    print("\nAttempting Snowflake SNOWFLAKE.ML.CLASSIFICATION META_CALIBRATOR...")
    sf_positive = np.asarray([], dtype=float)
    if not synthetic:
        trained = False
        try:
            with snowflake_cursor() as cur:
                cur.execute(
                    """CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION META_CALIBRATOR (
                          INPUT_DATA => (
                             SELECT CATEGORY, Q, P_LLM, DISAGREEMENT, BASE_RATE, Y
                               FROM META_TRAIN_TRAIN
                          ),
                          TARGET_COLNAME => 'Y',
                          CONFIG_OBJECT => OBJECT_CONSTRUCT('on_error','SKIP'))"""
                )
                trained = True
        except Exception as e:  # noqa: BLE001
            trained = False
            snow_note += f" | SNOWFLAKE.ML.CLASSIFICATION: {e}"

        snowflake_automl_ok = trained
        if trained:
            last_pointer = "Snowflake META_CALIBRATOR (SNOWFLAKE.ML.CLASSIFICATION)"
            print("  Snowflake AutoML trained successfully.")
            try:
                with snowflake_cursor() as cur:
                    dfv = pd.read_sql(
                        """
                        SELECT META_CALIBRATOR!PREDICT(OBJECT_CONSTRUCT(
                            'CATEGORY', CATEGORY, 'Q', Q, 'P_LLM', P_LLM,
                            'DISAGREEMENT', DISAGREEMENT, 'BASE_RATE', BASE_RATE
                        )) v FROM META_TRAIN_HOLDOUT_FEATS
                        """,
                        cur.connection,
                    )
                    sf_positive = np.asarray(
                        [parse_variant(r) for r in dfv.iloc[:, 0]], dtype=float
                    )
            except Exception as e:  # noqa: BLE001
                snow_note += f" | SF predict: {e}"
        else:
            print("  Snowflake AutoML unavailable — sklearn is the active calibrator.")

    sf_map = from_positives(sf_positive, slmap)
    ok_shape = sf_positive.size == prob_pos_hold.size and prob_pos_hold.size > 0
    b_sf = pooled_brier(sf_map, wmap) if ok_shape else float("nan")

    # Step 5 — Brier leaderboard
    def _fmt(z: float) -> str:
        return "n/a (empty holdout)" if math.isnan(z) else f"{z:.6f}"

    print("\n--- Hold-out multiclass Brier (recent 20% markets) ---")
    rows = [
        ("raw ensemble softmax(p_llm)",        b_llm),
        ("Kalshi softmax(q)",                   b_k),
        ("Snowflake META_CALIBRATOR",           b_sf),
        ("sklearn GradientBoostingClassifier",  b_sk),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    print(f"  {'Model':<{col_w}}  Brier")
    print(f"  {'-'*col_w}  {'-----'}")
    for name, score in rows:
        print(f"  {name:<{col_w}}  {_fmt(score)}")
    print(f"\n  Snowflake AutoML trained: {'yes' if snowflake_automl_ok else 'no (sklearn shadow)'}")
    print(f"  Registry pointer: {last_pointer}")
    if snow_note:
        print(f"  Notes: {snow_note}")

    # Write metrics JSON
    METRICS_JSON.write_text(
        json.dumps(
            {
                "briers": [
                    {"name": n, "score": float(s) if not math.isnan(s) else None}
                    for n, s in rows
                ],
                "snowflake_automl": snowflake_automl_ok,
                "pointer": last_pointer,
                "snow_note": snow_note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Metrics JSON -> {METRICS_JSON}")

    # Step 6 — Populate META_PREDICTIONS_BACKFILL
    def _meta_payload(df_scores: pd.DataFrame, clf: GradientBoostingClassifier) -> list[dict]:
        merged: list[dict] = []
        grp = sorted(
            df_scores.groupby(["market_id", "source"]),
            key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
        )
        for (_, _), gg in grp:
            g = gg.sort_values("outcome_idx")
            mat, _ = build_calibrator_feature_rows(
                p_ensemble=g["p_llm"].astype(float).tolist(),
                q_kalshi=g["q"].astype(float).tolist(),
                base_rate=g["base_rate"].astype(float).tolist(),
                category=str(g["category"].iloc[0]),
                disagreement=float(g["disagreement"].iloc[0]),
            )
            pos = clf.predict_proba(np.asarray(mat))[:, 1].tolist()
            vec = softmax(pos)
            merged.append(
                dict(
                    market_id=_sify(g["market_id"].iloc[0]),
                    source=_sify(g["source"].iloc[0]),
                    p_meta=json.dumps(vec),
                )
            )
        return merged

    print("\nPopulating META_PREDICTIONS_BACKFILL...")
    if synthetic:
        _rows = _meta_payload(full, model)
        print(f"  (synthetic — local only) {len(_rows)} rows computed")
    else:
        try:
            with snowflake_cursor() as cur:
                cur.execute(SCORE_ROWS_SQL)
                cur.execute(
                    "SELECT MARKET_ID, SOURCE, CATEGORY, Q, P_LLM,"
                    " DISAGREEMENT, BASE_RATE, OUTCOME_IDX FROM META_SCORE_ROWS"
                )
                cols = [d[0].lower() for d in cur.description]
                score_pdf = pd.DataFrame(cur.fetchall(), columns=cols)
                score_pdf.columns = [c.lower() for c in score_pdf.columns]
            _rows = _meta_payload(score_pdf, model)
            with snowflake_cursor() as cur:
                cur.executemany(
                    MERGE_META_SQL,
                    [{"market_id": r["market_id"], "source": r["source"], "p_meta": r["p_meta"]}
                     for r in _rows],
                )
            print(f"  MERGED META_PREDICTIONS_BACKFILL: {len(_rows)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"  META_PREDICTIONS_BACKFILL merge deferred: {e}")

    print(f"\nDone.  sklearn joblib: {JOBLIB_PATH.resolve()}")
    print("Run notebooks/06_train_alpha_policy.py next.")


if __name__ == "__main__":
    main()
