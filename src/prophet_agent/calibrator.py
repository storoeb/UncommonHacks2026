"""Phase 4 — meta-calibrator inference (Snowflake AutoML + sklearn fallback)."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Must match notebooks/04_train_automl_calibrator.ipynb one-hot order.
CALIBRATOR_CATEGORY_ORDER: tuple[str, ...] = (
    "sports",
    "politics",
    "crypto",
    "science",
    "tech",
    "econ",
    "other",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKLEARN_ARTIFACT_PATH = _REPO_ROOT / "notebooks" / "_calibrator.joblib"
_SF_MODEL_NAME = "META_CALIBRATOR"

_backend: str | None = None
_snowflake_model_verified = False
_sklearn_bundle: dict[str, Any] | None = None
_warned_passthrough = False


def _normalize_category(category: str | None) -> str:
    cat = (category or "").strip().lower()
    if cat in CALIBRATOR_CATEGORY_ORDER:
        return cat
    return "other"


def calibrator_feature_column_names(
    categories: tuple[str, ...] = CALIBRATOR_CATEGORY_ORDER,
) -> list[str]:
    """Ordered sklearn / inference feature names (matches training notebook)."""
    return [f"cat__{c}" for c in categories] + ["q", "p_llm", "disagreement", "base_rate"]


def build_calibrator_feature_rows(
    *,
    p_ensemble: list[float],
    q_kalshi: list[float] | None,
    base_rate: list[float],
    category: str | None,
    disagreement: float,
    categories_order: tuple[str, ...] = CALIBRATOR_CATEGORY_ORDER,
) -> tuple[list[list[float]], list[str]]:
    """Pure-Python builder: one feature row per outcome (testable independently)."""
    n = len(p_ensemble)
    if n == 0:
        return [], calibrator_feature_column_names(categories_order)

    cols = calibrator_feature_column_names(categories_order)
    normed_cat = _normalize_category(category)

    fill_q = None
    if q_kalshi is None or len(q_kalshi) != n:
        fill_q = 1.0 / n
    br = base_rate[:n] + [1.0 / n] * max(0, n - len(base_rate))

    rows: list[list[float]] = []
    for i in range(n):
        oh = [1.0 if normed_cat == c else 0.0 for c in categories_order]
        q_i = float(fill_q if fill_q is not None else q_kalshi[i])
        pi = float(p_ensemble[i])
        bi = float(br[i])
        rows.append(
            [*oh, q_i, pi, float(disagreement), bi],
        )
    return rows, cols


def _softmax_normalize(raw: list[float], eps: float = 1e-12) -> list[float]:
    import math

    m = max(raw)
    exps = [math.exp(x - m) for x in raw]
    s = sum(exps)
    if s <= eps:
        return [1.0 / len(raw)] * len(raw) if raw else []
    return [e / s for e in exps]


def model_source() -> str:
    """\"snowflake_automl\" | \"sklearn_fallback\" | \"passthrough\"."""
    _ensure_backend_selected()
    assert _backend is not None  # noqa: S101 — internal invariant
    return _backend


def is_loaded() -> bool:
    """True once a calibrated backend is active (not passthrough)."""
    _ensure_backend_selected()
    return _backend not in {None, "passthrough"}


def _warn_passthrough_once() -> None:
    global _warned_passthrough
    if not _warned_passthrough:
        logger.warning(
            "prophet_agent.calibrator: no Snowflake META_CALIBRATOR and no notebooks/"
            "_calibrator.joblib; returning raw ensemble (passthrough)."
        )
        warnings.warn(
            "Calibrator using passthrough (no sklearn/META_CALIBRATOR).",
            stacklevel=2,
        )
        _warned_passthrough = True


def _ensure_backend_selected() -> None:
    global _backend
    if _backend is not None:
        return
    if _select_snowflake_backend():
        _backend = "snowflake_automl"
        return
    if _try_load_sklearn_artifact():
        _backend = "sklearn_fallback"
        return
    _backend = "passthrough"


def _select_snowflake_backend() -> bool:
    """Prefer Snowflake if credentials work and META_CALIBRATOR appears in-schema."""
    try:
        from prophet_agent.snowflake_client import snowflake_cursor  # noqa: PLC0415
    except Exception as e:
        logger.info("snowflake unavailable for calibrator probe: %s", e)
        return False

    db = os.getenv("SNOWFLAKE_DATABASE", "").strip()
    schema = os.getenv("SNOWFLAKE_SCHEMA", "").strip()
    if not db or not schema:
        return False

    fq_schema = f"{db}.{schema}"
    if fq_schema.count(".") != 1:
        return False
    db_part, schema_part = fq_schema.split(".", 1)
    allowed = "_"
    ok = lambda s: len(s) > 0 and all(c.isalnum() or c in allowed for c in s)
    if not ok(db_part) or not ok(schema_part):
        return False

    try:
        with snowflake_cursor() as cur:
            cur.execute(
                f"SHOW SNOWFLAKE.ML.CLASSIFICATION LIKE '{_SF_MODEL_NAME}' "
                f"IN SCHEMA {fq_schema}"
            )
            rows = cur.fetchall() or []
            columns = tuple(c[0].upper() for c in (cur.description or ()))
            name_idx = columns.index("NAME") if "NAME" in columns else 1
    except Exception as e:
        logger.info("META_CALIBRATOR Snowflake probe failed: %s", e)
        return False

    return any(
        len(r) > name_idx and str(r[name_idx]).upper() == _SF_MODEL_NAME for r in rows
    )


def _try_load_sklearn_artifact() -> bool:
    global _sklearn_bundle
    if _sklearn_bundle is not None:
        return True
    path = Path(os.getenv("CALIBRATOR_JOBLIB_PATH", _SKLEARN_ARTIFACT_PATH))
    if not path.is_file():
        return False
    try:
        import joblib  # noqa: PLC0415

        _sklearn_bundle = joblib.load(path)
        return isinstance(_sklearn_bundle, dict) and "model" in _sklearn_bundle
    except Exception as e:
        logger.warning("Failed to load sklearn calibrator artifact %s: %s", path, e)
        return False


def _extract_prob_class_one(pred_variant: Any) -> float | None:
    """Pull P(Y=1) from Snowflake VARIANT prediction output."""
    if pred_variant is None:
        return None
    blob = pred_variant
    if isinstance(blob, dict) and "probability" in blob:
        prob = blob.get("probability")
    elif hasattr(blob, "as_dict"):
        d = blob.as_dict()
        prob = d.get("probability") if isinstance(d, dict) else None
    else:
        prob = getattr(blob, "probability", None)
    if not isinstance(prob, dict):
        return None

    candidates = []
    if "1" in prob:
        candidates.append(float(prob["1"]))
    elif 1 in prob:
        candidates.append(float(prob[1]))
    if "TRUE" in prob:
        candidates.append(float(prob["TRUE"]))
    if "yes" in prob:
        candidates.append(float(prob["yes"]))
    if candidates:
        return candidates[0]
    # Fallback: maximize key numerically interpreted
    float_keys = [(float(k), v) for k, v in prob.items() if represents_number(k)]
    if not float_keys:
        return None
    fk = sorted(float_keys, key=lambda kv: kv[0])
    closest_one = fk[-1][1] if fk[-1][0] >= fk[0][0] else fk[0][1]
    return float(closest_one)


def represents_number(k: Any) -> bool:
    try:
        float(k)
        return True
    except (TypeError, ValueError):
        return False


def _snowflake_predict_positive_proba(
    category: str | None,
    *,
    q: float,
    p_llm: float,
    disagreement: float,
    base_rate: float,
) -> float | None:
    """Return P(Y=1) for one META_TRAIN row using META_CALIBRATOR!PREDICT."""
    global _snowflake_model_verified
    from prophet_agent.snowflake_client import snowflake_cursor  # noqa: PLC0415

    cat_str = _normalize_category(category)

    sql = f"""
        SELECT META_CALIBRATOR!PREDICT(
            OBJECT_CONSTRUCT(
                'CATEGORY', %s,
                'Q', %s::FLOAT,
                'P_LLM', %s::FLOAT,
                'DISAGREEMENT', %s::FLOAT,
                'BASE_RATE', %s::FLOAT
            )
        ) AS PRED
        """
    params = (cat_str, q, p_llm, disagreement, base_rate)
    try:
        with snowflake_cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception as e:
        if _snowflake_model_verified:
            logger.warning("Snowflake META_CALIBRATOR inference failed (%s)", e)
        else:
            logger.info(
                "META_CALIBRATOR present but inference failed (%s); try sklearn artifact.",
                e,
            )
        return None

    if not row:
        return None
    p = _extract_prob_class_one(row[0])
    if p is not None:
        _snowflake_model_verified = True
    return p


def _sklearn_predict_proba_positive(
    feature_rows: list[list[float]],
) -> list[float] | None:
    if not _try_load_sklearn_artifact():
        return None
    assert _sklearn_bundle is not None  # noqa: S101
    model = _sklearn_bundle["model"]

    pred_pos: list[float] = []
    for row in feature_rows:
        probs = getattr(model, "predict_proba", lambda X: None)([row])
        if probs is None:
            return None
        if probs.shape[-1] < 2:
            return None
        pred_pos.append(float(probs[0, -1]))

    return pred_pos


def calibrator_predict(
    p_ensemble: list[float],
    q_kalshi: list[float] | None,
    base_rate: list[float],
    category: str | None,
    disagreement: float,
) -> list[float]:
    rows, _cols = build_calibrator_feature_rows(
        p_ensemble=p_ensemble,
        q_kalshi=q_kalshi,
        base_rate=base_rate,
        category=category,
        disagreement=disagreement,
    )
    if not rows:
        return []

    _ensure_backend_selected()

    backend = model_source()

    raw_scores: list[float] | None = None
    if backend == "snowflake_automl":
        raw_scores = _snowflake_predict_scores(rows, category)

    # Sklearn artifact can still score if Snowflake is unavailable mid-request.
    if raw_scores is None:
        raw_scores = _sklearn_predict_proba_positive(rows)

    if raw_scores is None:
        _warn_passthrough_once()
        return list(p_ensemble)

    return _softmax_normalize(raw_scores)


def _snowflake_predict_scores(
    feature_rows: list[list[float]],
    category: str | None,
) -> list[float] | None:
    nh = len(CALIBRATOR_CATEGORY_ORDER)
    out: list[float] = []
    for row in feature_rows:
        if len(row) < nh + 4:
            return None
        q_i, p_llm, disc, base = row[nh], row[nh + 1], row[nh + 2], row[nh + 3]
        p = _snowflake_predict_positive_proba(
            category,
            q=float(q_i),
            p_llm=float(p_llm),
            disagreement=float(disc),
            base_rate=float(base),
        )
        if p is None:
            return None
        out.append(p)
    return out
