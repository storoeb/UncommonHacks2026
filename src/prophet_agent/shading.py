"""Inference-side α-policy: load Snowflake ML regression, sklearn fallback, or global α."""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from prophet_agent.alpha_helpers import apply_shading

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_JOBLIB_PATH = _PROJECT_ROOT / "notebooks" / "_alpha_policy.joblib"
_GLOBAL_JSON_PATH = _PROJECT_ROOT / "notebooks" / "_alpha_global.json"

_BACKEND: dict[str, Any] | None = None
_MODEL_SOURCE: str = "uninitialized"


def _fq_model_ref(database: str, schema: str, object_name: str) -> str:
    """Quote identifiers safely for SQL embedding."""

    def q(ident: str) -> str:
        return '"' + ident.replace('"', "") + '"'

    return f"{q(database)}.{q(schema)}.{q(object_name)}"


def _set_backend(payload: dict[str, Any]) -> None:
    global _BACKEND
    _BACKEND = payload


def _try_load_snowflake() -> bool:
    """Probe Snowflake REGRESSION instance ALPHA_POLICY with one prediction."""
    try:
        from prophet_agent.snowflake_client import snowflake_cursor  # noqa: PLC0415
    except Exception as exc:
        logger.debug("snowflake_client unavailable for α-policy: %s", exc)
        return False

    db = os.getenv("SNOWFLAKE_DATABASE", "").strip()
    schema = os.getenv("SNOWFLAKE_SCHEMA", "").strip()
    if not db or not schema:
        logger.debug(
            "SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA not set; skip Snowflake α-policy."
        )
        return False

    model_ref = _fq_model_ref(db, schema, "ALPHA_POLICY")

    probe_sql = f"""
        SELECT {model_ref}!PREDICT(
            INPUT_DATA => OBJECT_CONSTRUCT(
                'DISAGREEMENT', %(dis)s::FLOAT,
                'NEIGHBOR_COUNT', %(nc)s::NUMBER,
                'NEIGHBOR_SIM', %(ns)s::FLOAT,
                'Q_SPREAD', %(qs)s::FLOAT,
                'CATEGORY', %(cat)s::VARCHAR
            )
        ) AS pred
    """
    params = {
        "dis": 0.01,
        "nc": 10,
        "ns": 0.7,
        "qs": 0.2,
        "cat": "crypto",
    }
    try:
        with snowflake_cursor() as cur:
            cur.execute(probe_sql, params)
            row = cur.fetchone()
            if not row:
                return False
            raw = row[0]
    except Exception as exc:
        logger.debug("Snowflake ALPHA_POLICY probe failed: %s", exc)
        return False

    if _variant_to_alpha(raw) is None:
        logger.debug("Snowflake ALPHA_POLICY probe returned unparseable payload: %r", raw)
        return False

    _set_backend({"kind": "snowflake", "model_ref": model_ref})
    return True


def _variant_to_alpha(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            pass
    if isinstance(raw, dict):
        for key in (
            "prediction",
            "ALPHA_STAR",
            "OUTPUT_VALUE",
            "output",
            "value",
            "PREDICTION",
        ):
            if key in raw:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    continue
        if len(raw) == 1:
            try:
                return float(next(iter(raw.values())))
            except (TypeError, ValueError):
                return None
    return None


def _snowflake_predict(
    disagreement: float,
    neighbor_count: int,
    neighbor_sim: float,
    q_spread: float,
    category: str | None,
) -> float:
    assert _BACKEND is not None
    model_ref = _BACKEND["model_ref"]
    sql = f"""
        SELECT {model_ref}!PREDICT(
            INPUT_DATA => OBJECT_CONSTRUCT(
                'DISAGREEMENT', %(dis)s::FLOAT,
                'NEIGHBOR_COUNT', %(nc)s::NUMBER,
                'NEIGHBOR_SIM', %(ns)s::FLOAT,
                'Q_SPREAD', %(qs)s::FLOAT,
                'CATEGORY', %(cat)s::VARCHAR
            )
        ) AS pred
    """
    cat_val = category if category is not None else ""
    params = {
        "dis": float(disagreement),
        "nc": int(neighbor_count),
        "ns": float(neighbor_sim),
        "qs": float(q_spread),
        "cat": cat_val,
    }
    from prophet_agent.snowflake_client import snowflake_cursor  # noqa: PLC0415

    with snowflake_cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        if not row:
            msg = "ALPHA_POLICY PREDICT returned no row"
            raise RuntimeError(msg)
        alpha = _variant_to_alpha(row[0])
        if alpha is None:
            msg = f"Unparseable Snowflake PREDICT payload: {row[0]!r}"
            raise RuntimeError(msg)
        return alpha


def _try_load_sklearn() -> bool:
    try:
        import joblib  # noqa: PLC0415
    except ImportError:
        logger.debug("joblib not installed; skip sklearn α-policy.")
        return False

    if not _JOBLIB_PATH.is_file():
        return False
    try:
        bundle = joblib.load(_JOBLIB_PATH)
    except Exception as exc:
        logger.warning("Failed to load sklearn α-policy bundle: %s", exc)
        return False

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns")
    if model is None or feature_columns is None:
        logger.warning("Invalid sklearn α-policy bundle (missing keys).")
        return False

    _set_backend(
        {
            "kind": "sklearn",
            "model": model,
            "feature_columns": list(feature_columns),
        }
    )
    return True


def _sklearn_predict(
    disagreement: float,
    neighbor_count: int,
    neighbor_sim: float,
    q_spread: float,
    category: str | None,
) -> float:
    import pandas as pd  # noqa: PLC0415

    assert _BACKEND is not None
    model = _BACKEND["model"]
    feature_columns = _BACKEND["feature_columns"]

    cat = category if category is not None else "nan"
    row = pd.DataFrame(
        [
            {
                "disagreement": float(disagreement),
                "neighbor_count": int(neighbor_count),
                "neighbor_sim": float(neighbor_sim),
                "q_spread": float(q_spread),
                "category": cat,
            }
        ]
    )
    xd = pd.get_dummies(row, columns=["category"], dummy_na=True)
    # Deduplicate columns before reindexing (get_dummies can create dupes on
    # a single-row frame with dummy_na=True when the value is NaN).
    xd = xd.loc[:, ~xd.columns.duplicated()]
    xd = xd.reindex(columns=feature_columns, fill_value=0)
    pred = model.predict(xd.values)[0]
    return float(pred)


def _try_load_global_json() -> bool:
    if not _GLOBAL_JSON_PATH.is_file():
        return False
    try:
        raw = json.loads(_GLOBAL_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", _GLOBAL_JSON_PATH, exc)
        return False

    if isinstance(raw, dict) and "alpha" in raw:
        alpha = raw["alpha"]
    else:
        alpha = raw
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError):
        return False

    _set_backend({"kind": "global", "alpha": alpha_f})
    return True


def _resolve_backend_once() -> None:
    global _MODEL_SOURCE
    if _BACKEND is not None:
        return

    if _try_load_snowflake():
        _MODEL_SOURCE = "snowflake_automl"
        return
    if _try_load_sklearn():
        _MODEL_SOURCE = "sklearn_fallback"
        return
    if _try_load_global_json():
        _MODEL_SOURCE = "global_constant"
        return

    warnings.warn(
        "α-policy: using hardcoded α=0.5 (no Snowflake model, sklearn bundle, "
        f"or {_GLOBAL_JSON_PATH.name}). Train via notebooks/06_train_alpha_policy.ipynb.",
        stacklevel=2,
    )
    _set_backend({"kind": "hardcoded", "alpha": 0.5})
    _MODEL_SOURCE = "global_constant"


def alpha_policy_predict(
    disagreement: float,
    neighbor_count: int,
    neighbor_sim: float,
    q_spread: float,
    category: str | None,
) -> float:
    """Predict blending weight α ∈ [0, 1] from observable market features."""
    _resolve_backend_once()
    assert _BACKEND is not None

    if _MODEL_SOURCE == "snowflake_automl":
        alpha = _snowflake_predict(
            disagreement,
            neighbor_count,
            neighbor_sim,
            q_spread,
            category,
        )
    elif _MODEL_SOURCE == "sklearn_fallback":
        alpha = _sklearn_predict(
            disagreement,
            neighbor_count,
            neighbor_sim,
            q_spread,
            category,
        )
    else:
        alpha = float(_BACKEND["alpha"])

    return max(0.0, min(1.0, alpha))


def is_loaded() -> bool:
    """True once a backend has been chosen (including hardcoded fallback)."""
    _resolve_backend_once()
    return _BACKEND is not None


def model_source() -> str:
    """``snowflake_automl`` | ``sklearn_fallback`` | ``global_constant``."""
    _resolve_backend_once()
    return _MODEL_SOURCE
