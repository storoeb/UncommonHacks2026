"""Forecasting pipeline orchestrator.

The single entry point `run_pipeline(content)` accepts the raw user-message
text from a chat-completions request and runs:

    parse → ensemble → retrieval → calibrator → α-shading → log

Calibrator: prophet_agent.calibrator (Snowflake AutoML or sklearn fallback).
α-policy:   prophet_agent.shading   (Snowflake AutoML or global-α fallback).
Both modules are loaded lazily at first call so the server starts even when
Snowflake is unreachable. If either module fails to import, the original
pass-through stubs below are used instead.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from prophet_agent.parser import ForecastPromptParsed, parse_forecast_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval fallback — base_rate.py is owned by another agent and may not be
# importable yet. We try to bind the real implementation, but the pipeline
# must still work without it so the server can start in isolation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubBaseRateFeatures:
    """Mirror of `prophet_agent.retrieval.base_rate.BaseRateFeatures`."""

    neighbor_count: int = 0
    mean_similarity: float = 0.0
    base_rate: list[float] = field(default_factory=list)
    kalshi_residual: list[float] = field(default_factory=list)
    neighbors: list[dict[str, Any]] = field(default_factory=list)


async def _retrieve_base_rate_stub(
    question_text: str,
    outcomes: list[str],
    category: str | None = None,
    k: int = 15,
    exclude_market_id: str | None = None,
) -> _StubBaseRateFeatures:
    """Zero-neighbor fallback used when retrieval module isn't importable."""
    n = len(outcomes)
    uniform = [1.0 / n] * n if n > 0 else []
    return _StubBaseRateFeatures(
        neighbor_count=0,
        mean_similarity=0.0,
        base_rate=uniform,
        kalshi_residual=[0.0] * n,
        neighbors=[],
    )


def _resolve_base_rate_fn():
    """Bind the real retrieval function at runtime, else fall back to stub."""
    try:
        from prophet_agent.retrieval.base_rate import (  # noqa: PLC0415
            get_base_rate,
        )

        return get_base_rate
    except Exception as e:  # noqa: BLE001 — any import failure → stub
        logger.warning(
            "retrieval.base_rate not importable (%s); using stub.", e
        )
        return _retrieve_base_rate_stub


# ---------------------------------------------------------------------------
# Calibrator — Phase 4 (prophet_agent.calibrator) with pass-through fallback.
# ---------------------------------------------------------------------------

def _passthrough_calibrator(
    p_ensemble: list[float],
    q_kalshi: list[float] | None,  # noqa: ARG001
    base_rate: list[float],  # noqa: ARG001
    category: str | None,  # noqa: ARG001
    disagreement: float,  # noqa: ARG001
) -> list[float]:
    """Pass-through used when calibrator.py isn't available."""
    return list(p_ensemble)


def _load_calibrator_predict():
    """Lazily import prophet_agent.calibrator; fall back to pass-through."""
    try:
        from prophet_agent.calibrator import (  # noqa: PLC0415
            calibrator_predict as real_fn,
        )
        return real_fn
    except Exception as exc:  # noqa: BLE001
        logger.warning("calibrator not importable (%s); using pass-through.", exc)
        return _passthrough_calibrator


_calibrator_predict_fn = None


def calibrator_predict(
    p_ensemble: list[float],
    q_kalshi: list[float] | None,
    base_rate: list[float],
    category: str | None,
    disagreement: float,
) -> list[float]:
    """Delegate to prophet_agent.calibrator, lazy-loaded on first call."""
    global _calibrator_predict_fn
    if _calibrator_predict_fn is None:
        _calibrator_predict_fn = _load_calibrator_predict()
    return _calibrator_predict_fn(p_ensemble, q_kalshi, base_rate, category, disagreement)


# ---------------------------------------------------------------------------
# α-policy + shading — Phase 6 (prophet_agent.shading) with stub fallback.
# ---------------------------------------------------------------------------

def _stub_alpha_policy(
    disagreement: float,  # noqa: ARG001
    neighbor_count: int,  # noqa: ARG001
    neighbor_sim: float,  # noqa: ARG001
    q_spread: float,  # noqa: ARG001
    category: str | None,  # noqa: ARG001
) -> float:
    """Stub: α=0.5 blend used when shading.py isn't available."""
    return 0.5


def _stub_apply_shading(
    p_meta: list[float],
    q_kalshi: list[float] | None,
    alpha: float,
) -> list[float]:
    if q_kalshi is None or len(q_kalshi) != len(p_meta):
        return list(p_meta)
    alpha = max(0.0, min(1.0, alpha))
    return [alpha * p + (1.0 - alpha) * q for p, q in zip(p_meta, q_kalshi, strict=True)]


def _load_shading_fns():
    """Lazily import prophet_agent.shading; fall back to stubs."""
    try:
        from prophet_agent.shading import (  # noqa: PLC0415
            alpha_policy_predict as real_alpha,
            apply_shading as real_shade,
        )
        return real_alpha, real_shade
    except Exception as exc:  # noqa: BLE001
        logger.warning("shading not importable (%s); using stubs.", exc)
        return _stub_alpha_policy, _stub_apply_shading


_shading_fns = None


def alpha_policy_predict(
    disagreement: float,
    neighbor_count: int,
    neighbor_sim: float,
    q_spread: float,
    category: str | None,
) -> float:
    """Delegate to prophet_agent.shading, lazy-loaded on first call."""
    global _shading_fns
    if _shading_fns is None:
        _shading_fns = _load_shading_fns()
    return _shading_fns[0](disagreement, neighbor_count, neighbor_sim, q_spread, category)


def apply_shading(
    p_meta: list[float],
    q_kalshi: list[float] | None,
    alpha: float,
) -> list[float]:
    """Delegate to prophet_agent.shading (single source of truth for the blend math)."""
    global _shading_fns
    if _shading_fns is None:
        _shading_fns = _load_shading_fns()
    return _shading_fns[1](p_meta, q_kalshi, alpha)


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """All artifacts a request produces — return so the server can emit the
    response body AND populate the debug dict."""

    request_id: str
    parsed: ForecastPromptParsed
    p_ensemble: list[float]
    p_ensemble_var: float
    base_rate: list[float]
    neighbor_count: int
    mean_similarity: float
    p_meta: list[float]
    alpha: float
    p_final: list[float]
    elapsed_s: float
    ensemble_raw: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main orchestrator.
# ---------------------------------------------------------------------------

async def run_pipeline(
    content: str,
    *,
    wafer_client: Any = None,
    base_rate_fn: Any = None,
    snowflake_logger: Any = None,
) -> PipelineResult:
    """Run parse → ensemble → retrieval → calibrator → shading → log.

    All collaborators are injectable so tests can substitute fakes. In
    production, wafer_client and base_rate_fn are bound lazily so module import
    doesn't require live credentials.
    """
    start = time.monotonic()

    parsed = parse_forecast_prompt(content)
    n = len(parsed.outcomes)

    # ---- Ensemble ----------------------------------------------------------
    if wafer_client is None:
        from prophet_agent.llm.wafer import WaferPool  # noqa: PLC0415
        wafer_client = WaferPool()
    ensemble = await wafer_client.ensemble(
        question=parsed.question,
        outcomes=list(parsed.outcomes),
        market_prices=parsed.market_prices,
        resolve_by=parsed.resolve_by,
    )
    p_ensemble = list(ensemble.mean)
    p_ensemble_var = float(ensemble.variance)
    ensemble_raw = [
        {
            "model": f.model,
            "probabilities": list(f.probabilities),
            "latency_s": f.latency_s,
        }
        for f in ensemble.forecasts
    ]

    # ---- Retrieval ---------------------------------------------------------
    if base_rate_fn is None:
        base_rate_fn = _resolve_base_rate_fn()
    try:
        # base_rate_fn may be sync (real get_base_rate) or async (test fakes).
        # Inspect before calling so we don't accidentally block the event loop.
        _kwargs = dict(
            question_text=parsed.question,
            outcomes=list(parsed.outcomes),
            category=None,
            k=15,
            exclude_market_id=parsed.market_id,
        )
        if inspect.iscoroutinefunction(base_rate_fn):
            br = await base_rate_fn(**_kwargs)
        else:
            loop = asyncio.get_event_loop()
            br = await loop.run_in_executor(None, lambda: base_rate_fn(**_kwargs))
        base_rate = list(getattr(br, "base_rate", [])) or [1.0 / n] * n
        neighbor_count = int(getattr(br, "neighbor_count", 0))
        mean_similarity = float(getattr(br, "mean_similarity", 0.0))
    except Exception as e:  # noqa: BLE001 — never fail the request on retrieval
        logger.warning("base_rate lookup failed (%s); using uniform.", e)
        base_rate = [1.0 / n] * n
        neighbor_count = 0
        mean_similarity = 0.0
    if len(base_rate) != n:
        base_rate = [1.0 / n] * n

    # ---- Calibrator (Phase 4 — prophet_agent.calibrator) -------------------
    p_meta = calibrator_predict(
        p_ensemble=p_ensemble,
        q_kalshi=parsed.market_prices,
        base_rate=base_rate,
        category=None,
        disagreement=p_ensemble_var,
    )

    # ---- α-policy + shading (Phase 6 — prophet_agent.shading) -------------
    if parsed.market_prices and len(parsed.market_prices) == n:
        q_spread = max(parsed.market_prices) - min(parsed.market_prices)
    else:
        q_spread = 0.0
    alpha = alpha_policy_predict(
        disagreement=p_ensemble_var,
        neighbor_count=neighbor_count,
        neighbor_sim=mean_similarity,
        q_spread=q_spread,
        category=None,
    )
    p_final = apply_shading(p_meta, parsed.market_prices, alpha)

    elapsed = time.monotonic() - start
    result = PipelineResult(
        request_id=str(uuid.uuid4()),
        parsed=parsed,
        p_ensemble=p_ensemble,
        p_ensemble_var=p_ensemble_var,
        base_rate=base_rate,
        neighbor_count=neighbor_count,
        mean_similarity=mean_similarity,
        p_meta=list(p_meta),
        alpha=alpha,
        p_final=p_final,
        elapsed_s=elapsed,
        ensemble_raw=ensemble_raw,
    )

    # ---- Log to Snowflake (best effort) ------------------------------------
    logger_fn = snowflake_logger if snowflake_logger is not None else log_prediction_to_snowflake
    try:
        logger_fn(result)
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] AGENT_PREDICTIONS log failed: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Snowflake logger.
# ---------------------------------------------------------------------------

def log_prediction_to_snowflake(result: PipelineResult) -> None:
    """Insert one row into AGENT_PREDICTIONS. Best-effort — swallows failures
    (caller wraps in try/except too, belt-and-suspenders)."""
    try:
        from prophet_agent.snowflake_client import snowflake_cursor  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] snowflake_client unavailable: {e}", file=sys.stderr)
        return

    parsed = result.parsed
    sql = """
        INSERT INTO AGENT_PREDICTIONS (
            request_id, market_id, category, question_text, outcomes,
            q_kalshi, p_ensemble, p_ensemble_var, base_rate,
            p_meta, p_final, alpha
        )
        SELECT
            %(request_id)s,
            %(market_id)s,
            %(category)s,
            %(question_text)s,
            PARSE_JSON(%(outcomes)s),
            PARSE_JSON(%(q_kalshi)s),
            PARSE_JSON(%(p_ensemble)s),
            %(p_ensemble_var)s,
            PARSE_JSON(%(base_rate)s),
            PARSE_JSON(%(p_meta)s),
            PARSE_JSON(%(p_final)s),
            %(alpha)s
    """
    params = {
        "request_id": result.request_id,
        "market_id": parsed.market_id,
        "category": None,
        "question_text": parsed.question,
        "outcomes": json.dumps(list(parsed.outcomes)),
        "q_kalshi": json.dumps(list(parsed.market_prices)) if parsed.market_prices else json.dumps(None),
        "p_ensemble": json.dumps(result.p_ensemble),
        "p_ensemble_var": result.p_ensemble_var,
        "base_rate": json.dumps(result.base_rate),
        "p_meta": json.dumps(result.p_meta),
        "p_final": json.dumps(result.p_final),
        "alpha": result.alpha,
    }
    try:
        with snowflake_cursor() as cur:
            cur.execute(sql, params)
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] AGENT_PREDICTIONS insert failed: {e}", file=sys.stderr)
