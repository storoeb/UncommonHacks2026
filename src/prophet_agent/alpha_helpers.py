"""Pure-Python helpers for α-shading targets and scoring (no Snowflake).

Used offline (notebook) and in tests; inference reuses ``apply_shading`` here.
"""

from __future__ import annotations

import math
from typing import Sequence

_LOG_EPS = 1e-12


def _coerce_probs(seq: Sequence[float]) -> list[float]:
    return [float(x) for x in seq]


def compute_optimal_alpha(
    p_meta: Sequence[float],
    q_kalshi: Sequence[float],
    realized_outcome: int,
    n_steps: int = 21,
    *,
    eps: float = _LOG_EPS,
) -> float:
    """Line-search α ∈ [0, 1] maximizing Kelly log-payoff vs the market.

    For candidate α: ``p_final[i] = α·p_meta[i] + (1-α)·q_kalshi[i]``.

    Payoff (CRRA γ=0): ``log(max(p_final[o], ε) / max(q[o], ε))`` where ``o``
    is ``realized_outcome``. Tie-break prefers larger α (trust calibrator).
    """
    p_meta_l = _coerce_probs(p_meta)
    q_l = _coerce_probs(q_kalshi)
    if len(p_meta_l) != len(q_l) or len(p_meta_l) == 0:
        msg = "p_meta and q_kalshi must be non-empty and equal length"
        raise ValueError(msg)
    if realized_outcome < 0 or realized_outcome >= len(p_meta_l):
        msg = "realized_outcome must index into probabilities"
        raise ValueError(msg)

    if n_steps < 2:
        alphas = [1.0]
    else:
        alphas = [i / (n_steps - 1) for i in range(n_steps)]

    best_payoff = float("-inf")
    best_alpha = 1.0
    tie_tol = 1e-15

    for alpha in alphas:
        po = _payoff_for_alpha(alpha, p_meta_l, q_l, realized_outcome, eps)
        if po > best_payoff + tie_tol:
            best_payoff = po
            best_alpha = alpha
        elif math.isclose(po, best_payoff, rel_tol=1e-12, abs_tol=tie_tol):
            best_alpha = max(best_alpha, alpha)

    return best_alpha


def _payoff_for_alpha(
    alpha: float,
    p_meta: list[float],
    q: list[float],
    realized_outcome: int,
    eps: float,
) -> float:
    p_final = [
        alpha * pm + (1.0 - alpha) * qk for pm, qk in zip(p_meta, q, strict=True)
    ]
    num = max(p_final[realized_outcome], eps)
    den = max(q[realized_outcome], eps)
    return math.log(num / den)


def brier_score(p: Sequence[float], realized_outcome: int) -> float:
    """Sum_i (p[i] - 1{realized==i})²."""
    seq = _coerce_probs(p)
    if realized_outcome < 0 or realized_outcome >= len(seq):
        msg = "realized_outcome must index into p"
        raise ValueError(msg)
    total = 0.0
    for i, pi in enumerate(seq):
        target = 1.0 if i == realized_outcome else 0.0
        d = pi - target
        total += d * d
    return total


def aver_score(
    p: Sequence[float],
    q: Sequence[float],
    realized_outcome: int,
    *,
    eps: float = _LOG_EPS,
) -> float:
    """Log payoff vs market prices on the realized outcome (risk-neutral Kelly)."""
    p_l = _coerce_probs(p)
    q_l = _coerce_probs(q)
    if len(p_l) != len(q_l) or len(p_l) == 0:
        msg = "p and q must be non-empty and equal length"
        raise ValueError(msg)
    if realized_outcome < 0 or realized_outcome >= len(p_l):
        msg = "realized_outcome must index into probabilities"
        raise ValueError(msg)
    num = max(p_l[realized_outcome], eps)
    den = max(q_l[realized_outcome], eps)
    return math.log(num / den)


def apply_shading(
    p_meta: Sequence[float],
    q_kalshi: Sequence[float] | None,
    alpha: float,
) -> list[float]:
    """Blend calibrator output with the market: p_final = α·p_meta + (1-α)·q."""
    p_list = list(p_meta)
    if q_kalshi is None or len(q_kalshi) != len(p_list):
        return list(p_list)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return [
        alpha_clamped * p + (1.0 - alpha_clamped) * q
        for p, q in zip(p_list, q_kalshi, strict=True)
    ]
