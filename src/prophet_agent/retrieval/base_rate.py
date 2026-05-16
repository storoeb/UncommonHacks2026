"""Base-rate retrieval over resolved historical markets.

Given a forecasting question, embed it via Snowflake Cortex and pull the
top-K most similar resolved markets from `HISTORICAL_MARKETS`. Aggregate
their realized outcomes into a base-rate prior, and compute a per-outcome
"how miscalibrated was the market on questions like this" residual.

This is Phase 3 of the ProphetHacks 2026 plan — the core Snowflake play.
The embedding model must match the one used to populate
`HISTORICAL_MARKETS.question_embedding`, currently
`snowflake-arctic-embed-m-v1.5` (768-dim).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from prophet_agent.snowflake_client import snowflake_cursor


# Must match the model used to populate HISTORICAL_MARKETS.question_embedding.
EMBED_MODEL = "snowflake-arctic-embed-m-v1.5"


@dataclass
class BaseRateFeatures:
    """Retrieval features for a forecasting question.

    Attributes:
        neighbor_count: How many resolved markets we matched against.
        mean_similarity: Mean cosine similarity of the matched neighbors.
        base_rate: Per-outcome aggregated rate across neighbors (fraction
            of neighbors whose realized outcome was index i). Sums to <=1
            (== 1 when every neighbor's realized_outcome is within the
            query's outcome range).
        kalshi_residual: Per-outcome mean(realized - q_kalshi_at_open[i])
            across neighbors with aligned outcome shapes. Captures how
            miscalibrated the market tends to be on similar questions.
        neighbors: List of {market_id, similarity, question_text,
            realized_outcome} dicts, sorted by similarity desc.
    """

    neighbor_count: int
    mean_similarity: float
    base_rate: list[float]
    kalshi_residual: list[float]
    neighbors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure-Python aggregation helpers (no Snowflake dependency — unit-testable).
# ---------------------------------------------------------------------------


def _empty_features(n_outcomes: int) -> BaseRateFeatures:
    """Default features when we have zero neighbors: uniform prior, zero residual."""
    n = max(1, n_outcomes)
    uniform = [1.0 / n] * n
    zeros = [0.0] * n
    return BaseRateFeatures(
        neighbor_count=0,
        mean_similarity=0.0,
        base_rate=uniform,
        kalshi_residual=zeros,
        neighbors=[],
    )


def _coerce_array(raw: Any) -> list[Any] | None:
    """Snowflake ARRAY columns can come back as a JSON string or a Python list."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, list):
            return parsed
        return None
    return None


def aggregate_neighbors(
    rows: list[dict],
    n_outcomes: int,
) -> BaseRateFeatures:
    """Aggregate raw neighbor rows into BaseRateFeatures.

    Each row is expected to have keys:
        market_id, similarity, question_text, realized_outcome,
        outcomes (list or None), q_kalshi_at_open (list or None).

    base_rate[i] = (# neighbors with realized_outcome == i) / neighbor_count
    kalshi_residual[i] = mean over neighbors of (o_ik - q_kalshi_at_open[i]),
        where o_ik = 1 if the neighbor's realized outcome index is i else 0.
        Neighbors with a different-length outcome list (relative to n_outcomes)
        or missing q_kalshi_at_open are skipped for residual purposes.
    """
    n = max(1, n_outcomes)
    if not rows:
        return _empty_features(n_outcomes)

    base_counts = [0] * n
    residual_sums = [0.0] * n
    residual_counts = [0] * n
    sim_sum = 0.0
    neighbors_out: list[dict] = []

    for row in rows:
        sim = float(row.get("similarity") or 0.0)
        sim_sum += sim
        realized = row.get("realized_outcome")
        if realized is not None:
            try:
                idx = int(realized)
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < n:
                base_counts[idx] += 1

        # Kalshi residual alignment: same outcome count + valid prices vector.
        nbr_outcomes = _coerce_array(row.get("outcomes"))
        nbr_q = _coerce_array(row.get("q_kalshi_at_open"))
        if (
            nbr_q is not None
            and len(nbr_q) == n
            and (nbr_outcomes is None or len(nbr_outcomes) == n)
            and realized is not None
        ):
            try:
                idx = int(realized)
            except (TypeError, ValueError):
                idx = -1
            for i in range(n):
                try:
                    q_i = float(nbr_q[i])
                except (TypeError, ValueError):
                    continue
                o_ik = 1.0 if idx == i else 0.0
                residual_sums[i] += o_ik - q_i
                residual_counts[i] += 1

        neighbors_out.append(
            {
                "market_id": row.get("market_id"),
                "similarity": sim,
                "question_text": row.get("question_text"),
                "realized_outcome": realized,
            }
        )

    total = len(rows)
    base_rate = [c / total for c in base_counts]
    kalshi_residual = [
        (residual_sums[i] / residual_counts[i]) if residual_counts[i] > 0 else 0.0
        for i in range(n)
    ]
    mean_sim = sim_sum / total if total > 0 else 0.0

    return BaseRateFeatures(
        neighbor_count=total,
        mean_similarity=mean_sim,
        base_rate=base_rate,
        kalshi_residual=kalshi_residual,
        neighbors=neighbors_out,
    )


# ---------------------------------------------------------------------------
# Snowflake-facing entry point.
# ---------------------------------------------------------------------------


def _build_query(category: str | None, exclude_market_id: str | None) -> tuple[str, list[Any]]:
    """Build the parameterized SQL + bound params for the neighbor lookup."""
    filters = [
        "h.question_embedding IS NOT NULL",
        "h.realized_outcome IS NOT NULL",
    ]
    params: list[Any] = []
    # Question text param for embedding.
    # NOTE: param order matches the order the placeholders appear in the SQL
    # (question_text, then optional filters, then k).
    if category is not None:
        filters.append("h.category = %s")
    if exclude_market_id is not None:
        filters.append("h.market_id <> %s")

    where_clause = " AND ".join(filters)
    sql = f"""
        WITH q AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s) AS qv
        )
        SELECT
            h.market_id,
            h.question_text,
            h.realized_outcome,
            h.outcomes,
            h.q_kalshi_at_open,
            VECTOR_COSINE_SIMILARITY(h.question_embedding, q.qv) AS similarity
        FROM HISTORICAL_MARKETS h, q
        WHERE {where_clause}
        ORDER BY similarity DESC
        LIMIT %s
    """
    return sql, params


def get_base_rate(
    question_text: str,
    outcomes: list[str],
    category: str | None = None,
    k: int = 15,
    exclude_market_id: str | None = None,
) -> BaseRateFeatures:
    """Retrieve top-K similar resolved markets and aggregate into base-rate features.

    Args:
        question_text: The natural-language forecasting question.
        outcomes: The query's outcome list (used to size base_rate/residual vectors).
        category: Optional category filter (e.g. 'crypto', 'sports').
        k: Number of neighbors to retrieve.
        exclude_market_id: Optional market_id to exclude (used during backfill
            so a historical market doesn't retrieve itself).

    Returns:
        A BaseRateFeatures dataclass. Returns the empty/uniform default if
        no neighbors are available (e.g. table is empty or embeddings haven't
        been populated yet).
    """
    n_outcomes = len(outcomes) if outcomes else 2

    sql, _unused = _build_query(category, exclude_market_id)
    # Build the full param list in the order placeholders appear in `sql`.
    params: list[Any] = [EMBED_MODEL, question_text]
    if category is not None:
        params.append(category)
    if exclude_market_id is not None:
        params.append(exclude_market_id)
    params.append(int(k))

    try:
        with snowflake_cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0].lower() for d in cur.description]
            raw_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — defensive: any Snowflake failure → empty.
        # Most likely cause early in the project: table doesn't exist yet or
        # no embeddings populated. Return the safe default so downstream code
        # has a well-shaped object to consume.
        print(f"[base_rate] Snowflake retrieval failed ({type(exc).__name__}): {exc}")
        return _empty_features(n_outcomes)

    return aggregate_neighbors(raw_rows, n_outcomes)


# ---------------------------------------------------------------------------
# CLI smoke test.
# ---------------------------------------------------------------------------


def _pretty_print(features: BaseRateFeatures) -> None:
    print("BaseRateFeatures:")
    print(f"  neighbor_count   : {features.neighbor_count}")
    print(f"  mean_similarity  : {features.mean_similarity:.4f}")
    print(f"  base_rate        : {[round(x, 4) for x in features.base_rate]}")
    print(f"  kalshi_residual  : {[round(x, 4) for x in features.kalshi_residual]}")
    print(f"  neighbors        : ({len(features.neighbors)} rows)")
    for nbr in features.neighbors[:5]:
        sim = nbr.get("similarity")
        sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
        print(
            f"    - [{sim_str}] outcome={nbr.get('realized_outcome')} "
            f"id={nbr.get('market_id')} :: {nbr.get('question_text')}"
        )


if __name__ == "__main__":
    features = get_base_rate(
        "Will Bitcoin close above $100,000 USD on the last day of this year?",
        ["Yes", "No"],
        category=None,
        k=5,
    )
    _pretty_print(features)
