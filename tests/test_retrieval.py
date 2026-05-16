"""Unit tests for retrieval/base_rate.py — pure-Python aggregation only.

The Snowflake live path is exercised by the CLI smoke test
(`python -m prophet_agent.retrieval.base_rate`) and gated behind
PROPHET_LIVE_SNOWFLAKE in the optional `test_live_snowflake` below.
"""

from __future__ import annotations

import json
import os

import pytest

from prophet_agent.retrieval.base_rate import (
    BaseRateFeatures,
    _build_query,
    _coerce_array,
    _empty_features,
    aggregate_neighbors,
    get_base_rate,
)


# ---------------------------------------------------------------------------
# _empty_features
# ---------------------------------------------------------------------------


def test_empty_features_binary_uniform() -> None:
    feats = _empty_features(2)
    assert feats.neighbor_count == 0
    assert feats.mean_similarity == 0.0
    assert feats.base_rate == pytest.approx([0.5, 0.5])
    assert feats.kalshi_residual == [0.0, 0.0]
    assert feats.neighbors == []


def test_empty_features_three_outcomes() -> None:
    feats = _empty_features(3)
    assert feats.base_rate == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert feats.kalshi_residual == [0.0, 0.0, 0.0]


def test_empty_features_zero_outcomes_falls_back_to_one() -> None:
    # Defensive: a degenerate call shouldn't divide by zero.
    feats = _empty_features(0)
    assert len(feats.base_rate) == 1
    assert feats.base_rate == pytest.approx([1.0])


# ---------------------------------------------------------------------------
# _coerce_array
# ---------------------------------------------------------------------------


def test_coerce_array_passes_list_through() -> None:
    assert _coerce_array([1, 2, 3]) == [1, 2, 3]


def test_coerce_array_parses_json_string() -> None:
    assert _coerce_array("[0.4, 0.6]") == [0.4, 0.6]


def test_coerce_array_handles_none() -> None:
    assert _coerce_array(None) is None


def test_coerce_array_rejects_garbage() -> None:
    assert _coerce_array("not-json") is None
    assert _coerce_array("{}") is None  # dict is not a list


# ---------------------------------------------------------------------------
# aggregate_neighbors
# ---------------------------------------------------------------------------


def test_aggregate_neighbors_empty_returns_uniform() -> None:
    feats = aggregate_neighbors([], n_outcomes=2)
    assert feats.neighbor_count == 0
    assert feats.base_rate == pytest.approx([0.5, 0.5])


def test_aggregate_neighbors_basic_binary_base_rate() -> None:
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 0,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.4, 0.6],
        },
        {
            "market_id": "m2",
            "similarity": 0.8,
            "question_text": "Q2",
            "realized_outcome": 0,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.5, 0.5],
        },
        {
            "market_id": "m3",
            "similarity": 0.7,
            "question_text": "Q3",
            "realized_outcome": 1,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.3, 0.7],
        },
        {
            "market_id": "m4",
            "similarity": 0.6,
            "question_text": "Q4",
            "realized_outcome": 1,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.5, 0.5],
        },
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert feats.neighbor_count == 4
    assert feats.base_rate == pytest.approx([0.5, 0.5])  # 2 of each outcome.
    assert feats.mean_similarity == pytest.approx((0.9 + 0.8 + 0.7 + 0.6) / 4)

    # kalshi_residual[0] = mean(o_0 - q_0) across the 4 neighbors.
    # neighbor 1: realized=0 → o_0=1, q_0=0.4 → 0.6
    # neighbor 2: realized=0 → o_0=1, q_0=0.5 → 0.5
    # neighbor 3: realized=1 → o_0=0, q_0=0.3 → -0.3
    # neighbor 4: realized=1 → o_0=0, q_0=0.5 → -0.5
    expected_r0 = (0.6 + 0.5 - 0.3 - 0.5) / 4
    # kalshi_residual[1] = mean(o_1 - q_1):
    # n1: o_1=0, q_1=0.6 → -0.6
    # n2: o_1=0, q_1=0.5 → -0.5
    # n3: o_1=1, q_1=0.7 → 0.3
    # n4: o_1=1, q_1=0.5 → 0.5
    expected_r1 = (-0.6 - 0.5 + 0.3 + 0.5) / 4
    assert feats.kalshi_residual[0] == pytest.approx(expected_r0)
    assert feats.kalshi_residual[1] == pytest.approx(expected_r1)


def test_aggregate_neighbors_skips_misshaped_for_residual_but_counts_base() -> None:
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 0,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.4, 0.6],
        },
        # Different outcome arity → keep for base_rate, drop for residual.
        {
            "market_id": "m2",
            "similarity": 0.8,
            "question_text": "Q2",
            "realized_outcome": 1,
            "outcomes": ["A", "B", "C"],
            "q_kalshi_at_open": [0.3, 0.3, 0.4],
        },
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert feats.neighbor_count == 2
    # base_rate uses all rows: realized 0 → idx 0; realized 1 → idx 1 (within range).
    assert feats.base_rate == pytest.approx([0.5, 0.5])
    # Only neighbor m1 contributes to residual.
    # r[0] = 1 - 0.4 = 0.6
    # r[1] = 0 - 0.6 = -0.6
    assert feats.kalshi_residual[0] == pytest.approx(0.6)
    assert feats.kalshi_residual[1] == pytest.approx(-0.6)


def test_aggregate_neighbors_handles_json_string_arrays() -> None:
    # Snowflake ARRAY columns sometimes come back as JSON-encoded strings.
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 0,
            "outcomes": json.dumps(["Yes", "No"]),
            "q_kalshi_at_open": json.dumps([0.4, 0.6]),
        }
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert feats.neighbor_count == 1
    assert feats.base_rate == pytest.approx([1.0, 0.0])
    assert feats.kalshi_residual[0] == pytest.approx(0.6)
    assert feats.kalshi_residual[1] == pytest.approx(-0.6)


def test_aggregate_neighbors_realized_out_of_range_ignored_for_base() -> None:
    # A neighbor whose realized index exceeds the query's outcome count
    # should not contribute to base_rate counts.
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 5,  # out of [0,1] range
            "outcomes": ["A", "B", "C", "D", "E", "F"],
            "q_kalshi_at_open": [0.1] * 6,
        }
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert feats.neighbor_count == 1
    assert feats.base_rate == pytest.approx([0.0, 0.0])


def test_aggregate_neighbors_handles_missing_kalshi() -> None:
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 0,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": None,
        }
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert feats.neighbor_count == 1
    assert feats.base_rate == pytest.approx([1.0, 0.0])
    # No neighbor contributed → residual stays at 0.
    assert feats.kalshi_residual == [0.0, 0.0]


def test_aggregate_neighbors_neighbors_field_is_compact() -> None:
    rows = [
        {
            "market_id": "m1",
            "similarity": 0.9,
            "question_text": "Q1",
            "realized_outcome": 0,
            "outcomes": ["Yes", "No"],
            "q_kalshi_at_open": [0.4, 0.6],
        },
    ]
    feats = aggregate_neighbors(rows, n_outcomes=2)
    assert len(feats.neighbors) == 1
    nbr = feats.neighbors[0]
    # Expect just the four documented fields.
    assert set(nbr.keys()) == {"market_id", "similarity", "question_text", "realized_outcome"}
    assert nbr["market_id"] == "m1"
    assert nbr["similarity"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _build_query
# ---------------------------------------------------------------------------


def test_build_query_no_filters_uses_two_placeholders_pre_limit() -> None:
    sql, _ = _build_query(category=None, exclude_market_id=None)
    # Two placeholders for the EMBED call (model name + question text),
    # plus one for LIMIT k → 3 total.
    assert sql.count("%s") == 3
    assert "category" not in sql.lower() or "h.category" not in sql.lower()
    assert "VECTOR_COSINE_SIMILARITY" in sql
    assert "ORDER BY similarity DESC" in sql


def test_build_query_with_category_adds_filter() -> None:
    sql, _ = _build_query(category="crypto", exclude_market_id=None)
    assert "h.category = %s" in sql
    # 2 for embed + 1 for category + 1 for limit = 4.
    assert sql.count("%s") == 4


def test_build_query_with_exclude_adds_filter() -> None:
    sql, _ = _build_query(category=None, exclude_market_id="KX-FOO")
    assert "h.market_id <> %s" in sql
    assert sql.count("%s") == 4


def test_build_query_with_both_filters() -> None:
    sql, _ = _build_query(category="sports", exclude_market_id="KX-BAR")
    assert "h.category = %s" in sql
    assert "h.market_id <> %s" in sql
    # 2 for embed + 1 cat + 1 excl + 1 limit.
    assert sql.count("%s") == 5


# ---------------------------------------------------------------------------
# get_base_rate — Snowflake failure path returns the empty default.
# ---------------------------------------------------------------------------


def test_get_base_rate_returns_empty_on_snowflake_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Snowflake is unreachable, we should still return a well-shaped result."""

    class _Boom:
        def __enter__(self):
            raise RuntimeError("simulated snowflake outage")

        def __exit__(self, *_a):  # pragma: no cover — never reached
            return False

    monkeypatch.setattr(
        "prophet_agent.retrieval.base_rate.snowflake_cursor",
        lambda: _Boom(),
    )
    feats = get_base_rate("Will it rain tomorrow?", ["Yes", "No"], k=5)
    assert isinstance(feats, BaseRateFeatures)
    assert feats.neighbor_count == 0
    assert feats.base_rate == pytest.approx([0.5, 0.5])
    assert feats.kalshi_residual == [0.0, 0.0]
    assert feats.neighbors == []


def test_get_base_rate_aggregates_mocked_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire up a fake cursor that returns two rows; aggregation should work end-to-end."""
    captured: dict = {}

    class _FakeCursor:
        description = [
            ("MARKET_ID",), ("QUESTION_TEXT",), ("REALIZED_OUTCOME",),
            ("OUTCOMES",), ("Q_KALSHI_AT_OPEN",), ("SIMILARITY",),
        ]

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return [
                ("m1", "Q1", 0, ["Yes", "No"], [0.4, 0.6], 0.91),
                ("m2", "Q2", 1, ["Yes", "No"], [0.3, 0.7], 0.82),
            ]

    class _FakeCtx:
        def __enter__(self):
            return _FakeCursor()

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(
        "prophet_agent.retrieval.base_rate.snowflake_cursor",
        lambda: _FakeCtx(),
    )
    feats = get_base_rate(
        "Will X happen?",
        ["Yes", "No"],
        category="crypto",
        k=2,
        exclude_market_id="KX-SELF",
    )
    assert feats.neighbor_count == 2
    assert feats.base_rate == pytest.approx([0.5, 0.5])
    assert feats.mean_similarity == pytest.approx((0.91 + 0.82) / 2)
    # Params: [model, question, category, exclude, k]
    assert captured["params"][0] == "snowflake-arctic-embed-m-v1.5"
    assert captured["params"][1] == "Will X happen?"
    assert captured["params"][2] == "crypto"
    assert captured["params"][3] == "KX-SELF"
    assert captured["params"][4] == 2


# ---------------------------------------------------------------------------
# Optional live test — guarded behind an env var so CI / local runs stay quick.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("PROPHET_LIVE_SNOWFLAKE"),
    reason="set PROPHET_LIVE_SNOWFLAKE=1 to hit real Snowflake",
)
def test_live_snowflake_smoke() -> None:
    feats = get_base_rate(
        "Will Bitcoin close above $100,000 USD on the last day of this year?",
        ["Yes", "No"],
        k=3,
    )
    assert isinstance(feats, BaseRateFeatures)
    # Whatever the table state, the shape should be well-formed.
    assert len(feats.base_rate) == 2
    assert len(feats.kalshi_residual) == 2
