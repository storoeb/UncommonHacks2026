"""Integration tests for the FastAPI agent.

We never hit Wafer, Snowflake, or the retrieval module in tests. The
server-under-test exposes hooks via `app.state` so the pipeline picks up
in-memory fakes instead of real collaborators.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prophet_agent.server import app


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------

@dataclass
class _FakeForecast:
    model: str
    probabilities: list[float]
    raw_text: str = "fake"
    latency_s: float = 0.01


@dataclass
class _FakeEnsembleResult:
    forecasts: list[_FakeForecast]
    mean: list[float]
    variance: float


class _FakeWafer:
    """Minimal stand-in for `WaferClient`. Mirrors `ensemble` signature."""

    def __init__(self, mean: list[float] | None = None) -> None:
        self._mean = mean

    async def ensemble(
        self,
        question: str,
        outcomes: list[str],
        market_prices: list[float] | None = None,
        resolve_by: str | None = None,
    ) -> _FakeEnsembleResult:
        n = len(outcomes)
        mean = self._mean if (self._mean and len(self._mean) == n) else [1.0 / n] * n
        forecasts = [
            _FakeForecast(model="fake-model-A", probabilities=list(mean)),
            _FakeForecast(model="fake-model-B", probabilities=list(mean)),
        ]
        return _FakeEnsembleResult(forecasts=forecasts, mean=list(mean), variance=0.0)


@dataclass
class _FakeBaseRate:
    neighbor_count: int = 0
    mean_similarity: float = 0.0
    base_rate: list[float] | None = None
    kalshi_residual: list[float] | None = None
    neighbors: list[dict[str, Any]] | None = None


async def _fake_base_rate_fn(
    question_text: str,
    outcomes: list[str],
    category: str | None = None,
    k: int = 15,
    exclude_market_id: str | None = None,
) -> _FakeBaseRate:
    n = len(outcomes)
    return _FakeBaseRate(
        neighbor_count=3,
        mean_similarity=0.71,
        base_rate=[1.0 / n] * n,
        kalshi_residual=[0.0] * n,
        neighbors=[],
    )


def _silent_logger(_result: Any) -> None:
    # No-op — keeps the test from touching Snowflake.
    return None


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    app.state.wafer_client = _FakeWafer(mean=[0.42, 0.58])
    app.state.base_rate_fn = _fake_base_rate_fn
    app.state.snowflake_logger = _silent_logger
    with TestClient(app) as c:
        yield c
    # Cleanup so other tests don't inherit overrides.
    for attr in ("wafer_client", "base_rate_fn", "snowflake_logger"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_models(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["id"] == "prophet-agent" for m in data["data"])


def test_chat_completions_with_fixture(client: TestClient) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_event.json"
    req = json.loads(fixture_path.read_text(encoding="utf-8"))
    r = client.post("/v1/chat/completions", json=req)
    assert r.status_code == 200, r.text
    body = r.json()

    # OpenAI shape checks.
    assert body["object"] == "chat.completion"
    assert body["model"]
    assert body["choices"], body
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    content = msg["content"]

    # Probability JSON block must be parseable from the content.
    matches = re.findall(r'\{"probabilities":\s*\[[^\]]+\]\}', content)
    assert matches, f"no probabilities JSON found in: {content!r}"
    probs = json.loads(matches[-1])["probabilities"]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6

    # Debug block populated.
    dbg = body["prophet_debug"]
    assert dbg["request_id"]
    assert dbg["p_final"] == probs
    assert dbg["neighbor_count"] == 3
    assert dbg["alpha"] == pytest.approx(1.0)


def test_chat_completions_no_outcomes_defaults_binary(client: TestClient) -> None:
    req = {
        "model": "prophet-agent",
        "messages": [{"role": "user", "content": "Will it snow in Chicago next week?"}],
    }
    r = client.post("/v1/chat/completions", json=req)
    assert r.status_code == 200, r.text
    dbg = r.json()["prophet_debug"]
    assert dbg["outcomes"] == ["Yes", "No"]
    assert len(dbg["p_final"]) == 2


def test_chat_completions_missing_messages(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"model": "prophet-agent", "messages": []})
    assert r.status_code == 400


def test_chat_completions_no_user_content(client: TestClient) -> None:
    req = {
        "model": "prophet-agent",
        "messages": [{"role": "system", "content": "system only"}],
    }
    # System content is still used as fallback content, so this should succeed.
    r = client.post("/v1/chat/completions", json=req)
    assert r.status_code == 200


def test_alpha_shading_identity_when_alpha_one(client: TestClient) -> None:
    # With α=1.0 (stub), p_final == p_meta == p_ensemble (calibrator stub is identity).
    req = {
        "model": "prophet-agent",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Question: Will X happen?\n"
                    "1. Yes\n2. No\n"
                    "Market price: 0.9\n"
                ),
            }
        ],
    }
    r = client.post("/v1/chat/completions", json=req)
    assert r.status_code == 200
    dbg = r.json()["prophet_debug"]
    assert dbg["p_final"] == pytest.approx(dbg["p_meta"])
    assert dbg["p_meta"] == pytest.approx(dbg["p_ensemble"])
