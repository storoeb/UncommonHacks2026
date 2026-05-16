"""Wafer.ai client — OpenAI-compatible chat completions.

Endpoint: https://pass.wafer.ai/v1/
Auth:     Bearer token in `WAFER_API_KEY`
Models:   GLM-5.1, Qwen3.5-397B-A17B, Qwen3.6-35B-A3B
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv


WAFER_BASE_URL = "https://pass.wafer.ai/v1"

# Model IDs as returned by GET /v1/models on Wafer (case-sensitive).
# Note: all three are reasoning models — they emit `reasoning_content` before
# `content`, so they need a generous max_tokens budget (~3000+) to finish.
MODEL_GLM_51 = "GLM-5.1"
MODEL_QWEN_397B = "Qwen3.5-397B-A17B"
MODEL_QWEN_35B = "Qwen3.6-35B-A3B"

DEFAULT_ENSEMBLE = (MODEL_GLM_51, MODEL_QWEN_397B, MODEL_QWEN_35B)


_FORECASTER_SYSTEM = """You are a calibrated probabilistic forecaster.
You will be given a forecasting question and a list of mutually exclusive outcomes.
Reason briefly about base rates, current evidence, and timing.
Then emit ONLY a JSON object on the final line of your response with the shape:

{"probabilities": [<float for outcome 0>, <float for outcome 1>, ...]}

The probabilities must sum to 1.0 and align in order with the outcomes given."""


_FORECASTER_USER_TEMPLATE = """Question: {question}

Outcomes (in order):
{outcomes_block}

{market_block}Resolution date (if given): {resolve_by}

Think step by step, then return the JSON probability object as instructed."""


_JSON_RE = re.compile(r"\{[^{}]*\"probabilities\"[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ModelForecast:
    """One model's probability vector + raw response text."""
    model: str
    probabilities: list[float]
    raw_text: str
    latency_s: float


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregate of all model forecasts on one question."""
    forecasts: list[ModelForecast]
    mean: list[float]
    variance: float  # mean across outcomes of per-outcome variance


def _load_key() -> str:
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    key = os.getenv("WAFER_API_KEY", "").strip()
    if not key:
        msg = "WAFER_API_KEY not set in environment"
        raise RuntimeError(msg)
    return key


def _load_all_keys() -> list[str]:
    """Return all configured WAFER_API_KEY, WAFER_API_KEY_2, WAFER_API_KEY_3, etc."""
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    keys: list[str] = []
    # Primary key
    k = os.getenv("WAFER_API_KEY", "").strip()
    if k:
        keys.append(k)
    # Additional keys
    i = 2
    while True:
        k = os.getenv(f"WAFER_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
    if not keys:
        raise RuntimeError("WAFER_API_KEY not set in environment")
    return keys


def _build_messages(
    question: str,
    outcomes: list[str],
    market_prices: list[float] | None,
    resolve_by: str | None,
) -> list[dict[str, str]]:
    outcomes_block = "\n".join(f"  {i}. {o}" for i, o in enumerate(outcomes))
    if market_prices is not None and len(market_prices) == len(outcomes):
        prices_block = "\n".join(
            f"  {i}. {o}: market price {p:.3f}"
            for i, (o, p) in enumerate(zip(outcomes, market_prices, strict=True))
        )
        market_block = f"Current market-implied probabilities (Kalshi-style):\n{prices_block}\n\n"
    else:
        market_block = ""
    user = _FORECASTER_USER_TEMPLATE.format(
        question=question,
        outcomes_block=outcomes_block,
        market_block=market_block,
        resolve_by=resolve_by or "unspecified",
    )
    return [
        {"role": "system", "content": _FORECASTER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse_probabilities(text: str | None, n_outcomes: int) -> list[float]:
    """Extract `{"probabilities": [...]}` from a model response. Last match wins."""
    if not text:
        msg = "empty/None response content"
        raise ValueError(msg)
    matches = _JSON_RE.findall(text)
    if not matches:
        msg = "no probabilities JSON found in response"
        raise ValueError(msg)
    raw = matches[-1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"could not parse probabilities JSON: {e}: {raw!r}"
        raise ValueError(msg) from e
    probs = obj.get("probabilities")
    if not isinstance(probs, list) or len(probs) != n_outcomes:
        msg = f"probabilities shape mismatch: expected {n_outcomes}, got {probs!r}"
        raise ValueError(msg)
    probs = [float(p) for p in probs]
    total = sum(probs)
    if total <= 0:
        msg = f"probabilities sum to {total}, cannot normalize"
        raise ValueError(msg)
    return [p / total for p in probs]


class WaferClient:
    """Async client over Wafer.ai OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = WAFER_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self._key = api_key or _load_key()
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 3500,
    ) -> tuple[str, float]:
        """Returns (response_text, latency_seconds).

        Wafer models are reasoning models — `content` is what we want, but they
        first emit `reasoning_content`. With too-low max_tokens, `content` is
        null. Default 3500 leaves room after reasoning.
        """
        url = f"{self._base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        loop = asyncio.get_event_loop()
        start = loop.time()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        latency = loop.time() - start
        msg = data["choices"][0]["message"]
        text = msg.get("content")
        if not text:
            # Reasoning hit the token cap before content was emitted. Fall back
            # to reasoning_content so the parser at least has something to chew on.
            text = msg.get("reasoning_content") or ""
        return text, latency

    async def forecast_one(
        self,
        model: str,
        question: str,
        outcomes: list[str],
        market_prices: list[float] | None = None,
        resolve_by: str | None = None,
        retries: int = 2,
    ) -> ModelForecast:
        messages = _build_messages(question, outcomes, market_prices, resolve_by)
        last_err: Exception | None = None
        last_text: str = ""
        for attempt in range(retries + 1):
            try:
                # Bump tokens on retry — usually the problem is reasoning ate them.
                text, latency = await self.chat(
                    model,
                    messages,
                    temperature=0.4 + 0.1 * attempt,
                    max_tokens=3500 + 1500 * attempt,
                )
                last_text = text
                probs = _parse_probabilities(text, len(outcomes))
                return ModelForecast(model=model, probabilities=probs, raw_text=text, latency_s=latency)
            except (ValueError, httpx.HTTPError) as e:
                last_err = e
        # Final fallback: uniform prior so the ensemble can still proceed.
        uniform = [1.0 / len(outcomes)] * len(outcomes)
        return ModelForecast(
            model=model,
            probabilities=uniform,
            raw_text=f"[FALLBACK uniform — last_err={last_err} last_text={last_text!r:.200s}]",
            latency_s=0.0,
        )

    async def ensemble(
        self,
        question: str,
        outcomes: list[str],
        market_prices: list[float] | None = None,
        resolve_by: str | None = None,
        models: tuple[str, ...] = DEFAULT_ENSEMBLE,
    ) -> EnsembleResult:
        forecasts = await asyncio.gather(
            *(self.forecast_one(m, question, outcomes, market_prices, resolve_by) for m in models)
        )
        n = len(outcomes)
        mean = [sum(f.probabilities[i] for f in forecasts) / len(forecasts) for i in range(n)]
        # Per-outcome variance across models, averaged across outcomes.
        per_out_var = [
            sum((f.probabilities[i] - mean[i]) ** 2 for f in forecasts) / len(forecasts)
            for i in range(n)
        ]
        variance = sum(per_out_var) / n
        return EnsembleResult(forecasts=list(forecasts), mean=mean, variance=variance)


class WaferPool:
    """Multi-key pool with the same interface as WaferClient.

    Distributes each model call to a different API key in round-robin order.
    With 3 keys and 3 models, each model gets its own key — no single key
    handles more than one concurrent call per ensemble request.

    Reads WAFER_API_KEY, WAFER_API_KEY_2, WAFER_API_KEY_3 (and so on) from
    the environment. Falls back to a single WaferClient if only one key exists.

    Drop-in replacement for WaferClient everywhere (same ensemble/forecast_one
    signatures).
    """

    def __init__(
        self,
        base_url: str = WAFER_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        keys = _load_all_keys()
        self._clients = [
            WaferClient(api_key=k, base_url=base_url, timeout_s=timeout_s)
            for k in keys
        ]
        self._n = len(self._clients)

    def _client_for(self, idx: int) -> WaferClient:
        """Return the client at position idx % pool_size."""
        return self._clients[idx % self._n]

    async def forecast_one(
        self,
        model: str,
        question: str,
        outcomes: list[str],
        market_prices: list[float] | None = None,
        resolve_by: str | None = None,
        retries: int = 2,
        _key_idx: int = 0,
    ) -> ModelForecast:
        return await self._client_for(_key_idx).forecast_one(
            model, question, outcomes, market_prices, resolve_by, retries
        )

    async def ensemble(
        self,
        question: str,
        outcomes: list[str],
        market_prices: list[float] | None = None,
        resolve_by: str | None = None,
        models: tuple[str, ...] = DEFAULT_ENSEMBLE,
        replicates: int = 1,
    ) -> EnsembleResult:
        """Fire each model on a different key in parallel.

        replicates=1 (default): one call per model, distributed across keys.
            3 models × 3 keys → 3 parallel calls (key 0→model 0, key 1→model 1, …)
        replicates=N: each model runs N times, each on a different key.
            3 models × replicates=3 → 9 parallel calls, all keys used for every model.
            Gives a more robust ensemble at N× cost.
        """
        # Build (client, model) pairs
        calls: list[tuple[WaferClient, str]] = []
        for rep in range(max(1, replicates)):
            for i, m in enumerate(models):
                key_idx = (rep * len(models) + i)
                calls.append((self._client_for(key_idx), m))

        raw_forecasts = await asyncio.gather(
            *(client.forecast_one(m, question, outcomes, market_prices, resolve_by)
              for client, m in calls)
        )
        n = len(outcomes)
        mean = [sum(f.probabilities[i] for f in raw_forecasts) / len(raw_forecasts) for i in range(n)]
        per_out_var = [
            sum((f.probabilities[i] - mean[i]) ** 2 for f in raw_forecasts) / len(raw_forecasts)
            for i in range(n)
        ]
        variance = sum(per_out_var) / n
        return EnsembleResult(forecasts=list(raw_forecasts), mean=mean, variance=variance)


async def _smoke() -> None:
    """Manual smoke test entry point."""
    client = WaferClient()
    result = await client.ensemble(
        question="Will Bitcoin close above $100,000 USD on the last day of this year?",
        outcomes=["Yes", "No"],
        market_prices=[0.55, 0.45],
        resolve_by="end of current calendar year",
    )
    for f in result.forecasts:
        snippet = f.raw_text.replace("\n", " ")[:160]
        print(f"[{f.model}]  probs={[round(p, 3) for p in f.probabilities]}  "
              f"latency={f.latency_s:.2f}s  raw={snippet!r}")
    print(f"ENSEMBLE mean = {[round(p, 3) for p in result.mean]}  variance = {result.variance:.4f}")


if __name__ == "__main__":
    asyncio.run(_smoke())
