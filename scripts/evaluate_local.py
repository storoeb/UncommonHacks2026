"""Evaluate the running local agent against a resolved event slate.

Reads events from a JSON file produced by:
    prophet forecast retrieve --dataset sample-resolved --include-resolved -o resolved.json

Fires all requests concurrently against http://localhost:8000, extracts
probabilities, computes Brier score and AVER, and prints a summary table.

Usage:
    python scripts/evaluate_local.py resolved.json
    python scripts/evaluate_local.py resolved.json --url http://localhost:8000
    python scripts/evaluate_local.py resolved.json --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx


def _brier(probs: list[float], outcome_idx: int) -> float:
    return sum((p - (1.0 if i == outcome_idx else 0.0)) ** 2 for i, p in enumerate(probs))


def _aver(probs: list[float], outcome_idx: int, epsilon: float = 1e-9) -> float:
    n = len(probs)
    p = max(probs[outcome_idx], epsilon)
    return math.log(p / (1.0 / n))


def _outcome_index(event: dict, resolved_outcome: dict) -> int | None:
    outcomes = event.get("outcomes", [])
    resolved_values = resolved_outcome.get("value", [])
    if not resolved_values:
        return None
    target = resolved_values[0]
    for i, o in enumerate(outcomes):
        if o == target:
            return i
    return None


def _build_prompt(event: dict) -> str:
    title = event.get("title") or event.get("description") or ""
    outcomes = event.get("outcomes", ["Yes", "No"])
    rules = event.get("rules") or event.get("description") or ""
    close_time = event.get("close_time") or ""
    market_ticker = event.get("market_ticker") or event.get("event_ticker") or ""

    outcomes_block = "\n".join(f"{i+1}. {o}" for i, o in enumerate(outcomes))
    lines = [f"Question: {title}", "", f"Outcomes:\n{outcomes_block}"]
    if close_time:
        lines.append(f"Resolve by: {close_time}")
    if market_ticker:
        lines.append(f"Market ID: {market_ticker}")
    if rules and rules != title:
        lines.append(f"\nResolution rules: {rules}")
    return "\n".join(lines)


def _parse_probs(content: str, n_outcomes: int) -> list[float] | None:
    matches = re.findall(r'\{"probabilities":\s*\[[^\]]+\]\}', content)
    if not matches:
        return None
    try:
        probs = [float(p) for p in json.loads(matches[-1])["probabilities"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if len(probs) != n_outcomes:
        return None
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else None


async def call_agent_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    idx: int,
    event: dict,
    timeout: float,
) -> dict:
    outcomes = event.get("outcomes", ["Yes", "No"])
    n = len(outcomes)
    ro = event.get("resolved_outcome") or {}
    outcome_idx = _outcome_index(event, ro)
    title = (event.get("title") or "")[:50]
    category = (event.get("category") or "?")[:14]

    if outcome_idx is None:
        return {"idx": idx, "status": "skip", "category": category, "title": title}

    prompt = _build_prompt(event)
    body = {
        "model": "prophet-agent",
        "messages": [
            {"role": "system", "content": "You are a calibrated probabilistic forecaster."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    t0 = time.monotonic()
    async with semaphore:
        try:
            r = await client.post(
                f"{url.rstrip('/')}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return {"idx": idx, "status": "error", "error": str(e),
                    "category": category, "title": title, "elapsed": elapsed}

    probs = _parse_probs(content, n)
    if probs is None:
        return {"idx": idx, "status": "parse_error", "category": category,
                "title": title, "elapsed": elapsed}

    return {
        "idx": idx,
        "status": "ok",
        "category": category,
        "title": title,
        "elapsed": elapsed,
        "brier": _brier(probs, outcome_idx),
        "aver": _aver(probs, outcome_idx),
        "probs": probs,
        "outcome_idx": outcome_idx,
    }


async def run_eval(resolved: list[dict], url: str, concurrency: int, timeout: float) -> None:
    print(f"Evaluating {len(resolved)} events concurrently (concurrency={concurrency}) against {url}")
    print(f"{'#':<4}  {'Category':<14}  {'Brier':>7}  {'AVER':>7}  {'s':>5}  Title")
    print("-" * 90)

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(call_agent_async(client, semaphore, url, i+1, e, timeout))
            for i, e in enumerate(resolved)
        ]
        results_raw = await asyncio.gather(*tasks)

    # Sort by original index for clean output
    results = sorted(results_raw, key=lambda r: r["idx"])

    briers, avers = [], []
    errors = 0
    for r in results:
        idx = r["idx"]
        cat = r["category"]
        title = r["title"]
        if r["status"] == "skip":
            print(f"{idx:<4}  {cat:<14}  {'SKIP':>7}  {'?':>7}  {'?':>5}  {title}")
            errors += 1
        elif r["status"] in ("error", "parse_error"):
            msg = r.get("error", "parse?")[:20]
            elapsed = r.get("elapsed", 0)
            print(f"{idx:<4}  {cat:<14}  {'ERR':>7}  {'ERR':>7}  {elapsed:>5.1f}  {title}  [{msg}]")
            errors += 1
        else:
            b, a, elapsed = r["brier"], r["aver"], r["elapsed"]
            briers.append(b)
            avers.append(a)
            print(f"{idx:<4}  {cat:<14}  {b:>7.4f}  {a:>7.4f}  {elapsed:>5.1f}  {title}")

    print("-" * 90)
    if briers:
        mean_b = sum(briers) / len(briers)
        mean_a = sum(avers) / len(avers)
        print(f"\nResults over {len(briers)} scored events ({errors} errors/skips):")
        print(f"  Mean Brier : {mean_b:.4f}  (lower is better; always-Kalshi ≈ 0.50 for binary)")
        print(f"  Mean AVER  : {mean_a:.4f}  (higher is better; matching market = 0.00)")
        print(f"\n  Paste into PITCH.md:  Brier={mean_b:.3f}  AVER={mean_a:.3f}")
    else:
        print("\nNo events scored successfully.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events_file", help="resolved.json from prophet forecast retrieve")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=26, help="Parallel requests (default: all at once)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    events_path = Path(args.events_file)
    if not events_path.is_file():
        print(f"File not found: {events_path}", file=sys.stderr)
        sys.exit(1)

    events = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        print("Expected a JSON array", file=sys.stderr)
        sys.exit(1)

    resolved = [e for e in events if e.get("resolved_outcome") is not None]
    if args.limit:
        resolved = resolved[:args.limit]

    asyncio.run(run_eval(resolved, args.url, args.concurrency, args.timeout))


if __name__ == "__main__":
    main()
