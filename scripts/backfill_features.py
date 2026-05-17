"""Backfill ensemble + base-rate features for resolved historical markets.

For every row in HISTORICAL_MARKETS that doesn't yet have a corresponding row
in ENSEMBLE_BACKFILL (or is stale), this job:

  1. Runs the LLM "ensemble" — for cost reasons during backfill we use a single
     model (`Qwen3.6-35B-A3B`, the cheapest on Wafer). Live inference uses all 3.
  2. Runs the Snowflake retrieval base-rate (excluding the row itself so a
     historical market doesn't retrieve itself as a neighbor).
  3. MERGE-upserts the results into ENSEMBLE_BACKFILL and BASE_RATE_BACKFILL.

CLI:
  python scripts/backfill_features.py [--limit N] [--concurrency K] [--dry-run] [--yes]

Cost safety: at startup we print an estimated cost and prompt for confirmation
(TTY only; piped/redirected stdin auto-confirms, and `--yes` skips the prompt).

This script never modifies HISTORICAL_MARKETS, only the two backfill tables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.llm.wafer import WaferClient, WaferPool  # noqa: E402
from prophet_agent.retrieval.base_rate import BaseRateFeatures, get_base_rate  # noqa: E402
from prophet_agent.snowflake_client import _load_env_once, snowflake_cursor  # noqa: E402

_load_env_once()


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

# Single-model "ensemble" for cost-controlled backfill.
BACKFILL_MODELS: tuple[str, ...] = ("Qwen3.6-35B-A3B",)

# Roughly: ~2000 tokens per market on the 35B model.
# Wafer's posted price for that tier is ~$0.19/1M tokens.
COST_PER_MARKET_USD = 2000 * 0.19 / 1_000_000  # ~$0.00038/market

PROGRESS_EVERY = 25


# ---------------------------------------------------------------------------
# Work-queue query.
# ---------------------------------------------------------------------------

_WORK_QUEUE_SQL = """
    SELECT m.market_id, m.source, m.category, m.question_text, m.outcomes
    FROM HISTORICAL_MARKETS m
    LEFT JOIN ENSEMBLE_BACKFILL e
      ON e.market_id = m.market_id AND e.source = m.source
    WHERE e.market_id IS NULL
      AND m.question_text IS NOT NULL
      AND m.realized_outcome IS NOT NULL
      {category_filter}
    LIMIT %(limit)s
"""


def fetch_work_queue(limit: int, only_categories: list[str] | None = None) -> list[dict[str, Any]]:
    """Return rows that need backfilling, up to `limit`."""
    if only_categories:
        placeholders = ", ".join(f"'{c}'" for c in only_categories)
        cat_filter = f"AND m.category IN ({placeholders})"
    else:
        cat_filter = ""
    sql = _WORK_QUEUE_SQL.format(category_filter=cat_filter)
    with snowflake_cursor() as cur:
        cur.execute(sql, {"limit": int(limit)})
        cols = [d[0].lower() for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Outcome coercion — Snowflake ARRAY columns come back as JSON strings or lists.
# ---------------------------------------------------------------------------

def _coerce_outcomes(raw: Any) -> list[str]:
    if raw is None:
        return ["Yes", "No"]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return ["Yes", "No"]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return ["Yes", "No"]
    return ["Yes", "No"]


# ---------------------------------------------------------------------------
# Upsert SQL (PARSE_JSON over JSON-string binds for ARRAY/VARIANT columns).
# ---------------------------------------------------------------------------

_ENSEMBLE_MERGE_SQL = """
MERGE INTO ENSEMBLE_BACKFILL t
USING (
    SELECT
        %(market_id)s                      AS market_id,
        %(source)s                         AS source,
        PARSE_JSON(%(model_outputs_json)s) AS model_outputs,
        PARSE_JSON(%(p_ensemble_json)s)    AS p_ensemble,
        %(p_ensemble_var)s                 AS p_ensemble_var
) s
ON t.market_id = s.market_id AND t.source = s.source
WHEN MATCHED THEN UPDATE SET
    model_outputs   = s.model_outputs,
    p_ensemble      = s.p_ensemble,
    p_ensemble_var  = s.p_ensemble_var,
    backfilled_at   = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
    market_id, source, model_outputs, p_ensemble, p_ensemble_var
) VALUES (
    s.market_id, s.source, s.model_outputs, s.p_ensemble, s.p_ensemble_var
)
"""

_BASE_RATE_MERGE_SQL = """
MERGE INTO BASE_RATE_BACKFILL t
USING (
    SELECT
        %(market_id)s                       AS market_id,
        %(source)s                          AS source,
        %(neighbor_count)s                  AS neighbor_count,
        %(mean_similarity)s                 AS mean_similarity,
        PARSE_JSON(%(base_rate_json)s)      AS base_rate,
        PARSE_JSON(%(kalshi_residual_json)s) AS kalshi_residual
) s
ON t.market_id = s.market_id AND t.source = s.source
WHEN MATCHED THEN UPDATE SET
    neighbor_count  = s.neighbor_count,
    mean_similarity = s.mean_similarity,
    base_rate       = s.base_rate,
    kalshi_residual = s.kalshi_residual,
    backfilled_at   = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN INSERT (
    market_id, source, neighbor_count, mean_similarity, base_rate, kalshi_residual
) VALUES (
    s.market_id, s.source, s.neighbor_count, s.mean_similarity, s.base_rate, s.kalshi_residual
)
"""


def upsert_one(
    cur: Any,
    market_id: str,
    source: str,
    ensemble_payload: dict[str, Any],
    base_rate: BaseRateFeatures,
) -> None:
    """Upsert one market's backfill features into both tables.

    Both MERGE statements use parameterized binds (no f-string SQL with user
    data) — ARRAY/VARIANT values are passed as JSON strings and PARSE_JSON'd
    inside Snowflake.
    """
    cur.execute(
        _ENSEMBLE_MERGE_SQL,
        {
            "market_id": market_id,
            "source": source,
            "model_outputs_json": json.dumps(ensemble_payload["model_outputs"]),
            "p_ensemble_json": json.dumps(ensemble_payload["p_ensemble"]),
            "p_ensemble_var": float(ensemble_payload["p_ensemble_var"]),
        },
    )
    cur.execute(
        _BASE_RATE_MERGE_SQL,
        {
            "market_id": market_id,
            "source": source,
            "neighbor_count": int(base_rate.neighbor_count),
            "mean_similarity": float(base_rate.mean_similarity),
            "base_rate_json": json.dumps(list(base_rate.base_rate)),
            "kalshi_residual_json": json.dumps(list(base_rate.kalshi_residual)),
        },
    )


# ---------------------------------------------------------------------------
# Per-row backfill (async).
# ---------------------------------------------------------------------------

async def backfill_one(
    client: WaferClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Compute backfill features for one historical market.

    Returns a result dict with either {"ok": True, ...} or {"ok": False, ...}.
    Errors are returned (not raised) so the outer gather can keep going.
    """
    market_id = row["market_id"]
    source = row["source"]
    category = row.get("category")
    question = row["question_text"]
    outcomes = _coerce_outcomes(row.get("outcomes"))

    try:
        async with semaphore:
            ensemble_result = await client.ensemble(
                question,
                outcomes,
                models=BACKFILL_MODELS,
            )

        # Base-rate retrieval is a synchronous Snowflake call. Push it to the
        # default executor so we don't block the event loop.
        loop = asyncio.get_event_loop()
        base_rate = await loop.run_in_executor(
            None,
            lambda: get_base_rate(
                question,
                outcomes,
                category=category,
                k=15,
                exclude_market_id=market_id,
            ),
        )

        model_outputs = [
            {
                "model": f.model,
                "probabilities": list(f.probabilities),
                "latency_s": float(f.latency_s),
            }
            for f in ensemble_result.forecasts
        ]
        ensemble_payload = {
            "model_outputs": model_outputs,
            "p_ensemble": list(ensemble_result.mean),
            "p_ensemble_var": float(ensemble_result.variance),
        }
        # Sum of latencies of the (single-model) ensemble for progress stats.
        latency = sum(float(f.latency_s) for f in ensemble_result.forecasts)
        return {
            "ok": True,
            "market_id": market_id,
            "source": source,
            "ensemble_payload": ensemble_payload,
            "base_rate": base_rate,
            "latency_s": latency,
        }
    except Exception as exc:  # noqa: BLE001 — defensive: never crash the batch.
        return {
            "ok": False,
            "market_id": market_id,
            "source": source,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Progress reporting.
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _print_progress(processed: int, total: int, latencies: list[float], errors: int) -> None:
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = _percentile(latencies, 95.0)
    print(
        f"  progress {processed}/{total}  "
        f"latency p50={p50:.2f}s p95={p95:.2f}s  errors={errors}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Cost-confirmation prompt.
# ---------------------------------------------------------------------------

def _confirm_cost(limit: int, auto_yes: bool) -> bool:
    cost = limit * COST_PER_MARKET_USD
    print(
        f"Estimated cost: ~{limit} markets x ~2000 tokens x $0.19/M = ${cost:.2f}"
    )
    if auto_yes:
        print("  --yes passed; skipping prompt.")
        return True
    if not sys.stdin.isatty():
        print("  stdin is not a TTY; auto-confirming.")
        return True
    print("Proceed? [y/N] ", end="", flush=True)
    try:
        ans = input().strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------

async def run_backfill(rows: list[dict[str, Any]], concurrency: int) -> None:
    try:
        from prophet_agent.llm.wafer import WaferPool  # noqa: PLC0415
        client = WaferPool()
        print(f"  WaferPool loaded with {client._n} key(s).")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] WaferPool failed to load: {exc}")
        print("  Falling back to single WaferClient...")
        from prophet_agent.llm.wafer import WaferClient  # noqa: PLC0415
        client = WaferClient()  # type: ignore[assignment]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    tasks = [asyncio.create_task(backfill_one(client, semaphore, row)) for row in rows]

    processed = 0
    errors = 0
    latencies: list[float] = []
    total = len(tasks)

    with snowflake_cursor() as cur:
        with tqdm(total=total, desc="backfill markets", unit="mkt", dynamic_ncols=True) as pbar:
            for fut in asyncio.as_completed(tasks):
                result = await fut
                processed += 1
                if not result["ok"]:
                    errors += 1
                    tqdm.write(
                        f"  [error] {result['market_id']} ({result['source']}): "
                        f"{result['error']}"
                    )
                    # Print first few errors immediately so problems are visible
                    if errors <= 5:
                        tqdm.write(f"  [error #{errors}] {result['market_id']}: {result['error'][:200]}")
                else:
                    try:
                        upsert_one(
                            cur,
                            result["market_id"],
                            result["source"],
                            result["ensemble_payload"],
                            result["base_rate"],
                        )
                        latencies.append(float(result["latency_s"]))
                        # Explicit commit every 25 successful upserts so kills
                        # don't roll back the entire session's work.
                        if len(latencies) % 25 == 0:
                            cur.connection.commit()
                    except Exception as exc:  # noqa: BLE001
                        errors += 1
                        tqdm.write(
                            f"  [upsert error] {result['market_id']} "
                            f"({result['source']}): {type(exc).__name__}: {exc}"
                        )
                pbar.update(1)
                if latencies:
                    p50 = statistics.median(latencies)
                    pbar.set_postfix(err=errors, p50=f"{p50:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill ensemble + base-rate features for historical markets."
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=100,
        help="total rows to backfill this run",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="simultaneous Wafer calls",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch the work queue and report; do not call Wafer or write to Snowflake",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="skip the cost-confirmation prompt",
    )
    ap.add_argument(
        "--only-categories",
        type=str,
        default=None,
        help="comma-separated category filter e.g. 'Sports,Entertainment'",
    )
    args = ap.parse_args()

    only_cats = [c.strip() for c in args.only_categories.split(",")] if args.only_categories else None

    t0 = time.time()

    print(f">>> Fetching work queue (limit={args.limit}" + (f", categories={only_cats}" if only_cats else "") + ") ...")
    try:
        rows = fetch_work_queue(args.limit, only_categories=only_cats)
    except Exception as exc:  # noqa: BLE001
        print(f"  failed to fetch work queue: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"  {len(rows)} rows in work queue")

    if not rows:
        print("Nothing to do. Exiting.")
        return

    if args.dry_run:
        print("--dry-run: skipping cost prompt, Wafer calls, and Snowflake writes.")
        # Show a tiny sample so the operator can sanity-check.
        preview = rows[: min(5, len(rows))]
        for r in preview:
            outcomes = _coerce_outcomes(r.get("outcomes"))
            qt = (r.get("question_text") or "").replace("\n", " ")
            if len(qt) > 120:
                qt = qt[:117] + "..."
            print(
                f"  - {r['market_id']:30}  source={r['source']:8}  "
                f"outcomes={outcomes}  q={qt!r}"
            )
        return

    if not _confirm_cost(args.limit, args.yes):
        print("Aborted by user.")
        return

    print(
        f">>> Backfilling {len(rows)} markets with concurrency={args.concurrency} "
        f"using models={BACKFILL_MODELS} ..."
    )
    asyncio.run(run_backfill(rows, args.concurrency))

    print(f"\n=== Backfill summary ===")
    print(f"  wall time (s): {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
