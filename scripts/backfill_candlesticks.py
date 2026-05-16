"""Backfill q_kalshi_at_open and q_kalshi_at_24h_pre via Kalshi /candlesticks.

`scripts/import_kalshi_history.py` lands resolved Kalshi markets with
`q_kalshi_at_open = q_kalshi_at_24h_pre = [0.5, 0.5]` placeholders because the
bulk import path doesn't include intra-market prices. Those placeholders are
useless as features for the downstream AutoML calibrator. This script fills
them in by hitting Kalshi's per-market candlesticks endpoint:

    GET /series/{series_ticker}/markets/{market_ticker}/candlesticks

For each placeholder row we fetch two windows:

  * Open price:    first candle in [open_ts, open_ts + 1h], period 60s
  * 24h-pre price: candle whose end_period_ts is closest to (resolve_ts - 24h)
                   within a +/- 30 min window around the target

The response uses *_dollars decimal-string fields in [0, 1]; we parse those
directly as Yes-probabilities and store [p_yes, 1 - p_yes]. Markets with no
candles in either window are left on the placeholder so a later run can retry.

URL note: the candlesticks endpoint is keyed on series_ticker, not
event_ticker, despite what `scripts/probe_kalshi_candlesticks.py` suggests.
Hitting `/events/{ev}/markets/{tk}/candlesticks` returns 404 for every market
we tested; `/series/{series}/markets/{tk}/candlesticks` returns 200.

CLI:
  python scripts/backfill_candlesticks.py --dry-run --limit 5
  python scripts/backfill_candlesticks.py --limit 500 --throttle 0.25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.snowflake_client import _load_env_once, snowflake_cursor  # noqa: E402

_load_env_once()


BASE = "https://api.elections.kalshi.com/trade-api/v2"
PROGRESS_EVERY = 25
DRY_RUN_PRINT_LIMIT = 5
ONE_HOUR_S = 3600
ONE_DAY_S = 86400
HALF_HOUR_S = 1800


# Selects only Kalshi rows still holding the [0.5, 0.5] placeholder. Using
# element-wise float compare instead of ARRAY_TO_STRING avoids any ambiguity
# about how Snowflake serialises 0.5 inside an ARRAY.
#
# `market_id` is the Kalshi market ticker (set in import_kalshi_history.py via
# `"market_id": m.get("ticker")`), and `series_ticker` is already a top-level
# column — no need to dig into raw_payload.
_WORK_QUEUE_SQL = """
SELECT
    market_id,
    source,
    series_ticker,
    open_ts,
    resolve_ts
FROM HISTORICAL_MARKETS
WHERE source = 'kalshi'
  AND realized_outcome IS NOT NULL
  AND series_ticker IS NOT NULL
  AND ARRAY_SIZE(q_kalshi_at_open) = 2
  AND q_kalshi_at_open[0]::FLOAT = 0.5
  AND q_kalshi_at_open[1]::FLOAT = 0.5
LIMIT %(limit)s
"""

# We only touch the two ARRAY columns — everything else on the row was set
# by import_kalshi_history.py and remains authoritative.
_MERGE_SQL = """
MERGE INTO HISTORICAL_MARKETS t
USING (
    SELECT
        %(market_id)s                 AS market_id,
        %(source)s                    AS source,
        PARSE_JSON(%(q_open_json)s)   AS q_open,
        PARSE_JSON(%(q_24h_json)s)    AS q_24h
) s
ON t.market_id = s.market_id AND t.source = s.source
WHEN MATCHED THEN UPDATE SET
    q_kalshi_at_open    = s.q_open,
    q_kalshi_at_24h_pre = s.q_24h
"""


# ---------------------------------------------------------------------------
# HTTP — throttled, 429-aware, 404-tolerant.
# ---------------------------------------------------------------------------

def _get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    throttle_s: float,
    max_retries: int = 5,
) -> dict[str, Any] | None:
    """GET with throttle + 429 backoff. Returns None on 404 so callers can skip
    cleanly (some markets vanish from /candlesticks even though /markets still
    lists them — usually delisted exotic instruments)."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        time.sleep(throttle_s)
        try:
            r = client.get(url, params=params, timeout=30.0)
        except httpx.HTTPError as e:
            sleep_for = delay + random.uniform(0, 0.5)
            print(f"  network error {e!r} — sleeping {sleep_for:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)
            continue
        if r.status_code == 429:
            sleep_for = delay + random.uniform(0, 0.5)
            print(f"  429 — backing off {sleep_for:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    msg = f"giving up on {url} after {max_retries} retries"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Candle parsing.
# ---------------------------------------------------------------------------
#
# As of this writing the Kalshi candlesticks response uses *_dollars fields
# (decimal strings in [0, 1]) inside `price`, `yes_bid`, `yes_ask`:
#
#   {"end_period_ts": 1776186000,
#    "price":   {"close_dollars": "0.7000", "open_dollars": "0.6900", ...},
#    "yes_bid": {"close_dollars": "0.6700", ...},
#    "yes_ask": {"close_dollars": "0.7500", ...},
#    "volume_fp": "281.00", ...}
#
# In zero-trade buckets `price` can be empty (`{}`) while yes_bid/yes_ask are
# still populated, so we fall back to a mid-quote. We also tolerate the legacy
# integer-cents form (`"close": 70`) just in case Kalshi flips back.

_PRICE_FIELDS = ("close_dollars", "close")
_QUOTE_FIELDS = ("close_dollars", "close")


def _prob_from_candle(candle: dict[str, Any]) -> float | None:
    """Return a Yes-probability in [0, 1] for one candle, or None if unparseable."""
    trade = _extract_price(candle.get("price"), _PRICE_FIELDS)
    if trade is not None:
        return _clip01(trade)

    bid = _extract_price(candle.get("yes_bid"), _QUOTE_FIELDS)
    ask = _extract_price(candle.get("yes_ask"), _QUOTE_FIELDS)
    if bid is not None and ask is not None:
        return _clip01((bid + ask) / 2.0)
    if ask is not None:
        return _clip01(ask)
    if bid is not None:
        return _clip01(bid)
    return None


def _extract_price(obj: Any, fields: tuple[str, ...]) -> float | None:
    """Return a probability (0..1) from a nested price/quote dict.

    Prefers *_dollars (already in [0, 1]); falls back to bare cents (0..100)
    and divides by 100. Returns None if nothing parseable is present.
    """
    if not isinstance(obj, dict):
        return None
    for f in fields:
        if f not in obj:
            continue
        v = _to_float(obj.get(f))
        if v is None:
            continue
        # Heuristic: *_dollars is already in [0, 1]; bare int is cents.
        return v if f.endswith("_dollars") else v / 100.0
    return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clip01(p: float) -> float:
    return max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Per-market fetches.
# ---------------------------------------------------------------------------

def _candles_url(series_ticker: str, market_ticker: str) -> str:
    return f"{BASE}/series/{series_ticker}/markets/{market_ticker}/candlesticks"


def fetch_open_prob(
    client: httpx.Client,
    series_ticker: str,
    market_ticker: str,
    open_ts_unix: int,
    throttle_s: float,
) -> float | None:
    """First usable candle close in the first hour of trading."""
    url = _candles_url(series_ticker, market_ticker)
    params = {
        "start_ts": open_ts_unix,
        "end_ts": open_ts_unix + ONE_HOUR_S,
        "period_interval": 60,
    }
    data = _get_with_retry(client, url, params, throttle_s)
    if data is None:
        return None
    for c in data.get("candlesticks") or []:
        p = _prob_from_candle(c)
        if p is not None:
            return p
    return None


def fetch_24h_pre_prob(
    client: httpx.Client,
    series_ticker: str,
    market_ticker: str,
    resolve_ts_unix: int,
    throttle_s: float,
) -> float | None:
    """Candle whose end_period_ts is closest to (resolve - 24h)."""
    target = resolve_ts_unix - ONE_DAY_S
    url = _candles_url(series_ticker, market_ticker)
    params = {
        "start_ts": target - HALF_HOUR_S,
        "end_ts": target + HALF_HOUR_S,
        "period_interval": 60,
    }
    data = _get_with_retry(client, url, params, throttle_s)
    if data is None:
        return None
    candles = data.get("candlesticks") or []
    best_prob: float | None = None
    best_dist: int | None = None
    for c in candles:
        end_raw = c.get("end_period_ts")
        try:
            end_ts = int(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end_ts = None
        if end_ts is None:
            continue
        p = _prob_from_candle(c)
        if p is None:
            continue
        dist = abs(end_ts - target)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_prob = p
    return best_prob


# ---------------------------------------------------------------------------
# Snowflake helpers.
# ---------------------------------------------------------------------------

def fetch_work_queue(cur: Any, limit: int) -> list[dict[str, Any]]:
    cur.execute(_WORK_QUEUE_SQL, {"limit": limit})
    cols = [c[0].lower() for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _flush_batch(cur: Any, batch: list[dict[str, Any]]) -> None:
    for params in batch:
        cur.execute(_MERGE_SQL, params)
    print(f"  flushed {len(batch)} MERGEs")


def _ts_to_unix(dt: Any) -> int | None:
    """Snowflake TIMESTAMP_NTZ rows come back as naive datetimes. They were
    stored via `.astimezone(tz=None).replace(tzinfo=None)` in the importer,
    i.e. machine-local time; `datetime.timestamp()` on a naive value uses the
    local tz too, so the round-trip yields the original unix ts."""
    if dt is None:
        return None
    try:
        return int(dt.timestamp())
    except (AttributeError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1000,
                    help="max work-queue rows to process per run (default: 1000)")
    ap.add_argument("--throttle", type=float, default=0.25,
                    help="seconds to sleep between Kalshi requests (default: 0.25)")
    ap.add_argument("--commit-every", type=int, default=50,
                    help="MERGE rows per Snowflake batch flush (default: 50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch candles + print decisions only; no Snowflake writes")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    with snowflake_cursor() as cur:
        print(f">>> Loading work queue (limit={args.limit})...")
        rows = fetch_work_queue(cur, args.limit)
        n_total = len(rows)
        print(f"  found {n_total} markets still on the [0.5, 0.5] placeholder")
        if n_total == 0:
            print("Nothing to do.")
            return 0

        n_success = 0
        n_no_series = 0
        n_no_window = 0
        n_no_open = 0
        n_no_24h = 0
        n_errors = 0
        n_short_window = 0  # market shorter than 24h; reused open price
        n_dry_printed = 0
        batch: list[dict[str, Any]] = []
        t0 = time.time()

        with httpx.Client() as client:
            with tqdm(rows, desc="candlestick backfill", unit="mkt", dynamic_ncols=True) as pbar:
                for idx, row in enumerate(pbar, start=1):
                    market_id = row["market_id"]
                    source = row["source"]
                    series_ticker = row.get("series_ticker")
                    open_ts_unix = _ts_to_unix(row.get("open_ts"))
                    resolve_ts_unix = _ts_to_unix(row.get("resolve_ts"))
                    pbar.set_postfix(ok=n_success, skip=n_no_open + n_no_24h, err=n_errors)

                    if not series_ticker or not market_id:
                        n_no_series += 1
                        continue
                    if open_ts_unix is None or resolve_ts_unix is None:
                        n_no_window += 1
                        continue

                    try:
                        p_open = fetch_open_prob(
                            client, series_ticker, market_id,
                            open_ts_unix, args.throttle,
                        )
                    except Exception as e:  # noqa: BLE001
                        n_errors += 1
                        tqdm.write(f"  {market_id}: open fetch errored: {e}")
                        continue

                    if p_open is None:
                        n_no_open += 1
                        continue

                    # 24h-pre window. If the market lived for <24h, fall back to
                    # the open price rather than scanning a window that overlaps
                    # the open / pre-open period.
                    src_24h: str
                    if resolve_ts_unix - open_ts_unix < ONE_DAY_S:
                        p_24h = p_open
                        src_24h = "open-fallback"
                        n_short_window += 1
                    else:
                        try:
                            p_24h = fetch_24h_pre_prob(
                                client, series_ticker, market_id,
                                resolve_ts_unix, args.throttle,
                            )
                        except Exception as e:  # noqa: BLE001
                            n_errors += 1
                            tqdm.write(f"  {market_id}: 24h-pre fetch errored: {e}")
                            continue
                        if p_24h is None:
                            n_no_24h += 1
                            continue
                        src_24h = "candle"

                    # If the real open mid is exactly 0.5 (typically a no-trade
                    # candle with bid=0/ask=1), nudge by 1e-6 so the row no longer
                    # matches the work-queue's `= 0.5` placeholder filter and we
                    # don't loop on it forever. 1e-6 is below any modelling
                    # precision we'll ever care about.
                    p_open_w = 0.500001 if p_open == 0.5 else p_open
                    q_open = [round(p_open_w, 6), round(1.0 - p_open_w, 6)]
                    q_24h = [round(p_24h, 6), round(1.0 - p_24h, 6)]

                    if args.dry_run:
                        if n_dry_printed < DRY_RUN_PRINT_LIMIT:
                            tqdm.write(
                                f"  {market_id} (series={series_ticker}): "
                                f"q_open={q_open}  q_24h={q_24h}  src_24h={src_24h}"
                            )
                            n_dry_printed += 1
                        n_success += 1
                        if n_dry_printed >= DRY_RUN_PRINT_LIMIT:
                            tqdm.write(f"  (dry-run print cap {DRY_RUN_PRINT_LIMIT} reached; stopping early)")
                            break
                        continue

                    batch.append({
                        "market_id": market_id,
                        "source": source,
                        "q_open_json": json.dumps(q_open),
                        "q_24h_json": json.dumps(q_24h),
                    })
                    n_success += 1

                    if len(batch) >= args.commit_every:
                        _flush_batch(cur, batch)
                        batch.clear()

                    if (idx % PROGRESS_EVERY) == 0:
                        pbar.set_postfix(ok=n_success, no_open=n_no_open, no_24h=n_no_24h, err=n_errors)

            if batch and not args.dry_run:
                _flush_batch(cur, batch)

        elapsed = time.time() - t0
        print("\n=== Backfill summary ===")
        print(f"  {'processed':32} {n_total}")
        print(f"  {'updated (or would update)':32} {n_success}")
        print(f"  {'skipped: missing series/market':32} {n_no_series}")
        print(f"  {'skipped: missing open/resolve':32} {n_no_window}")
        print(f"  {'skipped: no candle near open':32} {n_no_open}")
        print(f"  {'skipped: no candle near 24h-pre':32} {n_no_24h}")
        print(f"  {'short window (<24h, reused open)':32} {n_short_window}")
        print(f"  {'errors':32} {n_errors}")
        print(f"  {'wall time (s)':32} {elapsed:.1f}")
        if args.dry_run:
            print("  (dry-run: no Snowflake writes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
