"""Import resolved Kalshi markets into HISTORICAL_MARKETS.

Strategy:
  Phase A: Page all settled EVENTS (one paginated stream). Build
           event_ticker -> {category, series_ticker, title} dict in memory.
           This avoids per-event API calls.
  Phase B: Page all settled MARKETS (one paginated stream). For each market,
           join against the event dict. Skip:
             - market_type != 'binary'
             - is_provisional
             - result not in {yes, no}
             - high-frequency series (15M, 30M, MVE, etc.)
             - excluded categories (Sports, Crypto for v1)
           MERGE qualifying markets into HISTORICAL_MARKETS in batches.

Throttle: each request sleeps `--throttle` seconds (default 0.25 = ~4 req/s).
Retries: 429 responses get exponential backoff (1s, 2s, 4s, 8s).

v1 limitations (tracked as TODOs):
  - q_kalshi_at_open / q_kalshi_at_24h_pre are stored as [0.5, 0.5] placeholders
    until backfilled from the /candlesticks endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import httpx
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.snowflake_client import _load_env_once, snowflake_cursor

_load_env_once()


BASE = "https://api.elections.kalshi.com/trade-api/v2"

INCLUDE_CATEGORIES = frozenset({
    "Elections", "Politics", "Economics", "Climate and Weather",
    "Science and Technology", "Financials", "Companies", "World", "Mentions",
    "Entertainment", "Health", "Social", "Commodities", "Education",
    "Sports",  # added: Prophet Arena eval dataset is sports-heavy
})

HIGH_FREQ_MARKERS = ("15M", "30M", "5M", "1H", "MVE", "MULTIVARIATE")


# ---------------------------------------------------------------------------
# HTTP helpers — throttled + 429-retrying.
# ---------------------------------------------------------------------------

def _get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    throttle_s: float,
    max_retries: int = 5,
) -> dict[str, Any]:
    delay = 1.0
    for attempt in range(max_retries + 1):
        time.sleep(throttle_s)
        r = client.get(url, params=params, timeout=30.0)
        if r.status_code == 429:
            sleep_for = delay + random.uniform(0, 0.5)
            print(f"  429 — backing off {sleep_for:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)
            continue
        r.raise_for_status()
        return r.json()
    msg = f"giving up on {url} after {max_retries} retries"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Phase A: bulk event listing.
# ---------------------------------------------------------------------------

def fetch_all_events(
    client: httpx.Client,
    throttle_s: float,
    max_pages: int = 30,
    page_limit: int = 200,
) -> dict[str, dict[str, Any]]:
    """Return event_ticker -> {category, series_ticker, title}. Paginates settled events."""
    out: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    pages = 0
    while pages < max_pages:
        params: dict[str, Any] = {"status": "settled", "limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        data = _get_with_retry(client, f"{BASE}/events", params, throttle_s)
        evs = data.get("events") or []
        for ev in evs:
            tk = ev.get("event_ticker")
            if not tk:
                continue
            out[tk] = {
                "category": ev.get("category"),
                "series_ticker": ev.get("series_ticker"),
                "title": ev.get("title"),
            }
        cursor = data.get("cursor")
        pages += 1
        print(f"  events page {pages}: +{len(evs)} (total {len(out)})")
        if not cursor or not evs:
            break
    return out


# ---------------------------------------------------------------------------
# Phase B: bulk market listing + transform.
# ---------------------------------------------------------------------------

def _is_high_freq(ticker: str | None) -> bool:
    if not ticker:
        return False
    up = ticker.upper()
    return any(m in up for m in HIGH_FREQ_MARKERS)


def _iso_to_naive_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.astimezone(tz=None).replace(tzinfo=None)


def market_to_row(
    m: dict[str, Any],
    event_info: dict[str, Any],
) -> dict[str, Any] | None:
    if m.get("market_type") != "binary":
        return None
    if m.get("is_provisional"):
        return None
    result = (m.get("result") or "").lower()
    if result not in ("yes", "no"):
        return None
    if _is_high_freq(m.get("ticker")) or _is_high_freq(event_info.get("series_ticker")):
        return None
    category = event_info.get("category")
    if category not in INCLUDE_CATEGORIES:
        return None

    title = m.get("title") or event_info.get("title") or m.get("ticker")
    # Kalshi sometimes generates comma-joined exotic titles; fall back to event title.
    if title and "," in title and "Target Price" in title:
        title = event_info.get("title") or m.get("ticker")

    return {
        "market_id": m.get("ticker"),
        "source": "kalshi",
        "category": category,
        "series_ticker": event_info.get("series_ticker"),
        "question_text": title,
        "outcomes": ["Yes", "No"],
        "open_ts": _iso_to_naive_utc(m.get("open_time")),
        "resolve_ts": _iso_to_naive_utc(m.get("settlement_ts") or m.get("close_time")),
        "q_kalshi_at_open": [0.5, 0.5],     # TODO: backfill via /candlesticks
        "q_kalshi_at_24h_pre": [0.5, 0.5],  # TODO: backfill via /candlesticks
        "realized_outcome": 0 if result == "yes" else 1,
        "raw_payload": m,
    }


def iter_markets(
    client: httpx.Client,
    throttle_s: float,
    max_pages: int = 80,
    page_limit: int = 200,
) -> Iterator[dict[str, Any]]:
    """Bulk-paginate all settled markets. Useful when you want a wide sweep,
    but expect heavy filtering — recent settled markets are dominated by
    high-frequency auto-settle instruments."""
    cursor: str | None = None
    pages = 0
    while pages < max_pages:
        params: dict[str, Any] = {"status": "settled", "limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        data = _get_with_retry(client, f"{BASE}/markets", params, throttle_s)
        ms = data.get("markets") or []
        for m in ms:
            yield m
        cursor = data.get("cursor")
        pages += 1
        print(f"  markets page {pages}: +{len(ms)}")
        if not cursor or not ms:
            return


def iter_markets_per_event(
    client: httpx.Client,
    event_tickers: list[str],
    throttle_s: float,
    page_limit: int = 200,
) -> Iterator[dict[str, Any]]:
    """For each event ticker, fetch its markets. Use this when you have a
    pre-filtered event list (our case) — far more efficient than bulk-paging
    settled markets and filtering."""
    for idx, ev_tk in enumerate(event_tickers):
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "event_ticker": ev_tk,
                "status": "settled",
                "limit": page_limit,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                data = _get_with_retry(client, f"{BASE}/markets", params, throttle_s)
            except Exception as e:  # noqa: BLE001
                print(f"  event {ev_tk}: fetch failed ({e}); skipping")
                break
            ms = data.get("markets") or []
            for m in ms:
                yield m
            cursor = data.get("cursor")
            if not cursor:
                break
        if (idx + 1) % 50 == 0:
            print(f"  fetched markets for {idx + 1}/{len(event_tickers)} events")


# ---------------------------------------------------------------------------
# Snowflake upsert.
# ---------------------------------------------------------------------------

_MERGE_SQL = """
MERGE INTO HISTORICAL_MARKETS t
USING (
    SELECT
        %(market_id)s        AS market_id,
        %(source)s           AS source,
        %(category)s         AS category,
        %(series_ticker)s    AS series_ticker,
        %(question_text)s    AS question_text,
        PARSE_JSON(%(outcomes_json)s)            AS outcomes,
        %(open_ts)s          AS open_ts,
        %(resolve_ts)s       AS resolve_ts,
        PARSE_JSON(%(q_open_json)s)              AS q_kalshi_at_open,
        PARSE_JSON(%(q_24h_json)s)               AS q_kalshi_at_24h_pre,
        %(realized_outcome)s AS realized_outcome,
        PARSE_JSON(%(raw_payload_json)s)         AS raw_payload
) s
ON t.market_id = s.market_id AND t.source = s.source
WHEN MATCHED THEN UPDATE SET
    category            = s.category,
    series_ticker       = s.series_ticker,
    question_text       = s.question_text,
    outcomes            = s.outcomes,
    open_ts             = s.open_ts,
    resolve_ts          = s.resolve_ts,
    q_kalshi_at_open    = s.q_kalshi_at_open,
    q_kalshi_at_24h_pre = s.q_kalshi_at_24h_pre,
    realized_outcome    = s.realized_outcome,
    raw_payload         = s.raw_payload
WHEN NOT MATCHED THEN INSERT (
    market_id, source, category, series_ticker, question_text, outcomes,
    open_ts, resolve_ts, q_kalshi_at_open, q_kalshi_at_24h_pre,
    realized_outcome, raw_payload
) VALUES (
    s.market_id, s.source, s.category, s.series_ticker, s.question_text, s.outcomes,
    s.open_ts, s.resolve_ts, s.q_kalshi_at_open, s.q_kalshi_at_24h_pre,
    s.realized_outcome, s.raw_payload
)
"""


def upsert_rows(cur: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        params = {
            "market_id": row["market_id"],
            "source": row["source"],
            "category": row["category"],
            "series_ticker": row["series_ticker"],
            "question_text": row["question_text"],
            "outcomes_json": json.dumps(row["outcomes"]),
            "open_ts": row["open_ts"],
            "resolve_ts": row["resolve_ts"],
            "q_open_json": json.dumps(row["q_kalshi_at_open"]),
            "q_24h_json": json.dumps(row["q_kalshi_at_24h_pre"]),
            "realized_outcome": row["realized_outcome"],
            "raw_payload_json": json.dumps(row["raw_payload"], default=str),
        }
        cur.execute(_MERGE_SQL, params)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500,
                    help="cap total markets ingested")
    ap.add_argument("--throttle", type=float, default=0.25,
                    help="seconds to sleep between API requests")
    ap.add_argument("--commit-every", type=int, default=50,
                    help="rows per Snowflake batch")
    ap.add_argument("--event-pages", type=int, default=30)
    ap.add_argument("--market-pages", type=int, default=80,
                    help="(bulk strategy only) max market pages to walk")
    ap.add_argument("--strategy", choices=("per-event", "bulk"), default="per-event",
                    help="per-event: fetch /markets?event_ticker=X for each interesting event "
                         "(efficient, deterministic). bulk: paginate all settled markets and "
                         "filter (wide but most pages are garbage).")
    ap.add_argument("--only-category", type=str, default=None,
                    help="restrict import to a single category (e.g. Sports)")
    args = ap.parse_args()

    t0 = time.time()
    with httpx.Client() as client:
        print(">>> Phase A: listing settled events")
        events = fetch_all_events(client, args.throttle, max_pages=args.event_pages)
        print(f"  total events cached: {len(events)}")
        # Quick distribution by category
        cat_counts: dict[str, int] = {}
        for info in events.values():
            cat_counts[info.get("category") or "(none)"] = cat_counts.get(info.get("category") or "(none)", 0) + 1
        print("  events by category:")
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:25}  {n}")
        keep_cats = {c for c in cat_counts if c in INCLUDE_CATEGORIES}
        kept_events = sum(cat_counts.get(c, 0) for c in keep_cats)
        print(f"  events in included categories: {kept_events}")

        # Build the list of event tickers to fetch (per-event strategy).
        interesting_events = [
            tk for tk, info in events.items()
            if info.get("category") in INCLUDE_CATEGORIES
        ]
        # --only-category filter: restrict to a single category
        if args.only_category:
            interesting_events = [
                tk for tk in interesting_events
                if events[tk].get("category") == args.only_category
            ]
            print(f"  --only-category={args.only_category}: {len(interesting_events)} events")
        # Bias toward smaller, higher-signal categories first so we don't burn
        # all our budget on Entertainment.
        priority = {"Elections": 0, "Politics": 1, "Economics": 2,
                    "Science and Technology": 3, "Financials": 4, "World": 5,
                    "Sports": 6, "Companies": 7, "Climate and Weather": 8, "Health": 9,
                    "Commodities": 10, "Education": 11, "Social": 12,
                    "Mentions": 13, "Entertainment": 14}
        interesting_events.sort(key=lambda tk: priority.get(
            events[tk].get("category") or "", 99))
        print(f"  interesting events to fetch: {len(interesting_events)}")

        print(f"\n>>> Phase B: fetching markets, strategy={args.strategy}")
        if args.strategy == "per-event":
            market_iter = iter_markets_per_event(client, interesting_events, args.throttle)
        else:
            market_iter = iter_markets(client, args.throttle, max_pages=args.market_pages)

        kept = 0
        seen_markets = 0
        batch: list[dict[str, Any]] = []
        by_cat: dict[str, int] = {}
        with snowflake_cursor() as cur:
            with tqdm(total=args.limit, desc="markets kept", unit="mkt", dynamic_ncols=True) as pbar:
                for m in market_iter:
                    seen_markets += 1
                    ev_tk = m.get("event_ticker")
                    if not ev_tk or ev_tk not in events:
                        continue
                    row = market_to_row(m, events[ev_tk])
                    if row is None:
                        continue
                    batch.append(row)
                    kept += 1
                    by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1
                    pbar.update(1)
                    pbar.set_postfix(seen=seen_markets, cat=row["category"][:8])
                    if len(batch) >= args.commit_every:
                        upsert_rows(cur, batch)
                        batch.clear()
                    if kept >= args.limit:
                        break
                if batch:
                    upsert_rows(cur, batch)

    print("\n=== Import summary ===")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:25}  {n}")
    print(f"  {'TOTAL KEPT':25}  {kept}")
    print(f"  {'TOTAL SEEN':25}  {seen_markets}")
    print(f"  {'WALL TIME (s)':25}  {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
