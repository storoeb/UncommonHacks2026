"""Probe Kalshi candlesticks endpoint for historical pricing.

Picks a non-exotic settled market from a real category (Elections / Politics /
Sports / Crypto) and fetches its price history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx


BASE = "https://api.elections.kalshi.com/trade-api/v2"


def find_real_settled_market() -> dict | None:
    """Scan settled markets until we find one with non-zero volume + simple binary type."""
    cursor = None
    for page in range(5):
        params = {"status": "settled", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        r = httpx.get(f"{BASE}/markets", params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        for m in data.get("markets", []):
            volume = float(m.get("volume_fp", 0) or 0)
            if (
                m.get("market_type") == "binary"
                and not m.get("is_provisional", False)
                and "MVE" not in m.get("ticker", "")
                and "MULTIVARIATE" not in m.get("ticker", "").upper()
                and volume > 100  # any real activity
            ):
                return m
        cursor = data.get("cursor")
        if not cursor:
            break
        print(f"  paging past {page+1}...")
    return None


def fetch_candlesticks(ticker: str, event_ticker: str, start_ts: int, end_ts: int) -> None:
    # The endpoint is /events/{event_ticker}/markets/{market_ticker}/candlesticks
    # per https://docs.kalshi.com/getting_started/quick_start_market_data
    url = f"{BASE}/events/{event_ticker}/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60}
    print(f"\nGET {url}")
    print(f"  params={params}")
    r = httpx.get(url, params=params, timeout=30.0)
    print(f"  status={r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return
    data = r.json()
    print(f"  keys: {list(data.keys())}")
    candles = data.get("candlesticks") or []
    print(f"  n candles: {len(candles)}")
    if candles:
        print(f"  === First candle ===")
        print(json.dumps(candles[0], indent=2)[:600])
        print(f"  === Last candle ===")
        print(json.dumps(candles[-1], indent=2)[:600])


def main() -> None:
    print("Searching for a real settled binary market with non-zero volume...")
    m = find_real_settled_market()
    if not m:
        print("Could not find one in first few pages")
        return
    print("\n=== Picked market ===")
    print(f"  ticker:      {m.get('ticker')}")
    print(f"  event:       {m.get('event_ticker')}")
    print(f"  title:       {m.get('title')}")
    print(f"  open_time:   {m.get('open_time')}")
    print(f"  close_time:  {m.get('close_time')}")
    print(f"  result:      {m.get('result')}")
    print(f"  volume_fp:   {m.get('volume_fp')}")

    # Convert times to unix ts for the candlestick query.
    def iso_to_ts(s: str) -> int:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

    start_ts = iso_to_ts(m["open_time"])
    end_ts = iso_to_ts(m["close_time"])
    if end_ts - start_ts > 60 * 60 * 24 * 7:
        # Limit window to ~1 week to avoid massive payloads
        end_ts = start_ts + 60 * 60 * 24
    fetch_candlesticks(m["ticker"], m["event_ticker"], start_ts, end_ts)


if __name__ == "__main__":
    main()
