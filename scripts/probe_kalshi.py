"""Probe Kalshi public API to confirm endpoint, fields, and pagination."""

from __future__ import annotations

import json

import httpx


BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Alternative: https://external-api.kalshi.com/trade-api/v2


def probe_markets(status: str = "settled", limit: int = 5) -> None:
    url = f"{BASE}/markets"
    params = {"status": status, "limit": limit}
    print(f"GET {url}  params={params}")
    r = httpx.get(url, params=params, timeout=30.0)
    print(f"status={r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return
    data = r.json()
    print(f"keys: {list(data.keys())}")
    markets = data.get("markets") or data.get("data") or []
    print(f"n markets in page: {len(markets)}")
    cursor = data.get("cursor")
    print(f"cursor: {cursor!r}\n")
    if markets:
        print("=== Field names on first market ===")
        print(sorted(markets[0].keys()))
        print("\n=== First market (full) ===")
        print(json.dumps(markets[0], indent=2)[:3000])
        print("\n=== Distinct series_ticker prefixes in this page ===")
        prefixes: dict[str, int] = {}
        for m in markets:
            t = m.get("ticker", "")
            # Series ticker is everything before the first dash usually.
            pref = t.split("-")[0] if "-" in t else t[:6]
            prefixes[pref] = prefixes.get(pref, 0) + 1
        for k, v in sorted(prefixes.items()):
            print(f"  {k}: {v}")


def probe_events(limit: int = 5) -> None:
    url = f"{BASE}/events"
    params = {"limit": limit, "status": "settled"}
    print(f"\nGET {url}  params={params}")
    r = httpx.get(url, params=params, timeout=30.0)
    print(f"status={r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return
    data = r.json()
    print(f"keys: {list(data.keys())}")
    events = data.get("events") or []
    if events:
        print(f"=== First event keys ===")
        print(sorted(events[0].keys()))
        print("\n=== First event (snippet) ===")
        print(json.dumps(events[0], indent=2)[:1500])


if __name__ == "__main__":
    probe_markets()
    probe_events()
