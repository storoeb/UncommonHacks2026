"""Probe the Kalshi events endpoint by category — find clean settled events."""

from __future__ import annotations

import json
from collections import Counter

import httpx


BASE = "https://api.elections.kalshi.com/trade-api/v2"


def categories_and_counts() -> None:
    """Page through settled events, tally categories."""
    cursor = None
    cat_counts: Counter[str] = Counter()
    series_counts: Counter[str] = Counter()
    sample_per_cat: dict[str, dict] = {}
    pages = 0
    while pages < 20:
        params = {"status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = httpx.get(f"{BASE}/events", params=params, timeout=30.0)
        if r.status_code != 200:
            print(f"status {r.status_code}: {r.text[:300]}")
            return
        data = r.json()
        events = data.get("events") or []
        for ev in events:
            cat = ev.get("category") or "(none)"
            cat_counts[cat] += 1
            series_counts[ev.get("series_ticker") or "(none)"] += 1
            sample_per_cat.setdefault(cat, ev)
        cursor = data.get("cursor")
        pages += 1
        if not cursor or not events:
            break

    print(f"Pages walked: {pages}")
    print(f"Total settled events seen: {sum(cat_counts.values())}\n")
    print("=== Category counts ===")
    for cat, n in cat_counts.most_common():
        print(f"  {cat:20}  {n}")
    print("\n=== Top 20 series ===")
    for s, n in series_counts.most_common(20):
        print(f"  {s:40}  {n}")
    print("\n=== Sample event per category ===")
    for cat, ev in sample_per_cat.items():
        print(f"  [{cat}] {ev.get('title')}  ({ev.get('event_ticker')})")


if __name__ == "__main__":
    categories_and_counts()
