"""Probe Wafer.ai: list models, then try a chat completion with each.

Helps us pin down the canonical model IDs Wafer expects.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

KEY = os.environ["WAFER_API_KEY"].strip()
BASE = "https://pass.wafer.ai/v1"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def list_models() -> list[str]:
    r = httpx.get(f"{BASE}/models", headers=HEADERS, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    print("=== GET /v1/models ===")
    print(json.dumps(data, indent=2)[:2000])
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    print(f"\n=> {len(ids)} model id(s): {ids}\n")
    return ids


def try_chat(model: str) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly the word 'hello' and nothing else."}
        ],
        "max_tokens": 20,
        "temperature": 0.1,
    }
    print(f"--- POST /chat/completions  model={model!r} ---")
    try:
        r = httpx.post(f"{BASE}/chat/completions", json=body, headers=HEADERS, timeout=60.0)
        print(f"status={r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2)[:1200])
        except Exception:
            print(r.text[:600])
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
    print()


def main() -> None:
    ids = list_models()
    sys.stdout.flush()
    for m in ids[:6]:  # cap so we don't spam
        if m:
            try_chat(m)


if __name__ == "__main__":
    main()
