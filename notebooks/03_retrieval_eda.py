"""Phase 3 — Retrieval / Base-Rate sanity check.

Sanity-checks `get_base_rate` against live Snowflake data.

Expected early in the project: HISTORICAL_MARKETS is empty or has no embeddings
yet, so neighbor_count=0 and the uniform prior is returned. Once
`scripts/import_kalshi_history.py` + `scripts/embed_questions.py` have run,
this script should surface semantically reasonable neighbors.

Usage (from repo root):
    python notebooks/03_retrieval_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from prophet_agent.retrieval.base_rate import BaseRateFeatures, get_base_rate  # noqa: E402


SAMPLE_QUESTIONS = [
    {
        "question": "Will Bitcoin close above $100,000 USD on the last day of this year?",
        "outcomes": ["Yes", "No"],
        "category": "crypto",
    },
    {
        "question": "Will the Federal Reserve cut rates at the next FOMC meeting?",
        "outcomes": ["Yes", "No"],
        "category": "econ",
    },
    {
        "question": "Will the Lakers win the NBA championship this season?",
        "outcomes": ["Yes", "No"],
        "category": "sports",
    },
    {
        "question": "Will a major US presidential candidate withdraw before election day?",
        "outcomes": ["Yes", "No"],
        "category": "politics",
    },
]


def show(features: BaseRateFeatures) -> None:
    print(f"  neighbor_count  : {features.neighbor_count}")
    print(f"  mean_similarity : {features.mean_similarity:.4f}")
    print(f"  base_rate       : {[round(x, 4) for x in features.base_rate]}")
    print(f"  kalshi_residual : {[round(x, 4) for x in features.kalshi_residual]}")
    for n in features.neighbors[:5]:
        sim = n.get("similarity")
        sim_s = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
        print(
            f"    - [{sim_s}] outcome={n.get('realized_outcome')} "
            f":: {n.get('question_text')}"
        )


def main() -> None:
    print("=== With category filter ===")
    for sample in SAMPLE_QUESTIONS:
        print("=" * 80)
        print(f"Q: {sample['question']}")
        print(f"   outcomes={sample['outcomes']}  category={sample['category']}")
        feats = get_base_rate(
            sample["question"],
            sample["outcomes"],
            category=sample["category"],
            k=5,
        )
        show(feats)

    print("\n=== No category filter (first 2 questions) ===")
    for sample in SAMPLE_QUESTIONS[:2]:
        print("=" * 80)
        print(f"Q (no cat): {sample['question']}")
        feats = get_base_rate(sample["question"], sample["outcomes"], category=None, k=10)
        show(feats)

    print(
        "\nWhat to look for once Phase 1 data lands:\n"
        "  - Crypto questions should pull crypto neighbors (BTC/ETH price markets).\n"
        "  - mean_similarity > ~0.6 is healthy; < ~0.3 means we're matching noise.\n"
        "  - kalshi_residual near 0 → well-calibrated market; consistent bias → signal\n"
        "    the calibrator should learn to exploit."
    )


if __name__ == "__main__":
    main()
