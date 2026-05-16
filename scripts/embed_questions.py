"""Backfill question_embedding for HISTORICAL_MARKETS rows.

Computes embeddings entirely inside Snowflake via Cortex
(`SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', ...)`) so
no question text leaves the warehouse.

Designed to run after `scripts/import_kalshi_history.py` lands rows. Idempotent:
only touches rows where `question_embedding IS NULL`. Safe to re-run.

Usage:
    python scripts/embed_questions.py
    python scripts/embed_questions.py --batch-size 500 --max-batches 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Make src/ importable when running as `python scripts/embed_questions.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.snowflake_client import _load_env_once, snowflake_cursor

_load_env_once()


EMBED_MODEL = "snowflake-arctic-embed-m-v1.5"

# Single batched UPDATE. The IN-subquery limits the number of rows updated per
# call without needing OFFSET/cursors. Snowflake re-evaluates the subquery each
# call, so as rows get embedded they drop out of the candidate set.
UPDATE_SQL = """
UPDATE HISTORICAL_MARKETS
SET question_embedding =
    SNOWFLAKE.CORTEX.EMBED_TEXT_768(%(model)s, question_text)
WHERE question_embedding IS NULL
  AND question_text IS NOT NULL
  AND market_id IN (
      SELECT market_id FROM HISTORICAL_MARKETS
      WHERE question_embedding IS NULL AND question_text IS NOT NULL
      LIMIT %(batch_size)s
  )
"""

COUNT_MISSING_SQL = """
SELECT COUNT(*) FROM HISTORICAL_MARKETS
WHERE question_embedding IS NULL AND question_text IS NOT NULL
"""

VERIFY_SQL = """
SELECT
    COUNT(*)                                       AS total_rows,
    COUNT(question_embedding)                      AS embedded,
    COUNT(*) - COUNT(question_embedding)           AS missing
FROM HISTORICAL_MARKETS
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=500,
        help="Rows to embed per UPDATE batch (default: 500).",
    )
    p.add_argument(
        "--max-batches",
        type=positive_int,
        default=50,
        help="Safety cap on number of batches per run (default: 50).",
    )
    return p.parse_args(argv)


def positive_int(value: str) -> int:
    """argparse type: positive integer."""
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"expected an integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if n <= 0:
        msg = f"expected a positive integer, got {n}"
        raise argparse.ArgumentTypeError(msg)
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch_size: int = args.batch_size
    max_batches: int = args.max_batches

    total_updated = 0
    start = time.monotonic()

    with snowflake_cursor() as cur:
        # Fast-path: nothing to embed.
        cur.execute(COUNT_MISSING_SQL)
        row = cur.fetchone()
        missing_before = int(row[0]) if row and row[0] is not None else 0
        if missing_before == 0:
            print("0 rows needed embedding. Nothing to do.")
            _print_verification(cur)
            return 0

        print(
            f"Found {missing_before} rows missing embeddings. "
            f"Embedding up to {batch_size * max_batches} this run "
            f"(batch_size={batch_size}, max_batches={max_batches}).",
        )

        with tqdm(total=missing_before, desc="embedding rows", unit="row", dynamic_ncols=True) as pbar:
            for batch_idx in range(1, max_batches + 1):
                batch_start = time.monotonic()
                cur.execute(
                    UPDATE_SQL,
                    {"model": EMBED_MODEL, "batch_size": batch_size},
                )
                updated = int(cur.rowcount or 0)
                total_updated += updated
                batch_elapsed = time.monotonic() - batch_start
                pbar.update(updated)
                pbar.set_postfix(batch=batch_idx, secs=f"{batch_elapsed:.1f}s")
                if updated == 0:
                    tqdm.write("  no more rows to embed; stopping early.")
                    break

        elapsed = time.monotonic() - start
        print(
            f"\nDone. {total_updated} rows embedded in {elapsed:.2f}s "
            f"across {batch_idx} batch(es).",
        )

        _print_verification(cur)

    return 0


def _print_verification(cur) -> None:
    cur.execute(VERIFY_SQL)
    row = cur.fetchone()
    if not row:
        print("Verification query returned no rows (table may be empty).")
        return
    total, embedded, missing = (int(x) if x is not None else 0 for x in row)
    print(
        f"Verification: total_rows={total} embedded={embedded} missing={missing}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
