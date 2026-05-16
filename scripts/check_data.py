"""Quick data status check — run at any point during the pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prophet_agent.snowflake_client import snowflake_cursor

with snowflake_cursor() as cur:
    cur.execute("""
        SELECT
            COALESCE(category, '(none)') AS category,
            COUNT(*) AS n_rows,
            SUM(CASE WHEN question_embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded,
            SUM(CASE WHEN q_kalshi_at_open[0]::FLOAT != 0.5 THEN 1 ELSE 0 END) AS real_prices,
            SUM(CASE WHEN realized_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved
        FROM HISTORICAL_MARKETS
        GROUP BY 1
        ORDER BY n_rows DESC
    """)
    rows = cur.fetchall()
    print(f"\n  {'Category':<28}  {'Rows':>5}  {'Embedded':>8}  {'RealPrices':>10}  {'Resolved':>8}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}")
    total = embedded = real = resolved = 0
    for r in rows:
        print(f"  {str(r[0]):<28}  {r[1]:>5}  {r[2]:>8}  {r[3]:>10}  {r[4]:>8}")
        total += r[1]; embedded += r[2]; real += r[3]; resolved += r[4]  # type: ignore[assignment]
    print(f"  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  {'TOTAL':<28}  {total:>5}  {embedded:>8}  {real:>10}  {resolved:>8}")

    # Backfill status
    print()
    for tbl, label in [
        ("ENSEMBLE_BACKFILL",         "ensemble backfill rows"),
        ("BASE_RATE_BACKFILL",        "base_rate backfill rows"),
        ("META_PREDICTIONS_BACKFILL", "meta_predictions rows"),
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        n = cur.fetchone()[0]
        print(f"  {label:<30}  {n:>6}")
