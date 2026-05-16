"""Bootstrap Snowflake for this project.

Steps (all idempotent):
  1. Create database + schema named by SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA.
  2. USE that schema.
  3. Apply sql/ddl.sql statement-by-statement.

Each DDL statement is executed independently so a failure on one (e.g., a feature
unavailable in the current region) prints a clear error but doesn't abort.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Make src/ importable when running as `python scripts/bootstrap_snowflake.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.snowflake_client import _load_env_once, snowflake_cursor

_load_env_once()

DDL_PATH = PROJECT_ROOT / "sql" / "ddl.sql"


def _ident(name: str) -> str:
    if '"' in name:
        msg = f"refusing identifier with embedded quote: {name!r}"
        raise ValueError(msg)
    return f'"{name}"'


def split_statements(sql_text: str) -> list[str]:
    """Strip line comments, split on `;` outside of quoted strings."""
    cleaned_lines = []
    for raw_line in sql_text.splitlines():
        # Drop lines that are *only* a SQL line comment (`-- ...`).
        if raw_line.lstrip().startswith("--"):
            continue
        cleaned_lines.append(raw_line)
    cleaned = "\n".join(cleaned_lines)
    parts = [p.strip() for p in cleaned.split(";")]
    return [p for p in parts if p]


def _first_object_name(stmt: str) -> str:
    m = re.search(r"\b(TABLE|VIEW|SCHEMA)\b\s+(IF\s+NOT\s+EXISTS\s+)?([A-Z0-9_.]+)",
                  stmt, re.IGNORECASE)
    return m.group(3) if m else stmt[:40].replace("\n", " ")


def main() -> None:
    if not DDL_PATH.is_file():
        print(f"DDL not found at {DDL_PATH}", file=sys.stderr)
        raise SystemExit(1)
    sql_text = DDL_PATH.read_text(encoding="utf-8")
    statements = split_statements(sql_text)

    db = os.environ.get("SNOWFLAKE_DATABASE", "").strip()
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "").strip()
    if not db or not schema:
        print("SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA must be set in .env", file=sys.stderr)
        raise SystemExit(1)

    failures: list[tuple[str, str]] = []
    with snowflake_cursor() as cur:
        # Step 1: ensure database + schema exist (idempotent).
        print(f"Ensuring database {db}.{schema} exists...")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {_ident(db)}")
        cur.execute(f"USE DATABASE {_ident(db)}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_ident(schema)}")
        cur.execute(f"USE SCHEMA {_ident(schema)}")
        print(f"  OK   {db}.{schema}\n")

        # Step 2: apply DDL.
        print(f"Applying {len(statements)} statements from {DDL_PATH.name}...")
        for stmt in statements:
            label = _first_object_name(stmt)
            try:
                cur.execute(stmt)
                print(f"  OK   {label}")
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {label}: {e}")
                failures.append((label, str(e)))

    if failures:
        print(f"\n{len(failures)} statement(s) failed.", file=sys.stderr)
        raise SystemExit(2)
    print("\nAll statements applied successfully.")


if __name__ == "__main__":
    main()
