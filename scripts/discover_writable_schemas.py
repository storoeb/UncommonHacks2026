"""List databases/schemas where the current Snowflake role has CREATE TABLE."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prophet_agent.snowflake_client import snowflake_cursor


def main() -> None:
    with snowflake_cursor() as cur:
        cur.execute("SELECT CURRENT_ROLE()")
        role = cur.fetchone()[0]
        print(f"Current role: {role}\n")

        print("=== Grants to role ===")
        cur.execute(f"SHOW GRANTS TO ROLE {role}")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        # Find the schema/database-level CREATE TABLE-relevant grants.
        for row in rows:
            rec = dict(zip(cols, row, strict=True))
            priv = rec.get("privilege") or rec.get("PRIVILEGE")
            kind = rec.get("granted_on") or rec.get("GRANTED_ON")
            name = rec.get("name") or rec.get("NAME")
            if priv in ("CREATE TABLE", "CREATE VIEW", "CREATE STAGE", "USAGE", "OWNERSHIP") \
               and kind in ("SCHEMA", "DATABASE"):
                print(f"  {priv:14}  {kind:10}  {name}")

        print("\n=== Databases visible ===")
        cur.execute("SHOW DATABASES")
        db_cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            rec = dict(zip(db_cols, row, strict=True))
            print(f"  {rec.get('name') or rec.get('NAME')}")

        # For each visible database, list schemas.
        cur.execute("SHOW DATABASES")
        db_cols = [d[0] for d in cur.description]
        db_names = [
            (dict(zip(db_cols, r, strict=True)).get("name")
             or dict(zip(db_cols, r, strict=True)).get("NAME"))
            for r in cur.fetchall()
        ]
        for db in db_names:
            if not db:
                continue
            print(f"\n=== Schemas in {db} ===")
            try:
                cur.execute(f'SHOW SCHEMAS IN DATABASE "{db}"')
                sc_cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(sc_cols, row, strict=True))
                    name = rec.get("name") or rec.get("NAME")
                    print(f"  {name}")
            except Exception as e:  # noqa: BLE001
                print(f"  (cannot list: {e})")


if __name__ == "__main__":
    main()
