"""Verify Snowflake credentials from `.env` by running a trivial query.

Prefer a **Snowflake programmatic access token (PAT)** to skip TOTP each run:

    SNOWFLAKE_USER=...
    SNOWFLAKE_PAT=<token from Snowsight; passed as connector `password`; MFA passcode omitted>

Fallback — password login + optional one-shot TOTP (avoid stale codes in `.env`):

    $env:SNOWFLAKE_MFA_PASSCODE = "423842"
    python scripts/check_snowflake_connection.py
    Remove-Item Env:\\SNOWFLAKE_MFA_PASSCODE"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def load_env(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as e:
        print("Missing dependency: python-dotenv. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from e

    env_path = project_root / ".env"
    if not env_path.is_file():
        print(f"No .env at {env_path}", file=sys.stderr)
        raise SystemExit(1)
    load_dotenv(env_path)


def build_connect_kwargs() -> dict[str, Any]:
    """Login only — omit warehouse/database/schema so bad names do not silently clear session."""

    account = os.getenv("SNOWFLAKE_ACCOUNT", "").strip()
    user = os.getenv("SNOWFLAKE_USER", "").strip()
    pat = os.getenv("SNOWFLAKE_PAT", "").strip()
    password_plain = os.getenv("SNOWFLAKE_PASSWORD", "").strip()

    missing_base = [n for n, v in (("SNOWFLAKE_ACCOUNT", account), ("SNOWFLAKE_USER", user)) if not v]
    if missing_base:
        print(f"Set these in .env: {', '.join(missing_base)}", file=sys.stderr)
        raise SystemExit(1)

    if pat:
        credential = pat
        use_totp = False
    elif password_plain:
        credential = password_plain
        use_totp = True
    else:
        print(
            "Set either SNOWFLAKE_PAT or SNOWFLAKE_PASSWORD in `.env`.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    conn: dict[str, Any] = {
        "account": account,
        "user": user,
        "password": credential,
    }
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        conn["role"] = role

    totp = os.getenv("SNOWFLAKE_MFA_PASSCODE", "").strip()
    if use_totp and totp:
        conn["passcode"] = totp

    return conn


def identifier_sql(name: str) -> str:
    """Double‑quoted Snowflake identifier; rejects embedded quotes."""
    if '"' in name or "\x00" in name:
        msg = "Refusing ambiguous identifier containing quote or NUL"
        raise ValueError(msg)
    return f'"{name}"'


def print_requested_session_context() -> None:
    wh = os.getenv("SNOWFLAKE_WAREHOUSE", "").strip()
    db = os.getenv("SNOWFLAKE_DATABASE", "").strip()
    sc = os.getenv("SNOWFLAKE_SCHEMA", "").strip()
    parts = []
    if wh:
        parts.append(f"warehouse={wh}")
    if db:
        parts.append(f"database={db}")
    if sc:
        parts.append(f"schema={sc}")
    if parts:
        print(f"Applying session context ({', '.join(parts)})...", flush=True)


def emit_inventory_hints(cursor: Any, database_from_env: str) -> None:
    """Best-effort listing so users can paste real warehouse/schema names into .env."""

    from snowflake.connector.errors import ProgrammingError

    print("\n--- Discovery (paste one of these into .env if appropriate) ---", file=sys.stderr, flush=True)

    try:
        cursor.execute("SHOW WAREHOUSES IN ACCOUNT LIMIT 100")
        desc = cursor.description or ()
        colnames = [(d[0] or "").lower() for d in desc]
        if "name" not in colnames:
            print(f"SHOW WAREHOUSES unexpected columns: {colnames}", file=sys.stderr)
        else:
            ni = colnames.index("name")
            warehouses = sorted({str(row[ni]).strip() for row in cursor.fetchall()})
            print(
                f"SNOWFLAKE_WAREHOUSE candidates ({len(warehouses)}):\n"
                + ("\n".join(f"    {w}" for w in warehouses) if warehouses else "    (none)"),
                file=sys.stderr,
                flush=True,
            )
    except ProgrammingError as e:
        print(f"(Could not list warehouses: {e})", file=sys.stderr, flush=True)

    if database_from_env:
        try:
            cursor.execute(f"SHOW SCHEMAS IN DATABASE {identifier_sql(database_from_env)} LIMIT 100")
            desc = cursor.description or ()
            colnames = [(d[0] or "").lower() for d in desc]
            if "name" not in colnames:
                print(f"SHOW SCHEMAS unexpected columns: {colnames}", file=sys.stderr)
            else:
                ni = colnames.index("name")
                schemas = sorted({str(row[ni]).strip() for row in cursor.fetchall()})
                print(
                    f"SNOWFLAKE_SCHEMA candidates in {database_from_env} ({len(schemas)}):\n"
                    + ("\n".join(f"    {s}" for s in schemas) if schemas else "    (none)"),
                    file=sys.stderr,
                    flush=True,
                )
        except ProgrammingError as e:
            print(f"(Could not list schemas for {database_from_env}: {e})", file=sys.stderr, flush=True)


def validate_session(cursor: Any) -> None:
    from snowflake.connector.errors import ProgrammingError

    mappings = (
        ("WAREHOUSE", "warehouse", "SNOWFLAKE_WAREHOUSE"),
        ("DATABASE", "database", "SNOWFLAKE_DATABASE"),
        ("SCHEMA", "schema", "SNOWFLAKE_SCHEMA"),
    )
    failures: list[str] = []
    print_requested_session_context()

    db_name = os.getenv("SNOWFLAKE_DATABASE", "").strip()

    for kind, label, env_key in mappings:
        name = os.getenv(env_key, "").strip()
        if not name:
            continue
        try:
            if kind == "SCHEMA" and db_name:
                # Fully-qualified avoids odd session state; matches Snowsight DATABASE.SCHEMA picker.
                cursor.execute(f"USE SCHEMA {identifier_sql(db_name)}.{identifier_sql(name)}")
            else:
                cursor.execute(f"USE {kind} {identifier_sql(name)}")
        except ProgrammingError as e:
            msg = getattr(e, "msg", None) or str(e)
            failures.append(f"{label} `{name}`: {msg}")

    if failures:
        hint = (
            "\nHint:\n"
            "  • Warehouse: Run `SHOW WAREHOUSES IN ACCOUNT;` Copy an exact **name** you have USAGE "
            "on — set SNOWFLAKE_WAREHOUSE to that string (often not COMPUTE_WH on training tenants).\n"
            "  • Schema: Run `SHOW SCHEMAS IN DATABASE TRAINING_DB` (adjust DB name). "
            "Set SNOWFLAKE_SCHEMA to a listed schema (PUBLIC may not exist everywhere).\n"
            "  • Grants: If objects exist but `USE` fails, check `SHOW GRANTS TO ROLE TRAINING_ROLE`."
        )
        joined = "\n".join(f"  - {msg}" for msg in failures)
        print(f"\nSession context failed:\n{joined}{hint}", file=sys.stderr, flush=True)
        emit_inventory_hints(cursor, database_from_env=db_name)
        raise SystemExit(3)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_env(project_root)

    try:
        import snowflake.connector
        from snowflake.connector.errors import Error as SnowflakeError
    except ImportError:
        print(
            "Missing dependency: snowflake-connector-python. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    params = build_connect_kwargs()

    print(f"Connecting to account {params['account']} as {params['user']} ...", flush=True)
    using_pat = bool(os.getenv("SNOWFLAKE_PAT", "").strip())
    if using_pat:
        print("Auth: PAT (connector `password` field; MFA passcode omitted).", flush=True)

    try:
        conn = snowflake.connector.connect(**params)
    except SnowflakeError as e:
        print(f"Connection failed: {e}", file=sys.stderr, flush=True)
        if not using_pat and ("MFA" in str(e) or "TOTP" in str(e)):
            print(
                "\nMFA/TOTP required for password login. Options:\n"
                "  • Set SNOWFLAKE_PAT (PAT substitutes for password; Snowflake PAT docs), or\n"
                "  • Export current 6‑digit code: $env:SNOWFLAKE_MFA_PASSCODE = \"...\"; rerun.",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(2) from e

    with conn.cursor() as cur:
        validate_session(cur)

        cur.execute(
            """
            SELECT
              CURRENT_ACCOUNT(),
              CURRENT_REGION(),
              CURRENT_USER(),
              CURRENT_ROLE(),
              CURRENT_WAREHOUSE(),
              CURRENT_DATABASE(),
              CURRENT_SCHEMA()
            """
        )
        row = cur.fetchone()

    conn.close()

    cols = ["account", "region", "user", "role", "warehouse", "database", "schema"]
    lines = "\n".join(f"  {c}: {v}" for c, v in zip(cols, row, strict=True))
    print(f"Connected successfully.\n{lines}")


if __name__ == "__main__":
    main()
