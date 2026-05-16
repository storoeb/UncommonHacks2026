"""Shared Snowflake connection helper.

Reuses the credential-loading pattern from `scripts/check_snowflake_connection.py`.
Prefer PAT auth (SNOWFLAKE_PAT) — no MFA prompt per run.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    _ENV_LOADED = True


def _connect_kwargs() -> dict[str, Any]:
    _load_env_once()
    account = os.environ["SNOWFLAKE_ACCOUNT"].strip()
    user = os.environ["SNOWFLAKE_USER"].strip()
    pat = os.getenv("SNOWFLAKE_PAT", "").strip()
    password = os.getenv("SNOWFLAKE_PASSWORD", "").strip()
    if not (pat or password):
        msg = "Set SNOWFLAKE_PAT or SNOWFLAKE_PASSWORD in .env"
        raise RuntimeError(msg)

    kwargs: dict[str, Any] = {
        "account": account,
        "user": user,
        "password": pat or password,
    }
    for env_key, conn_key in (
        ("SNOWFLAKE_ROLE", "role"),
        ("SNOWFLAKE_WAREHOUSE", "warehouse"),
        ("SNOWFLAKE_DATABASE", "database"),
        ("SNOWFLAKE_SCHEMA", "schema"),
    ):
        v = os.getenv(env_key, "").strip()
        if v:
            kwargs[conn_key] = v
    return kwargs


@contextmanager
def snowflake_connection() -> Iterator[Any]:
    """Yield an open Snowflake connection, auto-closing on exit."""
    import snowflake.connector

    conn = snowflake.connector.connect(**_connect_kwargs(), autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def snowflake_cursor() -> Iterator[Any]:
    with snowflake_connection() as conn, conn.cursor() as cur:
        yield cur
