"""
Thin SQLite helper – read-only connection to prevent prompt-injection SQL writes.
"""

import sqlite3
from contextlib import contextmanager
from app.config import settings

# Introspect the schema once at startup so agents can inject it into prompts
_SCHEMA_CACHE: str | None = None


@contextmanager
def get_conn():
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_schema() -> str:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE

    lines: list[str] = []
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        for name, ddl in cur.fetchall():
            if ddl:
                lines.append(ddl.strip())
    _SCHEMA_CACHE = "\n\n".join(lines)
    return _SCHEMA_CACHE


def run_query(sql: str) -> list[dict]:
    """Execute a read-only SELECT and return rows as list-of-dicts."""
    sql_stripped = sql.strip().rstrip(";")
    if not sql_stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are permitted.")
    with get_conn() as conn:
        cur = conn.execute(sql_stripped)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
