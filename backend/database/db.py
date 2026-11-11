"""SQLite connection helpers.

Application code uses `connect_readonly()`: the database is opened with
`mode=ro` and `PRAGMA query_only`, so no retrieval code path (and certainly no
model output) can modify records. Only the generator opens it for writing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.config import get_settings


def _configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or get_settings().database_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run `python scripts/generate_synthetic_data.py`."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    _configure(conn)
    conn.execute("PRAGMA query_only = ON")
    return conn


def connect_writable(path: Path) -> sqlite3.Connection:
    """Used by the synthetic data generator only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return _configure(sqlite3.connect(str(path)))
