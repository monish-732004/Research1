"""Left/right-truncated job detection and flagging.

Jobs that were already running at trace start (left-truncated) or still
running at trace end (right-truncated) must be flagged before lifetime
statistics are computed to avoid bias.

Usage:
    python -m src.preprocess.truncation
"""

import duckdb
from src.config import INTERIM_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: detect trace window boundaries; add is_left_trunc / is_right_trunc cols
    raise NotImplementedError("truncation flagging not yet implemented")


if __name__ == "__main__":
    run()
