"""Job state transition cleaner.

Validates that each job/task follows a legal state machine and repairs or
flags rows with illegal transitions (e.g. Running → Waiting).

Usage:
    python -m src.preprocess.state_machine
"""

import duckdb
from src.config import INTERIM_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS

# Legal forward transitions for a job/task
LEGAL_TRANSITIONS: dict[str, set[str]] = {
    "Waiting":   {"Running", "Failed", "Cancelled"},
    "Running":   {"Succeeded", "Failed", "Cancelled", "Preempted"},
    "Preempted": {"Waiting", "Failed", "Cancelled"},
    "Succeeded": set(),
    "Failed":    set(),
    "Cancelled": set(),
}


def is_legal(src: str, dst: str) -> bool:
    return dst in LEGAL_TRANSITIONS.get(src, set())


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: implement state-machine cleaning with DuckDB window functions
    raise NotImplementedError("state_machine cleaner not yet implemented")


if __name__ == "__main__":
    run()
