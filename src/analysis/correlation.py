"""Statistical correlation tests between fragmentation and carbon waste.

Tests performed:
  - Spearman rank correlation (frag_idle vs carbon_intensity)
  - Granger causality (frag time-series → carbon time-series)
  - Diurnal pattern ANOVA (fragmentation by hour-of-day)

Usage:
    python -m src.analysis.correlation
"""

import numpy as np
import scipy.stats as stats
import duckdb
from src.config import PROCESSED_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, RANDOM_SEED


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (rho, p_value)."""
    return stats.spearmanr(x, y)


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: load carbon_waste.parquet; run Spearman + Granger + ANOVA
    raise NotImplementedError("correlation tests not yet implemented")


if __name__ == "__main__":
    run()
