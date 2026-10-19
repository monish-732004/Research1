"""Central configuration: paths, constants, region lists, and secrets."""

import os
from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ─────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
INTERIM_DIR   = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR   = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
LOG_DIR   = ROOT / "outputs" / "logs"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED     = 42
NUMPY_SEED      = 42
SIMPY_SEED      = 42

# ── Dataset metadata ─────────────────────────────────────────────────────────
TRACE_NAME      = "cluster-trace-gpu-v2026"
TRACE_N_GPUS    = 155_000
TRACE_DURATION_DAYS = 180       # ~6 months

# ── Alibaba cluster regions present in the trace ─────────────────────────────
ALIBABA_REGIONS = [
    "cn-hangzhou",
    "cn-beijing",
    "cn-shanghai",
    "cn-shenzhen",
]

# ── Carbon / grid intensity ───────────────────────────────────────────────────
# Set ELECTRICITY_MAPS_API_KEY in a .env file or your shell environment.
ELECTRICITY_MAPS_API_KEY: str | None = os.environ.get("ELECTRICITY_MAPS_API_KEY")

# China grid carbon intensity defaults (gCO₂eq/kWh) — used as fallback when
# the live API is unavailable.  Values from IEA 2023 country-level estimates.
GRID_INTENSITY_DEFAULTS: dict[str, float] = {
    "cn-hangzhou":  581.0,   # Zhejiang province
    "cn-beijing":   586.0,   # North China grid
    "cn-shanghai":  556.0,   # East China grid
    "cn-shenzhen":  490.0,   # Guangdong province (higher hydro mix)
}

# ── DuckDB settings ───────────────────────────────────────────────────────────
DUCKDB_MEMORY_LIMIT = "12GB"    # leave headroom on a 16 GB laptop
DUCKDB_THREADS      = 4

# ── Fragmentation metric parameters ──────────────────────────────────────────
# Minimum GPU utilisation threshold below which a GPU is considered "idle"
GPU_IDLE_THRESHOLD = 0.10       # 10%
# Window size for rolling fragmentation aggregation
FRAG_WINDOW_HOURS  = 1

# ── Simulation (CADS) parameters ─────────────────────────────────────────────
# Maximum deferral window a job may be pushed back under CADS
CADS_MAX_DEFER_MINUTES = 30
# Carbon intensity percentile below which CADS considers the grid "green"
CADS_GREEN_PERCENTILE  = 0.25
