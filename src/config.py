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

# ── Trace temporal structure (hourly-aggregated ASI schema) ───────────────────
TRACE_DAYS          = 185                           # actual days in the trace
TRACE_HOURS_PER_DAY = 24
TRACE_START_HOUR    = 0
TRACE_END_HOUR      = TRACE_DAYS * TRACE_HOURS_PER_DAY - 1   # 4439

NETWORK_DAYS        = 7
NETWORK_END_HOUR    = NETWORK_DAYS * TRACE_HOURS_PER_DAY - 1  # 167

# ── Pod state space ───────────────────────────────────────────────────────────
POD_STATES   = ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "EVICTED", "UNKNOWN"]
ACTIVE_STATES = frozenset({"PENDING", "RUNNING"})

# ── Server tier GPU thresholds ────────────────────────────────────────────────
SERVER_TIER_SMALL_MAX  = 4   # ≤ 4 GPUs → "small"
SERVER_TIER_MEDIUM_MAX = 8   # 5–8 GPUs → "medium"  (> 8 → "large")

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

# Synthetic grid profiles for sensitivity analysis (Phase 4).
# Each entry: annual_avg (gCO₂/kWh), peak_hour (0–23 local), peak_factor (×).
# Diurnal multiplier: 1 + (peak_factor-1)*sin((h-6)*π/18) for h∈[6,24], 0.85 for h<6.
GRID_PROFILES: dict[str, dict] = {
    "cn-north":     {"annual_avg": 800, "peak_hour": 19, "peak_factor": 1.15},
    "cn-east":      {"annual_avg": 650, "peak_hour": 20, "peak_factor": 1.12},
    "cn-south":     {"annual_avg": 550, "peak_hour": 19, "peak_factor": 1.10},
    "cn-northwest": {"annual_avg": 750, "peak_hour": 18, "peak_factor": 1.20},
}

# Calendar anchors for sensitivity: maps label → ISO date string
CALENDAR_ANCHORS: dict[str, str] = {
    "2025-01-01": "2025-01-01",
    "2025-04-01": "2025-04-01",
    "2025-07-01": "2025-07-01",
}

CARBON_RAW_DIR = RAW_DIR / "carbon"

# ── Raw data glob helper ──────────────────────────────────────────────────────

def table_glob(table_name: str, trace: str = "v2026") -> str:
    """Return a glob pattern for all Parquet files for *table_name*.

    Handles both extraction layouts:
      single-level: data/raw/<trace>/<table>/**/*.parquet
      double-level: data/raw/<trace>/<table>/<table>/**/*.parquet  (ZIP quirk)
    """
    import glob as _glob
    double = RAW_DIR / trace / table_name / table_name / "**" / "*.parquet"
    single = RAW_DIR / trace / table_name / "**" / "*.parquet"
    if _glob.glob(str(double), recursive=True):
        return str(double)
    return str(single)


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
