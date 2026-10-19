# gpu-carbon-waste

**Measuring GPU resource fragmentation and its carbon cost on the Alibaba
cluster-trace-gpu-v2026, with a carbon-aware deferral scheduler (CADS).**

Target venue: HotCarbon 2027 / IEEE IGSC 2027

---

## Project goal

Modern GPU clusters waste significant capacity through *resource fragmentation*:
idle GPU-hours that arise from scheduling gaps, job preemptions, and
heterogeneous workload shapes.  This project:

1. **Quantifies** fragmentation across 155 K GPUs over six months using the
   Alibaba cluster-trace-gpu-v2026.
2. **Prices** that waste in carbon by joining the trace with grid electricity
   intensity data (Electricity Maps API).
3. **Simulates** a lightweight Carbon-Aware Deferral Scheduler (CADS) that
   shifts deferrable jobs to low-carbon windows and measures the resulting
   CO₂eq savings.

---

## Dataset

> Alibaba Group. *Alibaba Cluster Trace GPU v2026* (2026).
> Available via the Alibaba cluster-trace programme:
> <https://github.com/alibaba/clusterdata>

The trace covers ~155 000 GPU instances across four Chinese data-centre regions
over approximately 180 days.  Raw Parquet dumps belong in `data/raw/` and are
**not** committed to this repository.

---

## Quick-start reproduction

```bash
# 1. Clone and enter the repo
git clone <this-repo> gpu-carbon-waste
cd gpu-carbon-waste

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make setup

# 3. Copy the environment template and add your API key
cp .env.example .env
#   → edit .env and set ELECTRICITY_MAPS_API_KEY=<your key>

# 4. Place the raw Parquet files into data/raw/
#   (download instructions: src/ingest/download.py --help)

# 5. Run the full pipeline
make all

# 6. Open the notebooks for interactive exploration
jupyter lab
```

Individual pipeline stages:

| Command          | What it does                                    |
|------------------|-------------------------------------------------|
| `make preprocess`| Clean timestamps, states, truncation flags      |
| `make analyze`   | Compute fragmentation metrics + carbon join     |
| `make simulate`  | Replay baseline + CADS discrete-event sim       |
| `make figures`   | Render publication-quality PDFs to outputs/     |
| `make test`      | Run pytest suite                                |
| `make lint`      | Ruff lint + format check                        |

---

## Directory map

```
gpu-carbon-waste/
├── data/
│   ├── raw/          ← original Parquet dumps (gitignored)
│   ├── interim/      ← cleaned intermediate tables
│   └── processed/    ← analysis-ready aggregates
├── src/
│   ├── config.py     ← paths, constants, region list
│   ├── ingest/       ← download + DuckDB schema definitions
│   ├── preprocess/   ← clock align, state machine, truncation, aggregation
│   ├── analysis/     ← fragmentation metrics, carbon join, stats
│   ├── simulation/   ← baseline replay + CADS SimPy sim
│   └── viz/          ← matplotlib/seaborn figures, IEEE style
├── notebooks/        ← Jupyter EDA notebooks (01–05)
├── tests/            ← pytest unit tests
├── paper/            ← LaTeX source (IEEE template)
└── outputs/          ← generated figures, tables, logs (gitignored)
```

---

## Environment variables

| Variable                   | Required | Description                              |
|----------------------------|----------|------------------------------------------|
| `ELECTRICITY_MAPS_API_KEY` | Yes      | API key for grid carbon intensity data   |

Create a `.env` file at the project root (see `.env.example`).

---

## License

Code: MIT.  Data: subject to the Alibaba cluster-trace terms of use.
