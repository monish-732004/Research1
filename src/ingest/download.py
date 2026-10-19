"""Download Alibaba cluster-trace-gpu-v2026 Parquet files.

Usage:
    python -m src.ingest.download --help
    python -m src.ingest.download --dest data/raw/
"""

import argparse
import sys
from pathlib import Path


TRACE_HOME = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026"

# Table names expected in the trace release.
EXPECTED_TABLES = [
    "machine_spec",
    "job_table",
    "task_table",
    "instance_table",
    "machine_metric",
]


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Alibaba cluster-trace-gpu-v2026 must be downloaded manually.")
    print(f"Visit: {TRACE_HOME}")
    print(f"Place Parquet files in: {dest.resolve()}")
    print(f"\nExpected tables: {', '.join(EXPECTED_TABLES)}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download trace data.")
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    main(args.dest)
