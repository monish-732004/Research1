"""Fix double-nested extraction for already-downloaded archives.

Moves data/raw/v2026/<table>/<table>/* → data/raw/v2026/<table>/
for each of the three small archives.

Usage
-----
    python scripts/fix_extraction.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_V2026 = ROOT / "data" / "raw" / "v2026"

TABLES = [
    "asi_opensource_server_hourly",
    "asi_opensource_network_hourly",
    "asi_opensource_job_execution_summary",
]


def flatten(table: str) -> None:
    outer = RAW_V2026 / table
    inner = outer / table

    if not outer.exists():
        print(f"  SKIP {table}: outer dir does not exist")
        return
    if not inner.is_dir():
        print(f"  OK   {table}: already flat (no double-nesting)")
        return

    items = list(inner.iterdir())
    print(f"  Flattening {table} ({len(items)} items) …")
    for item in items:
        target = outer / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(outer))
    inner.rmdir()
    print(f"  Done → {outer}")


if __name__ == "__main__":
    for t in TABLES:
        flatten(t)
    print("\nAll done.")
