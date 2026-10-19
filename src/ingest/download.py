"""Resumable downloader for Alibaba cluster-trace Parquet files.

Usage
-----
    python -m src.ingest.download --trace v2026 --out data/raw/v2026/
    python -m src.ingest.download --trace v2023 --out data/raw/v2023/
    python -m src.ingest.download --trace v2026 --dry-run

The script reads download URLs and expected MD5 hashes from a manifest YAML:
    data/raw/manifests/{trace}.yaml

If only the template exists, the script explains how to fill it in and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from src.config import RAW_DIR

MANIFEST_DIR = RAW_DIR / "manifests"
CHUNK_SIZE = 8 * 1024 * 1024   # 8 MiB per chunk


# ── Manifest loading ──────────────────────────────────────────────────────────

class ManifestError(Exception):
    pass


def _load_manifest(trace: str) -> dict:
    yaml_path = MANIFEST_DIR / f"{trace}.yaml"
    template_path = MANIFEST_DIR / f"{trace}.yaml.template"

    if not yaml_path.exists():
        if template_path.exists():
            print(
                f"No manifest found at {yaml_path}\n"
                f"A template exists at {template_path}\n"
                f"Fill in the `url` and `md5` fields, then rename it to {trace}.yaml"
            )
        else:
            print(f"No manifest or template found for trace '{trace}' in {MANIFEST_DIR}")
        sys.exit(1)

    with yaml_path.open() as fh:
        manifest = yaml.safe_load(fh)

    # Basic validation
    for entry in manifest.get("files", []):
        if not entry.get("url"):
            raise ManifestError(
                f"Entry for table '{entry['table']}' has no URL. "
                f"Edit {yaml_path} and fill in the `url` field."
            )
        if not entry.get("md5"):
            raise ManifestError(
                f"Entry for table '{entry['table']}' has no MD5. "
                f"Edit {yaml_path} and fill in the `md5` field."
            )
    return manifest


# ── MD5 verification ─────────────────────────────────────────────────────────

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Resumable download ────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, expected_md5: str) -> None:
    """Download *url* to *dest* with resume support and MD5 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # How much is already on disk?
    existing_bytes = dest.stat().st_size if dest.exists() else 0

    headers: dict[str, str] = {}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416:
            # Range not satisfiable → file may already be complete
            pass
        elif resp.status_code not in (200, 206):
            resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0)) + existing_bytes
        mode = "ab" if existing_bytes and resp.status_code == 206 else "wb"

        with (
            dest.open(mode) as fh,
            tqdm(
                total=total or None,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                leave=True,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                pbar.update(len(chunk))

    # Verify integrity
    print(f"  Verifying MD5 for {dest.name} …", end=" ", flush=True)
    actual_md5 = _md5_file(dest)
    if actual_md5 != expected_md5:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"MD5 mismatch for {dest.name}: "
            f"expected {expected_md5}, got {actual_md5}. "
            f"File deleted — re-run to retry."
        )
    print("OK")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(trace: str, out_dir: Path, dry_run: bool = False) -> None:
    manifest = _load_manifest(trace)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = manifest.get("files", [])
    print(f"Trace: {trace}  ({len(files)} file(s))  →  {out_dir.resolve()}\n")

    for entry in files:
        dest = out_dir / entry["filename"]
        url = entry["url"]
        md5 = entry["md5"]

        if dry_run:
            status = "EXISTS" if dest.exists() else "MISSING"
            print(f"  [{status}] {entry['filename']}  ←  {url}")
            continue

        # Skip if already downloaded and verified
        if dest.exists():
            print(f"  Checking existing {dest.name} …", end=" ", flush=True)
            if _md5_file(dest) == md5:
                print("already complete, skipping.")
                continue
            print("MD5 mismatch, re-downloading.")

        _download_file(url, dest, md5)

    if not dry_run:
        print("\nAll files downloaded and verified.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download Alibaba cluster-trace Parquet files."
    )
    parser.add_argument(
        "--trace",
        choices=["v2026", "v2023"],
        required=True,
        help="Which trace version to download.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/raw/{trace}/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out or (RAW_DIR / args.trace)
    run(args.trace, out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
