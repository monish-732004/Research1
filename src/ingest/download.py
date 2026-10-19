"""Resumable downloader for Alibaba cluster-trace ZIP archives.

Usage
-----
    python -m src.ingest.download --trace v2026
    python -m src.ingest.download --trace v2026 --out data/raw/v2026/
    python -m src.ingest.download --trace v2026 --dry-run

The manifest YAML (data/raw/manifests/{trace}.yaml) lists each archive.
When ``md5`` is null the downloader computes the hash after a successful
download, prints it, and writes it back into the YAML for future runs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from src.config import RAW_DIR

MANIFEST_DIR = RAW_DIR / "manifests"
CHUNK_SIZE = 8 * 1024 * 1024   # 8 MiB


# ── Manifest ──────────────────────────────────────────────────────────────────

class ManifestError(Exception):
    pass


def _load_manifest(trace: str) -> tuple[Path, dict]:
    yaml_path = MANIFEST_DIR / f"{trace}.yaml"
    template_path = MANIFEST_DIR / f"{trace}.yaml.template"

    if not yaml_path.exists():
        if template_path.exists():
            print(
                f"No manifest found at {yaml_path}\n"
                f"A template exists at {template_path}\n"
                f"Fill in the URLs, then rename it to {trace}.yaml"
            )
        else:
            print(f"No manifest or template found for trace '{trace}' in {MANIFEST_DIR}")
        sys.exit(1)

    with yaml_path.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    archives = manifest.get("archives") or manifest.get("files") or []
    if not archives:
        raise ManifestError(f"Manifest {yaml_path} has no 'archives' entries.")

    for entry in archives:
        if not entry.get("url"):
            raise ManifestError(
                f"Entry '{entry.get('name', '?')}' has no URL. Edit {yaml_path}."
            )

    return yaml_path, manifest


def _save_md5(yaml_path: Path, name: str, md5: str) -> None:
    """Write a computed MD5 back into the manifest YAML."""
    with yaml_path.open(encoding="utf-8") as fh:
        text = fh.read()
    # Replace the null for this specific archive name
    # Targeted replacement of `md5: null` that follows this archive's name
    lines = text.splitlines(keepends=True)
    inside = False
    result = []
    for line in lines:
        if f"name: {name}" in line:
            inside = True
        if inside and "md5: null" in line:
            line = line.replace("md5: null", f"md5: {md5}")
            inside = False
        result.append(line)
    with yaml_path.open("w", encoding="utf-8") as fh:
        fh.writelines(result)
    print(f"  MD5 written back to manifest: {md5}")


# ── MD5 ───────────────────────────────────────────────────────────────────────

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Download ──────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, expected_size: int | None) -> None:
    """Resumable streaming download of *url* → *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers: dict[str, str] = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if resp.status_code == 416:
            print(f"  {dest.name}: already fully downloaded (416 Range Not Satisfiable).")
            return
        resp.raise_for_status()

        content_len = int(resp.headers.get("Content-Length", 0))
        total = content_len + existing if content_len else (expected_size or None)
        mode = "ab" if existing and resp.status_code == 206 else "wb"

        with (
            dest.open(mode) as fh,
            tqdm(
                total=total,
                initial=existing,
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


# ── Extract ───────────────────────────────────────────────────────────────────

def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*, skipping already-extracted files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} → {dest_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, unit="file", desc="extract"):
            out_path = dest_dir / member.filename
            if out_path.exists() and out_path.stat().st_size == member.file_size:
                continue  # already extracted
            zf.extract(member, dest_dir)
    print(f"  Extracted {len(members)} file(s).")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    trace: str,
    out_dir: Path,
    dry_run: bool = False,
    only: str | None = None,
) -> None:
    yaml_path, manifest = _load_manifest(trace)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = manifest.get("archives") or manifest.get("files") or []

    if only:
        # Match by name prefix or exact stem (without .zip)
        archives = [
            e for e in archives
            if e["name"] == only
            or e["name"] == f"{only}.zip"
            or Path(e["name"]).stem == only
        ]
        if not archives:
            print(f"No archive matching '{only}' found in manifest.")
            sys.exit(1)

    print(f"Trace: {trace}  ({len(archives)} archive(s))\n")

    for entry in archives:
        name = entry["name"]
        url = entry["url"]
        size = entry.get("size_bytes")
        expected_md5 = entry.get("md5")  # may be None
        # unpacks_to is relative to project root
        unpacks_to = Path(entry.get("unpacks_to", f"data/raw/{trace}/{Path(name).stem}/"))
        zip_dest = out_dir / name

        if dry_run:
            status = "ZIP-EXISTS" if zip_dest.exists() else "MISSING"
            extracted = "EXTRACTED" if unpacks_to.exists() else "NOT-YET"
            print(f"  [{status}|{extracted}] {name}  ({size/1e9:.1f} GB)" if size else f"  [{status}|{extracted}] {name}")
            continue

        # ── 1. Download ZIP ───────────────────────────────────────────────────
        if zip_dest.exists():
            actual_size = zip_dest.stat().st_size
            if size and actual_size >= size:
                print(f"  {name}: ZIP already complete ({actual_size/1e9:.2f} GB), skipping download.")
            else:
                print(f"  {name}: resuming download ({actual_size/1e9:.2f} / {(size or 0)/1e9:.2f} GB) …")
                _download_file(url, zip_dest, size)
        else:
            print(f"  {name}: downloading ({(size or 0)/1e9:.2f} GB) …")
            _download_file(url, zip_dest, size)

        # ── 2. MD5 verify / compute ───────────────────────────────────────────
        print(f"  Computing MD5 for {name} …", end=" ", flush=True)
        actual_md5 = _md5_file(zip_dest)
        print(actual_md5)

        if expected_md5:
            if actual_md5 != expected_md5:
                zip_dest.unlink(missing_ok=True)
                raise ValueError(
                    f"MD5 mismatch for {name}: expected {expected_md5}, got {actual_md5}. "
                    f"ZIP deleted — re-run to retry."
                )
            print(f"  MD5 OK.")
        else:
            # Write computed hash back so future runs can verify
            _save_md5(yaml_path, name, actual_md5)

        # ── 3. Extract ────────────────────────────────────────────────────────
        _extract_zip(zip_dest, unpacks_to)

    if not dry_run:
        print("\nAll archives downloaded, verified, and extracted.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract Alibaba cluster-trace archives."
    )
    parser.add_argument(
        "--trace", choices=["v2026", "v2023"], required=True,
        help="Which trace version to download.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Directory for downloaded ZIPs (default: data/raw/{trace}/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="NAME",
        help="Download only the archive whose name matches (with or without .zip).",
    )
    args = parser.parse_args(argv)
    out_dir = args.out or (RAW_DIR / args.trace)
    run(args.trace, out_dir, dry_run=args.dry_run, only=args.only)


if __name__ == "__main__":
    main()
