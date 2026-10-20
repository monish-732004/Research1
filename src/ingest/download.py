"""Resumable downloader for Alibaba cluster-trace ZIP archives.

Usage
-----
    python -m src.ingest.download --trace v2026
    python -m src.ingest.download --trace v2026 --out data/raw/v2026/
    python -m src.ingest.download --trace v2026 --dry-run
    python -m src.ingest.download --trace v2026 --only asi_opensource_pod_hourly --stream-extract

The manifest YAML (data/raw/manifests/{trace}.yaml) lists each archive.
When ``md5`` is null the downloader computes the hash after a successful
download, prints it, and writes it back into the YAML for future runs.

--stream-extract: download and extract simultaneously without saving the ZIP.
  Resumes automatically by skipping already-extracted files.
  Saves ~350 GB of disk space for large archives.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from src.config import RAW_DIR

MANIFEST_DIR = RAW_DIR / "manifests"
CHUNK_SIZE = 8 * 1024 * 1024   # 8 MiB
_STREAM_CHUNK = 10 * 1024 * 1024  # 10 MiB download chunks for stream-extract
MAX_RETRIES = 20                # max reconnects per archive
RETRY_BACKOFF = (5, 15, 30, 60, 120)  # seconds between retries (last value repeated)


# ── Streaming ZIP extractor ───────────────────────────────────────────────────

def _stream_extract_zip(
    url: str,
    dest_dir: Path,
    expected_size: int | None,
    skip_existing: bool = True,
) -> None:
    """Download *url* and extract to *dest_dir* in one streaming pass.

    Uses stream-unzip to parse the ZIP on-the-fly — never writes the full
    archive to disk (saves ~350 GB for large archives).
    Already-extracted files (size > 0) are skipped; their bytes are consumed
    and discarded so the stream stays in sync.
    Partial output files are deleted on KeyboardInterrupt or extraction error.
    Re-running resumes automatically by skipping existing files.

    Auto-retries up to MAX_RETRIES times on connection drops.
    """
    from stream_unzip import stream_unzip

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = skipped = errors = 0

    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            pbar = tqdm(
                total=expected_size, unit='B', unit_scale=True,
                unit_divisor=1024, desc='stream-extract', leave=True,
                initial=0,
            )

            def _chunks():
                with requests.get(url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=_STREAM_CHUNK):
                        if chunk:
                            pbar.update(len(chunk))
                            yield chunk

            current_out: Path | None = None
            try:
                for file_name_raw, _file_size, unzipped_chunks in stream_unzip(_chunks()):
                    fname    = (file_name_raw.decode('utf-8', errors='replace')
                                if isinstance(file_name_raw, bytes) else file_name_raw)
                    out_path = dest_dir / fname

                    if fname.endswith('/'):
                        out_path.mkdir(parents=True, exist_ok=True)
                        for _ in unzipped_chunks:   # must consume
                            pass
                        continue

                    # ── Skip already-extracted files ──────────────────────────
                    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                        for _ in unzipped_chunks:   # must consume to keep stream in sync
                            pass
                        skipped += 1
                        pbar.set_postfix(ext=extracted, skip=skipped, err=errors)
                        continue

                    # ── Extract ───────────────────────────────────────────────
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    current_out = out_path
                    try:
                        with out_path.open('wb') as fh:
                            for chunk in unzipped_chunks:
                                fh.write(chunk)
                        extracted += 1
                        current_out = None
                    except Exception as exc:
                        for _ in unzipped_chunks:   # drain remaining chunks
                            pass
                        errors += 1
                        if current_out and current_out.exists():
                            current_out.unlink(missing_ok=True)
                        current_out = None
                        pbar.write(f'  [warn] {fname}: {exc}')

                    pbar.set_postfix(ext=extracted, skip=skipped, err=errors)

            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                attempt += 1
                if attempt > MAX_RETRIES:
                    pbar.close()
                    raise
                delay = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                pbar.write(
                    f'  Connection dropped ({exc.__class__.__name__}) — '
                    f'retry {attempt}/{MAX_RETRIES} in {delay}s …'
                )
                if current_out and current_out.exists():
                    current_out.unlink(missing_ok=True)
                pbar.close()
                time.sleep(delay)
                continue  # restart

            except KeyboardInterrupt:
                if current_out and current_out.exists():
                    current_out.unlink(missing_ok=True)
                pbar.close()
                raise

            pbar.close()
            break  # success

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ):
            pass  # handled above

    print(
        f'  Stream extract complete: '
        f'{extracted} extracted, {skipped} skipped, {errors} errors'
    )


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
    """Resumable streaming download with automatic reconnect on dropped connections.

    On every connection error the file offset is re-read from disk, a new
    Range request is issued, and the tqdm bar resumes from where it left off.
    Up to MAX_RETRIES reconnects are attempted before giving up.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    pbar: tqdm | None = None

    try:
        while attempt <= MAX_RETRIES:
            existing = dest.stat().st_size if dest.exists() else 0

            headers: dict[str, str] = {}
            if existing:
                headers["Range"] = f"bytes={existing}-"

            try:
                with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                    if resp.status_code == 416:
                        print(f"  {dest.name}: already fully downloaded.")
                        return
                    resp.raise_for_status()

                    content_len = int(resp.headers.get("Content-Length", 0))
                    total = content_len + existing if content_len else (expected_size or None)
                    mode = "ab" if existing and resp.status_code == 206 else "wb"

                    if pbar is None:
                        pbar = tqdm(
                            total=total,
                            initial=existing,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=dest.name,
                            leave=True,
                        )
                    else:
                        # Update total/initial in case Content-Length changed on resume
                        pbar.total = total
                        pbar.n = existing
                        pbar.last_print_n = existing
                        pbar.refresh()

                    with dest.open(mode) as fh:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            fh.write(chunk)
                            pbar.update(len(chunk))

                    return  # success

            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                delay = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                restarted = dest.stat().st_size if dest.exists() else 0
                if pbar is not None:
                    pbar.write(
                        f"  Connection dropped ({exc.__class__.__name__}) after "
                        f"{restarted/1e9:.2f} GB — retry {attempt}/{MAX_RETRIES} "
                        f"in {delay}s …"
                    )
                time.sleep(delay)

    finally:
        if pbar is not None:
            pbar.close()


# ── Extract ───────────────────────────────────────────────────────────────────

def _flatten_if_double_nested(dest_dir: Path, archive_stem: str) -> None:
    """If dest_dir/archive_stem/ exists, move its contents up and remove it."""
    inner = dest_dir / archive_stem
    if not inner.is_dir():
        return
    import shutil
    print(f"  Flattening double-nested extraction for {archive_stem} …")
    for item in list(inner.iterdir()):
        target = dest_dir / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(dest_dir))
    inner.rmdir()
    print(f"  Flattened: {dest_dir}")


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
    _flatten_if_double_nested(dest_dir, zip_path.stem)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    trace: str,
    out_dir: Path,
    dry_run: bool = False,
    only: str | None = None,
    stream_extract: bool = False,
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

        # ── Stream-extract mode: download + extract simultaneously ───────────
        if stream_extract:
            print(
                f"  {name}: stream-extract ({(size or 0)/1e9:.2f} GB) "
                f"→ {unpacks_to} …"
            )
            _stream_extract_zip(url, unpacks_to, size)
            continue  # skip ZIP save / MD5 / regular extract

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
    parser.add_argument(
        "--stream-extract",
        action="store_true",
        help=(
            "Download and extract simultaneously without saving the ZIP. "
            "Saves ~350 GB of disk space. Skips already-extracted files on resume."
        ),
    )
    args = parser.parse_args(argv)
    out_dir = args.out or (RAW_DIR / args.trace)
    run(args.trace, out_dir, dry_run=args.dry_run, only=args.only,
        stream_extract=args.stream_extract)


if __name__ == "__main__":
    main()
