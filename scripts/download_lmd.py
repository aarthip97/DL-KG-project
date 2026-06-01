#!/usr/bin/env python3
"""
scripts/download_lmd.py
=======================
Download and unpack the three Lakh MIDI Dataset assets needed by this project:

  lmd_matched       ~1.5 GB tar.gz  → data/raw/lmd_matched/
  lmd_matched_h5    ~2.5 GB tar.gz  → data/raw/lmd_matched_h5/
  match_scores.json ~9  MB          → data/raw/match_scores.json

Usage
-----
    python scripts/download_lmd.py                    # full download
    python scripts/download_lmd.py --skip-existing    # skip already-present files
    python scripts/download_lmd.py --only match_scores
    python scripts/download_lmd.py --dest /mnt/data/lmd

References
----------
    https://colinraffel.com/projects/lmd/
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

ROOT = Path(__file__).parents[1]   # project root (DL-KG-project/)

ASSETS: Dict[str, Dict] = {
    "lmd_matched": {
        "url":       "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
        "filename":  "lmd_matched.tar.gz",
        "unpack":    True,
        "unpack_to": "lmd_matched",
        "description": "LMD-matched  (~45 k MIDI files matched to MSD)",
    },
    "lmd_matched_h5": {
        "url":       "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz",
        "filename":  "lmd_matched_h5.tar.gz",
        "unpack":    True,
        "unpack_to": "lmd_matched_h5",
        "description": "MSD HDF5 files for every LMD-matched entry",
    },
    "match_scores": {
        "url":       "http://hog.ee.columbia.edu/craffel/lmd/match_scores.json",
        "filename":  "match_scores.json",
        "unpack":    False,
        "description": "DTW match scores (track_id → md5 → score)",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _rp(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _download(url: str, dest: Path, skip_existing: bool) -> Path:
    if dest.exists() and skip_existing:
        print(f"    ↩  Already exists — skipping  ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    req  = Request(url, headers={"User-Agent": "Mozilla/5.0 DL-KG-project-downloader"})
    resp = urlopen(req, timeout=60)

    total = int(resp.headers.get("Content-Length", 0))
    chunk = 1 << 20  # 1 MB

    print(f"    ↓  {url}")
    print(f"       → {_rp(dest)}  ({_human(total) if total else 'size unknown'})")

    downloaded = 0
    t0 = time.time()
    bar = _tqdm(total=total, unit="B", unit_scale=True,
                unit_divisor=1024, ncols=80) if HAS_TQDM and total else None

    with open(dest, "wb") as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            downloaded += len(block)
            if bar:
                bar.update(len(block))
            elif total:
                pct   = downloaded / total * 100
                speed = downloaded / max(time.time() - t0, 1e-3)
                print(f"\r       {pct:5.1f}%  {_human(downloaded)}/{_human(total)}"
                      f"  @ {_human(int(speed))}/s", end="", flush=True)

    if bar:
        bar.close()
    else:
        print()

    print(f"       ✓  {_human(downloaded)} in {time.time()-t0:.1f}s")
    return dest


def _unpack(archive: Path, target_dir: Path, skip_existing: bool) -> None:
    if target_dir.exists() and skip_existing:
        n = sum(1 for _ in target_dir.rglob("*") if _.is_file())
        print(f"    ↩  Already unpacked  ({n:,} files in {_rp(target_dir)})")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Unpacking {archive.name} → {_rp(target_dir)} …", flush=True)

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        bar     = _tqdm(total=len(members), unit="files",
                        ncols=80) if HAS_TQDM else None
        for i, member in enumerate(members, 1):
            # strip the top-level tar directory so files land directly in target_dir
            parts = Path(member.name).parts
            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
            tf.extract(member, path=target_dir)
            if bar:
                bar.update(1)
            elif i % 5000 == 0:
                print(f"\r       {i:,}/{len(members):,} …", end="", flush=True)
        if bar:
            bar.close()
        else:
            print()

    n = sum(1 for _ in target_dir.rglob("*") if _.is_file())
    print(f"    ✓  {n:,} files extracted → {_rp(target_dir)}")


# ── manifest ──────────────────────────────────────────────────────────────────

def _write_manifest(dest: Path) -> Path:
    manifest = {
        key: {
            "archive":  str(dest / asset["filename"]),
            "unpacked": str(dest / asset["unpack_to"]) if asset.get("unpack") else None,
            "url":      asset["url"],
        }
        for key, asset in ASSETS.items()   # always write all three entries
    }
    path = dest / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download & unpack the Lakh MIDI Dataset (matched only)."
    )
    p.add_argument("--dest", default=str(ROOT / "data" / "raw"),
                   help="Destination directory  (default: data/raw/)")
    p.add_argument("--only", nargs="+", choices=list(ASSETS), metavar="ASSET",
                   help=f"Download only: {', '.join(ASSETS)}")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip", dest="skip_existing", action="store_false")
    p.add_argument("--no-unpack", action="store_true",
                   help="Download archives but skip extraction.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    dest   = Path(args.dest).resolve()
    assets = {k: v for k, v in ASSETS.items()
              if not args.only or k in args.only}

    dest.mkdir(parents=True, exist_ok=True)

    w = 65
    print(f"\n{'='*w}")
    print("  DL-KG-project — Lakh MIDI Dataset downloader")
    print(f"  Destination : {_rp(dest)}")
    print(f"  Assets      : {', '.join(assets)}")
    print(f"{'='*w}\n")

    for key, asset in assets.items():
        print(f"\n── {key}  ──  {asset['description']}")
        archive = dest / asset["filename"]
        try:
            _download(asset["url"], archive, skip_existing=args.skip_existing)
        except Exception as exc:
            print(f"    ✗  Download failed: {exc}", file=sys.stderr)
            continue

        if asset.get("unpack") and not args.no_unpack:
            try:
                _unpack(archive, dest / asset["unpack_to"],
                        skip_existing=args.skip_existing)
            except Exception as exc:
                print(f"    ✗  Unpack failed: {exc}", file=sys.stderr)

    manifest_path = _write_manifest(dest)
    print(f"\n{'='*w}")
    print(f"  ✓  Done.  Manifest → {_rp(manifest_path)}")
    print(f"{'='*w}\n")


if __name__ == "__main__":
    main()
