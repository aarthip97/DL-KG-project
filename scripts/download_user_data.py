#!/usr/bin/env python3
"""
scripts/download_user_data.py
==============================
Download the Echo Nest Taste Profile (user–song interaction data) and the
MSD/Echo Nest mismatch list needed to clean it.

Assets
------
  train_triplets.txt.zip   ~500 MB  → data/raw/train_triplets.txt
  sid_mismatches.txt       ~150 KB  → data/raw/sid_mismatches.txt

Sources
-------
  http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip
  http://millionsongdataset.com/sites/default/files/tasteprofile/sid_mismatches.txt

Usage
-----
    python scripts/download_user_data.py
    python scripts/download_user_data.py --dest /mnt/data/user
    python scripts/download_user_data.py --only mismatches
    python scripts/download_user_data.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

ROOT = Path(__file__).parents[1]   # project root  (DL-KG-project/)

ASSETS: Dict[str, Dict] = {
    "triplets": {
        "url": "http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip",
        "filename": "train_triplets.txt.zip",
        "unpack": True,
        "unpack_member": "train_triplets.txt",   # file inside the zip
        "description": "Echo Nest Taste Profile  (~48 M user–song–play-count rows)",
    },
    "mismatches": {
        "url": "http://millionsongdataset.com/sites/default/files/tasteprofile/sid_mismatches.txt",
        "filename": "sid_mismatches.txt",
        "unpack": False,
        "description": "Echo Nest / MSD mismatch list  (~19 k bad song-ID pairs)",
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
    """Download *url* to *dest*, with a progress bar."""
    if dest.exists() and skip_existing:
        print(f"    ↩  Already exists — skipping  ({_human(dest.stat().st_size)})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    req  = Request(url, headers={"User-Agent": "Mozilla/5.0 DL-KG-project-downloader"})
    resp = urlopen(req, timeout=120)

    total = int(resp.headers.get("Content-Length", 0))
    chunk = 1 << 20   # 1 MB

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

    print(f"       ✓  {_human(downloaded)} in {time.time() - t0:.1f}s")
    return dest


def _unzip_member(archive: Path, member_name: str, dest_dir: Path,
                  skip_existing: bool) -> Path:
    """Extract a single member from a zip archive."""
    out = dest_dir / member_name
    if out.exists() and skip_existing:
        print(f"    ↩  Already unpacked  ({_human(out.stat().st_size)}  →  {_rp(out)})")
        return out

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"    📦 Extracting {member_name} from {archive.name} …", flush=True)
    t0 = time.time()

    with zipfile.ZipFile(archive, "r") as zf:
        # find the member (may be nested inside a directory in the zip)
        names = zf.namelist()
        match = next((n for n in names if Path(n).name == member_name), None)
        if match is None:
            # fall back to exact name
            if member_name in names:
                match = member_name
            else:
                raise FileNotFoundError(
                    f"{member_name} not found in {archive}. "
                    f"Available: {names[:10]}"
                )
        info = zf.getinfo(match)
        info.filename = member_name   # extract flat (no sub-dirs)
        zf.extract(info, dest_dir)

    size = out.stat().st_size
    print(f"    ✓  {_human(size)} extracted → {_rp(out)}  ({time.time()-t0:.1f}s)")
    return out


# ── manifest ──────────────────────────────────────────────────────────────────

def _write_manifest(dest: Path) -> Path:
    """Append user-data entries to (or create) data/raw/user_manifest.json."""
    manifest: dict = {}
    for key, asset in ASSETS.items():
        archive = dest / asset["filename"]
        unpacked: str | None = None
        if asset.get("unpack") and asset.get("unpack_member"):
            unpacked = str(dest / asset["unpack_member"])
        manifest[key] = {
            "archive":  str(archive),
            "unpacked": unpacked,
            "url":      asset["url"],
            "exists":   archive.exists(),
        }
        if unpacked:
            manifest[key]["unpacked_exists"] = Path(unpacked).exists()

    path = dest / "user_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Echo Nest Taste Profile + MSD mismatch list."
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
    print("  DL-KG-project — Echo Nest Taste Profile downloader")
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

        if asset.get("unpack") and asset.get("unpack_member") and not args.no_unpack:
            try:
                _unzip_member(archive, asset["unpack_member"], dest,
                              skip_existing=args.skip_existing)
            except Exception as exc:
                print(f"    ✗  Unpack failed: {exc}", file=sys.stderr)

    manifest_path = _write_manifest(dest)
    print(f"\n{'='*w}")
    print(f"  ✓  Done.  Manifest → {_rp(manifest_path)}")
    print(f"{'='*w}\n")


if __name__ == "__main__":
    main()
