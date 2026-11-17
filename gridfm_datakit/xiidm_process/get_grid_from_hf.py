#!/usr/bin/env python3
"""Download one .xiidm.bz2 snapshot from OpenSynth/D-GITT-RTE7000-2023, verify its MD5,
and store it as data_rte/raw/YYYY-MM-DD-HHMM.xiidm.bz2.

Usage:
  python get_grid_from_hf.py                          # first snapshot of 2023-01-01
  python get_grid_from_hf.py --date 2023-02-15        # first snapshot of that day
  python get_grid_from_hf.py --date 2023-12-31 --time 2355
  python get_grid_from_hf.py --path 2023/12/31/recollement-auto-20231231-2355-enrichi.xiidm.bz2
  python get_grid_from_hf.py --decompress             # also write the .xiidm

Requires: pip install huggingface_hub
"""
import argparse
import bz2
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from grid_paths import RAW_DIR, snapshot_stem

REPO = "OpenSynth/D-GITT-RTE7000-2023"
REPO_TYPE = "dataset"


def pick_path(api: HfApi, date: str, time: str | None, variant: str | None) -> str:
    y, m, d = date.split("-")
    folder = f"{y}/{m}/{d}"
    files = sorted(
        e.path
        for e in api.list_repo_tree(REPO, repo_type=REPO_TYPE, path_in_repo=folder)
        if e.path.endswith(".xiidm.bz2")
    )
    if time:
        files = [f for f in files if f"-{y}{m}{d}-{time}-" in f]
    if variant:
        files = [f for f in files if f.endswith(f"-{variant}.xiidm.bz2")]
    if not files:
        sys.exit(f"No matching .bz2 in {folder} (time={time}, variant={variant}). "
                 "The snapshot may be listed in the dataset's missing-files list.")
    return files[0]


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="exact repo path, e.g. 2023/12/31/...xiidm.bz2")
    ap.add_argument("--date", default="2023-01-01", help="YYYY-MM-DD (ignored if --path)")
    ap.add_argument("--time", help="HHMM, e.g. 1425 (ignored if --path)")
    ap.add_argument("--variant", choices=["brut", "enrichi"])
    ap.add_argument("--out", type=Path, default=RAW_DIR, help=f"output directory (default: {RAW_DIR})")
    ap.add_argument("--decompress", action="store_true", help="also write the .xiidm")
    args = ap.parse_args()

    api = HfApi()
    path = args.path.lstrip("./") if args.path else pick_path(api, args.date, args.time, args.variant)
    final = args.out / f"{snapshot_stem(path)}.xiidm.bz2"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO}/{path}")

    # hf_hub_download mirrors the repo's folder layout, so fetch into a scratch folder
    # and move the verified file to its final, flat name.
    scratch = args.out / ".download"
    try:
        local = Path(hf_hub_download(REPO, path, repo_type=REPO_TYPE, local_dir=scratch))

        # Verify against the .md5 sidecar shipped next to each file.
        try:
            md5_file = Path(hf_hub_download(REPO, path + ".md5", repo_type=REPO_TYPE, local_dir=scratch))
            expected = md5_file.read_text().split()[0].strip().lower()
            actual = md5sum(local)
            if actual != expected:
                sys.exit(f"MD5 mismatch: expected {expected}, got {actual}")
            print(f"MD5 OK ({actual})")
        except Exception as e:  # sidecar absent or unreadable
            print(f"MD5 check skipped: {e}")

        if final.exists():
            print(f"Replacing existing {final.name}")
        shutil.move(str(local), final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"Saved {final} ({final.stat().st_size / 1e6:.2f} MB)")

    if args.decompress:
        xiidm = final.with_suffix("")  # strip .bz2
        with bz2.open(final, "rb") as src, xiidm.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"Decompressed {xiidm} ({xiidm.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
