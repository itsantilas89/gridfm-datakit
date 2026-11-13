"""Shared folder layout and file-naming convention for the grid snapshots.

  <repo>/scripts/grids/data_rte/raw/        snapshots exactly as downloaded           YYYY-MM-DD-HHMM.xiidm.bz2
  <repo>/scripts/grids/data_rte/connected/  the same snapshots after connect_grid.py  YYYY-MM-DD-HHMM.xiidm.bz2
  <repo>/scripts/grids/data_rte/base/       snapshots after make_base_case.py         YYYY-MM-DD-HHMM-base.xiidm.bz2

Generated output stays out of git: images/ (plot_grid.py) and reports/ (check_network.py)
next to this file, and scripts/grids/data_rte/ (all three gitignored).

A connected snapshot has the same file name as the raw one it came from. Non-default
variants of the dataset ("brut" instead of "enrichi") get a "-brut" suffix so they never
overwrite the default one.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "scripts" / "grids" / "data_rte"
RAW_DIR = DATA_DIR / "raw"
CONNECTED_DIR = DATA_DIR / "connected"
BASE_DIR = DATA_DIR / "base"
IMAGES_DIR = HERE / "images"
REPORTS_DIR = HERE / "reports"

EXTENSIONS = (".xiidm.bz2", ".xiidm")
DEFAULT_VARIANT = "enrichi"

# recollement-auto-20210103-0000-enrichi.xiidm.bz2 -> date, time, variant
_REPO_NAME = re.compile(r"-(\d{4})(\d{2})(\d{2})-(\d{4})(?:-(\w+))?\.xiidm")


def snapshot_stem(repo_path: str) -> str:
    """'2021/01/03/recollement-auto-20210103-0000-enrichi.xiidm.bz2' -> '2021-01-03-0000'."""
    m = _REPO_NAME.search(Path(repo_path).name)
    if not m:
        raise ValueError(f"Cannot read date/time from {repo_path!r}")
    y, mo, d, hhmm, variant = m.groups()
    stem = f"{y}-{mo}-{d}-{hhmm}"
    return stem if variant in (None, DEFAULT_VARIANT) else f"{stem}-{variant}"


def extension_of(p: Path) -> str:
    return next((e for e in EXTENSIONS if p.name.endswith(e)), p.suffix)


def list_snapshots(folder: Path) -> list[Path]:
    """Snapshot files directly inside `folder`, sorted by name (= by date)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith(EXTENSIONS))


def newest_snapshot(*folders: Path) -> Path | None:
    """The most recently modified snapshot across the given folders (None if there are none)."""
    files = [p for f in folders for p in list_snapshots(f)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None
