#!/usr/bin/env python
"""Publish a datakit run, with the grids it was made from, to a HuggingFace dataset repo.

Repo layout (one folder per snapshot, one folder per generation run):

  README.md, LICENSE                  dataset card and licence (regenerated on every publish)
  <snapshot>/                         e.g. 2023-12-31-2355
    latest.json                       which scenario version and grid are the current ones
    grids/
      raw/                            the OpenSynth snapshot, unchanged
      connected/                      after connect_grid.py
      base/                           after make_base_case.py (+ report)
      opf/                            after relax_to_opf_feasible.py: the grid datakit ran on (+ reports, CSVs)
    scenarios/
      <version>/                      v1, v2, ... never overwritten
        README.md, run.json           notes, known issues, seed, counts, commit
        config.yaml                   the datakit config as it ran (paths made relative; seed included)
        data/                         *_data.parquet (partitioned), runtime, load profile, stats, plots

HuggingFace has no symlinks, so "latest" is a pointer file, not a copy of the data.
Files are hard-linked into a staging folder (no extra disk space) and uploaded in several
commits (at most ~2 GB / 250 files each). The metadata files (README, run.json, config, latest.json,
LICENSE) go in the last commit, so `latest.json` only moves once all the data is there. After the
upload every remote file is compared with the staged one (size, and SHA-256 for LFS files).
If an upload is interrupted, re-run the same command with --resume: files already on the
remote are skipped. Needs a write token: `hf auth login` or HF_TOKEN.

  python gridfm_datakit/xiidm_process/publish_to_hf.py \
      --repo-id Juanillo02/opensynth_rte7k_synthetic_scenarios \
      --run-dir data_out/rte7k_full_1000/2023-12-31-2355-full-opf/raw --version v1 \
      --note "Generated before the anti-parallel branch flow fix: 4 branches have pf/pt, qf/qt at swapped ends." \
      --dry-run
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_paths import BASE_DIR, CONNECTED_DIR, RAW_DIR, REPO_ROOT, sibling  # noqa: E402

SOURCE_REPO = "OpenSynth/D-GITT-RTE7000-2023"
LICENSE_ID = "cdla-permissive-2.0"
CDLA_TEXT = """Community Data License Agreement – Permissive – Version 2.0

This is the Community Data License Agreement – Permissive, Version 2.0 (the “agreement”). Data Provider(s) and Data Recipient(s) agree as follows:

1. Provision of the Data

1.1. A Data Recipient may use, modify, and share the Data made available by Data Provider(s) under this agreement if that Data Recipient follows the terms of this agreement.

1.2. This agreement does not impose any restriction on a Data Recipient’s use, modification, or sharing of any portions of the Data that are in the public domain or that may be used, modified, or shared under any other legal exception or limitation.

2. Conditions for Sharing Data

2.1. A Data Recipient may share Data, with or without modifications, so long as the Data Recipient makes available the text of this agreement with the shared Data.

3. No Restrictions on Results

3.1. This agreement does not impose any restriction or obligations with respect to the use, modification, or sharing of Results.

4. No Warranty; Limitation of Liability

4.1. All Data Recipients receive the Data subject to the following terms:

THE DATA IS PROVIDED ON AN “AS IS” BASIS, WITHOUT REPRESENTATIONS, WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

NO DATA PROVIDER SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE DATA OR RESULTS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

5. Definitions

5.1. “Data” means the material received by a Data Recipient under this agreement.

5.2. “Data Provider” means any person who is the source of Data provided under this agreement and in reliance on a Data Recipient’s agreement to its terms.

5.3. “Data Recipient” means any person who receives Data directly or indirectly from a Data Provider and agrees to the terms of this agreement.

5.4. “Results” means any outcome obtained by computational analysis of Data, including for example machine learning models and models’ insights.

Copyright © 2017 The Linux Foundation®.
"""

# run-folder entries that go to scenarios/<version>/data/ (logs with absolute paths and
# solver logs are left out; checkpoint.json only matters for resuming)
DATA_ENTRIES = (
    "bus_data.parquet", "branch_data.parquet", "gen_data.parquet", "y_bus_data.parquet",
    "runtime_data.parquet", "scenarios_agg_load_profile.parquet", "stats.parquet",
    "stats_plot.png", "feature_plots", "error.log", "n_scenarios.txt", "args.log",
)


def link(src: Path, dst: Path) -> None:
    """Hard-link a file (or a folder tree) into the staging area, falling back to a copy."""
    if src.is_dir():
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            link(f, dst / f.relative_to(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


MAX_COMMIT_BYTES = 2_000_000_000
MAX_COMMIT_FILES = 250
METADATA_NAMES = {"README.md", "run.json", "config.yaml", "latest.json", "LICENSE"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def is_metadata(rel: Path) -> bool:
    """Small text files written by this script (not the datakit logs under data/ nor grid reports)."""
    return rel.name in METADATA_NAMES and "data" not in rel.parts and "grids" not in rel.parts


def plan_commits(files: list[Path], stage: Path) -> list[list[Path]]:
    """Group data files into commits of bounded size, metadata last."""
    data = [f for f in files if not is_metadata(f.relative_to(stage))]
    meta = [f for f in files if is_metadata(f.relative_to(stage))]
    batches, cur, size = [], [], 0
    for f in data:
        sz = f.stat().st_size
        if cur and (size + sz > MAX_COMMIT_BYTES or len(cur) >= MAX_COMMIT_FILES):
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += sz
    if cur:
        batches.append(cur)
    if meta:
        batches.append(meta)
    return batches


def remote_files(api, repo_id: str) -> dict:
    return {e.path: e for e in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True) if hasattr(e, "size")}


def verify_remote(api, repo_id: str, stage: Path, files: list[Path]) -> list[str]:
    """Return the problems found comparing the staged files with the remote repo (empty list = all good)."""
    remote = remote_files(api, repo_id)
    problems = []
    for f in files:
        rel = f.relative_to(stage).as_posix()
        e = remote.get(rel)
        if e is None:
            problems.append(f"missing on remote: {rel}")
        elif e.size != f.stat().st_size:
            problems.append(f"size differs: {rel} (remote {e.size}, local {f.stat().st_size})")
        elif getattr(e, "lfs", None) and e.lfs.sha256 != sha256(f):
            problems.append(f"sha256 differs: {rel}")
    return problems


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def relative_config(run_dir: Path) -> dict:
    """The run's config.yaml with the absolute paths of this machine made repo-relative."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    root = str(REPO_ROOT) + "/"

    def fix(v):
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v.replace(root, "") if isinstance(v, str) else v

    return fix(cfg)


def parquet_counts(run_dir: Path) -> dict:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    out = {}
    for name in ("bus_data", "branch_data", "gen_data"):
        out[f"{name}_rows"] = ds.dataset(run_dir / f"{name}.parquet", format="parquet").count_rows()
    rt = pq.read_table(run_dir / "runtime_data.parquet", columns=["scenario"])
    out["scenarios_generated"] = len(set(rt.column("scenario").to_pylist()))
    return out


def remote_versions(api, repo_id: str, snapshot: str) -> list[str]:
    try:
        tree = api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=f"{snapshot}/scenarios")
        return sorted(Path(e.path).name for e in tree if e.path.count("/") == 2)
    except Exception:
        return []


def dataset_card(snapshots: dict[str, list[str]], latest: dict[str, str]) -> str:
    rows = "\n".join(
        f"| `{snap}` | " + ", ".join(f"`{v}`" + (" (latest)" if latest.get(snap) == v else "") for v in vs) + " |"
        for snap, vs in sorted(snapshots.items())
    )
    return f"""---
license: {LICENSE_ID}
pretty_name: OpenSynth RTE7000 synthetic scenarios
tags:
  - power-systems
  - ac-opf
  - power-flow
  - gridfm
  - synthetic
size_categories:
  - 1M<n<10M
---

# OpenSynth RTE7000: synthetic scenarios generated with gridfm-datakit

Power-flow scenarios generated with [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) on the French
transmission grid snapshots of [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}),
together with the grid files they were made from.

**The data is synthetic, not real RTE operating data.** The raw snapshots have no load and no dispatch, so a
synthetic load (and a plausible dispatch, which datakit itself does not need) was invented, and a few limits were widened so that the AC OPF has a solution
(each change is reported in the `grids/` folder of every snapshot).

## Layout

```
<snapshot>/
  latest.json                  current scenario version and grid for this snapshot
  grids/raw/                   the OpenSynth snapshot, unchanged
  grids/connected/             switches closed, split busbars joined        (connect_grid.py)
  grids/base/                  load, dispatch, slack, regulation, limits    (make_base_case.py)
  grids/opf/                   limits widened until the OPF solves          (relax_to_opf_feasible.py)
  scenarios/<version>/         one generation run; versions are never overwritten
    README.md, run.json        notes and known issues, seed, counts, datakit commit
    config.yaml                the datakit configuration (seed included)
    data/                      bus_data, branch_data, gen_data, y_bus_data, runtime_data (parquet),
                               scenarios_agg_load_profile.parquet, stats and plots, error.log
```

| Snapshot | Scenario versions |
|---|---|
{rows}

Always read `<snapshot>/latest.json` to find the current version (HuggingFace has no symlinks, so it is a pointer
and not a copy). Each version's `README.md` lists its known issues.

## Reading the data

```python
import json
import pandas as pd
from huggingface_hub import snapshot_download

repo = "Juanillo02/opensynth_rte7k_synthetic_scenarios"
snap = "2023-12-31-2355"
latest = json.load(open(snapshot_download(repo, repo_type="dataset", allow_patterns=f"{{snap}}/latest.json") + f"/{{snap}}/latest.json"))
root = snapshot_download(repo, repo_type="dataset", allow_patterns=f"{{snap}}/scenarios/{{latest['scenarios']}}/data/*")
bus = pd.read_parquet(f"{{root}}/{{snap}}/scenarios/{{latest['scenarios']}}/data/bus_data.parquet")
```

The tables are partitioned by `scenario_partition` (100 scenarios each); filter on `scenario` to read a subset.
To regenerate, build the grid with the `gridfm_datakit/xiidm_process` pipeline of
[gridfm-datakit](https://github.com/itsantilas89/gridfm-datakit/tree/exp/rte-base-case) (or take it from `grids/opf/`)
and run `gridfm_datakit generate` with the version's `config.yaml` (change the seed for new samples).

## Provenance and licence

- Grid snapshots: [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}), licence CDLA-Permissive-2.0.
  The `raw/` files are the original ones (MD5 checked); the other grid files are derived from them.
- This repository is distributed under the same licence, CDLA-Permissive-2.0 (text in `LICENSE`).
"""


def version_readme(snapshot: str, version: str, run: dict, notes: list[str]) -> str:
    cfg = run["config"]
    issues = "\n".join(f"- {n}" for n in notes) if notes else "- none known"
    return f"""# {snapshot} / {version}

| | |
|---|---|
| Scenarios requested / generated | {cfg['load']['scenarios']} / {run['counts']['scenarios_generated']} |
| Mode | `{cfg['settings']['mode']}` (`pf`: limits may be violated) |
| Seed | {cfg['settings']['seed']} |
| Load noise sigma / global range | {cfg['load']['sigma']} / {cfg['load']['global_range']} |
| Grid | `grids/opf/{run['grid']['opf']}` |
| Bus / branch / generator rows | {run['counts']['bus_data_rows']:,} / {run['counts']['branch_data_rows']:,} / {run['counts']['gen_data_rows']:,} |
| datakit commit | `{run['datakit_commit']}` |
| Published | {run['published']} |

Scenarios that failed in the OPF are missing from the tables; `data/error.log` lists them.

## Known issues

{issues}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="HuggingFace dataset repo, e.g. user/name")
    ap.add_argument("--run-dir", type=Path, required=True, help="datakit output folder holding the *_data.parquet files (…/raw)")
    ap.add_argument("--version", required=True, help="scenario version folder to create, e.g. v1 (refused if it already exists)")
    ap.add_argument("--note", action="append", default=[], help="known issue to list for this version (repeatable)")
    ap.add_argument("--commit", help="datakit commit the run was made with (default: current HEAD; set it when publishing an older run)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted upload of this version: skip files already on the remote")
    ap.add_argument("--no-latest", action="store_true", help="do not point latest.json at this version")
    ap.add_argument("--stage-dir", type=Path, default=REPO_ROOT / "data_out" / "hf_stage")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and print the tree, upload nothing")
    a = ap.parse_args()

    run_dir = a.run_dir.resolve()
    cfg = relative_config(run_dir)
    opf = Path(cfg["network"]["file"])
    m = re.match(r"(\d{4}-\d{2}-\d{2}-\d{4})", opf.name)
    if not m:
        sys.exit(f"cannot read the snapshot timestamp from {opf.name}")
    snapshot = m.group(1)
    base_name = opf.name.replace("-opf.xiidm.bz2", ".xiidm.bz2")
    grid_files = {
        "raw": [RAW_DIR / f"{snapshot}.xiidm.bz2"],
        "connected": [CONNECTED_DIR / f"{snapshot}.xiidm.bz2"],
        "base": [BASE_DIR / base_name, sibling(BASE_DIR / base_name, ".report.json")],
        "opf": [REPO_ROOT / opf] + [
            sibling(REPO_ROOT / opf, t)
            for t in (".report.json", ".relaxed_branches.csv", ".relaxed_voltage_levels.csv", ".relaxed_generators.csv")
        ],
    }
    missing = [str(p) for fs in grid_files.values() for p in fs if not p.exists()]
    missing += [str(run_dir / e) for e in DATA_ENTRIES[:6] if not (run_dir / e).exists()]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    existing = remote_versions(api, a.repo_id, snapshot)
    if a.version in existing and not a.resume:
        sys.exit(f"{snapshot}/scenarios/{a.version} already exists in {a.repo_id}; versions are never overwritten (use --resume to finish an interrupted upload)")

    # the raw file must be the OpenSynth original
    remote_md5 = Path(hf_hub_download(
        SOURCE_REPO, f"{snapshot[:4]}/{snapshot[5:7]}/{snapshot[8:10]}/recollement-auto-{snapshot[:10].replace('-', '')}-{snapshot[11:]}-enrichi.xiidm.bz2.md5",
        repo_type="dataset", local_dir=a.stage_dir.parent / "hf_md5")).read_text().split()[0]
    if md5(grid_files["raw"][0]) != remote_md5:
        sys.exit("the raw snapshot does not match the OpenSynth MD5")

    if a.stage_dir.exists():
        shutil.rmtree(a.stage_dir)
    snap_dir = a.stage_dir / snapshot
    for kind, files in grid_files.items():
        for f in files:
            link(f, snap_dir / "grids" / kind / f.name)

    vdir = snap_dir / "scenarios" / a.version
    for e in DATA_ENTRIES:
        if (run_dir / e).exists():
            link(run_dir / e, vdir / "data" / e)
    (vdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    run = {
        "snapshot": snapshot, "version": a.version, "config": cfg,
        "grid": {"opf": opf.name, "base": base_name, "raw_md5": remote_md5},
        "counts": parquet_counts(run_dir),
        "datakit_commit": a.commit or git("rev-parse", "--short", "HEAD"),
        "known_issues": a.note,
        "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    (vdir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (vdir / "README.md").write_text(version_readme(snapshot, a.version, run, a.note))

    latest = {} if a.no_latest else {snapshot: a.version}
    if a.no_latest:
        try:
            latest_file = Path(hf_hub_download(a.repo_id, f"{snapshot}/latest.json", repo_type="dataset", local_dir=a.stage_dir.parent / "hf_md5"))
            latest = {snapshot: json.loads(latest_file.read_text())["scenarios"]}
        except Exception:
            pass
    else:
        (snap_dir / "latest.json").write_text(json.dumps(
            {"scenarios": a.version, "grid": f"grids/opf/{opf.name}", "config": f"scenarios/{a.version}/config.yaml"}, indent=2) + "\n")
    (a.stage_dir / "README.md").write_text(dataset_card({snapshot: sorted(set(existing + [a.version]))}, latest))
    (a.stage_dir / "LICENSE").write_text(CDLA_TEXT)

    files = sorted(p for p in a.stage_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files) / 1e9
    for p in files:
        rel = p.relative_to(a.stage_dir)
        if len(rel.parts) < 6 or "partition" not in str(rel):
            print(f"  {rel}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n{len(files)} files, {total:.2f} GB staged in {a.stage_dir}")
    if a.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import CommitOperationAdd

    api.create_repo(a.repo_id, repo_type="dataset", exist_ok=True)
    done = {}
    if a.resume:
        done = {p: e.size for p, e in remote_files(api, a.repo_id).items()}
    todo = [f for f in files if done.get(f.relative_to(a.stage_dir).as_posix()) != f.stat().st_size]
    # the version pointer and cards are always re-sent last, they are tiny
    todo += [f for f in files if is_metadata(f.relative_to(a.stage_dir)) and f not in todo]
    batches = plan_commits(todo, a.stage_dir)
    print(f"{len(todo)} files to upload in {len(batches)} commits ({len(files) - len(todo)} already on the remote)")
    for i, batch in enumerate(batches, 1):
        ops = [CommitOperationAdd(path_in_repo=f.relative_to(a.stage_dir).as_posix(), path_or_fileobj=str(f)) for f in batch]
        msg = f"Add {snapshot} scenarios {a.version} ({i}/{len(batches)})"
        for attempt in range(1, 4):
            try:
                api.create_commit(repo_id=a.repo_id, repo_type="dataset", operations=ops, commit_message=msg)
                break
            except Exception as e:  # network hiccup: retry, then give up (re-run with --resume)
                print(f"  commit {i}/{len(batches)} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
                if attempt == 3:
                    sys.exit(f"upload stopped at commit {i}/{len(batches)}; re-run with --resume")
        print(f"  commit {i}/{len(batches)} done: {len(batch)} files, {sum(f.stat().st_size for f in batch) / 1e9:.2f} GB")

    problems = verify_remote(api, a.repo_id, a.stage_dir, files)
    if problems:
        sys.exit("verification FAILED:\n  " + "\n  ".join(problems[:20]))
    print(f"verified: all {len(files)} files on the remote match the staged ones (size, SHA-256 for LFS)")
    print(f"uploaded: https://huggingface.co/datasets/{a.repo_id}")


if __name__ == "__main__":
    main()
