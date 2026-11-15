# xiidm_process

Turns an RTE snapshot from the HuggingFace dataset
[OpenSynth/D-GITT-RTE7000-2023](https://huggingface.co/datasets/OpenSynth/D-GITT-RTE7000-2023)
into a grid that gridfm-datakit can generate data on (`reader: powsybl`).

The raw snapshots are not directly usable: every load is 0 MW, generators have no setpoints,
there are about a hundred islands, most generators are non-regulating and the slack terminal sits on a busbar
without a generator. Datakit scales the loads that already exist and solves an OPF per scenario
(also in `mode: pf`, where the OPF supplies the dispatch), so the grid needs a synthetic operating
point and an OPF-feasible set of limits. **The result is synthetic, not real RTE data**; every step
records what it changed (see Reports).

## Scripts

Run them from the repo root as `python gridfm_datakit/xiidm_process/<script>.py`.

| Script | What it does |
|---|---|
| `get_grid_from_hf.py` | Downloads one snapshot, verifies its MD5, stores it in `scripts/grids/data_rte/raw/`. |
| `connect_grid.py` | Closes switches so no element is left disconnected (and joins split busbars) -> `data_rte/connected/`. |
| `make_base_case.py` | Keeps the main component (optionally only `--min-kv`), sets loads and dispatch, voltage regulation on all dispatched generators, slack on the largest one, reactive and voltage limits -> `data_rte/base/`. |
| `relax_to_opf_feasible.py` | Raises only the thermal limits that make the AC OPF infeasible (~120 of 2469 branches on the 225/380 kV grid), re-checks the saved file -> `<name>-opf.xiidm.bz2`. |
| `check_network.py` | Read-only check of a file, ending in a "Datakit readiness" verdict (one component, one REF bus with generator, no generator on a PQ bus, non-zero load). |
| `diagnose_opf.py` | Troubleshooting: relaxes one OPF constraint class at a time to see which one causes infeasibility. Only needed if `relax_to_opf_feasible.py` says the case is infeasible even without thermal limits. |
| `plot_grid.py` | Optional: draws the connected grid (needs `matplotlib`, `igraph`). |
| `grid_paths.py`, `lf_violations.py` | Shared helpers (folder layout, file names; AC load flow and limit violations). |

## Typical workflow

```bash
pip install -r gridfm_datakit/xiidm_process/requirements.txt
X=gridfm_datakit/xiidm_process; G=scripts/grids/data_rte

python $X/get_grid_from_hf.py --date 2023-12-31 --time 2355
python $X/connect_grid.py $G/raw/2023-12-31-2355.xiidm.bz2
python $X/make_base_case.py $G/connected/2023-12-31-2355.xiidm.bz2 $G/base/2023-12-31-2355-225kv.xiidm.bz2 \
    --min-kv 200 --load-fraction 0.5 --power-factor 0.98
python $X/relax_to_opf_feasible.py $G/base/2023-12-31-2355-225kv.xiidm.bz2
python $X/check_network.py $G/base/2023-12-31-2355-225kv-opf.xiidm.bz2
```

Then point a datakit config at the result and generate:

```yaml
network:
  name: "2023-12-31-2355-225kv-opf"
  reader: "powsybl"
  source: "file"
  network_dir: "scripts/grids/data_rte/base"
  file: "scripts/grids/data_rte/base/2023-12-31-2355-225kv-opf.xiidm.bz2"
```

```bash
gridfm_datakit generate scripts/config/default.yaml
```

Notes:

- `--min-kv 200` keeps the 225/380 kV transmission grid (1666 buses). Without it the whole connected grid (about 6600 buses) is used; that was not tested end to end.
- `connect_grid.py` is not optional. PowSyBl's MATPOWER export skips branches disconnected at both ends, which shifts datakit's branch-to-element mapping on a raw snapshot, and only about half of the generators are connected there (so the base case ends up with far fewer dispatchable units). `relax_to_opf_feasible.py` refuses to write a grid when it detects this.
- `generate` prints "Data generation complete" and exits 0 even if every scenario failed. A run worked only if `data_out/<name>/raw/` contains `bus_data.parquet`, `gen_data.parquet` and the other data files and `error.log` has no `Caught an exception` lines.

## Reports

Each step writes next to its output file (`<name>` is the output stem):

- `<name>.report.json` — arguments, what was pruned/changed, OPF statuses; `-opf` reports embed the base-case report as `parent_report`.
- `<name>.relaxed_branches.csv` — every branch whose limit was raised, with rating before and after and the OPF flow that required it.
- `reports/<name>.txt` (`check_network.py`) and `images/<name>.png` (`plot_grid.py`) in this folder, git-ignored.
