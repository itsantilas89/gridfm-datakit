# xiidm_process

Turns an RTE snapshot from the HuggingFace dataset
[OpenSynth/D-GITT-RTE7000-2023](https://huggingface.co/datasets/OpenSynth/D-GITT-RTE7000-2023)
into a grid that gridfm-datakit can generate data on (`reader: powsybl`).

The raw snapshots are not directly usable: every load is 0 MW, generators have no setpoints
(datakit does not need them, but the loads it scales must exist), there are about a hundred islands, most generators are non-regulating and the slack terminal sits on a busbar
without a generator. Datakit scales the loads that already exist and solves an OPF per scenario
(also in `mode: pf`, where the OPF supplies the dispatch), so the grid needs a synthetic operating
point and an OPF-feasible set of limits. **The result is synthetic, not real RTE data**; every step
records what it changed (see Reports, and [FULL_GRID_REPORT.md](FULL_GRID_REPORT.md) for the whole-grid run).

## Scripts

Run them from the repo root as `python gridfm_datakit/xiidm_process/<script>.py`.

| Script | What it does |
|---|---|
| `get_grid_from_hf.py` | Downloads one snapshot, verifies its MD5, stores it in `scripts/grids/private_data/opensynth_rte7k/raw/`. |
| `connect_grid.py` | Closes switches so no element is left disconnected (and joins split busbars) -> `private_data/opensynth_rte7k/connected/`. |
| `make_base_case.py` | Keeps the main component (optionally only `--min-kv`), sets loads (and, optionally, a dispatch: `--no-dispatch` skips it, datakit does not need it), voltage regulation on all generators with Pmax > 0, slack on the largest one, disconnects Pmax = 0 units, reactive and voltage limits -> `private_data/opensynth_rte7k/base/`. |
| `relax_to_opf_feasible.py` | Makes the case OPF-feasible by widening only the limits that bind: thermal ratings first, and if that is not enough also bus voltage bounds and generator Q limits. Checks every written limit and re-solves the saved file -> `<name>-opf.xiidm.bz2`. |
| `check_network.py` | Read-only check of a file, ending in a "Datakit readiness" verdict (one component, one REF bus with generator, no generator on a PQ bus, non-zero load). |
| `diagnose_opf.py` | Troubleshooting: relaxes one OPF constraint class at a time to see which one causes infeasibility. Only needed if `relax_to_opf_feasible.py` says the case is infeasible even with everything relaxed. |
| `plot_grid.py` | Optional: draws the connected grid (needs `matplotlib`, `igraph`). |
| `grid_paths.py`, `lf_violations.py` | Shared helpers (folder layout, file names; AC load flow and limit violations). |

## Typical workflow (whole grid, 6582 buses)

```bash
pip install -r gridfm_datakit/xiidm_process/requirements.txt
X=gridfm_datakit/xiidm_process; G=scripts/grids/private_data/opensynth_rte7k

python $X/get_grid_from_hf.py --date 2023-12-31 --time 2355
python $X/connect_grid.py $G/raw/2023-12-31-2355.xiidm.bz2
python $X/make_base_case.py $G/connected/2023-12-31-2355.xiidm.bz2 $G/base/2023-12-31-2355-full.xiidm.bz2 \
    --load-fraction 0.2 --power-factor 0.98
python $X/relax_to_opf_feasible.py $G/base/2023-12-31-2355-full.xiidm.bz2
python $X/check_network.py $G/base/2023-12-31-2355-full-opf.xiidm.bz2
```

Then point a datakit config at the result and generate:

```yaml
network:
  name: "2023-12-31-2355-full-opf"
  reader: "powsybl"
  source: "file"
  network_dir: "scripts/grids/private_data/opensynth_rte7k/base"
  file: "scripts/grids/private_data/opensynth_rte7k/base/2023-12-31-2355-full-opf.xiidm.bz2"
```

```bash
gridfm_datakit generate scripts/config/opensynth_rte7k.yaml
```

Notes:

- **Coverage.** Without `--min-kv` the base case keeps 6582 of the 6588 buses of the connected grid (the whole 20-380 kV system; datakit's tables have 6626 bus rows because the MATPOWER export adds 44). `--min-kv 60` keeps 6118 buses, `--min-kv 200` the 1666-bus 225/380 kV grid. See the trade-off table in [FULL_GRID_REPORT.md](FULL_GRID_REPORT.md).
- **Load level.** `--load-fraction 0.2` (about 30 GW) is what the whole grid takes with few limit changes. At 0.3-0.5 the OPF needs voltages above 1.1 pu at almost every bus. The 225/380 kV subgrid also works at 0.5.
- `connect_grid.py` is **not** optional. PowSyBl's MATPOWER export skips branches disconnected at both ends, which shifts the branch-to-element mapping on a raw snapshot, and only about half of the generators are connected there. `relax_to_opf_feasible.py` refuses to write a grid when it detects a mismatch.
- `generate` prints "Data generation complete" and exits 0 even if every scenario failed. A run worked only if `data_out/<name>/raw/` contains `bus_data.parquet`, `gen_data.parquet` and the other data files and `error.log` has no `Caught an exception` lines.
- `mode: pf` allows limit violations, so the generated voltages are not confined to the 0.9-1.1 pu limits (the whole grid gives 0.56-1.34 pu in the 10-scenario test).

## Reports

Each step writes next to its output file (`<name>` is the output stem):

- `<name>.report.json` — arguments, what was pruned/changed, OPF statuses; `-opf` reports embed the base-case report as `parent_report`.
- `<name>.relaxed_branches.csv`, `<name>.relaxed_voltage_levels.csv`, `<name>.relaxed_generators.csv` — every limit that was widened, with its value before and after and the OPF value that required it.
- `reports/<name>.txt` (`check_network.py`) and `images/<name>.png` (`plot_grid.py`) in this folder, git-ignored.
