# D-GITT RTE7000 snapshot 2023-12-31 23:55: from raw to a datakit-compliant grid

**Result.** A grid with **6582 buses** (99.9 % of the 6588 buses of the connected snapshot) on which datakit generates data in `mode: pf`:
10 scenarios, 0 exceptions, 0 NaNs, 61 770 generator rows, 96 610 branch rows
(datakit's tables hold 6626 bus rows per scenario because the MATPOWER export adds 44 buses).
It is **not real RTE data**: the operating point and some limits were invented or widened, listed below.

Files (git-ignored data, `scripts/grids/data_rte/base/`):

| File | What it is |
|---|---|
| `2023-12-31-2355-full.xiidm.bz2`, `.report.json` | base case (operating point) and its report |
| `2023-12-31-2355-full-opf.xiidm.bz2` | the grid to use with datakit |
| `…-full-opf.report.json`, `.relaxed_branches.csv`, `.relaxed_voltage_levels.csv`, `.relaxed_generators.csv` | every widened limit, before and after |

## Coverage versus changes

All rows use the same scripts (`make_base_case.py` then `relax_to_opf_feasible.py`), load fraction 0.2 unless noted.

| Grid (`--min-kv`) | Buses | Result | Branch ratings raised | Voltage levels widened | Generator Q limits widened |
|---|---|---|---|---|---|
| 200 kV and above (225/380 kV) | 1666 | OPF feasible | 20 (119 at load 0.5) | 0 | 0 |
| 80 kV and above | 2903 | **not obtained**: the relaxed OPF needs the upper voltage bound raised on 2875 of 2937 buses, and the written-file check refused the result (5 buses in `PRINQP6`, `SAINNP6`, `SSMR P6` did not get the intended voltage bounds; cause not investigated) | (35) | (about 2875) | (108 beyond 50 MVAr, 1192 in total) |
| 60 kV and above | 6118 | OPF feasible, 5 scenarios generated | 16 | 2 | 74 |
| **none (whole connected grid)** | **6582** | **OPF feasible, 10 scenarios generated** | **20** | **17** | **53** |

Counts in brackets are from the relaxation that did not produce a file. Total number of branches:
2469 / 4260 / 9115 / 9661, generators: 602 / 2357 / 5599 / 6177.

Two findings explain the shape of the table:

- **Cutting the grid does not make it easier.** The 80 kV cut needs far more limit changes (voltage bounds on almost every
  bus) than the full grid, and the 60 kV cut needs more generator Q changes (74) than the full grid (53). The full
  grid is both the largest and about the cheapest to repair.
- **The load level matters more than the cut.** At load fraction 0.5 the larger grids need voltages above 1.1 pu at
  more than 6000 buses; at 0.2 only 5 buses are above 1.1 pu and 16 below 0.9 pu in the relaxed solution.

## What was changed for the 6582-bus grid

### 1. Topology (`connect_grid.py`, `make_base_case.py`)

- All switches closed so no element is left disconnected, split busbars joined.
- Main connected component kept: 6588 -> 6582 buses, 7017 -> 7002 loads, 6185 -> 6180 generators
  (4 disconnected voltage levels dropped). No voltage level was removed.
- 12 connected generators with Pmax = 0 disconnected (they cannot produce and sat on PQ buses, which datakit's
  PF rejects): `ARGOEIN2, ARGOEIN3, AUSSO3B.AUS3G1, BAGE .HG1, BAGE .HG2, BEUVRIN1, FAMARIN3, PORTI5PORT9P1+2, PRAG5.HG1, PRAG5.HG2, PRAG5.HG3, SINCEIN2`.

### 2. Operating point (invented, `make_base_case.py`)

| Parameter | Value | Why |
|---|---|---|
| total load | 0.2 x connected Pmax = 29 531 MW, spread over 7002 loads with log-normal weights (sigma 0.5, seed 0) | the raw loads are all 0 MW |
| load power factor | 0.98 | reactive load for the above |
| dispatch | proportional to Pmax, clipped to [Pmin, Pmax], total = load x 1.02 = 30 122 MW (losses 2 %) | **not required by datakit**: its OPF solves Pg itself (in `mode: pf` the OPF result overwrites it) and the load is sized from the Pmax sum, not from the dispatch. Kept only as a balanced, plausible starting point for load-flow checks. Checked on this grid: with every `target_p` = 0 the OPF relaxation gives byte-identical limit changes and 20-scenario `pf` and `opf` runs give identical output. `make_base_case.py --no-dispatch` skips it |
| voltage regulation | switched on for 5426 generators that were non-regulating, setpoint 1.0 pu of the nominal voltage; applies to every connected generator with Pmax > 0, whatever its dispatch | datakit's PF needs every in-service generator on a PV bus (the MATPOWER export types a bus PV only if the generator regulates); only a minority were regulating in D-GITT. This is the flag `voltage_regulator_on`, not the in-service state |
| slack | moved to generator `SSAVO6E.HU6` (368 MW, voltage level `E.HUCP6`) | the file's slack sat on a busbar without generator |
| generator Q limits | capability curves collapsed to min/max; units without usable limits get +/-0.4 x Pmax (4977 generators) | the OPF needs real reactive ranges |
| bus voltage limits | 0.9 - 1.1 pu on all 5960 voltage levels | the raw limits are inconsistent |
| batteries | set to 0 MW (17) | no injection data |

### 3. Limits widened so the OPF is feasible (`relax_to_opf_feasible.py`)

The OPF on the base case did not converge. With thermal limits, voltage limits and generator Q limits all
relaxed it solves; the script then widens only the limits that solution violates, by the violation plus 5 %,
and checks the saved file (each written limit against the intended one, then the OPF again: `LOCALLY_SOLVED`).

**Thermal ratings: 20 of 9661 branches raised** (factor 1.08 to 15.99, median 1.29). Largest ones:

| Branch | Rating before (MVA) | After (MVA) | Factor |
|---|---|---|---|
| `PERR5L21ZAIG5` | 0.42 | 6.71 | 15.99 |
| `ORELLL61P.AND` | 86.4 | 380.1 | 4.40 |
| `P.ANDL71DEPHG` | 684.0 | 1921.8 | 2.81 |
| `MANDAL71PEUP5` | 650.2 | 1766.0 | 2.72 |
| `ALLE5L31PAIRE` | 5.6 | 13.0 | 2.32 |

**Bus voltage bounds: 17 of 5960 voltage levels widened** (all others keep 0.9 - 1.1 pu). Lower bound: 0.56 to 0.89 pu
on 13 levels (the 20 kV and 150 kV levels `M.TERP1` 0.56, `CPNI5P1` 0.65, `.G.ROP5` 0.73, `HOSPIP5` 0.73, `PRINQP1` 0.82 and others).
Upper bound: 1.13 to 1.36 pu on 4 levels (`JAZENP1` 1.36, `LUSSAP1` 1.18, `SAINNP1` 1.13, `SSMC P1` 1.13).
The full list is in `…-full-opf.relaxed_voltage_levels.csv`.

**Generator Q limits: 53 of 6177 generators widened** (the relaxed solution pushed 2351 past a limit; 55 exceed it by
more than 50 MVAr, 2 of those have no min/max limits in the file, and the other 2296 exceed theirs by less than 50 MVAr and stayed as
they were, which still gives a feasible OPF). Median widening 75 MVAr, largest:

| Generator | Q range before (MVAr) | After (MVAr) |
|---|---|---|
| `PEUP59.CHER1` | -361 .. 330 | -1571 .. 330 |
| `SAUS5INF` | -0.08 .. 0.08 | -193 .. 0.08 |
| `GAUDIINF` | -0.16 .. 0.16 | -188 .. 0.16 |

Several small units (Pmax 0.2 to 0.4 MW) were widened to absorb 50 to 190 MVAr: they act as reactive sinks in the
OPF. This is the least physical change in the case; limit it by lowering the load or by treating those units separately
if the reactive behaviour of the 20 kV network matters for your use.

## How the 6582-bus grid was reached (what failed on the way)

1. The raw file and the connected file are not usable directly (zero loads, islands, slack without generator).
2. The 225/380 kV cut converged with only thermal ratings raised; cutting at 200 kV had been used because everything
   larger failed.
3. The whole grid failed on the OPF even without thermal limits (`ITERATION_LIMIT` / `LOCALLY_INFEASIBLE`); no single
   limit class fixed it, only thermal, voltage and Q limits relaxed together did.
4. Targeted widening (instead of widening everything) needed 2350 generator Q limits at first; widening only the excesses
   above a tolerance cut this to 53 (tolerance 50 MVAr; 70 MVAr already fails).
5. Making the write-back reliable required fixing three datakit/export problems, below.

## Bugs found in the loading code (fixed in `gridfm_datakit/powsybl/mapping.py`)

pypowsybl's MATPOWER export does not list buses and generators in the order of `pp_net.get_buses()` /
`get_generators()`, but `build_p2g_maps` paired the i-th element with row i. On this grid, positional pairing put
the wrong load or base voltage on 566 / 855 of 6582 buses and the right limits on only 1666 of 6180 generators.
The map for generators and buses is now read from the export itself (unique tags in a temporary variant, restored afterwards).
After the fix: wrong loads 3 (one voltage level where the export itself drops a 3.9 MW load), wrong base voltages 0, generator
rows 6166 of 6168 consistent. The 124 tests in `tests/powsybl` pass.

Not changed: the branch map is still positional (lines, then transformers). It is correct on the connected and base
files (2 of 9616 inconsistent) but wrong on raw-like files with elements disconnected at both ends, which is why `connect_grid.py`
has to run first.

## Caveats

- `mode: pf` allows limit violations: generated voltages range from 0.56 to 1.34 pu on the whole grid (0.74 to 1.04 on
  the 6118-bus grid). Use `mode: opf` if you need limit-feasible samples (not tested here).
- Only one snapshot (2023-12-31 23:55), load fraction 0.2 and seed 0 were tested; 10 scenarios, not a large run.
- The 44 export-only buses and 2 generators without min/max Q limits in the file were left as they come from the export.
- `get_grid_from_hf.py` and `connect_grid.py` were not changed in this iteration beyond file-name helpers; `connect_grid.py` was run
  to build the connected file used here.

## Reproduce

See the workflow in [README.md](README.md). Time on `ugpu2` (8 cores): base case 15 s, relaxation about 10 min,
10 scenarios about 4 min.
