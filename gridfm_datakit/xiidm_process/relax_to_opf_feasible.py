"""Make a base-case snapshot AC-OPF-feasible by raising only the thermal limits that bind.

gridfm-datakit runs an OPF for every scenario, also in `mode: pf` (the OPF gives the
generator setpoints), so the base case must be OPF-feasible. On the D-GITT grids with the
synthetic load/dispatch of make_base_case.py the OPF is infeasible because a handful of
branches would have to carry more than their rating (see diagnose_opf.py). Instead of scaling
every limit, this script

  1. solves the OPF with thermal limits switched off (must be solvable, otherwise the
     problem is not thermal: use diagnose_opf.py),
  2. raises the rating of each branch whose flow in that solution exceeds it to
     flow * margin (the margin grows until the OPF with limits is solved),
  3. writes the scaled permanent current limits back to the PowSyBl network and saves it,
  4. reloads the saved file and re-solves the OPF to confirm it is feasible,
  5. writes a report of every change next to the output file:
       <out>.relaxed_branches.csv   one row per branch whose limit was raised
       <out>.report.json            parameters, counts and the make_base_case report, if any

Usage:
    python relax_to_opf_feasible.py IN.xiidm.bz2 [OUT.xiidm.bz2] [--margin 1.05] [--max-iter 200]

Default output: IN with "-opf" appended to the name, in the same folder.
"""

import argparse
import bz2
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gridfm_datakit.powsybl import load_net  # noqa: E402
from gridfm_datakit.process.process_network import init_julia  # noqa: E402
from grid_paths import sibling  # noqa: E402

RATE_A = 5  # MATPOWER branch column; 0 = unlimited
F_BUS, T_BUS = 0, 1
SOLVED = "LOCALLY_SOLVED"


def solve(net, jl):
    """AC OPF through the same PowerModels call datakit uses; returns the raw result."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".m", delete=False) as t:
        path = t.name
    try:
        net.to_mpc(path)
        return jl.run_opf(path)
    finally:
        os.unlink(path)


def branch_flows(net, res):
    """max(|S_from|, |S_to|) in MVA for every branch row of `net` (0 for out-of-service branches)."""
    sol = res["solution"]
    base = float(sol["baseMVA"]) if "baseMVA" in sol else 100.0
    flow = np.zeros(net.branches.shape[0])
    for i in range(flow.size):
        if str(i + 1) not in sol["branch"]:  # PowerModels leaves out-of-service branches out of the solution
            continue
        br = sol["branch"][str(i + 1)]
        flow[i] = base * max(
            np.hypot(float(br["pf"]), float(br["qf"])),
            np.hypot(float(br["pt"]), float(br["qt"])),
        )
    return flow


def with_rates(net, rates):
    n = deepcopy(net)
    n.branches[:, RATE_A] = rates
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", nargs="?", type=Path)
    ap.add_argument("--margin", type=float, default=1.05, help="new rating = OPF flow * margin (grows by --margin-step until the OPF is solved)")
    ap.add_argument("--margin-step", type=float, default=0.05)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=200)
    a = ap.parse_args()

    out = a.output or sibling(a.input, "-opf.xiidm.bz2")
    if out.resolve() == a.input.resolve():
        raise SystemExit("output would overwrite the input")

    loaded = load_net(str(a.input))
    net, pp_net, row_of = loaded.gfm_net, loaded.pp_net, loaded.mapping_p2g.branch
    rates0 = net.branches[:, RATE_A].copy()
    jl = init_julia(max_iter=a.max_iter, solver_log_dir=None, coinhsl=None)

    status = str(solve(net, jl)["termination_status"])
    print(f"OPF on the input: {status}")
    if status == SOLVED:
        raise SystemExit("already OPF-feasible, nothing to relax")

    free = solve(with_rates(net, np.zeros_like(rates0)), jl)
    free_status = str(free["termination_status"])
    print(f"OPF without thermal limits: {free_status}")
    if free_status != SOLVED:
        raise SystemExit("infeasible even without thermal limits: the cause is not thermal, run diagnose_opf.py")
    flow = branch_flows(net, free)

    over = (rates0 > 0) & (flow > rates0 * 1.0001)
    print(f"branches above their rating when thermal limits are off: {int(over.sum())} of {rates0.size}")

    margin, rates1, trial_status = a.margin, rates0.copy(), "not run"
    for _ in range(a.max_attempts):
        rates1 = rates0.copy()
        rates1[over] = flow[over] * margin
        trial_status = str(solve(with_rates(net, rates1), jl)["termination_status"])
        print(f"  margin {margin:.2f}: OPF {trial_status}")
        if trial_status == SOLVED:
            break
        margin += a.margin_step
    else:
        raise SystemExit(f"OPF still {trial_status} after {a.max_attempts} attempts; try a larger --margin")

    # ---- write the new ratings back as scaled permanent current limits
    id_of = {row: pid for pid, row in row_of.items()}
    factor_of = {id_of[i]: rates1[i] / rates0[i] for i in np.flatnonzero(over)}
    lim = pp_net.get_operational_limits()
    ix = lim.index
    keep = (ix.get_level_values("type") == "CURRENT") & (ix.get_level_values("acceptable_duration") == -1)
    keep &= (lim.value < 1e300).to_numpy()
    keep &= ix.get_level_values("element_id").isin(list(factor_of))
    sel = lim[keep]
    factors = np.array([factor_of[e] for e in sel.index.get_level_values("element_id")])
    pp_net.update_operational_limits(pd.DataFrame({"value": sel.value.to_numpy() * factors}, index=sel.index))
    scaled_ids = set(sel.index.get_level_values("element_id"))
    print(f"permanent current limits raised on {len(scaled_ids)} of {len(factor_of)} branches")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "net.xiidm"
        pp_net.save(str(tmp), format="XIIDM")
        with open(tmp, "rb") as fi, bz2.open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("wrote", out)

    # ---- verify on the file as written
    check = load_net(str(out)).gfm_net
    # The report must name the branches that really changed. PowSyBl's MATPOWER export skips branches that
    # are disconnected at both ends, which shifts the table-order row -> element mapping used above.
    wrong = ~np.isclose(check.branches[:, RATE_A], rates1, rtol=1e-4, atol=1e-6)
    if wrong.any():
        out.unlink()
        raise SystemExit(
            f"{int(wrong.sum())} branch ratings in the written file differ from the intended ones, so the element mapping is off "
            "(typical for a snapshot with elements disconnected at both ends: run connect_grid.py first). Nothing was kept."
        )
    final = str(solve(check, jl)["termination_status"])
    print(f"OPF on the written file: {final}")

    # ---- report
    csv_path, json_path = sibling(out, ".relaxed_branches.csv"), sibling(out, ".report.json")
    rows = np.flatnonzero(over)
    pd.DataFrame(
        {
            "element_id": [id_of[i] for i in rows],
            "from_bus": net.branches[rows, F_BUS].astype(int),
            "to_bus": net.branches[rows, T_BUS].astype(int),
            "rating_mva_before": rates0[rows],
            "rating_mva_after": rates1[rows],
            "factor": rates1[rows] / rates0[rows],
            "flow_mva_in_unconstrained_opf": flow[rows],
            "current_limit_written": [id_of[i] in scaled_ids for i in rows],
        }
    ).sort_values("factor", ascending=False).to_csv(csv_path, index=False)

    parent = {}
    parent_report = sibling(a.input, ".report.json")
    if parent_report.exists():
        parent = json.loads(parent_report.read_text())
    json_path.write_text(
        json.dumps(
            {
                "tool": "relax_to_opf_feasible.py",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input": str(a.input),
                "output": str(out),
                "synthetic": "operating point and limits are NOT real RTE data",
                "opf_status_input": status,
                "opf_status_without_thermal_limits": free_status,
                "opf_status_output": final,
                "margin": margin,
                "branches_total": int(rates0.size),
                "branches_with_raised_limit": int(over.sum()),
                "max_factor": float((rates1[rows] / rates0[rows]).max()) if len(rows) else 1.0,
                "relaxed_branches_csv": str(csv_path),
                "parent_report": parent,
            },
            indent=2,
        )
    )
    print("wrote", csv_path)
    print("wrote", json_path)
    if final != SOLVED:
        raise SystemExit("the written file is not OPF-feasible: round-trip through XIIDM changed the case")


if __name__ == "__main__":
    main()
