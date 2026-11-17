"""Make a base-case snapshot AC-OPF-feasible by widening only the limits that bind.

gridfm-datakit runs an OPF for every scenario, also in `mode: pf` (the OPF gives the
generator setpoints), so the base case must be OPF-feasible. With the synthetic load/dispatch of
make_base_case.py it is not, because some limits are too tight for the operating point. Instead of
scaling every limit, this script

  1. solves the OPF on the input (stop if it is already feasible),
  2. solves it with the thermal limits switched off. If that works, only thermal limits are
     relaxed (small grids, e.g. the 225/380 kV subgrid). Otherwise it solves with thermal limits
     off, voltage limits widened to 0.5-1.5 pu and generator Q limits removed. If that is still
     infeasible the cause is elsewhere: use diagnose_opf.py,
  3. takes every limit the relaxed solution violates (branch rating, bus voltage bound, generator
     Q limit) and widens just that limit to the solution value plus a margin (the margin grows until
     the OPF with the widened limits is solved). The relaxed OPF has no cost on Q and spreads it
     freely, so most generators exceed their Q limit only slightly; only the large excesses are
     widened (--q-tolerance, default: try 50, 30, 10, then 0 MVAr),
  4. writes the new limits back to the PowSyBl network (permanent current limits, voltage-level
     low/high voltage limits, generator min_q/max_q) and saves it,
  5. reloads the saved file, checks every written limit against the intended one, and re-solves
     the OPF to confirm that it is feasible,
  6. writes a report of every change next to the output file:
       <out>.relaxed_branches.csv   one row per branch whose rating was raised
       <out>.relaxed_voltage_levels.csv, <out>.relaxed_generators.csv   likewise (if any)
       <out>.report.json            parameters, counts and the make_base_case report, if any

Usage:
    python relax_to_opf_feasible.py IN.xiidm.bz2 [OUT.xiidm.bz2] [--margin 1.05] [--q-tolerance 50] [--max-iter 200]

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

# MATPOWER columns
RATE_A = 5  # 0 = unlimited
F_BUS, T_BUS = 0, 1
VMAX, VMIN = 11, 12
QMAX, QMIN = 3, 4
SOLVED = "LOCALLY_SOLVED"
V_WIDE = (0.5, 1.5)


def solve(net, jl):
    """AC OPF through the same PowerModels call datakit uses; returns the raw result."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".m", delete=False) as t:
        path = t.name
    try:
        net.to_mpc(path)
        return jl.run_opf(path)
    finally:
        os.unlink(path)


def solution_values(net, res):
    """(branch flow MVA, bus vm, gen qg MVAr) per row of `net`; flow 0 and qg NaN for out-of-service elements.

    PowerModels keys every element by its 1-based row and leaves out-of-service branches/generators out.
    """
    sol = res["solution"]
    base = float(sol["baseMVA"]) if "baseMVA" in sol else 100.0
    flow = np.zeros(net.branches.shape[0])
    for i in range(flow.size):
        br = sol["branch"][str(i + 1)] if str(i + 1) in sol["branch"] else None
        if br is not None:
            flow[i] = base * max(np.hypot(float(br["pf"]), float(br["qf"])), np.hypot(float(br["pt"]), float(br["qt"])))
    vm = np.array([float(sol["bus"][str(i + 1)]["vm"]) for i in range(net.buses.shape[0])])
    qg = np.full(net.gens.shape[0], np.nan)
    for j in range(qg.size):
        if str(j + 1) in sol["gen"]:
            qg[j] = base * float(sol["gen"][str(j + 1)]["qg"])
    return flow, vm, qg


def relaxed_copy(net, voltage_and_q):
    n = deepcopy(net)
    n.branches[:, RATE_A] = 0.0
    if voltage_and_q:
        n.buses[:, VMIN], n.buses[:, VMAX] = V_WIDE
        n.gens[:, QMAX], n.gens[:, QMIN] = 9999.0, -9999.0
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", nargs="?", type=Path)
    ap.add_argument("--margin", type=float, default=1.05, help="new rating / Q limit = OPF value * margin (grows by --margin-step until the OPF is solved)")
    ap.add_argument("--margin-step", type=float, default=0.05)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--q-tolerance", type=float, default=None, help="MVAr: only widen generator Q limits the relaxed solution exceeds by more than this (default: try 50, 30, 10, then 0)")
    a = ap.parse_args()

    out = a.output or sibling(a.input, "-opf.xiidm.bz2")
    if out.resolve() == a.input.resolve():
        raise SystemExit("output would overwrite the input")

    loaded = load_net(str(a.input))
    net, pp_net, maps = loaded.gfm_net, loaded.pp_net, loaded.mapping_p2g
    row_of = maps.branch
    rates0 = net.branches[:, RATE_A].copy()
    vmin0, vmax0 = net.buses[:, VMIN].copy(), net.buses[:, VMAX].copy()
    qmin0, qmax0 = net.gens[:, QMIN].copy(), net.gens[:, QMAX].copy()
    jl = init_julia(max_iter=a.max_iter, solver_log_dir=None, coinhsl=None)

    status = str(solve(net, jl)["termination_status"])
    print(f"OPF on the input: {status}")
    if status == SOLVED:
        raise SystemExit("already OPF-feasible, nothing to relax")

    free = solve(relaxed_copy(net, False), jl)
    free_status = str(free["termination_status"])
    print(f"OPF without thermal limits: {free_status}")
    wide = free_status != SOLVED
    if wide:
        free = solve(relaxed_copy(net, True), jl)
        free_status = str(free["termination_status"])
        print(f"OPF without thermal limits, voltage limits 0.5-1.5 pu, no generator Q limits: {free_status}")
        if free_status != SOLVED:
            raise SystemExit("infeasible even with thermal, voltage and Q limits relaxed: the cause is elsewhere, run diagnose_opf.py")
    flow, vm, qg = solution_values(net, free)

    pp_buses0 = pp_net.get_buses()
    vl_of_row0 = {maps.bus[b] for b in pp_buses0.index if b in maps.bus}
    over = (rates0 > 0) & (flow > rates0 * 1.0001)
    v_lo = wide & (vm < vmin0 - 1e-6)
    v_hi = wide & (vm > vmax0 + 1e-6)
    q_viol = wide & ~np.isnan(qg) & ((qg > qmax0 + 1e-6) | (qg < qmin0 - 1e-6))
    q_excess = np.where(np.isnan(qg), 0.0, np.maximum(qg - qmax0, qmin0 - qg))
    writable = np.zeros(vm.size, dtype=bool)
    writable[[r for r in vl_of_row0 if r < vm.size]] = True
    unwritable = int(((v_lo | v_hi) & ~writable).sum())
    v_lo, v_hi = v_lo & writable, v_hi & writable
    print(f"limits violated in the relaxed solution: branches {int(over.sum())} of {rates0.size}, "
          f"bus voltage {int(v_lo.sum())} low + {int(v_hi.sum())} high of {vm.size}, generator Q {int(q_viol.sum())} of {qg.size}"
          + (f" (+{unwritable} voltage violations on buses that are not PowSyBl buses, left as they are)" if unwritable else ""))

    # ---- row -> element maps, needed to write the limits back
    id_of_branch = {row: pid for pid, row in row_of.items()}
    id_of_gen = {row: pid for pid, row in maps.gen.items()}
    pp_buses = pp_net.get_buses()
    vl_of_row = {maps.bus[b]: pp_buses.voltage_level_id[b] for b in pp_buses.index if b in maps.bus}
    vls = pp_net.get_voltage_levels()
    nominal = vls.nominal_v

    # Most generators the relaxed solution pushes past their Q limit do so only slightly (the relaxed OPF has no cost on Q
    # and spreads it freely), so first widen only the large excesses and fall back to smaller tolerances if needed.
    tolerances = [a.q_tolerance] if a.q_tolerance is not None else [50.0, 30.0, 10.0, 0.0]
    solved_trial = False
    for tol in tolerances:
        q_out = q_viol & (q_excess > tol)
        attempts = a.max_attempts if tol == tolerances[-1] else 2
        margin, trial_status = a.margin, "not run"
        for _ in range(attempts):
            delta_v = 0.01 + 0.2 * (margin - 1.05)
            rates1, vmin1, vmax1, qmin1, qmax1 = rates0.copy(), vmin0.copy(), vmax0.copy(), qmin0.copy(), qmax0.copy()
            rates1[over] = flow[over] * margin
            vmin1[v_lo] = vm[v_lo] - delta_v
            vmax1[v_hi] = vm[v_hi] + delta_v
            # a voltage level has one pair of limits: give every bus in it the widest bounds of the group
            for vl in {vl_of_row[r] for r in np.flatnonzero(v_lo | v_hi) if r in vl_of_row}:
                rows = [r for r, v in vl_of_row.items() if v == vl]
                vmin1[rows], vmax1[rows] = vmin1[rows].min(), vmax1[rows].max()
            qmax1[q_out] = np.maximum(qmax0[q_out], qg[q_out] * margin + 1.0)
            qmin1[q_out] = np.minimum(qmin0[q_out], qg[q_out] * margin - 1.0)
            trial = deepcopy(net)
            trial.branches[:, RATE_A] = rates1
            trial.buses[:, VMIN], trial.buses[:, VMAX] = vmin1, vmax1
            trial.gens[:, QMIN], trial.gens[:, QMAX] = qmin1, qmax1
            trial_status = str(solve(trial, jl)["termination_status"])
            print(f"  Q tolerance {tol:g} MVAr ({int(q_out.sum())} generators), margin {margin:.2f}: OPF {trial_status}")
            if trial_status == SOLVED:
                solved_trial = True
                break
            margin += a.margin_step
        if solved_trial:
            break
    if not solved_trial:
        raise SystemExit(f"OPF still {trial_status} after the attempts; try a larger --margin")
    q_tol_used = tol

    # ---- write the new limits back to the PowSyBl network
    # thermal: scaled permanent current limits
    factor_of = {id_of_branch[i]: rates1[i] / rates0[i] for i in np.flatnonzero(over)}
    if factor_of:
        lim = pp_net.get_operational_limits()
        ix = lim.index
        keep = (ix.get_level_values("type") == "CURRENT") & (ix.get_level_values("acceptable_duration") == -1)
        keep &= (lim.value < 1e300).to_numpy()
        keep &= ix.get_level_values("element_id").isin(list(factor_of))
        sel = lim[keep]
        factors = np.array([factor_of[e] for e in sel.index.get_level_values("element_id")])
        pp_net.update_operational_limits(pd.DataFrame({"value": sel.value.to_numpy() * factors}, index=sel.index))
        scaled_ids = set(sel.index.get_level_values("element_id"))
    else:
        scaled_ids = set()
    print(f"permanent current limits raised on {len(scaled_ids)} of {len(factor_of)} branches")

    # voltage levels: new limits in kV
    vl_rows = sorted({r for r in np.flatnonzero(v_lo | v_hi) if r in vl_of_row})
    changed_vl = {}
    for r in vl_rows:
        changed_vl[vl_of_row[r]] = (vmin1[r], vmax1[r])
    if changed_vl:
        ids = list(changed_vl)
        pp_net.update_voltage_levels(
            id=ids,
            low_voltage_limit=[changed_vl[i][0] * nominal[i] for i in ids],
            high_voltage_limit=[changed_vl[i][1] * nominal[i] for i in ids],
        )
    print(f"voltage limits widened on {len(changed_vl)} voltage levels")

    # generators: min_q / max_q
    # generators without min/max reactive limits in the network (e.g. Pmax = 0 units) cannot be updated: leave them
    # (the export also rewrites degenerate limits, min_q == max_q, so those are left alone too)
    lims = pp_net.get_generators(attributes=["min_q", "max_q"])
    has_q_limits = lims.notna().all(axis=1) & (lims.max_q > lims.min_q)
    q_rows = [r for r in np.flatnonzero(q_out) if r in id_of_gen and has_q_limits[id_of_gen[r]]]
    q_skipped = np.setdiff1d(np.flatnonzero(q_out), q_rows)
    qmin1[q_skipped], qmax1[q_skipped] = qmin0[q_skipped], qmax0[q_skipped]
    if q_rows:
        pp_net.update_generators(
            id=[id_of_gen[r] for r in q_rows],
            min_q=qmin1[q_rows],
            max_q=qmax1[q_rows],
        )
    print(f"reactive limits widened on {len(q_rows)} of {int(q_out.sum())} generators ({len(q_skipped)} without min/max limits left as they are)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "net.xiidm"
        pp_net.save(str(tmp), format="XIIDM")
        with open(tmp, "rb") as fi, bz2.open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("wrote", out)

    # ---- verify on the file as written
    check = load_net(str(out)).gfm_net
    # PowSyBl's MATPOWER export skips branches that are disconnected at both ends, which shifts the
    # table-order row -> element mapping used above; any mismatch in a written limit shows up here.
    problems = {
        "branch ratings": ~np.isclose(check.branches[:, RATE_A], rates1, rtol=1e-4, atol=1e-6),
        # buses that are not PowSyBl buses take the limits of their voltage level, so they may only get wider
        "bus voltage bounds": np.where(
            writable,
            ~(np.isclose(check.buses[:, VMIN], vmin1, atol=1e-5) & np.isclose(check.buses[:, VMAX], vmax1, atol=1e-5)),
            (check.buses[:, VMIN] > vmin0 + 1e-5) | (check.buses[:, VMAX] < vmax0 - 1e-5),
        ),
        "generator Q limits": ~(np.isclose(check.gens[:, QMIN], qmin1, rtol=1e-4, atol=1e-3) & np.isclose(check.gens[:, QMAX], qmax1, rtol=1e-4, atol=1e-3)),
    }
    bad = {k: int(v.sum()) for k, v in problems.items() if v.any()}
    if bad:
        for r in np.flatnonzero(problems["bus voltage bounds"])[:5]:
            print(f"  bus row {r} (voltage level {vl_of_row.get(r)}): intended [{vmin1[r]:.4f}, {vmax1[r]:.4f}], "
                  f"written [{check.buses[r, VMIN]:.4f}, {check.buses[r, VMAX]:.4f}], original [{vmin0[r]:.4f}, {vmax0[r]:.4f}]")
        out.unlink()
        raise SystemExit(
            f"limits in the written file differ from the intended ones ({bad}), so the element mapping is off "
            "(typical for a snapshot with elements disconnected at both ends: run connect_grid.py first). Nothing was kept."
        )
    final = str(solve(check, jl)["termination_status"])
    print(f"OPF on the written file: {final}")

    # ---- report
    csv_path, json_path = sibling(out, ".relaxed_branches.csv"), sibling(out, ".report.json")
    rows = np.flatnonzero(over)
    pd.DataFrame(
        {
            "element_id": [id_of_branch[i] for i in rows],
            "from_bus": net.branches[rows, F_BUS].astype(int),
            "to_bus": net.branches[rows, T_BUS].astype(int),
            "rating_mva_before": rates0[rows],
            "rating_mva_after": rates1[rows],
            "factor": rates1[rows] / rates0[rows],
            "flow_mva_in_unconstrained_opf": flow[rows],
            "current_limit_written": [id_of_branch[i] in scaled_ids for i in rows],
        }
    ).sort_values("factor", ascending=False).to_csv(csv_path, index=False)

    vl_csv = gen_csv = None
    if changed_vl:
        vl_csv = sibling(out, ".relaxed_voltage_levels.csv")
        pd.DataFrame(
            {
                "voltage_level_id": list(changed_vl),
                "nominal_kv": [nominal[i] for i in changed_vl],
                "vmin_pu_before": [vls.low_voltage_limit[i] / nominal[i] for i in changed_vl],
                "vmax_pu_before": [vls.high_voltage_limit[i] / nominal[i] for i in changed_vl],
                "vmin_pu_after": [changed_vl[i][0] for i in changed_vl],
                "vmax_pu_after": [changed_vl[i][1] for i in changed_vl],
            }
        ).to_csv(vl_csv, index=False)
    if q_rows:
        gen_csv = sibling(out, ".relaxed_generators.csv")
        pd.DataFrame(
            {
                "generator_id": [id_of_gen[r] for r in q_rows],
                "qmin_mvar_before": qmin0[q_rows],
                "qmax_mvar_before": qmax0[q_rows],
                "qmin_mvar_after": qmin1[q_rows],
                "qmax_mvar_after": qmax1[q_rows],
                "qg_mvar_in_relaxed_opf": qg[q_rows],
            }
        ).to_csv(gen_csv, index=False)

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
                "relaxed_classes": "thermal, voltage, reactive" if wide else "thermal",
                "opf_status_relaxed": free_status,
                "opf_status_output": final,
                "margin": margin,
                "q_tolerance_mvar": q_tol_used,
                "branches_total": int(rates0.size),
                "branches_with_raised_limit": int(over.sum()),
                "max_factor": float((rates1[rows] / rates0[rows]).max()) if len(rows) else 1.0,
                "voltage_levels_widened": len(changed_vl),
                "generators_q_widened": len(q_rows),
                "generators_total": int(qg.size),
                "relaxed_branches_csv": str(csv_path),
                "relaxed_voltage_levels_csv": str(vl_csv) if vl_csv else None,
                "relaxed_generators_csv": str(gen_csv) if gen_csv else None,
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
