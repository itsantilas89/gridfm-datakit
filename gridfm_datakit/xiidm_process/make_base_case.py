"""Give an injection-free PowSyBl snapshot (e.g. D-GITT RTE7000) a synthetic nominal state.

gridfm-datakit scales the *existing* per-bus loads (p_i * profile * noise), so the network
needs non-zero nominal loads and generator setpoints; the raw D-GITT files have neither.
This script keeps only the main connected component (the default; optionally only voltage levels
>= --min-kv) and writes:

  * load p0/q0: total load = --load-fraction * sum(max_p of connected generators),
    split over connected loads (uniform, or noisy with --weight-sigma), q from --power-factor
  * generator target_p: proportional to max_p, clipped to [min_p, max_p], total = load * (1 + losses)
  * connected generators with Pmax = 0 are disconnected (they cannot produce, and would sit on a PQ bus)
  * voltage regulation on every dispatched generator, target_v = nominal kV * --vm-setpoint
    (datakit's PF needs each in-service generator on a PV bus; D-GITT marks most as non-regulating)
  * slack terminal on the largest dispatched generator (the files' own one sits on a busbar
    without generator, which leaves the PF with nothing to balance on)
  * symmetric reactive limits (+/- --q-limit-ratio * Pmax) for generators that have none
  * voltage limits --vlimits on every voltage level; dangling-line and battery injections = 0

It is a synthetic operating point, NOT real RTE data. A <out>.report.json next to the output
records the arguments and what was changed. The result is usually still not OPF-feasible:
run relax_to_opf_feasible.py on it next.

Usage:
    python make_base_case.py [IN.xiidm[.bz2]] [OUT.xiidm.bz2] [--min-kv 0] [--load-fraction 0.2]
                             [--power-factor 0.98] [--seed 0]

Default input: newest snapshot in data_rte/connected (else raw). Default output:
data_rte/base/<name>-base.xiidm.bz2.
"""

import argparse
import bz2
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl as pp
import lf_violations
from grid_paths import BASE_DIR, CONNECTED_DIR, RAW_DIR, newest_snapshot, sibling, stem_of


def remove_voltage_levels(net, vl_ids):
    """Remove voltage levels together with every branch that touches them."""
    vl = set(vl_ids)
    if not vl:
        return
    doomed = []
    for getter, cols in (
        (net.get_lines, ("voltage_level1_id", "voltage_level2_id")),
        (net.get_2_windings_transformers, ("voltage_level1_id", "voltage_level2_id")),
        (net.get_3_windings_transformers, ("voltage_level1_id", "voltage_level2_id", "voltage_level3_id")),
        (net.get_dangling_lines, ("voltage_level_id",)),
        (net.get_hvdc_lines, ("converter_station1_id", "converter_station2_id")),
    ):
        df = getter()
        if len(df) and all(c in df for c in cols):
            doomed += df.index[df[list(cols)].isin(vl).any(axis=1)].tolist()
    if doomed:
        net.remove_elements(doomed)
    net.remove_elements(sorted(vl))


def keep_main_component(net):
    """Drop everything outside the largest connected component."""
    b = net.get_buses()
    comp = b.connected_component
    main = comp.value_counts().idxmax()
    vls = set(b.voltage_level_id[comp != main]) - set(b.voltage_level_id[comp == main])
    remove_voltage_levels(net, vls)
    return len(vls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("output", nargs="?")
    ap.add_argument("--load-fraction", type=float, default=0.2, help="total load / connected Pmax (0.2 works on the whole grid; the 225/380 kV subgrid also takes 0.5)")
    ap.add_argument("--losses", type=float, default=0.02, help="dispatch = load * (1 + losses)")
    ap.add_argument("--power-factor", type=float, default=0.95)
    ap.add_argument("--weight-sigma", type=float, default=0.5, help="lognormal sigma of per-load weights (0 = uniform)")
    ap.add_argument("--vm-setpoint", type=float, default=1.0, help="p.u. voltage target of regulating generators")
    ap.add_argument("--q-limit-ratio", type=float, default=0.4, help="Qmax=-Qmin=ratio*Pmax for gens without reactive limits (0 = skip)")
    ap.add_argument("--vlimits", type=float, nargs=2, metavar=("LO", "HI"), default=(0.9, 1.1),
                    help="p.u. voltage bounds written to every voltage level (0 0 = keep the file's own limits)")
    ap.add_argument("--min-kv", type=float, default=0.0,
                    help="keep only voltage levels with nominal_v >= this (kV), e.g. 200 for the 225/380 kV grid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = a.input or newest_snapshot(CONNECTED_DIR) or newest_snapshot(RAW_DIR)
    if src is None:
        raise SystemExit("no input given and no snapshot in data_rte/{connected,raw}")
    rng = np.random.default_rng(a.seed)
    net = pp.network.load(str(src))
    report = {
        "tool": "make_base_case.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(src),
        "synthetic": "operating point and limits are NOT real RTE data",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
    }

    n_before = {"voltage_levels": len(net.get_voltage_levels()), "buses": len(net.get_buses()),
                "loads": len(net.get_loads()), "generators": len(net.get_generators())}
    if a.min_kv > 0:
        vls = net.get_voltage_levels(attributes=["nominal_v"])
        remove_voltage_levels(net, vls.index[vls.nominal_v < a.min_kv])
    dropped = keep_main_component(net)  # datakit needs a single connected component
    report["pruning"] = {
        "min_kv": a.min_kv, "before": n_before, "disconnected_levels_dropped": dropped,
        "after": {"voltage_levels": len(net.get_voltage_levels()), "buses": len(net.get_buses()),
                  "loads": len(net.get_loads()), "generators": len(net.get_generators())},
    }
    print(f"kept {report['pruning']['after']} (from {n_before}; {dropped} disconnected levels dropped)")

    gens = net.get_generators(attributes=["max_p", "min_p", "voltage_regulator_on", "voltage_level_id", "connected"])
    # a connected unit with Pmax = 0 produces nothing but would sit on a PQ bus, which datakit's PF rejects
    idle = gens.index[gens.connected & (gens.max_p <= 0)]
    if len(idle):
        net.update_generators(id=idle, connected=[False] * len(idle))
    report["idle_generators_disconnected"] = list(idle)
    print(f"{len(idle)} connected generators with Pmax = 0 disconnected")
    gens = gens[gens.connected & (gens.max_p > 0)]
    loads = net.get_loads(attributes=["connected"])
    loads = loads[loads.connected]
    if gens.empty or loads.empty:
        raise SystemExit("no connected generators or loads")

    # ---- loads
    total_load = a.load_fraction * gens.max_p.sum()
    w = rng.lognormal(0.0, a.weight_sigma, len(loads)) if a.weight_sigma > 0 else np.ones(len(loads))
    p0 = total_load * w / w.sum()
    q0 = p0 * np.tan(np.arccos(a.power_factor))
    all_loads = net.get_loads(attributes=[])
    z = np.zeros(len(all_loads))
    net.update_loads(id=all_loads.index, p0=z, q0=z)  # disconnected loads: defined, zero
    net.update_loads(id=loads.index, p0=p0, q0=q0)

    # ---- generators: proportional dispatch, iterate clipping to [min_p, max_p]
    target = total_load * (1 + a.losses)
    lo, hi = gens.min_p.clip(lower=0).to_numpy(), gens.max_p.to_numpy()
    scale = target / hi.sum()
    for _ in range(50):
        pg = np.clip(hi * scale, lo, hi)
        gap = target - pg.sum()
        free = (pg > lo) & (pg < hi)
        if abs(gap) < 1e-6 * target or not free.any():
            break
        scale *= 1 + gap / pg[free].sum()
    vls = net.get_voltage_levels(attributes=["nominal_v"])
    nom = vls.nominal_v.reindex(gens.voltage_level_id).to_numpy()
    reg = gens.voltage_regulator_on.to_numpy()
    all_gens = net.get_generators(attributes=[])
    z = np.zeros(len(all_gens))
    net.update_generators(id=all_gens.index, target_p=z, target_q=z)
    net.update_generators(id=gens.index, target_p=pg, target_q=np.zeros(len(gens)))
    # PowerModels' PF asserts that every in-service generator sits on a PV/slack bus, and the
    # MATPOWER export types buses from voltage regulation; D-GITT marks most units as non-regulating.
    off = gens.index[~reg]
    net.update_generators(id=off, voltage_regulator_on=np.ones(len(off), dtype=bool), target_v=nom[~reg] * a.vm_setpoint)
    report["regulation"] = {"generators_switched_on": int(len(off)), "vm_setpoint": a.vm_setpoint}
    print(f"voltage regulation switched on for {len(off)} dispatched generators")

    # ---- slack bus: put it on the largest dispatched generator.
    # The D-GITT files carry a slackTerminal on a busbar section with no generator, and the MATPOWER
    # export turns it into the REF bus; PowerModels' PF then has nothing to balance on (bus_gens[REF]).
    old_slack = net.get_extensions("slackTerminal")
    if len(old_slack):
        net.remove_extensions("slackTerminal", list(old_slack.index))
    top = gens.index[int(np.argmax(pg))]
    top_vl = gens.voltage_level_id[top]
    net.create_extensions("slackTerminal", pd.DataFrame({"element_id": [top]}, index=pd.Index([top_vl], name="voltage_level_id")))
    report["slack"] = {
        "generator": top, "voltage_level": top_vl, "pg_mw": float(pg.max()),
        "replaced": old_slack.reset_index().to_dict("records"),
    }
    print(f"slack terminal moved to generator {top} ({pg.max():.0f} MW, {top_vl}); replaced {len(old_slack)} existing terminal(s)")

    # voltage targets for every regulating generator (also disconnected / Pmax=0 ones,
    # otherwise PowSyBl validation rejects a regulator that is on without a setpoint)
    allg = net.get_generators(attributes=["voltage_regulator_on", "voltage_level_id"])
    allg = allg[allg.voltage_regulator_on]
    nom_all = vls.nominal_v.reindex(allg.voltage_level_id).to_numpy()
    net.update_generators(id=allg.index, target_v=nom_all * a.vm_setpoint)

    # ---- reactive limits for generators that have none
    # PowSyBl's MATPOWER export drops capability curves (Qmin = Qmax = 0), so
    # collapse each curve to its overall [min_q, max_q] range.
    curves = net.get_reactive_capability_curve_points()
    if len(curves):
        rng_q = curves.groupby(level=0).agg(min_q=("min_q", "min"), max_q=("max_q", "max"))
        rng_q = rng_q[rng_q.index.isin(gens.index)]
        net.create_minmax_reactive_limits(id=rng_q.index, min_q=rng_q.min_q, max_q=rng_q.max_q)
        print(f"capability curves collapsed to min/max for {len(rng_q)} generators")
    if a.q_limit_ratio > 0:
        full = net.get_generators(attributes=["reactive_limits_kind", "min_q", "max_q", "max_p"])
        bad = full[(full.reactive_limits_kind == "MIN_MAX") & ((full.max_q - full.min_q) <= 0) & full.index.isin(gens.index)]
        if len(bad):
            q = a.q_limit_ratio * bad.max_p
            net.create_minmax_reactive_limits(id=bad.index, min_q=-q, max_q=q)
        print(f"degenerate reactive limits replaced (+/-{a.q_limit_ratio}*Pmax) for {len(bad)} generators")

    # ---- other injections that need a defined state
    bat = net.get_batteries(attributes=[])
    if len(bat):
        z = np.zeros(len(bat))
        net.update_batteries(id=bat.index, target_p=z, target_q=z)
        print(f"batteries set to 0: {len(bat)}")

    # ---- boundary injections
    dl = net.get_dangling_lines(attributes=[])
    if len(dl):
        z = np.zeros(len(dl))
        net.update_dangling_lines(id=dl.index, p0=z, q0=z)

    # ---- voltage bounds (datakit takes VMIN/VMAX from the voltage levels)
    if tuple(a.vlimits) != (0.0, 0.0):
        vl = net.get_voltage_levels(attributes=["nominal_v"])
        net.update_voltage_levels(
            id=vl.index,
            low_voltage_limit=vl.nominal_v * a.vlimits[0],
            high_voltage_limit=vl.nominal_v * a.vlimits[1],
        )
        print(f"voltage bounds set to [{a.vlimits[0]}, {a.vlimits[1]}] p.u. on {len(vl)} voltage levels")

    try:
        net.validate()
    except Exception as e:  # noqa: BLE001
        print("validate() note:", e)
    print("validation level:", net.get_validation_level())
    print(f"load total {p0.sum():.0f} MW, dispatch {pg.sum():.0f} MW, Pmax {hi.sum():.0f} MW, dispatched generators {len(gens)}")

    out = Path(a.output) if a.output else BASE_DIR / f"{stem_of(src)}-base.xiidm.bz2"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "net.xiidm"
        net.save(str(tmp), format="XIIDM")
        with open(tmp, "rb") as fi, bz2.open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("wrote", out)
    report["totals_mw"] = {"load": float(p0.sum()), "dispatch": float(pg.sum()), "pmax": float(hi.sum())}
    report_path = sibling(out, ".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print("wrote", report_path)
    print("\n[load flow on the written case]")
    lf_violations.run(net)


if __name__ == "__main__":
    main()
