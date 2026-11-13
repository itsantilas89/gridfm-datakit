"""Give an injection-free PowSyBl snapshot (e.g. D-GITT RTE7000) a synthetic nominal state.

gridfm-datakit scales the *existing* per-bus loads (p_i * profile * noise), so the
network needs non-zero nominal loads, plus generator setpoints. This script writes:

  * load p0/q0: total load = --load-fraction * sum(max_p of connected generators),
    split over connected loads (uniform, or noisy with --weight-sigma), q from --power-factor
  * generator target_p: proportional to max_p, clipped to [min_p, max_p], total = load * (1 + losses)
  * generator target_v (kV): nominal voltage of the voltage level * --vm-setpoint (regulating units)
  * target_q = 0 for non-regulating units; dangling-line p0/q0 = 0
  * (optional) symmetric reactive limits for generators that have none

It is a synthetic operating point, NOT real RTE data.

Usage:
    python make_base_case.py [IN.xiidm[.bz2]] [OUT.xiidm.bz2] [--load-fraction 0.3] [--power-factor 0.95] [--relax-overloads] [--seed 0]

Default input: newest snapshot in data_rte/connected (else raw). Default output:
data_rte/base/<name>-base.xiidm.bz2.
"""

import argparse
import bz2
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl as pp
import lf_violations
from grid_paths import BASE_DIR, CONNECTED_DIR, RAW_DIR, extension_of, newest_snapshot


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
    ap.add_argument("--load-fraction", type=float, default=0.3, help="total load / connected Pmax")
    ap.add_argument("--losses", type=float, default=0.02, help="dispatch = load * (1 + losses)")
    ap.add_argument("--power-factor", type=float, default=0.95)
    ap.add_argument("--weight-sigma", type=float, default=0.5, help="lognormal sigma of per-load weights (0 = uniform)")
    ap.add_argument("--vm-setpoint", type=float, default=1.0, help="p.u. voltage target of regulating generators")
    ap.add_argument("--q-limit-ratio", type=float, default=0.4, help="Qmax=-Qmin=ratio*Pmax for gens without reactive limits (0 = skip)")
    ap.add_argument("--vlimits", type=float, nargs=2, metavar=("LO", "HI"), default=(0.9, 1.1),
                    help="p.u. voltage bounds written to every voltage level (0 0 = keep the file's own limits)")
    ap.add_argument("--controls", action="store_true",
                    help="let transformer taps / shunt sections regulate voltage before saving (slow; did not converge reliably on D-GITT)")
    ap.add_argument("--limit-scale", type=float, default=1.0,
                    help="multiply every permanent current limit by this factor (2.0 made the OPF feasible on the 225/380 kV D-GITT grid)")
    ap.add_argument("--relax-overloads", action="store_true",
                    help="raise the permanent current limit of elements still overloaded after the load flow to 110%% of their loading")
    ap.add_argument("--min-kv", type=float, default=0.0,
                    help="keep only voltage levels with nominal_v >= this (kV), e.g. 200 for the 225/380 kV grid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = a.input or newest_snapshot(CONNECTED_DIR) or newest_snapshot(RAW_DIR)
    if src is None:
        raise SystemExit("no input given and no snapshot in data_rte/{connected,raw}")
    rng = np.random.default_rng(a.seed)
    net = pp.network.load(str(src))

    if a.min_kv > 0:
        vls = net.get_voltage_levels(attributes=["nominal_v"])
        remove_voltage_levels(net, vls.index[vls.nominal_v < a.min_kv])
        dropped = keep_main_component(net)
        print(f"kept voltage levels >= {a.min_kv} kV: {len(net.get_voltage_levels())} levels, {len(net.get_buses())} buses "
              f"({dropped} disconnected levels dropped), {len(net.get_loads())} loads, {len(net.get_generators())} generators")

    gens = net.get_generators(attributes=["max_p", "min_p", "voltage_regulator_on", "voltage_level_id", "connected"])
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

    # ---- solve once with tap/shunt control and keep the solved tap positions / shunt sections
    if a.controls:
        res = pp.loadflow.run_ac(net, lf_violations.lf_parameters(controls=True))[0]
        print(f"load flow with transformer/shunt voltage control: {res.status.name} ({res.iteration_count} iterations)")

    # ---- scale all permanent current limits (the OPF is thermally infeasible with the real ratings
    # and a synthetic load placement; see diagnose_opf.py)
    if a.limit_scale != 1.0:
        lim = net.get_operational_limits()
        ix = lim.index
        keep = (ix.get_level_values("type") == "CURRENT") & (ix.get_level_values("acceptable_duration") == -1)
        keep &= (lim.value < 1e300).to_numpy()
        sel = lim[keep]
        net.update_operational_limits(pd.DataFrame({"value": sel.value.to_numpy() * a.limit_scale}, index=sel.index))
        print(f"permanent current limits x{a.limit_scale} on {len(sel)} limits")

    if a.relax_overloads:
        _, v = lf_violations.run(net, verbose=False)
        over = v.get("overloaded", {})
        if over:
            lim = net.get_operational_limits()
            ix = lim.index
            keep = (ix.get_level_values("type") == "CURRENT") & (ix.get_level_values("acceptable_duration") == -1)
            keep &= ix.get_level_values("element_id").isin(list(over))
            sel = lim[keep]
            factor = np.array([over[e] * 1.1 for e in sel.index.get_level_values("element_id")])
            net.update_operational_limits(pd.DataFrame({"value": sel.value.to_numpy() * factor}, index=sel.index))
            print(f"permanent current limit raised for {len(over)} overloaded elements")

    try:
        net.validate()
    except Exception as e:  # noqa: BLE001
        print("validate() note:", e)
    print("validation level:", net.get_validation_level())
    print(f"load total {p0.sum():.0f} MW, dispatch {pg.sum():.0f} MW, Pmax {hi.sum():.0f} MW, regulating gens {int(reg.sum())}")

    out = Path(a.output) if a.output else BASE_DIR / (src.name[: -len(extension_of(src))] + "-base.xiidm.bz2")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "net.xiidm"
        net.save(str(tmp), format="XIIDM")
        with open(tmp, "rb") as fi, bz2.open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("wrote", out)
    print("\n[load flow on the written case]")
    lf_violations.run(net)


if __name__ == "__main__":
    main()
