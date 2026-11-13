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
    python make_base_case.py [IN.xiidm[.bz2]] [OUT.xiidm.bz2] [--load-fraction 0.3] [--seed 0]

Default input: newest snapshot in data_rte/connected (else raw). Default output:
data_rte/base/<name>-base.xiidm.bz2.
"""

import argparse
import bz2
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pypowsybl as pp
from grid_paths import BASE_DIR, CONNECTED_DIR, RAW_DIR, extension_of, newest_snapshot


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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = a.input or newest_snapshot(CONNECTED_DIR, RAW_DIR)
    if src is None:
        raise SystemExit("no input given and no snapshot in data_rte/{connected,raw}")
    rng = np.random.default_rng(a.seed)
    net = pp.network.load(str(src))

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


if __name__ == "__main__":
    main()
