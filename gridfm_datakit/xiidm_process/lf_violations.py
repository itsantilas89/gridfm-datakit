"""AC load flow on a PowSyBl network + the operating-limit violations datakit's OPF would see.

Voltage limits follow gridfm_datakit.powsybl.convert.from_powsybl: the voltage level's own
limits (in p.u. of nominal_v), with non-positive values replaced by 0.9 / 1.1.
Thermal limit = lowest permanent current limit of a line / 2-winding transformer.
"""
import numpy as np
import pandas as pd
import pypowsybl as pp


def lf_parameters(controls=False):
    """Load-flow parameters; controls=True also lets transformer taps / shunt sections regulate voltage."""
    return pp.loadflow.Parameters(
        distributed_slack=True,
        transformer_voltage_control_on=controls,
        shunt_compensator_voltage_control_on=controls,
        provider_parameters={"maxOuterLoopIterations": "100", "maxNewtonRaphsonIterations": "30"},
    )


def run(net, verbose=True, params=None):
    """Return (status, dict of violation counts). Prints a summary when verbose."""
    out = {}
    res = pp.loadflow.run_ac(net, params or lf_parameters())[0]
    status = res.status.name
    out["status"] = status
    if verbose:
        print(f"  AC load flow: {status}, iterations={res.iteration_count}")
    if status != "CONVERGED":
        return status, out

    b = net.get_buses().join(net.get_voltage_levels()[["nominal_v", "low_voltage_limit", "high_voltage_limit"]], on="voltage_level_id")
    pu = b.v_mag / b.nominal_v
    lo = (b.low_voltage_limit / b.nominal_v).where(lambda s: s > 0, 0.9).fillna(0.9)
    hi = (b.high_voltage_limit / b.nominal_v).where(lambda s: s > 0, 1.1).fillna(1.1)
    out["v_low"], out["v_high"] = int((pu < lo - 1e-3).sum()), int((pu > hi + 1e-3).sum())
    out["v_min"], out["v_max"] = float(pu.min()), float(pu.max())
    out["v_below_0.9"], out["v_above_1.1"] = int((pu < 0.9).sum()), int((pu > 1.1).sum())

    lim = net.get_operational_limits().reset_index()
    cur = lim[(lim.type == "CURRENT") & (lim.acceptable_duration == -1)].groupby("element_id").value.min()
    over = {}
    for name, df in (("lines", net.get_lines()), ("trafos", net.get_2_windings_transformers())):
        ld = pd.concat([df.i1, df.i2], axis=1).max(axis=1) / cur.reindex(df.index)
        out[f"{name}_over_100"] = int((ld > 1).sum())
        over.update(ld[ld > 1].to_dict())
    out["overloaded"] = dict(sorted(over.items(), key=lambda kv: -kv[1]))

    g = net.get_generators(attributes=["q", "min_q", "max_q", "reactive_limits_kind"])
    mm = g[g.reactive_limits_kind == "MIN_MAX"]
    out["gen_q_out"] = int(((-mm.q > mm.max_q + 1e-3) | (-mm.q < mm.min_q - 1e-3)).sum())

    if verbose:
        print(f"  buses: {len(b)}  V outside own/default limits: low={out['v_low']} high={out['v_high']}"
              f"  (below 0.9: {out['v_below_0.9']}, above 1.1: {out['v_above_1.1']}; min={out['v_min']:.3f} max={out['v_max']:.3f} p.u.)")
        print(f"  overloaded lines: {out['lines_over_100']}  transformers: {out['trafos_over_100']}  (>100% of permanent current limit)")
        for k, v in list(out["overloaded"].items())[:8]:
            print(f"     {k}: {v:.0%}")
        print(f"  generators outside min/max Q: {out['gen_q_out']}")
    return status, out
