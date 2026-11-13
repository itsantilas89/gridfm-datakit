"""Find which constraint class makes the AC OPF infeasible on a base-case snapshot.

Runs the same PowerModels OPF datakit uses on the unmodified case, then on copies with
one constraint class relaxed at a time, and prints Ipopt's termination status for each.
Heavy (one OPF per variant, ~minutes on a large grid): run it on the server, not a laptop.

Usage (from this folder):
    python diagnose_opf.py [snapshot.xiidm[.bz2]]        # default: newest in data_rte/base
    python diagnose_opf.py FILE --max-iter 300 --only baseline,no_shunts
"""
import argparse
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gridfm_datakit.powsybl import load_net  # noqa: E402
from gridfm_datakit.process.process_network import init_julia  # noqa: E402
from grid_paths import BASE_DIR, newest_snapshot  # noqa: E402

# MATPOWER column indices
GS, BS, VMAX, VMIN = 4, 5, 11, 12
QMAX, QMIN = 3, 4
RATE_A = 5


def variants(net):
    def mod(fn):
        n = deepcopy(net)
        fn(n)
        return n

    def no_thermal(n):
        n.branches[:, RATE_A] = 0.0  # 0 = unlimited in MATPOWER

    def wide_v(n):
        n.buses[:, VMIN], n.buses[:, VMAX] = 0.7, 1.3

    def no_shunts(n):
        n.buses[:, GS] = 0.0
        n.buses[:, BS] = 0.0

    def wide_q(n):
        n.gens[:, QMAX], n.gens[:, QMIN] = 9999.0, -9999.0

    def no_q_load(n):
        n.Qd = n.Qd * 0.0

    def light_load(n):
        n.Pd = n.Pd * 0.2
        n.Qd = n.Qd * 0.2

    def all_relaxed(n):
        no_thermal(n), wide_v(n), no_shunts(n), wide_q(n)

    return {
        "baseline": net,
        "no_thermal_limits": mod(no_thermal),
        "wide_voltage_0.7-1.3": mod(wide_v),
        "no_shunts": mod(no_shunts),
        "wide_gen_q_limits": mod(wide_q),
        "no_reactive_load": mod(no_q_load),
        "load_x0.2": mod(light_load),
        "all_four_relaxed": mod(all_relaxed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--only", help="comma-separated variant names")
    a = ap.parse_args()
    src = a.file or newest_snapshot(BASE_DIR)
    if src is None:
        raise SystemExit("no snapshot given and none found in data_rte/base")

    net = load_net(str(src)).gfm_net
    jl = init_julia(max_iter=a.max_iter, solver_log_dir=None, coinhsl=None)
    todo = variants(net)
    if a.only:
        todo = {k: v for k, v in todo.items() if k in a.only.split(",")}

    print(f"\nOPF on {src.name}")
    for name, n in todo.items():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".m", delete=False) as t:
            path = t.name
        try:
            n.to_mpc(path)
            try:
                res = jl.run_opf(path)
                status = str(res["termination_status"])
            except Exception as e:  # noqa: BLE001
                status = f"error: {e}"
        finally:
            os.unlink(path)
        print(f"  {name:24s} {status}", flush=True)


if __name__ == "__main__":
    main()
