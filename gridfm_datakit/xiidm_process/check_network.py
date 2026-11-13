"""Diagnostic checks for a PowSyBl network file (read-only, no scenario generation).

Usage (from this folder):
    python check_network.py [file.xiidm[.bz2]]   # default: newest in data_rte/base, connected, raw
The printed report is also saved to reports/<snapshot name>.txt (gitignored).
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pypowsybl as pp
import scipy.io
import scipy.sparse as sp
import lf_violations
from grid_paths import BASE_DIR, CONNECTED_DIR, RAW_DIR, REPORTS_DIR, extension_of, newest_snapshot
from scipy.sparse.csgraph import connected_components

BUS = ["BUS_I", "TYPE", "PD", "QD", "GS", "BS", "AREA", "VM", "VA", "BASEKV", "ZONE", "VMAX", "VMIN"]
GEN = ["GEN_BUS", "PG", "QG", "QMAX", "QMIN", "VG", "MBASE", "STATUS", "PMAX", "PMIN"]
BRA = ["F_BUS", "T_BUS", "R", "X", "B", "RATE_A", "RATE_B", "RATE_C", "TAP", "SHIFT", "STATUS", "ANGMIN", "ANGMAX"]


def nan_report(name, m, cols):
    print(f"\n[NaN per column] {name} ({m.shape[0]} rows)")
    for j, c in enumerate(cols):
        if j < m.shape[1]:
            n = int(np.isnan(m[:, j]).sum())
            if n:
                print(f"  {c:8s} {n}")


def main(path):
    net = pp.network.load(path)
    print("Loaded:", path)
    for k in ("buses", "generators", "loads", "lines", "2_windings_transformers", "shunt_compensators", "dangling_lines", "hvdc_lines"):
        print(f"  {k}: {len(getattr(net, 'get_' + k)())}")

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as t:
        mp = t.name
    net.save(mp, format="MATPOWER")
    mpc = scipy.io.loadmat(mp, struct_as_record=True, squeeze_me=False)["mpc"][0, 0]
    Path(mp).unlink()
    base = float(mpc["baseMVA"][0, 0])
    bus, gen, br = (mpc[k].astype(float) for k in ("bus", "gen", "branch"))
    print(f"\nbaseMVA={base}  buses={len(bus)} gens={len(gen)} branches={len(br)}")

    # 1. what the NaN fill touched
    nan_report("bus", bus, BUS)
    nan_report("gen", gen, GEN)
    nan_report("branch", br, BRA)
    bus0, gen0, br0 = (np.nan_to_num(x, nan=0.0) for x in (bus, gen, br))

    # 2. generator sanity (after zero-fill)
    st = gen0[:, 7] > 0
    pmax, pmin = gen0[:, 8], gen0[:, 9]
    qmax, qmin = gen0[:, 3], gen0[:, 4]
    print("\n[Generators after zero-fill]")
    print(f"  in service: {st.sum()}/{len(gen)}")
    print(f"  PMAX==0 (in service): {(st & (pmax == 0)).sum()}   PMAX<=PMIN: {(st & (pmax <= pmin)).sum()}")
    print(f"  QMAX==QMIN (in service): {(st & (qmax == qmin)).sum()}   QMAX==0&QMIN==0: {(st & (qmax == 0) & (qmin == 0)).sum()}")
    print(f"  PG outside [PMIN,PMAX]: {(st & ((gen0[:,1] > pmax + 1e-6) | (gen0[:,1] < pmin - 1e-6))).sum()}")
    print(f"  Sum PMAX={pmax[st].sum():.0f} MW  Sum PG={gen0[st,1].sum():.0f} MW  Sum PMIN={pmin[st].sum():.0f} MW")
    print(f"  Sum load PD={bus0[:,2].sum():.0f} MW  (headroom PMAX/PD = {pmax[st].sum()/max(bus0[:,2].sum(),1e-9):.2f})")
    print(f"  VG outside [0.9,1.1]: {(gen0[:,5] < 0.9).sum() + (gen0[:,5] > 1.1).sum()}")

    # 3. bus sanity
    print("\n[Buses]")
    print(f"  types: " + ", ".join(f"{int(t)}:{int((bus0[:,1]==t).sum())}" for t in np.unique(bus0[:, 1])))
    print(f"  VMAX<=0: {(bus0[:,11] <= 0).sum()}  VMIN<=0: {(bus0[:,12] <= 0).sum()}  VMIN>=VMAX: {(bus0[:,12] >= bus0[:,11]).sum()}  BASEKV<=0: {(bus0[:,9] <= 0).sum()}")
    print(f"  VM outside [0.8,1.2]: {((bus0[:,7] < 0.8) | (bus0[:,7] > 1.2)).sum()}")

    # 4. branch sanity
    on = br0[:, 10] > 0
    print("\n[Branches after zero-fill]")
    print(f"  in service: {on.sum()}/{len(br)}")
    print(f"  X==0: {(on & (br0[:,3] == 0)).sum()}   R==0&X==0: {(on & (br0[:,2] == 0) & (br0[:,3] == 0)).sum()}   |X|<1e-5: {(on & (np.abs(br0[:,3]) < 1e-5)).sum()}")
    print(f"  RATE_A==0 (unlimited): {(on & (br0[:,5] == 0)).sum()}")
    print(f"  TAP==0 (means 1): {(on & (br0[:,8] == 0)).sum()}   TAP>1.5 or <0.5: {(on & (br0[:,8] > 1.5)).sum() + (on & (br0[:,8] != 0) & (br0[:,8] < 0.5)).sum()}")

    # 5. connectivity / slack
    idx = {int(b): i for i, b in enumerate(bus0[:, 0])}
    f = np.array([idx[int(x)] for x in br0[on, 0]])
    tt = np.array([idx[int(x)] for x in br0[on, 1]])
    n = len(bus0)
    A = sp.coo_matrix((np.ones(len(f)), (f, tt)), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    print(f"\n[Connectivity] {nc} connected components; sizes (largest 10): {sorted(sizes, reverse=True)[:10]}")
    ref = np.where(bus0[:, 1] == 3)[0]
    print(f"  REF buses: {len(ref)} in components {sorted(set(lab[ref]))}")
    genbus = {int(idx[int(g)]) for g in gen0[st, 0]}
    for c in range(nc):
        members = np.where(lab == c)[0]
        if len(members) < max(sizes) :
            has_gen = any(m in genbus for m in members)
            has_ref = any(bus0[m, 1] == 3 for m in members)
            print(f"  island {c}: {len(members)} buses, gen={has_gen}, ref={has_ref}, load={bus0[members,2].sum():.1f} MW")
    big = int(np.argmax(sizes))
    mem = np.where(lab == big)[0]
    gin = [g for g in np.where(st)[0] if idx[int(gen0[g, 0])] in set(mem)]
    print(f"  main island: load={bus0[mem,2].sum():.0f} MW  PMAX={pmax[gin].sum():.0f} MW  PG={gen0[gin,1].sum():.0f} MW")

    # 6. PowSyBl AC load flow + operating-limit violations
    print("\n[AC load flow and limit violations]")
    try:
        lf_violations.run(net)
    except Exception as e:  # noqa: BLE001
        print("  loadflow raised:", e)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_snapshot(BASE_DIR, CONNECTED_DIR, RAW_DIR)
    if src is None:
        raise SystemExit("no snapshot given and none found in data_rte/{base,connected,raw}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            main(str(src))
        finally:
            pass
    text = buf.getvalue()
    print(text)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / (src.name[: -len(extension_of(src))] + ".txt")
    out.write_text(text, encoding="utf-8")
    print("report saved to", out)
