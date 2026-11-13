#!/usr/bin/env python3
"""Plot the electrically connected part of a grid snapshot (.xiidm / .xiidm.bz2) with pypowsybl.

"Electrically connected" is taken from powsybl's bus view: open switches and
disconnected terminals are respected, and the connected components are computed on
the resulting buses. By default only the largest connected component is drawn;
small islands and lines left open at one end are left out.

The graph is drawn at substation level: nodes are substations, edges are lines
(transformers join voltage levels *inside* a substation, so they only matter for
connectivity). RTE files carry no coordinates, so the layout is force-directed
and shows topology, not geography.

Usage:
  python plot_grid.py                          # newest snapshot in data_rte/connected (else raw) -> images/<name>.png
  python plot_grid.py data_rte/raw/2023-12-31-2355.xiidm.bz2
  python plot_grid.py --min-kv 225             # only lines >= 225 kV (clearer for big grids)
  python plot_grid.py --all-components         # draw every component, not just the largest
  python plot_grid.py --out grid.pdf --dpi 300

Requires: pip install pypowsybl networkx matplotlib igraph
"""
import argparse
import sys
from pathlib import Path

import random

import igraph as ig
import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import networkx as nx
import pypowsybl.network as pn
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from grid_paths import CONNECTED_DIR, IMAGES_DIR, RAW_DIR, extension_of, newest_snapshot


def default_image_path(snapshot: Path) -> Path:
    """images/<snapshot name>.png; a plot of a raw snapshot gets '-raw' so it can't clobber the connected one."""
    stem = snapshot.name[: -len(extension_of(snapshot))]
    suffix = "-raw" if snapshot.parent.resolve() == RAW_DIR.resolve() else ""
    return IMAGES_DIR / f"{stem}{suffix}.png"

# (minimum kV, label, colour, edge width, node size) - drawn lowest voltage first.
VOLTAGE_CLASSES = [
    (0, "< 63 kV", "#9aa5b1", 0.25, 2),
    (63, "63 kV", "#2a9d8f", 0.35, 3),
    (90, "90 kV", "#4361ee", 0.50, 4),
    (150, "150 kV", "#f4a261", 0.70, 6),
    (225, "225 kV", "#e76f51", 1.00, 9),
    (380, "400 kV", "#b5179e", 1.50, 14),
]


def voltage_class(kv: float) -> int:
    """Index of the highest class whose lower bound is <= kv."""
    return max(i for i, (lo, *_rest) in enumerate(VOLTAGE_CLASSES) if kv >= lo)


def find_default_file() -> Path:
    path = newest_snapshot(CONNECTED_DIR) or newest_snapshot(RAW_DIR)
    if path is None:
        sys.exit(f"No .xiidm / .xiidm.bz2 file found in {CONNECTED_DIR} or {RAW_DIR}. Pass a path explicitly.")
    return path


def build_graph(net, all_components: bool, min_kv: float):
    buses = net.get_buses()
    vls = net.get_voltage_levels()[["substation_id", "nominal_v"]]

    if all_components:
        keep = buses
    else:
        main = buses["connected_component"].value_counts().idxmax()
        keep = buses[buses["connected_component"] == main]
    n_islands = buses["connected_component"].nunique() - (0 if all_components else 1)
    keep_ids = set(keep.index)

    bus_info = keep.join(vls, on="voltage_level_id")
    sub_kv = bus_info.groupby("substation_id")["nominal_v"].max()

    # An edge exists only if BOTH terminals sit on a bus of the selected component
    # (bus id is empty when a terminal is disconnected).
    G = nx.Graph()
    n_open_ended = 0
    for kind, df in (("line", net.get_lines()), ("transformer", net.get_2_windings_transformers())):
        for row in df.itertuples():
            b1, b2 = row.bus1_id, row.bus2_id
            if not b1 or not b2:
                n_open_ended += kind == "line"
                continue
            if b1 not in keep_ids or b2 not in keep_ids:
                continue
            s1 = vls.at[row.voltage_level1_id, "substation_id"]
            s2 = vls.at[row.voltage_level2_id, "substation_id"]
            kv = vls.at[row.voltage_level1_id, "nominal_v"]
            if s1 == s2 or kind != "line" or kv < min_kv:
                continue  # intra-substation transformer, or below the voltage filter
            # Parallel circuits collapse into one edge; keep the highest voltage seen.
            if G.has_edge(s1, s2):
                G[s1][s2]["kv"] = max(G[s1][s2]["kv"], kv)
                G[s1][s2]["circuits"] += 1
            else:
                G.add_edge(s1, s2, kv=kv, circuits=1)

    isolated = [s for s in G.nodes if G.degree(s) == 0]  # (none expected, kept for safety)
    G.remove_nodes_from(isolated)
    for s in G.nodes:
        G.nodes[s]["kv"] = sub_kv.get(s, 0.0)

    stats = {
        "buses_total": len(buses),
        "buses_kept": len(keep),
        "islands_dropped": n_islands,
        "lines_open_ended": n_open_ended,
        "substations": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }
    return G, stats


def compute_layout(G, method: str, iterations: int, seed: int) -> dict:
    """Force-directed layout via igraph (C implementation; networkx's takes minutes at this size)."""
    nodes = list(G.nodes)
    index = {s: i for i, s in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in G.edges])
    random.seed(seed)  # igraph draws its random starting positions from Python's RNG
    if method == "drl":
        coords = g.layout_drl()
    else:
        # grid="nogrid" = exact repulsion between all node pairs. igraph's default grid
        # approximation is ~20x faster but gives a tangled hairball on this grid.
        coords = g.layout_fruchterman_reingold(niter=iterations, grid="nogrid")
    return {s: tuple(coords[i]) for s, i in index.items()}


def draw(G, pos, out: Path, dpi: int, title: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_axis_off()

    edge_class = {e: voltage_class(d["kv"]) for e, d in G.edges.items()}
    node_class = {s: voltage_class(d["kv"]) for s, d in G.nodes.items()}

    for i, (_lo, _label, colour, width, _size) in enumerate(VOLTAGE_CLASSES):
        segs = [(pos[u], pos[v]) for (u, v), c in edge_class.items() if c == i]
        if segs:
            ax.add_collection(LineCollection(segs, colors=colour, linewidths=width, alpha=0.8, zorder=1 + i))
    for i, (_lo, _label, colour, _w, size) in enumerate(VOLTAGE_CLASSES):
        pts = [pos[s] for s, c in node_class.items() if c == i]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=size, c=colour, linewidths=0, zorder=10 + i)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    present = sorted({*edge_class.values(), *node_class.values()})
    ax.legend(
        handles=[
            Line2D([0], [0], color=VOLTAGE_CLASSES[i][2], lw=3, label=VOLTAGE_CLASSES[i][1]) for i in present
        ],
        title="Highest voltage",
        loc="lower left",
        frameon=False,
    )
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, help=".xiidm or .xiidm.bz2 (default: newest in data_rte/connected, else raw)")
    ap.add_argument("--out", type=Path, help="output image, png/pdf/svg (default: images/<snapshot name>.png)")
    ap.add_argument("--min-kv", type=float, default=0, help="only draw lines with nominal voltage >= this (kV)")
    ap.add_argument("--all-components", action="store_true", help="draw all connected components, not just the largest")
    ap.add_argument("--layout", choices=["fr", "drl"], default="fr",
                    help="fr = Fruchterman-Reingold, best readability (~1 min for 5k substations); "
                         "drl = DrL, faster (~15 s) but squeezes substations into tight clusters")
    ap.add_argument("--iterations", type=int, default=500, help="iterations for --layout fr")
    ap.add_argument("--seed", type=int, default=42, help="layout seed (same seed = same picture)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    path = args.file or find_default_file()
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    print(f"Loading {path} ...")
    net = pn.load(str(path))

    G, st = build_graph(net, args.all_components, args.min_kv)
    if G.number_of_nodes() == 0:
        sys.exit("Nothing to draw (no connected lines match the filters).")
    print(
        f"Buses kept: {st['buses_kept']}/{st['buses_total']} "
        f"({st['islands_dropped']} other component(s) dropped); "
        f"lines open at one end ignored: {st['lines_open_ended']}"
    )
    print(f"Drawing {st['substations']} substations, {st['edges']} line connections")

    print("Computing layout ...")
    pos = compute_layout(G, args.layout, args.iterations, args.seed)

    scope = "all components" if args.all_components else "largest connected component"
    kv = f", lines >= {args.min_kv:g} kV" if args.min_kv else ""
    title = f"{net.id}\n{scope}{kv}: {st['substations']} substations, {st['edges']} line connections"
    out = args.out or default_image_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(G, pos, out, args.dpi, title)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
