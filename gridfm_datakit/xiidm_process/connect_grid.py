#!/usr/bin/env python3
"""Reconnect every disconnected element of a grid snapshot and save the result.

Pick a snapshot from data_rte/raw in a small terminal menu (or pass a path), and the
script closes switches so that no line, transformer, generator, load, ... is left
disconnected, then writes it to data_rte/connected under the same name
(YYYY-MM-DD-HHMM.xiidm.bz2).

How it works (node-breaker topology, as in the RTE snapshots):
  1. pypowsybl's network.connect() on each element that has a disconnected terminal;
     this may operate breakers, disconnectors and fictitious switches.
  2. connect() refuses to touch other switch kinds (e.g. open load-break switches), so
     for terminals that are still disconnected the script closes the open switches on
     the shortest path to a busbar section, preferring busbars of the main component.

  3. Islands that sit next to the main grid in the same voltage level (busbars split by
     an open coupler) are joined by closing the cheapest switch path between them.
     Skip this with --no-merge-buses if you only want elements reconnected.

What it does NOT do: invent equipment. Sub-networks with no equipment linking them to
the main grid stay separate; they are listed at the end.

Usage:
  python connect_grid.py                      # choose a raw snapshot from a menu
  python connect_grid.py data_rte/raw/2023-12-31-2355.xiidm.bz2
  python connect_grid.py --out fixed.xiidm.bz2   # custom output path
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
import pypowsybl.network as pn

from grid_paths import CONNECTED_DIR, RAW_DIR, extension_of, list_snapshots

# getter name on the Network -> elements that can have disconnected terminals.
# Terminal columns are discovered from the "connected*" columns of each table.
ELEMENT_GETTERS = [
    "get_lines",
    "get_2_windings_transformers",
    "get_3_windings_transformers",
    "get_generators",
    "get_loads",
    "get_shunt_compensators",
    "get_dangling_lines",
    "get_static_var_compensators",
    "get_batteries",
    "get_vsc_converter_stations",
    "get_lcc_converter_stations",
]


# ----------------------------------------------------------------------------- menu
def connected_path(p: Path) -> Path:
    """data_rte/connected/<same name as the raw snapshot>.xiidm.bz2"""
    return CONNECTED_DIR / (p.name[: -len(extension_of(p))] + ".xiidm.bz2")


def choose_snapshot() -> Path:
    files = list_snapshots(RAW_DIR)
    if not files:
        sys.exit(f"No snapshots found in {RAW_DIR}. Download one with get_grid_from_hf.py first.")
    print(f"Available raw snapshots in {RAW_DIR}:\n")
    for i, p in enumerate(files, 1):
        done = "  (already connected)" if connected_path(p).exists() else ""
        print(f"  {i:>2}) {p.name}{done}")
    print("   q) quit\n")
    while True:
        try:
            answer = input(f"Choose a snapshot [1-{len(files)}]: ").strip().lower()
        except EOFError:
            sys.exit("\nNo input available; pass the file path as an argument instead.")
        if answer in ("q", "quit", "exit"):
            sys.exit(0)
        if answer.isdigit() and 1 <= int(answer) <= len(files):
            return files[int(answer) - 1]
        print("  Not a valid choice, try again.")


# ------------------------------------------------------------------------- helpers
def terminal_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(connected column, voltage level column, node column) for each terminal of a table."""
    cols = []
    for c in df.columns:
        if c == "connected" or (c.startswith("connected") and c[len("connected"):].isdigit()):
            s = c[len("connected"):]
            cols.append((c, f"voltage_level{s}_id" if s else "voltage_level_id", f"node{s}"))
    return cols


def disconnected_terminals(net) -> list[tuple[str, str, str, int]]:
    """(element kind, element id, voltage level id, node) for every disconnected terminal."""
    out = []
    for getter in ELEMENT_GETTERS:
        df = getattr(net, getter)(all_attributes=True)
        for c, vl, node in terminal_columns(df):
            if node not in df.columns:  # e.g. bus-breaker voltage levels
                continue
            for i in df.index[~df[c].astype(bool)]:
                out.append((getter[4:], i, df.at[i, vl], int(df.at[i, node])))
    return out


def component_sizes(net) -> list[int]:
    return net.get_buses()["connected_component"].value_counts().tolist()


def try_connect(net, element_ids: list[str]) -> None:
    for i in element_ids:
        net.connect(i, operate_disconnectors=True, operate_fictitious=True)


def vl_graph(net, vl_id: str):
    """Graph of one voltage level: edge weight 0 = closed/internal, 1 = open switch."""
    topo = net.get_node_breaker_topology(vl_id)
    g = nx.Graph()
    for sid, r in topo.switches.iterrows():
        w = 1 if r["open"] else 0
        a, b = int(r["node1"]), int(r["node2"])
        if g.has_edge(a, b) and g[a][b]["w"] <= w:
            continue  # keep the cheapest of parallel switches
        g.add_edge(a, b, w=w, switch=sid)
    for r in topo.internal_connections.itertuples():
        g.add_edge(int(r.node1), int(r.node2), w=0, switch=None)
    busbars = set(topo.nodes.index[topo.nodes["connectable_type"] == "BUSBAR_SECTION"])
    node_to_busbar = topo.nodes.loc[list(busbars), "connectable_id"].to_dict()
    return g, node_to_busbar


def close_path_to_busbar(net, g, node_to_busbar, node, preferred_busbars) -> list[str]:
    """Close the open switches on the cheapest path from `node` to a busbar. Returns switch ids."""
    if node not in g:
        return []
    dist, paths = nx.single_source_dijkstra(g, node, weight="w")
    reachable = [n for n in node_to_busbar if n in dist]
    if not reachable:
        return []
    preferred = [n for n in reachable if node_to_busbar[n] in preferred_busbars]
    target = min(preferred or reachable, key=lambda n: dist[n])
    path = paths[target]
    to_close = []
    for a, b in zip(path, path[1:]):
        edge = g[a][b]
        if edge["w"] == 1:
            to_close.append(edge["switch"])
            edge["w"] = 0  # later terminals in this voltage level see it as closed
    if to_close:
        net.update_switches(pd.DataFrame({"open": False}, index=pd.Index(to_close, name="id")))
    return to_close


def close_remaining_paths(net, terminals) -> tuple[int, int]:
    """Fallback for terminals connect() could not handle. Returns (fixed, skipped)."""
    buses = net.get_buses()
    main = buses["connected_component"].value_counts().idxmax()
    main_buses = set(buses.index[buses["connected_component"] == main])
    bbs = net.get_busbar_sections()
    preferred = set(bbs.index[bbs["bus_id"].isin(main_buses)])

    graphs, fixed, skipped = {}, 0, 0
    for _kind, _eid, vl, node in terminals:
        try:
            if vl not in graphs:
                graphs[vl] = vl_graph(net, vl)
        except Exception:  # not a node-breaker voltage level
            skipped += 1
            continue
        g, node_to_busbar = graphs[vl]
        if close_path_to_busbar(net, g, node_to_busbar, node, preferred):
            fixed += 1
        else:
            skipped += 1
    return fixed, skipped


def node_bus_table(net) -> dict:
    """(voltage level, node) -> bus id, from every element terminal that sits on a bus."""
    table = {}
    for getter in ELEMENT_GETTERS:
        df = getattr(net, getter)(all_attributes=True)
        for c, vl, node in terminal_columns(df):
            bus = f"bus{c[len('connected'):]}_id"
            if node in df.columns and bus in df.columns:
                for v, nd, b in zip(df[vl], df[node], df[bus]):
                    if b:
                        table[(v, int(nd))] = b
    return table


def merge_islands(net, max_rounds: int = 10) -> int:
    """Join islands to the main grid by closing switches *inside* voltage levels.

    Typical case: a substation whose busbars are split by an open coupler, so one busbar
    (with its lines) forms an island next to the main grid. Per round, every island gets
    the cheapest (fewest open switches) path to a main-grid node in one of its voltage
    levels closed. Returns the number of switches closed.
    """
    total = 0
    for _ in range(max_rounds):
        buses = net.get_buses()
        vc = buses["connected_component"].value_counts()
        if len(vc) < 2:
            break
        main = vc.index[0]
        comp_of_bus = buses["connected_component"].to_dict()
        bbs = net.get_busbar_sections()
        table = node_bus_table(net)
        touched, closed_this_round = set(), 0

        for comp in vc.index[1:]:
            for vl in sorted(set(buses.loc[buses["connected_component"] == comp, "voltage_level_id"])):
                if vl in touched:
                    continue
                try:
                    g, node_to_busbar = vl_graph(net, vl)
                except Exception:  # not node-breaker
                    continue
                known = dict(table_for_vl(table, vl))
                for node, bb in node_to_busbar.items():
                    bus = bbs.at[bb, "bus_id"]
                    if bus:
                        known[node] = bus
                zero = nx.Graph()
                zero.add_nodes_from(g.nodes)
                zero.add_edges_from((a, b) for a, b, d in g.edges(data=True) if d["w"] == 0)
                src, dst = [], []
                for cluster in nx.connected_components(zero):
                    comps = {comp_of_bus[known[n]] for n in cluster if n in known and known[n] in comp_of_bus}
                    if comp in comps:
                        src += cluster
                    elif main in comps:
                        dst += cluster
                if not src or not dst:
                    continue
                dist, paths = nx.multi_source_dijkstra(g, src, weight="w")
                reachable = [n for n in dst if n in dist]
                if not reachable:
                    continue
                path = paths[min(reachable, key=lambda n: dist[n])]
                to_close = [g[a][b]["switch"] for a, b in zip(path, path[1:]) if g[a][b]["w"] == 1]
                if to_close:
                    net.update_switches(pd.DataFrame({"open": False}, index=pd.Index(to_close, name="id")))
                    closed_this_round += len(to_close)
                    touched.add(vl)
                    break  # island handled for this round
        total += closed_this_round
        if not closed_this_round:
            break
    return total


def table_for_vl(table: dict, vl: str):
    return ((node, bus) for (v, node), bus in table.items() if v == vl)


def describe_islands(net, limit: int = 10) -> None:
    buses = net.get_buses()
    vc = buses["connected_component"].value_counts()
    others = vc.iloc[1:]
    if others.empty:
        return
    print(f"\n{len(others)} island(s) remain that cannot be joined by closing switches "
          f"(no connecting equipment to the main grid). Sizes: {dict(Counter(others.tolist()))}")
    for comp in others.index[:limit]:
        vls = sorted(set(buses.loc[buses["connected_component"] == comp, "voltage_level_id"]))
        print(f"  component {comp}: {', '.join(vls)}")
    if len(others) > limit:
        print(f"  ... and {len(others) - limit} more")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, help=".xiidm / .xiidm.bz2 (default: choose from menu)")
    ap.add_argument("--out", type=Path, help="output path (default: <name>-connected.xiidm.bz2 next to the input)")
    ap.add_argument("--no-merge-buses", action="store_true",
                    help="skip step 3 (closing couplers to join islands to the main grid)")
    args = ap.parse_args()

    path = args.file or choose_snapshot()
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    out = args.out or connected_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {path} ...")
    net = pn.load(str(path))
    switches_before = net.get_switches()["open"].copy()
    sizes_before = component_sizes(net)

    terms = disconnected_terminals(net)
    print(f"Connected components: {len(sizes_before)} (main: {sizes_before[0]} buses)")
    print(f"Disconnected terminals: {len(terms)}  {dict(Counter(t[0] for t in terms))}")

    if terms:
        print("\nStep 1: connect() on each affected element ...")
        try_connect(net, sorted({t[1] for t in terms}))
        terms = disconnected_terminals(net)
        print(f"  still disconnected: {len(terms)}")

    if terms:
        print("Step 2: closing open switches on the shortest path to a busbar ...")
        fixed, skipped = close_remaining_paths(net, terms)
        terms = disconnected_terminals(net)
        print(f"  fixed {fixed}, skipped {skipped}; still disconnected: {len(terms)}")

    if not args.no_merge_buses and len(component_sizes(net)) > 1:
        print("Step 3: joining islands to the main grid (closing bus couplers) ...")
        n_closed = merge_islands(net)
        print(f"  closed {n_closed} switch(es); components now: {len(component_sizes(net))}")

    if terms:
        print("\nTerminals that could not be reconnected (no switch path to any busbar):")
        for kind, eid, vl, node in terms[:15]:
            print(f"  {kind} {eid} (voltage level {vl}, node {node})")
        if len(terms) > 15:
            print(f"  ... and {len(terms) - 15} more")

    switches_after = net.get_switches()["open"]
    closed = switches_before[switches_before & ~switches_after.reindex(switches_before.index)]
    kinds = net.get_switches().loc[closed.index, "kind"].value_counts().to_dict()
    sizes_after = component_sizes(net)
    print(f"\nSwitches closed: {len(closed)}  {kinds}")
    print(f"Connected components: {len(sizes_before)} -> {len(sizes_after)} "
          f"(main grid: {sizes_before[0]} -> {sizes_after[0]} buses)")
    describe_islands(net)

    net.save(str(out))
    print(f"\nSaved {out}")

    check = component_sizes(pn.load(str(out)))
    status = "OK" if check == sizes_after else "MISMATCH"
    print(f"Reloaded saved file: {len(check)} component(s), main {check[0]} buses [{status}]")


if __name__ == "__main__":
    main()
