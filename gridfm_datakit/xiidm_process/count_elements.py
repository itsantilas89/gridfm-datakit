#!/usr/bin/env python3
"""Count the network elements present in an .xiidm or .xiidm.bz2 file.

Usage:
  python count_elements.py                       # newest snapshot in data_rte/raw
  python count_elements.py data_rte/connected/2023-12-31-2355.xiidm.bz2
  python count_elements.py path/to/file.xiidm --all   # also list every other XML tag

Uses only the standard library; the file is streamed, so large grids are fine.
"""
import argparse
import bz2
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from grid_paths import RAW_DIR, newest_snapshot

# XML tag -> label shown in the report (in display order).
EQUIPMENT = {
    "substation": "substations",
    "voltageLevel": "voltage levels",
    "busbarSection": "busbars",
    "bus": "buses (bus-breaker)",
    "switch": "switches",
    "line": "lines",
    "tieLine": "tie lines",
    "twoWindingsTransformer": "2-winding transformers",
    "threeWindingsTransformer": "3-winding transformers",
    "danglingLine": "dangling lines",
    "hvdcLine": "HVDC lines",
    "generator": "generators",
    "load": "loads",
    "battery": "batteries",
    "shunt": "shunt compensators",
    "staticVarCompensator": "static var compensators",
    "lccConverterStation": "LCC converter stations",
    "vscConverterStation": "VSC converter stations",
}

# Elements that connect two (or more) nodes, i.e. the "edges" of the grid.
BRANCH_TAGS = ("line", "tieLine", "twoWindingsTransformer", "threeWindingsTransformer")


def find_default_file() -> Path:
    path = newest_snapshot(RAW_DIR)
    if path is None:
        sys.exit(f"No .xiidm / .xiidm.bz2 file found in {RAW_DIR}. Pass a path explicitly.")
    return path


def open_file(path: Path):
    return bz2.open(path, "rb") if path.suffix == ".bz2" else path.open("rb")


def count_elements(path: Path):
    tags: Counter = Counter()
    switch_kinds: Counter = Counter()
    network_attrs: dict = {}
    with open_file(path) as f:
        for _, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]  # drop the XML namespace
            tags[tag] += 1
            if tag == "switch":
                switch_kinds[el.get("kind", "?")] += 1
            elif tag == "network":
                network_attrs = dict(el.attrib)
            el.clear()  # free memory as we go
    return tags, switch_kinds, network_attrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, help=".xiidm or .xiidm.bz2 (default: newest in data_rte/raw)")
    ap.add_argument("--all", action="store_true", help="also list all remaining XML tags")
    args = ap.parse_args()

    path = args.file or find_default_file()
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    try:
        tags, switch_kinds, net = count_elements(path)
    except (ET.ParseError, OSError, EOFError) as e:
        sys.exit(f"Could not read {path}: {e}")

    print(f"File: {path}")
    if net.get("id"):
        print(f"Network: {net['id']}  (caseDate {net.get('caseDate', '?')})")
    print()

    width = max(len(v) for v in EQUIPMENT.values()) + 1
    for tag, label in EQUIPMENT.items():
        n = tags.get(tag, 0)
        if n:
            print(f"{label + ':':<{width + 1}} {n}")
            if tag == "switch":
                for kind, k in sorted(switch_kinds.items()):
                    print(f"    {kind.lower():<{width - 3}} {k}")

    branches = sum(tags.get(t, 0) for t in BRANCH_TAGS)
    print()
    print(f"{'branches (edges):':<{width + 1}} {branches}  (lines + tie lines + transformers)")

    if args.all:
        print("\nOther XML tags:")
        for tag, n in sorted(tags.items(), key=lambda kv: -kv[1]):
            if tag not in EQUIPMENT and tag != "network":
                print(f"  {tag + ':':<32} {n}")


if __name__ == "__main__":
    main()
