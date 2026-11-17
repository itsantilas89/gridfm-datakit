import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS
from gridfm_datakit.utils.idx_bus import BASE_KV, VMIN
from gridfm_datakit.utils.idx_gen import PG, PMAX

from .api import check_powsybl_available


@dataclass
class MappingP2G:
    """Index maps from pypowsybl element IDs to gridfm row indices.

    Attributes
    ----------
    bus : Dict[str, float]
        ``{pp_bus_id: gfm_bus_index}``
    branch : Dict[str, int]
        ``{pp_branch_id: gfm_branch_row}``
    gen : Dict[str, int]
        ``{pp_gen_id: gfm_gen_row}``
    reverse_bus_index_mapping : Dict[int, int]
        ``{gfm_bus_index: original_bus_number}``. Power-flow bus results are
        keyed by that original number, the same way PowerModels keys them.
    """

    bus: Dict[str, float]
    branch: Dict[str, int]
    gen: Dict[str, int]
    reverse_bus_index_mapping: Dict[int, int]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_p2g_maps(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm index maps for a network read by ``from_powsybl``.

    :func:`~gridfm_datakit.powsybl.convert.from_powsybl` takes its tables from
    pypowsybl's MATPOWER export.  Buses and branches (lines, then 2-winding
    transformers) come out in the order pypowsybl lists them, so each of those
    maps pairs the i-th pypowsybl element with gridfm row i.  Generators do
    **not**: the export orders them by bus (and in its own order within a
    bus), so their map is read from the export itself, see
    :func:`_generator_rows_from_export`.  A network built by
    :func:`~gridfm_datakit.powsybl.convert.to_powsybl` follows the gridfm
    tables instead, so use the maps ``to_powsybl`` returns.

    The maps can be built once on the **base network** and then reused across
    all perturbed scenarios, because perturbations preserve element identity
    and row ordering.

    Parameters
    ----------
    network:
        The gridfm_datakit Network returned by ``from_powsybl(pp_net)``.
    pp_net:
        The pypowsybl network passed to ``from_powsybl``.

    Returns
    -------
    MappingP2G
        Dataclass bundling the three maps: ``bus``, ``branch``, and ``gen``.

    Examples
    --------
    >>> import pypowsybl as pp
    >>> from gridfm_datakit.powsybl.convert import from_powsybl
    >>> from gridfm_datakit.powsybl.mapping import build_p2g_maps
    >>>
    >>> pp_net = pp.network.create_ieee14()
    >>> net = from_powsybl(pp_net)
    >>> mapping = build_p2g_maps(net, pp_net)
    >>> mapping.bus    # pp_bus_id → gfm index
    >>> mapping.branch # pp_branch_id → gfm row
    >>> mapping.gen    # pp_gen_id → gfm row
    """
    return _build_maps(network, pp_net, from_export=True)


def _build_maps(network: Network, pp_net, from_export: bool) -> MappingP2G:
    """Bus, branch and generator maps; see :func:`build_p2g_maps`.

    ``from_export`` selects how buses and generators are paired: by reading the
    export (networks built by ``from_powsybl``) or by position (networks built
    by ``to_powsybl``, which follow the gridfm tables).
    """
    check_powsybl_available()

    # -------------------------------------------------------------------------
    # 0. Bus map - read from the export (from_powsybl) or by position (to_powsybl)
    # -------------------------------------------------------------------------
    if from_export:
        map_bus_p2g = _bus_rows_from_export(network, pp_net)
    else:
        map_bus_p2g: Dict[str, float] = {
            pp_bus_id: gfm_row for gfm_row, pp_bus_id in enumerate(pp_net.get_buses().index)
        }

    # -------------------------------------------------------------------------
    # 1. Gen map - read from the export (from_powsybl) or by position (to_powsybl)
    # -------------------------------------------------------------------------
    if from_export:
        map_gen_p2g = _generator_rows_from_export(network, pp_net)
    else:
        map_gen_p2g = {
            pp_gen_id: gfm_row
            for gfm_row, pp_gen_id in enumerate(pp_net.get_generators().index)
        }

    # -------------------------------------------------------------------------
    # 2. Branch map — direct enumeration (lines, transformers, ...)
    # -------------------------------------------------------------------------
    offset = 0
    map_branch_p2g: Dict[str, int] = {}
    for df in (pp_net.get_lines(), pp_net.get_2_windings_transformers()):
        for row, pp_branch_id in enumerate(df.index):
            map_branch_p2g[pp_branch_id] = offset + row
        offset += len(df)

    return MappingP2G(
        bus=map_bus_p2g,
        branch=map_branch_p2g,
        gen=map_gen_p2g,
        reverse_bus_index_mapping={
            int(row): int(orig)
            for row, orig in network.reverse_bus_index_mapping.items()
        },
    )


def build_p2g_maps_by_bus_number(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm index maps for a network built by ``to_powsybl``.

    ``to_powsybl`` names each bus-breaker bus ``BUS-<n>`` after its original
    bus number.  pypowsybl lists buses by voltage level, and buses joined by a
    transformer share one, so position does not give the gridfm row.  Branches
    are matched by original from and to buses, parallel branches in table
    order, because pypowsybl lists lines before transformers.

    Args:
        network: The gridfm_datakit Network passed to ``to_powsybl``.
        pp_net: The pypowsybl network ``to_powsybl(network)`` built.

    Returns:
        MappingP2G with the bus and branch maps read from the original bus
        numbers.

    Raises:
        ValueError: If a pypowsybl bus groups several gridfm buses or has none,
            or a pypowsybl branch has no gridfm branch with the same ends.
    """
    maps = _build_maps(network, pp_net, from_export=False)
    maps.bus = {}
    bb_buses = pp_net.get_bus_breaker_view_buses(attributes=["bus_id"])
    for bb_bus_id, pp_bus_id in bb_buses["bus_id"].items():
        if not pp_bus_id:
            continue
        gfm_bus = network.bus_index_mapping[_bus_number(bb_bus_id)]
        if maps.bus.setdefault(pp_bus_id, gfm_bus) != gfm_bus:
            raise ValueError(f"pypowsybl bus {pp_bus_id} groups several gridfm buses")
    unmapped = set(pp_net.get_buses().index) - set(maps.bus)
    if unmapped:
        raise ValueError(f"No gridfm bus for pypowsybl buses {sorted(unmapped)}")

    rev = network.reverse_bus_index_mapping
    rows_by_ends: Dict[Tuple[int, int], List[int]] = {}
    for row, (f_bus, t_bus) in enumerate(network.branches[:, [F_BUS, T_BUS]]):
        rows_by_ends.setdefault((rev[int(f_bus)], rev[int(t_bus)]), []).append(row)
    ends = ["bus_breaker_bus1_id", "bus_breaker_bus2_id"]
    maps.branch = {}
    for df in (
        pp_net.get_lines(attributes=ends),
        pp_net.get_2_windings_transformers(attributes=ends),
    ):
        for pp_branch_id, bus1, bus2 in df.itertuples():
            key = (_bus_number(bus1), _bus_number(bus2))
            rows = rows_by_ends.get(key)
            if not rows:
                raise ValueError(
                    f"No gridfm branch from bus {key[0]} to bus {key[1]} "
                    f"for pypowsybl branch {pp_branch_id}",
                )
            maps.branch[pp_branch_id] = rows.pop(0)
    return maps


def _export_tagged(pp_net, tag_fn):
    """Export ``pp_net`` to MATPOWER after ``tag_fn(pp_net)`` has marked its elements; return the ``mpc`` struct.

    The marks are made in a temporary variant, so the caller's network is left unchanged.
    """
    work = pp_net.get_working_variant_id()
    tmp = "_row_probe"
    pp_net.clone_variant(work, tmp, True)
    try:
        pp_net.set_working_variant(tmp)
        tag_fn(pp_net)
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            pp_net.save(path, format="MATPOWER")
            return scipy.io.loadmat(path, struct_as_record=True, squeeze_me=False)["mpc"]
        finally:
            os.unlink(path)
    finally:
        pp_net.set_working_variant(work)
        pp_net.remove_variant(tmp)


def _bus_rows_from_export(network: Network, pp_net) -> Dict[str, float]:
    """Map every pypowsybl bus id to its row in the exported gridfm ``bus`` table.

    The export does not list buses in the order of ``pp_net.get_buses()``: with
    an order-based map about one bus in eight lands on another bus's row (wrong
    load, shunt, base voltage).  Every voltage level temporarily gets a unique low
    voltage limit (restored afterwards); the export writes it into ``VMIN``, so
    the voltage level of each row can be read back.  The buses of one voltage level
    are paired with its rows in table order.  Rows the export adds after the
    pypowsybl buses (e.g. for three-winding transformers) carry no tag and have
    no pypowsybl bus.
    """
    levels = list(pp_net.get_voltage_levels(attributes=[]).index)
    tag = 1.0 + np.arange(len(levels))  # kV

    def mark(net):
        net.update_voltage_levels(id=levels, high_voltage_limit=tag + 1e4)
        net.update_voltage_levels(id=levels, low_voltage_limit=tag)

    # voltage limits are not variant data: the marks change the network itself, so put the limits back
    limits = pp_net.get_voltage_levels(attributes=["low_voltage_limit", "high_voltage_limit"])
    try:
        bus = _export_tagged(pp_net, mark)["bus"][0, 0]
    finally:
        pp_net.update_voltage_levels(id=levels, low_voltage_limit=limits.low_voltage_limit.to_numpy())
        pp_net.update_voltage_levels(id=levels, high_voltage_limit=limits.high_voltage_limit.to_numpy())
    if bus.shape[0] != network.buses.shape[0]:
        raise ValueError(f"the export has {bus.shape[0]} bus rows but the network has {network.buses.shape[0]}")
    level_of_row = np.rint(bus[:, VMIN] * bus[:, BASE_KV]).astype(int) - 1  # -1: no tag

    buses = pp_net.get_buses(attributes=["voltage_level_id"])
    n_real = len(buses)
    # a real bus whose VMIN the export zeroes has no tag either: it takes the rows its level is short of
    rows_of_level: Dict[str, List[int]] = {}
    for row in range(n_real):
        if level_of_row[row] >= 0:
            rows_of_level.setdefault(levels[level_of_row[row]], []).append(row)
    untagged = [row for row in range(n_real) if level_of_row[row] < 0]
    wanted = buses.groupby("voltage_level_id").size()
    for level, count in wanted.items():
        missing = count - len(rows_of_level.get(level, []))
        for _ in range(max(missing, 0)):
            if not untagged:
                raise ValueError("could not pair every pypowsybl bus with a row of the export")
            rows_of_level.setdefault(level, []).append(untagged.pop(0))
    taken: Dict[str, int] = {}
    mapping: Dict[str, float] = {}
    for bus_id, level in zip(buses.index, buses.voltage_level_id):
        k = taken.get(level, 0)
        taken[level] = k + 1
        mapping[bus_id] = sorted(rows_of_level[level])[k]

    rows = np.array(list(mapping.values()))
    wrong = ~np.isclose(network.buses[rows, BASE_KV], pp_net.get_voltage_levels(attributes=["nominal_v"]).nominal_v[buses.voltage_level_id].to_numpy())
    if wrong.any():
        warnings.warn(f"{int(wrong.sum())} buses were paired with a row of another base voltage", stacklevel=3)
    return mapping


def _generator_rows_from_export(network: Network, pp_net) -> Dict[str, int]:
    """Map every pypowsybl generator id to its row in the exported gridfm ``gen`` table.

    pypowsybl's MATPOWER export lists generators sorted by bus, not in the order
    of ``pp_net.get_generators()``, so pairing the i-th generator with row i
    puts setpoints and limits on the wrong units.  Instead, every generator gets
    a unique ``target_p`` in a temporary variant, the variant is exported, and
    each generator is found at the row whose ``PG`` carries its tag.  The working
    variant, and so the caller's network, is left unchanged.

    The export cannot carry a tag for a generator with ``max_p == 0`` (its
    ``PG`` is clamped to 0).  Those units, which cannot produce anything, are
    paired with the rows left over, in table order (left unmapped, with a
    warning, if there are more of them than free rows).  Generators that are
    disconnected are not exported, so they have no row and are not in the map.
    """
    ids = list(pp_net.get_generators(attributes=[]).index)
    tag = 1000.0 + np.arange(len(ids))

    def mark(net):
        net.update_generators(id=ids, target_p=tag)
        # batteries are exported as generator rows too: tag them so they are not mistaken for a generator
        bat = list(net.get_batteries(attributes=[]).index)
        if bat:
            net.update_batteries(id=bat, target_p=1000.0 + len(ids) + np.arange(len(bat)))

    exported = _export_tagged(pp_net, mark)["gen"][0, 0]

    if exported.shape[0] != network.gens.shape[0]:
        raise ValueError(
            f"the export has {exported.shape[0]} generator rows but the network has {network.gens.shape[0]}",
        )
    row_of_tag = {int(round(pg)): row for row, pg in enumerate(exported[:, PG]) if pg >= 999.5}
    rows = {gid: row_of_tag[1000 + k] for k, gid in enumerate(ids) if 1000 + k in row_of_tag}

    # rows of batteries and of any other element the export writes as a generator are not generators
    tagged = {r for t, r in row_of_tag.items()}
    # a disconnected generator is not exported at all and has no row: leave it out of the map
    connected = pp_net.get_generators(attributes=["connected"]).connected
    left_ids = [gid for gid in ids if gid not in rows and connected[gid]]
    left_rows = sorted(set(range(exported.shape[0])) - tagged)
    if len(left_ids) <= len(left_rows):
        rows.update(zip(left_ids, left_rows))
    else:  # more untagged generators than free rows: pairing them would be a guess
        warnings.warn(
            f"{len(left_ids)} generators that the export cannot tag have only {len(left_rows)} free rows; left unmapped",
            stacklevel=3,
        )

    # sanity check: the paired row must hold the generator's own limit
    mapped = [gid for gid in ids if gid in rows]
    max_p = pp_net.get_generators(attributes=["max_p"]).max_p[mapped].to_numpy()
    paired = np.array([rows[gid] for gid in mapped])
    wrong = ~np.isclose(network.gens[paired, PMAX], max_p, rtol=1e-5, atol=1e-6)
    if wrong.any():
        warnings.warn(
            f"{int(wrong.sum())} generators were paired with a row whose PMAX differs from their max_p",
            stacklevel=3,
        )
    return rows


def _bus_number(bus_breaker_bus_id: str) -> int:
    """Return the original bus number of a ``BUS-<n>`` bus-breaker bus ID."""
    return int(bus_breaker_bus_id.rsplit("-", 1)[1])

