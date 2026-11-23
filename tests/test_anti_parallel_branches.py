"""Flows of anti-parallel branches must be reported at the case file's from/to bus.

PowerModels.parse_file reverses one branch of every pair of opposite-direction
branches (same two buses). The solution then reports that branch's flows at the
swapped ends, so ``_solution_arrays`` has to swap them back.
"""

import os

import numpy as np
import pandas as pd
import pytest
from matpowercaseframes import CaseFrames

from gridfm_datakit.network import Network
from gridfm_datakit.process.process_network import _solution_arrays, init_julia
from gridfm_datakit.process.solvers import run_dcopf, run_opf
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IEEE14_PATH = os.path.join(REPO_ROOT, "tests", "powsybl", "grids", "ieee14.m")

# anti-parallel twin of the 1 -> 2 branch (the one carrying the most power), as 2 -> 1
TWIN = "\t2\t1\t0.02\t0.06\t0.0\t472.0\t472.0\t472.0\t0\t0\t1\t-30\t30\n"


@pytest.fixture(scope="module")
def jl():
    return init_julia(max_iter=500, solver_log_dir=None)


@pytest.fixture()
def net(tmp_path):
    text = open(IEEE14_PATH).read()
    head, tail = text.split("mpc.branch = [\n")
    rows, rest = tail.split("];", 1)
    path = tmp_path / "ieee14_anti_parallel.m"
    path.write_text(head + "mpc.branch = [\n" + rows + TWIN + "];" + rest)
    # Not load_net_from_file: it runs correct_network, which already reverses
    # the twin in the file. Readers such as powsybl build the Network from the
    # export as it is, which is the case under test.
    frames = CaseFrames(str(path))
    return Network(
        {
            k: getattr(frames, k).values
            if isinstance(getattr(frames, k), pd.DataFrame)
            else getattr(frames, k)
            for k in frames._attributes
        },
    )


def _pair(net):
    """Row numbers of the 1 -> 2 branch and of its 2 -> 1 twin."""
    f = net.branches[:, F_BUS].astype(int)
    t = net.branches[:, T_BUS].astype(int)
    a, b = 0, 1  # buses 1 and 2 of the file (listed first, so ids 0 and 1)
    fwd = np.where((f == a) & (t == b))[0]
    rev = np.where((f == b) & (t == a))[0]
    assert len(fwd) == 1 and len(rev) == 1
    return fwd[0], rev[0]


@pytest.mark.parametrize("solve", [run_opf, run_dcopf])
def test_antiparallel_flows_at_file_orientation(net, jl, solve):
    res = solve(net, jl)
    flows, _, _ = _solution_arrays(res, net)
    fwd, rev = _pair(net)
    ids = list(net.idx_branches_in_service)
    pf_fwd, pt_fwd = flows[ids.index(fwd), [0, 2]]
    pf_rev, pt_rev = flows[ids.index(rev), [0, 2]]
    # both branches join the same buses, so they carry power the same way:
    # what one sends from bus 1 the other receives at bus 1
    assert abs(pf_fwd) > 0.1  # per unit; the branch is loaded
    assert pf_fwd * pt_rev > 0
    assert pt_fwd * pf_rev > 0
    # and every branch has its losses on the right side: sent >= received
    assert np.all(flows[:, 0] + flows[:, 2] > -1e-6)
