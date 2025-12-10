import pytest
import numpy as np
import sys

sys.path.append("../src/")
from Purity import Purity
import h5flow
import os
from pathlib import Path
import tempfile

hits_test = np.array(
    [np.arange(1, 10), np.arange(21, 30)], dtype=np.dtype([("id", "i4")])
)


@pytest.fixture
def my_purity():
    file = "../data/MiniRun6.5_1E19_RHC.flow.0000433.FLOW.proto_nd_flow.hdf5"
    f = h5flow.data.H5FlowDataManager(file, "r")

    wanted_sim = "MiniRun6.5"

    purity_file = "purity.png"
    p = Purity(f, wanted_sim, purity_file)
    return p


@pytest.fixture
def data_traj_makeup():
    file = "../data/MiniRun6.5_1E19_RHC.flow.0000433.FLOW.proto_nd_flow.hdf5"
    return h5flow.data.H5FlowDataManager(file, "r")


@pytest.mark.parametrize("indices", [(np.arange(0, 50)), (np.arange(75, 100))])
def test_grab_bt(my_purity, indices):
    hits = my_purity.f["charge/calib_prompt_hits/data"][indices]
    assert np.all(hits["id"] == indices)


@pytest.mark.parametrize(
    "pdg_dict, expected_pdg",
    [({"13": 0.9, "11": 0.1}, "13"), ({"13": 0.3, "-14": 0.6, "2": 0.1}, "-14")],
)
def test_get_max_pdg(my_purity, pdg_dict, expected_pdg):
    assert my_purity.get_max_pdg(pdg_dict) == expected_pdg


@pytest.mark.parametrize(
    "pdg_dict, expected_value",
    [({"12": 30, "13": 500, "-13": 20, "14": 1}, 94.3), ({"12": 30, "14": 1}, 0.0)],
)
def test_purity_measurement(my_purity, pdg_dict, expected_value):
    assert my_purity.purity_measurement(pdg_dict) == pytest.approx(expected_value, 0.1)


@pytest.mark.parametrize("n_colors, expected_n_colors", [(3, 3), (7, 7), (10, 10)])
def test_generate_random_colors(my_purity, n_colors, expected_n_colors):
    assert len(my_purity.generate_random_colors(n_colors)) == expected_n_colors


def test_make_purity_plot(my_purity):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, my_purity.purity_file)
        my_purity.make_purity_plot(
            {"13": 10, "32": 1}, my_purity.wanted_sim, 94.4, 26.3, tmp_path
        )
        assert os.path.exists(tmp_path)


@pytest.mark.parametrize("hits", [(hits_test)])
def test_produce_purity_and_plot(my_purity, hits):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, my_purity.purity_file)
        my_purity.purity_file = tmp_path
        print(f"{my_purity.purity_file=}")
        my_purity.produce_purity_and_plot(hits)
        assert os.path.exists(tmp_path)
