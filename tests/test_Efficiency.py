import pytest
import sys
import numpy as np
import h5flow

sys.path.append("../src/")
from Efficiency import is_point_outside, check_intersection, count_true_muons


x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
y_boundaries = np.array([-42 - 19.8543, -42 + 103.8543])
z_boundaries = np.array([-64.3163, -2.6837, 2.6837, 64.3163])

min_bounds = [min(x_boundaries), min(y_boundaries), min(z_boundaries)]
max_bounds = [max(x_boundaries), max(y_boundaries), max(z_boundaries)]

traj_test = np.array(
    [(-63.931, 63.931), (-42 - 19.8543, -42 + 103.8543), (-64.3163, 64.3163)],
    dtype=[("xyz_start", "f4"), ("xyz_end", "f4")],
)


@pytest.fixture
def grab_h5flow_data():
    file = "../data/MiniRun6.5_1E19_RHC.flow.0000433.FLOW.proto_nd_flow.hdf5"
    return h5flow.data.H5FlowDataManager(file, "r")


@pytest.mark.parametrize(
    "point, test_bool",
    [
        ([0, 0, 0], False),
        ([-70, 29, 0], True),
        ([5, 6, 8], False),
        ([-80, 10, -1000], True),
    ],
)
def test_is_point_outside(point, test_bool):
    assert (
        is_point_outside(point, x_boundaries, y_boundaries, z_boundaries) == test_bool
    )


@pytest.mark.parametrize(
    "traj, min_bounds, max_bounds", [(traj_test, min_bounds, max_bounds)]
)
def test_check_intersection(traj, min_bounds, max_bounds):
    assert check_intersection(traj, min_bounds, max_bounds)


def test_count_true_muons(grab_h5flow_data):
    assert count_true_muons(grab_h5flow_data) > 0
