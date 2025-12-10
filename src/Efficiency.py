import numpy as np
import h5flow


def is_point_outside(
    point: np.ndarray,
    x_boundaries: np.ndarray,
    y_boundaries: np.ndarray,
    z_boundaries: np.ndarray,
) -> bool:
    """Check if point is outside detector.

    This function will check to see if the end points of the trajectory is outside of the detector.

    Args:
        point: End or start point of trajectory.
        x_boundaries: X boundaries of detector.
        y_boundaries: Y boundaries of detector.
        z_boundaries: Z boundaries of detector.

    Returns:
        True if point outside of detector and false otherwise.
    """
    x, y, z = point[0], point[1], point[2]

    xmin, xmax, ymin, ymax, zmin, zmax = (
        x_boundaries.min(),
        x_boundaries.max(),
        y_boundaries.min(),
        y_boundaries.max(),
        z_boundaries.min(),
        z_boundaries.max(),
    )

    return x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax


def check_intersection(
    traj: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray
) -> bool:
    """Check for intersection.

    This function will check if the trajectory enters and exits the detector.

    Args:
        traj: True trajectory info.
        min_bounds: Minimum x, y, and z bounds.
        max_bounds: Maximum x, y, and z bounds.

    Returns:
        True if trajectory enters and exits and false otherwise.

    """
    direction = (traj["xyz_start"] - traj["xyz_end"]) / np.linalg.norm(
        traj["xyz_start"] - traj["xyz_end"]
    )

    origin = (traj["xyz_start"] + traj["xyz_end"]) / 2

    tmin, tmax = -np.inf, np.inf

    for i in range(3):
        if direction[i] == 0:
            direction[i] = 1e-15

        # Get points where the trajectory intersects the detector for an axes
        t1 = (min_bounds[i] - origin[i]) / direction[i]
        t2 = (max_bounds[i] - origin[i]) / direction[i]

        # Since we dont know which one is entry or exit based on whether the direction is negative or postive, take min() and max() for entry and exit
        t1, t2 = min(t1, t2), max(t1, t2)

        # Get the min/max of the tmax,t2 and tmin,t1 to see if the current trajectory is still in the box for this intervale
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)

    if tmax < tmin:
        return False
    else:
        return True  # Intersect


def count_true_muons(f: h5flow.data.h5flow_data_manager.H5FlowDataManager) -> int:
    """Get amount of true muons.

    This function will determine the amount of true through-going muons in the file.

    Args:
        f: hdf5 file data.

    Returns:
        Amount of true through-going muons in file.
    """
    # Detector boundaries
    x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
    y_boundaries = np.array([-42 - 19.8543, -42 + 103.8543])
    z_boundaries = np.array([-64.3163, -2.6837, 2.6837, 64.3163])

    min_bounds = [min(x_boundaries), min(y_boundaries), min(z_boundaries)]
    max_bounds = [max(x_boundaries), max(y_boundaries), max(z_boundaries)]

    counts_of_true_rock_muons = 0

    interactions = f["mc_truth/interactions/data"]

    trajs = f["mc_truth/trajectories/data"]

    muon_keys = [13, -13]
    mask = np.isin(trajs["pdg_id"], muon_keys)

    muon_trajs = trajs[mask]

    for muon_traj in muon_trajs:
        index_vertex = np.where(
            f["mc_truth/interactions/data"]["vertex_id"] == muon_traj["vertex_id"]
        )[0]
        try:
            vertex = interactions["vertex"][index_vertex][0][:3]
        except:
            vert_x = interactions[index_vertex]["x_vert"]
            vert_y = interactions[index_vertex]["y_vert"]
            vert_z = interactions[index_vertex]["z_vert"]

            vertex = np.array([vert_x, vert_y, vert_z])

        if is_point_outside(vertex, x_boundaries, y_boundaries, z_boundaries):
            intersect_check = check_intersection(muon_traj, min_bounds, max_bounds)
            if intersect_check:
                counts_of_true_rock_muons += 1

    return counts_of_true_rock_muons
