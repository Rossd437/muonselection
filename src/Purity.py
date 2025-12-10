"""This Python module contains function to help calculate purity."""

import numpy as np
from h5flow.data import dereference
import h5flow
import matplotlib.pyplot as plt
import random
import Efficiency


class Purity:
    """Purity of the muon selection.

    This class will have the functionality to produce the purity of the muon selection and make the purity plot.

    Attributes:
        f: H5FlowDataManager for hdf5 file.
        wanted_sim: Simulation version
    """

    def __init__(
        self,
        f: h5flow.data.h5flow_data_manager.H5FlowDataManager,
        wanted_sim: str,
        purity_file: str,
    ):
        """Initializes a new Purity Instance.

        Args:
            f: H5FlowDataManager for hdf5 file.
            wanted_sim: Simulation version.
            purity_file: File to save purity plot.
        """
        self.f = f
        self.wanted_sim = wanted_sim
        self.purity_file = purity_file

    def grab_bt(self, hits: np.ndarray, bt_info: np.ndarray) -> np.ndarray:
        """Return backtrack info for selected track.

        Args:
            hits: Hits of muon track.
            bt_info: Backtrack information of prompt hits.

        Returns:
            Backtrack information for hits of muon track.
        """
        hits_bt = bt_info[hits["id"]]

        return hits_bt

    def get_traj_makeup(
        self,
        segment_map: dict,
        traj_map: dict,
        bt_info: np.ndarray,
        seg_dtype: np.dtypes.VoidDType,
    ) -> dict:
        """Get true trajectory makeup.

        Get all of the true trajectory pdg codes and the percent makeup for each pdg code that \
        makes up the selected track.

        Args:
            segment_map: Dictionary map of the true segments.
            traj_map: Dictionary map of the true trajectories.
            bt_info: Backtrack information for prompt hits.
            seg_dtype:dtype of the true segments.

        Returns:
            Dictionary of the true trajectory makeup of the selected track.
        """
        track_makeup = {}
        trajs_of_track = []

        # Plot all of the backtracked segment positions
        for hit in bt_info:
            for cont in range(len(hit["fraction"])):
                if hit["fraction"][cont] > 0.0001:
                    seg_id = hit["segment_ids"][cont]
                    seg = segment_map.get(seg_id)

                    # Append trajectory information to the list
                    trajs_of_track.append(
                        [
                            seg["file_traj_id"],  # File trajectory ID
                            seg["n_electrons"],  # Number of electrons
                            hit["fraction"][cont],  # Fraction associated with the hit
                            seg_id,
                        ]
                    )

        traj_arr = np.array(trajs_of_track)

        unique_trajs = np.unique(traj_arr[:, 0])

        for i in range(len(unique_trajs)):
            traj = unique_trajs[i]
            mask = traj_arr[:, 0] == traj
            trajss = traj_arr[mask]

            # Get makeup of track
            wanted_traj = traj_map.get(traj)

            wanted_segments = np.array(
                [segment_map.get(seg_id) for seg_id in trajss[:, -1]], dtype=seg_dtype
            )

            pdg_of_traj = wanted_traj["pdg_id"]
            E_of_traj = sum(wanted_segments["dE"])

            if pdg_of_traj not in track_makeup.keys():
                track_makeup[f"{pdg_of_traj}"] = E_of_traj
            else:
                track_makeup[f"{pdg_of_traj}"] = (
                    track_makeup[f"{pdg_of_traj}"] + E_of_traj
                )

        return track_makeup

    def get_max_pdg(self, makeup: dict) -> int:
        """Get true pdg code of selected track.

        Args:
            makeup: true trajectory makeup

        Returns:
            True pdg code of selected track
        """
        return max(makeup, key=makeup.get)

    def purity_measurement(self, pdg_makeup: dict) -> float:
        """Get purity measurement.

        Get the purity measurement of the selected file(s).

        Args:
            pdg_makeup: true pdgs for each selected track

        Returns:
            The purity of the pdg makeup
        """
        total_amount_tracks = sum(pdg_makeup.values())

        amount_of_muons = pdg_makeup.get("13", 0) + pdg_makeup.get("-13", 0)

        return round(amount_of_muons / total_amount_tracks, 3) * 100

    def generate_random_colors(self, n_colors: int) -> list:
        """Generate random colors.

        Args:
            n_colors: Number of colors wanted.

        Returns:
            N random colors.
        """
        random_colors = [
            (random.random(), random.random(), random.random(), 0.9)
            for n in range(n_colors)
        ]
        return random_colors

    def make_purity_plot(
        self,
        sorted_pdg_makeup: dict,
        wanted_sim: str,
        purity: float,
        eff: float,
        output_file: str,
    ) -> int:
        """Make Purity Plot.

        Take in the pdg_makeup of the selection and produce a purity plots.

        Args:
            sorted_pdg_makeup: Sorted dictionary of the true pdg makeup (by length).
            wanted_sim: Simulation the selection ran on.
            purity: Purity of the selection
            eff: Efficiency of the selection.
            output_file: File to save purity plot

        Returns:
            Saved image of purity
        """

        n_colors = len(sorted_pdg_makeup)

        colors = self.generate_random_colors(n_colors)

        fig, ax = plt.subplots()

        # Set black spines
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        x = list(sorted_pdg_makeup.keys())
        y = list(sorted_pdg_makeup.values())
        labels = [f"{x_value}: {y_value}" for x_value, y_value in zip(x, y)]

        ax.bar(x, y, color=colors, edgecolor="k", linewidth=2, label=labels)

        text_str = f"Purity: {purity} \nEfficiency: {eff}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.3,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_ylabel("Count", fontsize=14)
        ax.set_title(f"{wanted_sim} Purity")

        ax.grid(axis="y")
        ax.tick_params(axis="x", rotation=45)
        ax.set_axisbelow(True)

        leg = ax.legend(facecolor="wheat", edgecolor="grey")
        leg.get_frame().set_alpha(0.5)
        fig.savefig(output_file)
        return

    def produce_purity_and_plot(self, hits_list: np.ndarray) -> int:
        """Extract purity.

        This function will make the purity plot and measure the purity and efficiency of selection.

        Args:
            hits_list: Nested list of hits for every selected track

        Returns:
            Plot of the purity and efficiency.
        """

        type_of_particles = {
            1: "d",
            2: "u",
            3: "s",
            4: "c",
            5: "b",
            6: "t",
            11: "$e^-$",
            -11: "$e^+$",
            12: "$\\nu_e$",
            13: "$\\mu^-$",
            -13: "$\\mu^+$",
            14: "$\\nu_\\mu$",
            15: "$\\tau^-$",
            16: "$\\nu_\\tau$",
            17: "$\\tau'^-$",
            18: "$\\nu_\\tau'$",
            21: "$g$",
            211: "$\\pi^{+}$",
            -211: "$\\pi^{-}$",
            111: "$\\pi^0$",
            2212: "$p$",
            2112: "$n$",
            22: "$\\gamma$",
            321: "$K^+$",
            -321: "$K^-$",
            311: "$K^0$",
            -311: "$\\bar{K}^0$",
        }

        calib_prompt_hits_bt = self.f["mc_truth/calib_prompt_hit_backtrack/data"]
        trajs = self.f["mc_truth/trajectories/data"]
        segments = self.f["mc_truth/segments/data"]

        traj_map = {traj["file_traj_id"]: traj for traj in trajs}
        seg_map = {seg["segment_id"]: seg for seg in segments}

        bts = [self.grab_bt(hit_array, calib_prompt_hits_bt) for hit_array in hits_list]

        # Get trajectory makeup
        trajectory_makeup = [
            self.get_traj_makeup(seg_map, traj_map, bt, segments.dtype) for bt in bts
        ]

        pdgs = [self.get_max_pdg(make) for make in trajectory_makeup]
        particles, counts = np.unique(pdgs, return_counts=True)

        pdg_makeup = {}

        for p, c in zip(particles, counts):
            pdg_makeup[p] = c

        purity = self.purity_measurement(pdg_makeup)

        amount_of_selected_muons = pdg_makeup.get("13", 0) + pdg_makeup.get("-13", 0)
        true_muon_count = Efficiency.count_true_muons(self.f)
        eff = round((amount_of_selected_muons / true_muon_count) * 100, 3)

        print(f"Efficiency = {eff}")
        print(f"Purity = {purity}")
        print("Making Purity plot")

        pdg_makeup = {
            type_of_particles.get(int(key), key): value
            for key, value in pdg_makeup.items()
        }

        integer_makeup = {
            key: value for key, value in pdg_makeup.items() if key.isdigit()
        }

        string_makeup = {
            key: value for key, value in pdg_makeup.items() if type(key) == str
        }

        string_makeup["else"] = sum(integer_makeup.values())

        pdg_makeup = string_makeup

        sorted_pdg_makeup = dict(
            sorted(pdg_makeup.items(), key=lambda value: value[1], reverse=True)
        )

        self.make_purity_plot(
            sorted_pdg_makeup,
            self.wanted_sim,
            purity,
            eff,
            output_file=self.purity_file,
        )

        return
