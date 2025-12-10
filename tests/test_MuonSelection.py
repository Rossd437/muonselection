import pytest
import pandas as pd
import numpy as np
import sys
import h5flow
sys.path.append('../src/')
from MuonSelection import MuonSelection

@pytest.fixture
def my_selection():
        segment_count, track_count, length_cut = 0, 0, 100
        x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
        y_boundaries = np.array([-42-19.8543, -42+103.8543])
        z_boundaries = np.array([-64.3163,  -2.6837, 2.6837, 64.3163]) 

        rock_muon_dtype = np.dtype([
            ('event_id','i4'),
            ('rock_muon_id', 'i4'),
            ('length','f8'),
            ('x_start', 'f8'),
            ('y_start','f8'),
            ('z_start', 'f8'),
            ('x_end','f8'),
            ('y_end', 'f8'),
            ('z_end', 'f8'),
            ('exp_var', 'f8'),
            ('theta_xz','f8'),
            ('theta_yz', 'f8'),
            ('theta_z','f8')
            ])

        segment_dtype =  np.dtype([
            ('rock_segment_id', 'i4'),
            ('x_start', 'f8'),
            ('y_start','f8'),
            ('z_start','f8'),
            ('dE', 'f8'),
            ('x_end', 'f8'),
            ('y_end','f8'),
            ('z_end', 'f8'),
            ('dQ','f8'),
            ('dN', 'i4'),
            ('dx','f8'),
            ('x_mid','f8'),
            ('y_mid','f8'),
            ('z_mid','f8'),
            ('t','f8'),
            ('io_group', 'i4')
        ])

        return MuonSelection(segment_count, track_count, length_cut, rock_muon_dtype, segment_dtype, 
                              x_boundaries, y_boundaries, z_boundaries)

@pytest.fixture
def get_track_df():
        file = '../results/csvs/MiniRun6.5_tracks.csv'
        return pd.read_csv(file)


@pytest.fixture
def get_segment_df():
        file = '../results/csvs/MiniRun6.5_segments.csv'
        return pd.read_csv(file)

@pytest.fixture
def event_hits():
    file = "../data/MiniRun6.5_1E19_RHC.flow.0000433.FLOW.proto_nd_flow.hdf5"
    f = h5flow.data.H5FlowDataManager(file, "r")
    return f['charge/events', 'charge/calib_prompt_hits', 0].data[0]

@pytest.mark.parametrize("track_index", [0, 2, 3, 4])
def test_length(get_track_df, track_index):
        track = get_track_df.iloc[track_index]

        xyz_start = np.array([track['x_start'], track['y_start'], track['z_start']])
        xyz_end = np.array([track['x_end'], track['y_end'], track['z_end']])

        test_length = np.linalg.norm([xyz_end - xyz_start])

        assert track['length'] == pytest.approx(test_length, 1)

@pytest.mark.parametrize("track_index", [0, 2, 3, 4])
def test_angle(my_selection, get_track_df, track_index):
        track = get_track_df.iloc[track_index]

        xyz_start = np.array([track['x_start'], track['y_start'], track['z_start']])
        xyz_end = np.array([track['x_end'], track['y_end'], track['z_end']])

        direction_vector = (xyz_start - xyz_end)/np.linalg.norm([xyz_end - xyz_start])

        assert my_selection.angle(direction_vector) == (pytest.approx(track['theta_xz'], 1), pytest.approx(track['theta_yz'], 1), pytest.approx(track['theta_z'], 1))

@pytest.mark.parametrize("track_index", [0, 2, 3, 4])
def test_close_to_two_faces(my_selection, get_track_df, track_index):
        track = get_track_df.iloc[track_index]
 
        xyz_start = (track['x_start'], track['y_start'], track['z_start'])
        xyz_end = (track['x_end'], track['y_end'], track['z_end'])
       
        mock_hits = np.array([xyz_start, xyz_end], dtype=np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8')]))

        min_boundaries = np.array([my_selection.x_boundaries.min(), my_selection.y_boundaries.min(), my_selection.z_boundaries.min()])
        max_boundaries = np.array([my_selection.x_boundaries.max(), my_selection.y_boundaries.max(), my_selection.z_boundaries.max()])

        faces_of_detector = np.concatenate((min_boundaries,max_boundaries))

        assert my_selection.close_to_two_faces(faces_of_detector, mock_hits)

@pytest.mark.parametrize("track_index", [0, 2, 3, 4])
def test_select_muon_track(my_selection, get_track_df, track_index):
        track = get_track_df.iloc[track_index]

        assert (track['length'] >= my_selection.length_cut) & (track['avg_distance'] <= 1.5)

@pytest.mark.parametrize(
                "io_groups, expected_value", 
                [
                        (np.array([1, 3, 6, 1], dtype=np.dtype([('io_group', 'i4')])), 3),
                        (np.array([1, 2, 3, 4], dtype=np.dtype([('io_group', 'i4')])), 4)
                ]
)
def test_TPC_separation(my_selection, io_groups, expected_value):
        assert len(my_selection.TPC_separation(io_groups)) == expected_value

@pytest.mark.parametrize(
                "positions, hits_mean, track_direction, expected_distance",
                [(
                        np.array([(1, 1, 1), (3, 3, 3)]),
                        np.array([2, 2, 2]),
                        np.array([2, 2, 2])/np.sqrt(12),
                        0
                )]
)
def test_average_distance(my_selection, positions, hits_mean, track_direction, expected_distance):
        assert my_selection.average_distance(positions, hits_mean, track_direction) == pytest.approx(expected_distance, .1)

@pytest.mark.parametrize(
                "positions, hits_mean, track_direction, expected_hits",
                [(
                        np.array([(1, 1, 1), (3, 3, 3), (43, 53, 52)]),
                        np.array([2, 2, 2]),
                        np.array([2, 2, 2])/np.sqrt(12),
                        2
                )]
)
def test_clean_noise_hits(my_selection, positions, track_direction, hits_mean, expected_hits):
        assert sum(my_selection.clean_noise_hits(positions, track_direction, hits_mean)) == expected_hits


@pytest.mark.parametrize(
                "hit_positions, expected_variance",
                [(
                        np.array([(1, 1, 1), (3, 3, 3)]),
                        1.0
                )]
)
def test_PCAs(my_selection, hit_positions, expected_variance):
    results = my_selection.PCAs(hit_positions)
    assert results[0]  == pytest.approx(expected_variance, .05)

def test_cluster(my_selection, event_hits):
       assert len(my_selection.cluster(event_hits, 2)) == 3

def test_segments(my_selection, get_segment_df):
       half_segments_index = int(get_segment_df.size/2)

       half_segment_df = get_segment_df.iloc[:half_segments_index]

       assert np.all(half_segment_df['z_start'] >= min(my_selection.z_boundaries))
       
def test_grab_segment_info(get_segment_df):
       assert (np.all(get_segment_df['t'] >= 0)) & (np.all(get_segment_df['t'] <= 2000))
