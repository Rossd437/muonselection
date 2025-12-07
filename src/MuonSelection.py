import numpy as np
import pandas as pd
import h5flow
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings
from scipy.spatial.distance import cdist
import sys
import Purity

class MuonSelection:
    """ Muon Selection class.

    This class will provide the functionality to select muon tracks for a given detector geometry.

    Attributes:
        segment_count: Amount of segments
        track_count: Amount of tracks
        length_cut: Minimum length for cluster to be considered a muon
        rock_muon_dtype: dtype of muon array
        segment_dtype: dtype of segment array
        x_boundaries: X boundaries of detector
        y_boundaries: Y boundaries of detector
        z_boundaries: Z boundaries of detector
    """
    def __init__(self, segment_count:int, track_count:int, length_cut:int,
                 rock_muon_dtype:np.dtypes.VoidDType, segment_dtype:np.dtypes.VoidDType,
                 x_boundaries:np.ndarray, y_boundaries:np.ndarray, z_boundaries:np.ndarray):
        """
        Initializes a new MuonSelection instance.
        
        Args:
            segment_count: Amount of segments
            track_count: Amount of tracks
            length_cut: Minimum length for cluster to be considered a muon
            rock_muon_dtype: dtype of muon array
            segment_dtype: dtype of segment array
            x_boundaries: X boundaries of detector
            y_boundaries: Y boundaries of detector
            z_boundaries: Z boundaries of detector
        """
        self.segment_count = segment_count
        self.track_count = track_count
        self.length_cut = length_cut
        self.rock_muon_dtype = rock_muon_dtype
        self.segment_dtype = segment_dtype
        self.x_boundaries = x_boundaries
        self.y_boundaries = y_boundaries
        self.z_boundaries = z_boundaries

    def merge_test(self, main_cluster_direction:np.ndarray, main_cluster_mean:np.ndarray, test_clusters:np.ndarray, average_dist:float) -> list:
        """Merge test clusters to main cluster.

        Merge test cluster with the main cluster if the average distance of the test \
        cluster hits fall below the average_dist threshold.

        Args:
            main_cluster_direction: The direction of the main cluster.
            main_cluster_mean: The mean position of the main cluster.
            test_clusters: Hit positions of the cluster to be tested.
            average_dist: The distance cut to see if the test cluster will be merged with the main cluster.

        Returns:
            Index of the test clusters to be merged with the main cluster.
        """
        
        distances = [
            self.average_distance(test_cluster,
                            main_cluster_mean, 
                            main_cluster_direction) 
            for test_cluster in test_clusters]
        
        
        indices = [index for index, dist in enumerate(distances) if dist <= average_dist]

        return indices

    def cluster(self, PromptHits_ev:np.ndarray, average_dist:float) -> list:
        """Cluster an event of hits.

        Take in either calib_prompt_hits or calib_final_hits and clusters them using DBSCAN.

        Args:
            PromptHits_ev: Hits of event
            average_dist: Average distance to determine if two test clusters will be merged. 
        
        Returns:
            A nested list of indices each list is a cluster and the indices within \
            the list is where to find the hit in PrompHits_ev.
         """
        positions = np.column_stack((
            PromptHits_ev['x'],
            PromptHits_ev['y'],
            PromptHits_ev['z']
        ))

        dbscan = DBSCAN(min_samples=6, eps=4*.4434)
        clusters = dbscan.fit(positions)
        labels = clusters.labels_

        remove_noise = (labels != -1)

        non_noise_hits = PromptHits_ev[remove_noise]
        non_noise_positions = positions[remove_noise]
        non_noise_labels = labels[remove_noise]

        indicies_of_clusters = []

        for label in np.unique(non_noise_labels):

            indicies_of_clusters.append(
                np.where(non_noise_labels==label)[0]
                )

        direction_each_cluster = np.array([
            self.PCAs(non_noise_positions[indices])[1]
            for indices in indicies_of_clusters
            ])
        
        sorted_indices_cluster_directions = sorted(
        range(len(direction_each_cluster)),
        key=lambda v: max(abs(x) for x in direction_each_cluster[v]),
        reverse=True
        )
        
        positions_per_cluster = [
        non_noise_positions[indicies_of_clusters[i]].data
        for i in sorted_indices_cluster_directions
        ]

        sorted_indices_of_cluster = [
            indicies_of_clusters[index]
            for index in sorted_indices_cluster_directions
        ]
        sorted_directions = [direction_each_cluster[index] for index in sorted_indices_cluster_directions]

        mean_per_cluster = [
            np.mean(cluster, axis=0)
            for cluster in positions_per_cluster
        ]
        
        new_cluster_indices = []

        test = list(range(len(sorted_indices_of_cluster)))

        merged_flags = [False] * len(sorted_indices_of_cluster)
        
        new_cluster_indices = []

        for main_idx in test:
            
            if merged_flags[main_idx]:
                continue

            main_cluster_mean = mean_per_cluster[main_idx]
            main_cluster_direction = sorted_directions[main_idx]

            test_indices = [i for i in range(len(sorted_indices_of_cluster)) if i != main_idx and not merged_flags[i]]
            test_clusters = [positions_per_cluster[i] for i in test_indices]
            
            indices_merge = self.merge_test(main_cluster_direction, main_cluster_mean, test_clusters, average_dist)

            if indices_merge:
                clusters_to_merge = [main_idx] + [test_indices[i] for i in indices_merge]
                merge_indices_flattened = np.concatenate([sorted_indices_of_cluster[index] for index in clusters_to_merge])
                
                for index in clusters_to_merge:
                    merged_flags[index] =True

                new_cluster_indices.append(merge_indices_flattened)
                
            else:
                new_cluster_indices.append(sorted_indices_of_cluster[main_idx])
                merged_flags[main_idx] = True
                continue
        if new_cluster_indices:
            for j in range(len(new_cluster_indices)):
                new_c = non_noise_hits[new_cluster_indices[j]]
                
                indices = np.where(np.isin(PromptHits_ev,new_c))[0]
                new_cluster_indices[j] = indices
        
            return new_cluster_indices
        else:
            return indicies_of_clusters
        
    #@staticmethod
    def PCAs(self, hit_positions:np.ndarray) -> tuple:
        """Compute the PCA.

        Compute the PCA for a cluster of hits.
         
        Args:
            hit_positions: Positions of the hits.
        
        Returns:
            The explained variance ratio, the direction of the cluster, and the mean position.
        """
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)

        #Scale data
        mean = np.mean(hit_positions, axis=0)
        std = np.std(hit_positions, axis=0)
        std1 = np.array([s if s != 0 else 1e-9 for s in std])

        X_train = (hit_positions - mean)/std1

        pca = PCA(1) # 1 component

        pca.fit(X_train)

        explained_var = pca.explained_variance_ratio_[0]
        scaled_direction_vector = pca.components_[0]
        unscaled_vector = std * scaled_direction_vector

        normalized_direction_vector = unscaled_vector/np.linalg.norm(unscaled_vector)

        return  explained_var, normalized_direction_vector, mean

    #@staticmethod
    def length(self, hits:np.ndarray) -> tuple:
        """Get length of cluster.

        Get the length of a particular cluster.

        Args:
            hits: Numpy array of the hits

        Returns:
            The length of the cluster.
        """

        hit_positions = np.column_stack((hits['x'], hits['y'], hits['z']))
        
        hdist = cdist(hit_positions, hit_positions)
         
        max_value_index = np.argmax(hdist)
        # Convert flattened index to row and column indices
        max_value_row = max_value_index // hdist.shape[1]
        max_value_col = max_value_index % hdist.shape[1]
        
        indices = [max_value_row, max_value_col]
        
        start_hit, end_hit = hit_positions[np.min(indices)], hit_positions[np.max(indices)]
        
        return np.max(hdist), start_hit, end_hit

    def close_to_two_faces(self, boundaries:np.ndarray, hits:np.ndarray) -> bool:
        """Test if a track goes through the detector.
        
        Test if the cluster is a maximum of 2.1 cm away from two cluster faces.

        Args:
            boundaries: The maximum and minimum detector boundaries.
            hits: The hits of the cluster.

        Returns:
            A boolean true or false where true is if the cluster went through the entire detector and false otherwise.
        """
        penetrated = False

        test_face = [False] * len(boundaries)
        threshold = 2.1
        for index, face in enumerate(boundaries):
            if (index == 0) or (index == 3):
                distance = np.abs(face - hits['x'])

                if np.any(distance <= threshold):
                    test_face[index] = True
        
            elif (index == 1) or (index == 4):
                distance = np.abs(face - hits['y'])
                if np.any(distance <= threshold):
                    test_face[index] = True
        
            elif (index == 2) or (index == 5): 
                distance = np.abs(face - hits['z'])
                if np.any(distance <= threshold):
                    test_face[index] = True

        if sum(test_face)>= 2:
            penetrated = True

        return penetrated

    def clean_noise_hits(self, positions:np.ndarray, track_direction:np.ndarray, hits_mean:np.ndarray) -> list:
        """Clean hits to far away from track.
        
        Remove hits that are more than 3.5 centimeters away from the track axis.

        Args:
            positions: Position of the hits.
            track_direction: Direction of the track.
            hits_mean: Mean positon of the hits

        Returns:
            A list of booleans that will show which hits are less than or equal to 3.5 centimeters away from track axis. 
        """
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        distances = np.linalg.norm(positions - projections, axis=1)
        
        mask_good = distances <= 3.5
        return mask_good

    def average_distance(self, positions:np.ndarray, hits_mean:np.ndarray, track_direction:np.ndarray) -> float:
        """Return average distance from track.
        
        This will return the average distance of the hits from the track axis.

        Args:
            positions: Position of the hits.
            track_direction: Direction of the track.
            hits_mean: Mean positon of the hits
    

        Returns:
            The average distance of the hits from the track.
        """
        projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

        distances = np.linalg.norm(positions - projections, axis=1)

        average_distances = np.mean(distances)

        return average_distances

    #@staticmethod
    def select_muon_track(self, hits:np.ndarray, min_max_detector_bounds:np.ndarray) -> tuple:
            """Test if cluster is a muon track.

            If the cluster goes through the detector, straight, and has a length of 100 centimeters \
            consider the cluster a muon track.

            Args:
                hits: The hits of a cluster
                min_max_detector_bounds: The minimum and maximum detector bounds
            
            Returns:
                The hits of the cluster (if the hits are from a muon track), the length, the start and end position \
                the explained variance, and the direction.
            """
            muon_hits = []

            min_boundaries = np.flip(min_max_detector_bounds[0]) 
            max_boundaries = np.flip(min_max_detector_bounds[1])
            
            faces_of_detector = np.concatenate((min_boundaries,max_boundaries))

            hit_positions = np.column_stack((
                hits['x'], hits['y'], hits['z']
            ))

            L_cut = self.length_cut 

            explained_var, direction_vector, hits_mean_position = self.PCAs(hit_positions)
            
            mask = self.clean_noise_hits(hit_positions, direction_vector, hits_mean_position)
            
            filtered_hits = hits[mask]
            
            avg_distance = self.average_distance(hit_positions[mask], hits_mean_position, direction_vector)
            
            l_track, start_point, end_point = self.length(filtered_hits)
            if (avg_distance <= 1.5) & (l_track >= L_cut):

                penetrated = self.close_to_two_faces(faces_of_detector, filtered_hits)

                if penetrated:

                    muon_hits.append(filtered_hits)

            return np.array(muon_hits), l_track, start_point, end_point, explained_var, direction_vector

    #@staticmethod
    def angle(self, direction_vector:np.ndarray) -> tuple:
        """Get angle of the muon.
        
        Args:
            direction_vector: Direction of the track.
        Returns:
            Returns that xz, yz, and z angles
        """
        magnitude = np.linalg.norm(direction_vector)

        normal_vector_xz = np.array([0, 1, 0])
        
        dot_product = np.dot(direction_vector, normal_vector_xz)

        theta_xz = np.arccos(dot_product / magnitude)

        theta_xz = np.degrees(theta_xz)
        
        normal_vector_yz = np.array([1, 0, 0])

        dot_product = np.dot(direction_vector, normal_vector_yz)

        theta_yz = np.arccos(dot_product / magnitude)

        theta_yz = np.degrees(theta_yz)

        theta_z = np.degrees(np.arctan2(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2), direction_vector[2]))
        
        return theta_xz, theta_yz, theta_z

    #@staticmethod
    def TPC_separation(self, hits:np.ndarray) -> list:
        """Separate into TPCs.

        This will separate the hits of a track into their individual io_groups.

        Args:
            hits: Hits of the track.

        Returns:
            A nested list where each list is a numpy array of hits and the index is a different io_group.  
        """
        hits_tpc = []

        io_groups = np.unique(hits['io_group'])
        
        for io_group in io_groups:
            mask = hits['io_group'] == io_group

            hits_of_tpc = hits[mask]
            if len(hits_of_tpc) != 0:
                hits_tpc.append(hits_of_tpc)

        return hits_tpc


    #@staticmethod
    def segments(self,muon_hits:np.ndarray) -> list:
        """Create rock muon segments.
        
        This is fit the hits with a PCA to fit the hits with a line (a track), and split this line into segments.

        Args:
            muon_hits: Hits of the muon

        Returns:
            A list of segments information where each index is a segment and the information contained is \
            segment id, x start, y start, z start, energy of segment, x end, y end, z end, charge of segment, \
            number of hits, dx, x mid, y mid, z mid, drift time, io group of segment
        """
        segment_info = []

        hit_ref = []
        segment_to_track_ref = []

        track = muon_hits[0]

        tpc_hits = self.TPC_separation(track)

        given_scale = 2

        for hits in tpc_hits:
            if len(hits) != 0:
                hit_positions = np.array([[hit['x'], hit['y'], hit['z']] for hit in hits])
            
                tpc_var, principal_component, tpc_mean = self.PCAs(hit_positions)
        
                centered_points = hit_positions - tpc_mean

                projections = np.dot(centered_points, principal_component)
                projected_hits = tpc_mean + np.outer(projections, principal_component)

                t_min = np.min(projections)
                t_max = np.max(projections)

                #End points
                line_point_1 = tpc_mean + t_min * principal_component
                line_point_2 = tpc_mean + t_max * principal_component
            
                line_defined_points = [line_point_1,line_point_2]

                line_start = line_defined_points[
                    np.argmax([line_point_1[2], line_point_2[2]])
                    ]
                
                line_end = line_defined_points[
                    np.argmin([line_point_1[2], line_point_2[2]])
                    ]
            
                #lets make segments
                if principal_component[2] < 0:
                    principal_component = -principal_component

                initial_jump_size = given_scale
                jump_vector = initial_jump_size * principal_component
            
                for i in range(1,1000):
                    break_out = False
                
                    segment_start = line_start - (i-1)*jump_vector
                    segment_end = segment_start - jump_vector
                
                    if segment_end[2] >= line_end[2]:
                        seg_info = self.grab_segment_info(segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref)

                        if seg_info is not None:
                            segment_info.append(seg_info)   
                    
                    
                    else:
                        segment_end = line_end
                        seg_info = self.grab_segment_info(segment_end, segment_start, projected_hits, hits, hit_ref, segment_to_track_ref)
                        break_out = True
                        if seg_info is not None:
                            segment_info.append(seg_info)  
                    
                    
                    if break_out:
                        break


        return segment_info, hit_ref, segment_to_track_ref

    def grab_segment_info(self, segment_end:np.ndarray, segment_start:np.ndarray, projected_hits:np.ndarray,
                           hits:np.ndarray, hit_ref:list, segment_to_track_ref:list) -> None | list:
            """Grab segment info.
            
            Computes all wanted segment info if there are hits within the segment ends.

            Args:
                segment_end: The x, y, z position of the segment end.
                segment_start: The x, y, z position of the segment start.
                projected_hits: Hits of track projected onto pca fit.
                hits: Non-projected hits of track.
                hit_ref: Reference of hits to segment.
                segment_to_track_ref: Reference of segment to track.

            Returns:
                The segment id, x start, y start, z start, energy, x end, y end, z end, charge, \
                number of hits, dx, x mid, y mid, z mid, drift time, and io group of the segment
            """
            min_bounds = [min([segment_end[i],segment_start[i]]) for i in range(0,3)]
            max_bounds = [max([segment_end[i],segment_start[i]]) for i in range(0,3)]
            condition = (projected_hits[:,2] >= min_bounds[2]) & (projected_hits[:,2] <= max_bounds[2])
        
            condition = (
                    (projected_hits[:, 0] >= min_bounds[0]) & (projected_hits[:, 0] <= max_bounds[0]) &
                    (projected_hits[:, 1] >= min_bounds[1]) & (projected_hits[:, 1] <= max_bounds[1]) &
                    (projected_hits[:, 2] >= min_bounds[2]) & (projected_hits[:, 2] <= max_bounds[2])
                )
        
            hits_of_segment = hits[condition]
        
            if len(hits_of_segment) != 0:
                x_start, y_start, z_start = segment_start[0], segment_start[1], segment_start[2]
                x_end, y_end, z_end = segment_end[0], segment_end[1], segment_end[2]
                x_mid, y_mid, z_mid = (x_start+x_end)/2, (y_start + y_end)/2, (z_start + z_end)/2

                Energy_of_segment = sum(hits_of_segment['E'])
                Q_of_segment = sum(hits_of_segment['Q'])
                drift_time = (max(hits_of_segment['t_drift'])+min(hits_of_segment['t_drift']))/2
            
                io_group_of_segment = np.unique(hits_of_segment['io_group'])[0]
                self.segment_count += 1

                
                for hit in hits_of_segment:
                    hit_ref.append([self.segment_count, hit['id']])
                segment_to_track_ref.append([self.track_count, self.segment_count])
                dx = np.linalg.norm(segment_start-segment_end)
            
                return [self.segment_count, x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Q_of_segment,len(hits_of_segment), dx, x_mid, y_mid,z_mid, drift_time, io_group_of_segment]
            else:
                return None

    def run(self, file:str) -> tuple:
        """Run the muon selection.

        Args:
            file: hdf5 file

        Returns:
            Selected muon tracks of file their segments, and their hits


        """
        muon_tracks = []
        muon_segments = []
        muon_hits = []
        f = h5flow.data.H5FlowDataManager(file, 'r')
        
        events = f['charge/events/data']
        
        Min_max_detector_bounds = [[min(self.z_boundaries),min(self.y_boundaries),min(self.x_boundaries)],
                                    [max(self.z_boundaries), max(self.y_boundaries), max(self.x_boundaries)]]
        for event in range(events.size):
            PromptHits_ev = f['charge/events', 'charge/calib_prompt_hits', event][0].data
 
            PromptHits_ev_positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))
            
            nan_indices = np.unique(np.argwhere(np.isnan(PromptHits_ev_positions))[:,0]) 
            
            if len(nan_indices) >   0:
                PromptHits_ev = np.delete(PromptHits_ev,nan_indices, axis = 0)

            unique_points, counts = np.unique(PromptHits_ev_positions, axis=0, return_counts=True)

            for unique_point, count in zip(unique_points, counts):

                if count > 1000:
                    mask = np.all(PromptHits_ev_positions != unique_point, axis =1)

                    PromptHits_ev = PromptHits_ev[mask]
                    
            if len(PromptHits_ev) >= 100:
                hit_indices = self.cluster(PromptHits_ev, 2)

                for indices in hit_indices:
                    if len(indices) > 10:
                       
                        hits = PromptHits_ev[indices]

                        if len(hits) < 1:
                            continue
                        muon_track,length_of_track, start_point, end_point, explained_var, direction_vector = self.select_muon_track(hits,Min_max_detector_bounds)
                        
                        if len(muon_track) != 0:
                            muon_hits.append(hits)
                            #Loop through tracks and changes the DBSCAN cluster_id to a given track number
                            self.track_count += 1 
                            track_number = self.track_count
                            
                            #Get angle of track
                            theta_xz, theta_yz,theta_z = self.angle(direction_vector)
                            
                            #Fill track info
                            track_info = [event, track_number,length_of_track, start_point[0],start_point[1],start_point[2], end_point[0],end_point[1],end_point[2], explained_var, theta_xz, theta_yz, theta_z]
                            
                            track_info = np.array([tuple(track_info)], dtype = self.rock_muon_dtype)
                            muon_tracks.append(track_info)

                            #Get segments
                            segments_list, segment_hit_ref, segment_track_ref = self.segments(muon_track)
                            segments_array = np.array([tuple(segment) for segment in segments_list], dtype = self.segment_dtype)
                            muon_segments.append(segments_array)
                            
        return np.concatenate(muon_tracks), np.concatenate(muon_segments), muon_hits
    


if __name__ == '__main__':
    segment_count = 0
    track_count = 0
    length_cut = 100
    
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

    x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
    y_boundaries = np.array([-42-19.8543, -42+103.8543])
    z_boundaries = np.array([-64.3163,  -2.6837, 2.6837, 64.3163]) 

    selection = MuonSelection(segment_count, track_count, length_cut, rock_muon_dtype, segment_dtype,
                              x_boundaries, y_boundaries, z_boundaries)
    file = sys.argv[1]
    hdf5_file_name = file.split('/')[-1]
    wanted_sim = hdf5_file_name.split('_')[0]

    f = h5flow.data.H5FlowDataManager(file, 'r')
    
    tracks, segments, hits = selection.run(file)
    print('Done w/ selection starting purity calc')

    save_tracks_name = hdf5_file_name + '.tracks.csv'
    save_segments_name = hdf5_file_name + '.segments.csv'

    track_df = pd.DataFrame(tracks)
    segment_df = pd.DataFrame(segments)

    
    track_df.to_csv(save_tracks_name)
    segment_df.to_csv(save_segments_name)

    p = Purity.Purity(f, wanted_sim)

    particle_stack = p.produce_purity_and_plot(hits)
    