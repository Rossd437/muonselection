import numpy as np

import h5flow

from h5flow.data import dereference

from scipy.optimize import curve_fit

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

from scipy.spatial.distance import cdist

segment_count = 0

def PCAs(hits_of_track:np.ndarray):
    scaler = StandardScaler()
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    positions = np.column_stack((hits_of_track['x'], hits_of_track['y'], hits_of_track['z']))

    X_train = positions
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    pca = PCA(1) # 1 component

    pca.fit(X_train)

    explained_var = pca.explained_variance_ratio_[0]

    scaled_vector = pca.components_[0]

    unscaled_vector = scaler.scale_ * scaled_vector

    normalized_direction_vector = unscaled_vector/np.linalg.norm(unscaled_vector)

    scaled_mean = pca.mean_

    original_mean = scaler.inverse_transform(scaled_mean.reshape(1, -1)).flatten()

    return  explained_var, normalized_direction_vector, original_mean

def cluster(PromptHits_ev):
    index_of_track_hits = []
    positions = np.column_stack((PromptHits_ev['x'], PromptHits_ev['y'], PromptHits_ev['z']))

    # Perform DBSCAN clustering
    hit_cluster = DBSCAN(eps=1, min_samples=3).fit(positions)

    cluster_labels = hit_cluster.labels_

    unique_labels = np.unique(cluster_labels)

    if len(unique_labels) < 150:

        # Collect indices of hits for each cluster
        for unique in unique_labels:
            index = np.where(cluster_labels == unique)[0]
            index_of_track_hits.append(index)

        index = 0
        while index < len(index_of_track_hits):
            center_of_masses = [np.mean(positions[cluster], axis=0) for cluster in index_of_track_hits]
            center_of_1 = np.mean(positions[index_of_track_hits[index]], axis=0)

            # Compute distances and lengths
            distances = np.linalg.norm(center_of_masses - center_of_1, axis=1)
            lengths = [len(cluster) for cluster in index_of_track_hits]
        
            combined_dist_length = [[distances[k], lengths[k]] for k in range(len(distances))]

            # Create a list of indices
            indices = list(range(len(combined_dist_length)))

            # Sort indices based on length (descending) and distance (ascending)
            sorted_indices = sorted(indices, key=lambda i: (-combined_dist_length[i][1], combined_dist_length[i][0]))
        
            explained_var, direction, original_mean = PCAs(PromptHits_ev[index_of_track_hits[index]])

            # Try merging with sorted clusters
            for j in sorted_indices:
                if (j == index) | (len(index_of_track_hits[j]) < 6) | (len(index_of_track_hits[index]) < 6) | (combined_dist_length[j][0] > 100) | (combined_dist_length[j][0] < 2):  # Skip merging with itself
                    continue
                explained_var, direction2, original_mean = PCAs(PromptHits_ev[index_of_track_hits[j]])
                hits_of_testing_merge = np.concatenate((positions[index_of_track_hits[index]], positions[index_of_track_hits[j]]))
                center_of_merge = np.mean(hits_of_testing_merge, axis=0)

                projections = np.dot(hits_of_testing_merge - center_of_merge, direction[:, np.newaxis]) * direction + center_of_merge
                distances = np.linalg.norm(hits_of_testing_merge - projections, axis=1)
                average_dist = np.mean(distances)
                sim_direction = np.rad2deg(np.arccos(np.abs(np.dot(direction, direction2))))

                if (average_dist <= 3) & (sim_direction <= 20):  # Adjust distance threshold as needed
                    index_of_track_hits[index] = np.concatenate([index_of_track_hits[index], index_of_track_hits[j]])
                    index_of_track_hits.pop(j)
                    center_of_masses.pop(j)
                    #print(f'Merging cluster {index} with cluster {j}')
                    break  # Recompute centers and distances after merge
            else:
                index += 1
    else:
        for unique in unique_labels:
            index = np.where(hit_cluster.labels_ == unique)[0]
            index_of_track_hits.append(index)

    return index_of_track_hits 
#@staticmethod
def TPC_separation(hits:np.ndarray):
    hits_tpc = []

    io_groups = np.unique(hits['io_group'])

    for io_group in io_groups:
        mask = hits['io_group'] == io_group

        hits_of_tpc = hits[mask]
        if len(hits_of_tpc) != 0:
            hits_tpc.append(hits_of_tpc)

    return hits_tpc


#@staticmethod
def segments(muon_hits:np.ndarray, scale:int) ->  np.ndarray:

    segment_info = []

    hit_ref = []
    
    random_values = []
    segment_count=0
    given_scale = scale
    track = muon_hits
    ex_var, direction, mean_point = PCAs(track)

    #print(sum(track['Q'])/l_track)
    number_of_hits = len(track)

    tpc_hits = TPC_separation(track)

    for hitss in tpc_hits:

        if len(hitss) != 0:
            hits = hitss

            io_group_of_tpc = np.unique(hitss['io_group'])

            tpc_var, principal_component, tpc_mean = PCAs(hitss)

            if principal_component[2] < 0:
                principal_component = -principal_component

            points = np.array([[hit['x'], hit['y'], hit['z']]for hit in hits])
            
            centered_points = points - tpc_mean

            projections = np.dot(centered_points, principal_component)
            projected_hits = tpc_mean + np.outer(projections, principal_component)
            
            # Step 7: Find the minimum and maximum projections
            t_min = np.min(projections)
            t_max = np.max(projections)
            
            # Step 8: Compute the endpoints of the finite line
            # Line endpoint 1: mean + t_min * principal_component
            line_point_1 = tpc_mean + t_min * principal_component

            # Line endpoint 2: mean + t_max * principal_component
            line_point_2 = tpc_mean + t_max * principal_component
            
            line_defined_points = [line_point_1,line_point_2]

            line_start = line_defined_points[np.argmax([line_point_1[2], line_point_2[2]])]
            line_end = line_defined_points[np.argmin([line_point_1[2], line_point_2[2]])]
            
            line_length = np.linalg.norm(line_start-line_end)
            
            initial_jump_size = given_scale

            jump_vector = initial_jump_size * principal_component
            
            #lets make segments
            if principal_component[2] < 0:
                principal_component = -principal_component
            
            
            for i in range(1,1000):
                break_out = False
                
                
                segment_start = line_start - (i-1)*jump_vector
                segment_end = segment_start - jump_vector
                
                if segment_end[2] >= line_end[2]:
                    seg_info = grab_segment_info(segment_end, segment_start, projected_hits, hits)

                    if seg_info is not None:
                        segment_info.append(seg_info)   
                    
                    
                else:
                    segment_end = line_end
                    seg_info = grab_segment_info(segment_end, segment_start, projected_hits, hits)
                    break_out = True
                    if seg_info is not None:
                        segment_info.append(seg_info)  
                    
                    
                if break_out == True:
                    break


    return segment_info

def grab_segment_info(segment_end, segment_start, projected_hits, hits):
        min_bounds = [min([segment_end[i],segment_start[i]]) for i in range(0,3)]
        max_bounds = [max([segment_end[i],segment_start[i]]) for i in range(0,3)]
        condition = (projected_hits[:,2] >= min_bounds[2]) & (projected_hits[:,2] <= max_bounds[2])
        
        condition = (
                    (projected_hits[:, 0] >= min_bounds[0]) & (projected_hits[:, 0] <= max_bounds[0]) &
                    (projected_hits[:, 1] >= min_bounds[1]) & (projected_hits[:, 1] <= max_bounds[1]) &
                    (projected_hits[:, 2] >= min_bounds[2]) & (projected_hits[:, 2] <= max_bounds[2])
                )
        
        hits_of_segment = hits[condition]
        

        #print(f'Segment: start={segment_start}, end={segment_end}, hits_found={len(hits_of_segment)}')
        if len(hits_of_segment) != 0:
            hits_positions = np.column_stack((hits_of_segment['x'],hits_of_segment['y'],hits_of_segment['z']))
    
            x_start, y_start, z_start = segment_start[0], segment_start[1], segment_start[2]
            x_end, y_end, z_end = segment_end[0], segment_end[1], segment_end[2]
            x_mid, y_mid, z_mid = (x_start+x_end)/2, (y_start + y_end)/2, (z_start + z_end)/2

            Energy_of_segment = sum(hits_of_segment['E'])
            Q_of_segment = sum(hits_of_segment['Q'])
            drift_time = np.mean(hits_of_segment['t_drift'])
            
            io_group_of_segment = np.unique(hits_of_segment['io_group'])[0]
            global segment_count
            segment_count += 1

            dx = np.linalg.norm(segment_start-segment_end)
            #dx = max(distance.cdist(projected_hits[condition],projected_hits[condition]).ravel())

            return [len(hits_of_segment), x_start, y_start, z_start, Energy_of_segment, x_end, y_end, z_end, Q_of_segment, dx, x_mid, y_mid,z_mid, drift_time, io_group_of_segment]
        else:
            #print(f'No hits found for segment: start={segment_start}, end={segment_end}')
            return None


def segment_reconstruction(flist: list[str], scale: int) -> np.ndarray:
    
    rock_muon_segments_dtype = np.dtype([
        ('rock_segment_id', 'i4'),
        ('x_start', 'f8'),
        ('y_start','f8'),
        ('z_start','f8'),
        ('dE', 'f8'),
        ('x_end', 'f8'),
        ('y_end','f8'),
        ('z_end', 'f8'),
        ('dQ','f8'),
        ('dx','f8'),
        ('x_mid','f8'),
        ('y_mid','f8'),
        ('z_mid','f8'),
        ('t','f8'),
        ('io_group', 'i4')
    ])



    segments_of_scale = []

    for file in flist:
        try:
            f = h5flow.data.H5FlowDataManager(file, 'r')
            #print(file)
            rock_tracks = f['analysis/rock_muon_tracks/data']
            calib_prompt_hits = f['charge/calib_prompt_hits/data']
            for track in rock_tracks:

                hits_of_track = dereference(
                track['rock_muon_id'],
                f['/analysis/rock_muon_tracks/ref/charge/calib_prompt_hits/ref/'],
                f['charge/calib_prompt_hits/data'],
                ref_direction = (0,1)
                )[0]


                hits = hits_of_track.data

                mask = np.logical_not([np.all(hits[name]) == 0 for name in hits.dtype.names])

                true_mask = ~np.all(mask == False, axis =0)
                wanted_hits = hits[true_mask]
                try:
                    segments_list= segments(wanted_hits, scale)#hits_bt,segments_map,segs_dtype, seg_hit_counts, segs_max_frac)
                    #print(segments_list
                    segments_array = np.array([tuple(sub) for sub in segments_list], dtype = rock_muon_segments_dtype)
                    segments_of_scale.append(segments_array)
                    #dx_real.append(random_values)
                except Exception as e:
                    print(f'{e}')
                
            
        except:
            continue

    
    return np.concatenate(segments_of_scale)

#@staticmethod
def length(hits):
    #Get Hit positions
    hit_positions = np.column_stack((hits['x'], hits['y'], hits['z']))
    
    hdist = cdist(hit_positions, hit_positions)
        
    max_value_index = np.argmax(hdist)
    # Convert flattened index to row and column indices
    max_value_row = max_value_index // hdist.shape[1]
    max_value_col = max_value_index % hdist.shape[1]
    
    indices = [max_value_row, max_value_col]
    
    start_hit, end_hit = hit_positions[np.min(indices)], hit_positions[np.max(indices)]
    
    return np.max(hdist), start_hit, end_hit

    '''
    Checks to see if start/end point of track are close to two different faces of detector. If they are this will return True. Note: >= -1 just in case if a hit is reconstructed outside of detector.
    '''
def close_to_two_faces(boundaries, hits):
    # Boundaries are in the order [xmin, ymin, zmin, xmax, ymax, zmax]
    penetrated = False

    test_face = [False] * len(boundaries)
    threshold = 2.1
    for index, face in enumerate(boundaries):
        if (index == 0) or (index == 3):
            distance = np.abs(face - hits['x'])
            #print(f"Checking x boundaries at index {index}: face = {face}, distances = {distance}")

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
    #print(test_face)
    if sum(test_face)>= 2:
        penetrated = True

    return penetrated

def average_distance(hits):
    positions = np.column_stack((hits['x'], hits['y'], hits['z']))

    # Perform PCA to find the principal component
    pca = PCA(n_components=1)
    pca.fit(positions)
    track_direction = pca.components_[0]
    hits_mean = pca.mean_



    # Project points onto the principal component (the line)
    projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

    # Calculate the Euclidean distance between each point and its projection on the line
    distances = np.linalg.norm(positions - projections, axis=1)
    #print(np.mean(distances))
    average_distances = np.mean(distances)

    return average_distances

#@staticmethod
def select_muon_track(hits):
        muon_hits = []
        z_boundaries = np.array([-48.686,48.686])
        y_boundaries = np.array([-154.05199731,150.05199731])
        x_boundaries = np.array([-46.7879982,46.7879982])

        min_boundaries = np.array([min(x_boundaries), min(y_boundaries), min(z_boundaries)]) #bounds are z,y,x and hits x,y,z, so bounds must be flipped
        max_boundaries = np.array([max(x_boundaries), max(y_boundaries), max(z_boundaries)])
        
        faces_of_detector = np.concatenate((min_boundaries,max_boundaries))

        L_cut = 100 #minimum track length requirement
        
        filtered_hits = hits#clean_noise_hits(hits)
        
        explained_var, direction_vector,mean_point = PCAs(filtered_hits)
            
        l_track, start_point, end_point = length(filtered_hits)
        
        avg_distance = average_distance(filtered_hits)

        if (avg_distance <= 1.5) & (l_track >= L_cut):

            penetrated = close_to_two_faces(faces_of_detector, filtered_hits)

            if penetrated == True:
                #filtered_hits = self.clean_noise_hits(hits)

                muon_hits.append(filtered_hits)

                #Get the new hits info
                #explained_var, direction_vector,mean_point = self.PCAs(filtered_hits)

                #l_track, start_point, end_point = self.length(filtered_hits)

        return np.array(muon_hits), l_track, start_point, end_point, explained_var, direction_vector

#@staticmethod
def angle(direction_vector):
    magnitude = np.linalg.norm(direction_vector)

    # Calculate the unit vector in the xz-plane
    normal_vector_xz = np.array([0, 1, 0])
    
    # Calculate the dot product between the direction vector and the unit vector in the yz-plane
    dot_product = np.dot(direction_vector, normal_vector_xz)

    # Calculate the angle between the direction vector and the yz-plane
    theta_xz = np.arccos(dot_product / magnitude)

    # Convert the angle from radians to degrees
    theta_xz = np.degrees(theta_xz)
    
    normal_vector_yz = np.array([1, 0, 0])

    # Calculate the dot product between the direction vector and the unit vector in the yz-plane
    dot_product = np.dot(direction_vector, normal_vector_yz)

    # Calculate the angle between the direction vector and the yz-plane
    theta_yz = np.arccos(dot_product / magnitude)

    # Convert the angle from radians to degrees
    theta_yz = np.degrees(theta_yz)

    if direction_vector[2] > 0:
        theta_z = np.degrees(np.arctan(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)/direction_vector[2]))
    elif direction_vector[2] < 0:
        theta_z = 180 + np.degrees(np.arctan(np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)/direction_vector[2]))
    else:
        theta_z = 90
    
    return theta_xz, theta_yz, theta_z

def clean_noise_hits(hits):
    positions = np.column_stack((hits['x'], hits['y'], hits['z']))

    # Perform PCA to find the principal component
    pca = PCA(n_components=1)
    pca.fit(positions)
    track_direction = pca.components_[0]
    hits_mean = pca.mean_

    # Project points onto the principal component (the line)
    projections = np.dot(positions - hits_mean, track_direction[:, np.newaxis]) * track_direction + hits_mean

    # Calculate the Euclidean distance between each point and its projection on the line
    distances = np.linalg.norm(positions - projections, axis=1)
    
    mask_good = distances <= 3.5

    filtered_hits = hits[mask_good]

    return filtered_hits
