#imports 

import numpy as np
from h5flow.data import dereference
import h5flow
import matplotlib.pyplot as plt
import glob
import mplhep as hep

class Purity:

    def __init__(self, filelist, nFiles, hits_dset_name, x_boundaries, y_boundaries, z_boundaries):
        
        #Initialization of attributes for Objects
        
        self.filelist = filelist
        self.nFiles = nFiles
        self.hits_dset_name = hits_dset_name

        self.x_boundaries = x_boundaries
        self.y_boundaries = y_boundaries
        self.z_boundaries = z_boundaries
    '''
    
    Methods that can be called by a object of this class
    
    '''
    
    '''

    Purity Methods
    
    '''
    def back_track_hits(self):
        makeup_of_selection = []
        energy_ratio = []
        purity_of_muon_segments = []
        segs_of_true_muon_tracks = []

        
        for file in self.filelist:

            try:
                # Load data sets
                f = h5flow.data.H5FlowDataManager(file, 'r')

                segments = f['mc_truth/segments/data']
                trajectories = f['mc_truth/trajectories/data']
                dset_hits = f['charge/'+self.hits_dset_name+'/data']
                hitss_bt = f['mc_truth/'+self.hits_dset_name[:-1]+'_backtrack/data']

                segment_map = {seg['segment_id']: seg for seg in segments}
                traj_map = {traj['file_traj_id']: traj for traj in trajectories}
                rock_tracks = f['analysis/rock_muon_tracks/data']

                track2hits = dereference(
                    rock_tracks['rock_muon_id'],     # indices of A to load references for, shape: (n,)
                    f['/analysis/rock_muon_tracks/ref/charge/calib_prompt_hits/ref'],  # references to use, shape: (L,)
                    f['/charge/calib_prompt_hits/data'],
                    ref_direction=(0,1)  # dataset to load, shape: (M,)
                )

                # Loop through the tracks
                for hits_of_track in track2hits:
                    track_makeup = {}

                    trajs_of_track = []

                    hits = np.array([tup for tup in hits_of_track.data if not all(np.all(elem == 0) for elem in tup)], dtype=hits_of_track.dtype)

                    total_charge = np.sum(hits['Q'])
                    total_energy = np.sum(hits['E'])

                    hit_ref = hits['id']
                    hits_bt = hitss_bt[hit_ref] 

                    # Check if hit['id'] matches index in dset_hits_name
                    indices = np.where(np.isin(dset_hits['id'], hits['id']))[0]

                    for i in range(len(indices)):
                        if indices[i] != hits['id'][i]:
                            print(f'WARNING: rock_muon hit id not the same as {hits_dset_name} index')
                    true_energy = []
                    # Plot all of the backtracked segment positions
                    for hit in hits_bt:
                        for cont in range(len(hit['fraction'])):
                            if hit['fraction'][cont] > 0.0001:
                                seg_id = hit['segment_ids'][cont]
                                seg = segment_map.get(seg_id)
                                
                                # Append trajectory information to the list
                                trajs_of_track.append([
                                    seg['file_traj_id'],  # File trajectory ID
                                    seg['n_electrons'],  # Number of electrons
                                    hit['fraction'][cont],  # Fraction associated with the hit
                                    seg_id
                                ])
                                
                                possible_true_energy = hit['fraction'][cont] * segments[seg_id]['n_electrons']
                                true_energy.append(possible_true_energy)
                                
                                if not seg['segment_id'] == seg_id:
                                    print(f'WARNING: segment id not the same as segment index!')
                    
                    traj_arr = np.array(trajs_of_track)
                    #print(traj_arr)
                    unique_trajs = np.unique(traj_arr[:, 0])

                    for i in range(len(unique_trajs)):
                        traj = unique_trajs[i]
                        mask = traj_arr[:,0] == traj
                        trajss = traj_arr[mask]
                        #print(trajss)
                        #Get makeup of track
                        wanted_traj = traj_map.get(traj)
                        #print(wanted_traj)
                        wanted_segments = np.array([segment_map.get(seg_id) for seg_id in trajss[:,-1]], dtype = segments.dtype)
                        #print(wanted_segments)
                        pdg_of_traj = wanted_traj['pdg_id'] #trajectories[index]['pdg_id'][0]
                        E_of_traj = sum(wanted_segments['dE'])
                        #E_of_traj = abs(trajectories[index]['E_end'] - trajectories[index]['E_start'])[0]
                        
                        if pdg_of_traj not in track_makeup.keys():
                            track_makeup[f"{pdg_of_traj}"] = E_of_traj
                        else:
                            track_makeup[f"{pdg_of_traj}"] = track_makeup[f"{pdg_of_traj}"] + E_of_traj
                    
                    muon_key = [13,-13]
                    max_key = int(max(track_makeup, key = track_makeup.get))
                    
                    if max_key in muon_key:
                        #Get purity of segments
                        all_segs_of_track = np.array([segment_map.get(seg_id) for seg_id in traj_arr[:,-1]], dtype = segments.dtype)#[np.unique(np.array(traj_arr[:,-1], dtype = int))]
                        mask_muon_segments = (all_segs_of_track['pdg_id'] == 13) | (all_segs_of_track['pdg_id'] == -13)
                        mask_electron_segments = (all_segs_of_track['pdg_id'] == 11) | (all_segs_of_track['pdg_id'] == -11)
                        muon_segments = all_segs_of_track[mask_muon_segments]
                        purity_segments = len(muon_segments)/len(all_segs_of_track)
                        purity_of_muon_segments.append(purity_segments) 
                        segs_of_true_muon_tracks.append(all_segs_of_track)
                    
                    all_segs_of_track = np.array([segment_map.get(seg_id) for seg_id in traj_arr[:,-1]], dtype = segments.dtype)#[np.unique(np.array(traj_arr[:,-1], dtype = int))]
                    
                    total_E_of_track = sum(all_segs_of_track['dE'])
                    
                    tot_E_of_track = sum(track_makeup.values())
                    
                    for key in track_makeup.keys():
                        track_makeup[f'{key}'] = round(track_makeup[f'{key}']/tot_E_of_track,6)
                    
                    recon_true_E_ratio = total_energy/tot_E_of_track
                    makeup_of_selection.append(track_makeup)
                    
                    energy_ratio.append(recon_true_E_ratio)
            except:
                continue

        return makeup_of_selection, np.array(energy_ratio), purity_of_muon_segments, np.concatenate(segs_of_true_muon_tracks)

    def amount_of_particle(self,pdg, pdgs_of_selection):
        
        mask = pdgs_of_selection == pdg
        
        trajs_with_pdg = [ptype for ptype in pdgs_of_selection if ptype == pdg or ptype == pdg]
        
        amount = len(trajs_with_pdg)
        
        percent = round((amount/len(pdgs_of_selection)) * 100, 2)
        
        return amount, percent

    '''
    
    Efficiency related methods

    '''

    def is_point_outside(self, point, x_boundaries, y_boundaries, z_boundaries):

        x, y, z = point[0], point[1], point[2]
        
        xmin, xmax, ymin, ymax, zmin, zmax = x_boundaries.min(), x_boundaries.max(), y_boundaries.min(), y_boundaries.max(), z_boundaries.min(), z_boundaries.max()
        
        return x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax
    
    def detector_eff(self):
        
        min_bounds = [min(self.x_boundaries), min(self.y_boundaries), min(self.z_boundaries)]
        max_bounds = [max(self.x_boundaries), max(self.y_boundaries), max(self.z_boundaries)]
        A = 'mc_truth/interactions'
        B = 'mc_truth/trajectories'
        #B = 'mc_truth/segments'
        C= 'charge/packets'
        D = 'charge/calib_prompt_hits'

        counts_of_true_rock_muons = 0
        
        n_rock_tracks = []
        
        for file in self.filelist[:self.nFiles]:
            try:
                f = h5flow.data.H5FlowDataManager(file, 'r')

                interactions = f['mc_truth/interactions/data']
                
                trajs = f['mc_truth/trajectories/data']
                rock_tracks = f['analysis/rock_muon_tracks/data']

                tracks = np.unique(rock_tracks['rock_muon_id'])
                muons_key = [13, -13]
                nu_keys = [14, -14]

                muon_trajs = [traj for traj in trajs if traj['pdg_id'] in muons_key]
            
                n_rock_tracks.append(len(tracks))
            
                for index, interaction in enumerate(interactions):

                    vertex = [interaction['vertex'][0], interaction['vertex'][1], interaction['vertex'][2]]
                
                    pdg = interaction['lep_pdg']
                    nu_pdg = interaction['nu_pdg']

                    if  self.is_point_outside(vertex, x_boundaries, y_boundaries, z_boundaries) & (pdg in muons_key) & (nu_pdg in nu_keys):
                        a2b_ref = dereference(
                        index,     # indices of A to load references for, shape: (n,)
                        f['/{}/ref/{}/ref'.format(A,B)],  # references to use, shape: (L,)
                        f['/{}/data'.format(B)],       # dataset to load, shape: (M,)
                        region = f['/{}/ref/{}/ref_region'.format(A,B)], # lookup regions in references, shape: (N,)
                        indices_only = True,
                        ref_direction = (0,1)
                        )
                        
                        trajs_of_interactions = (trajs[a2b_ref[0]])

                        mask_muon = abs(trajs_of_interactions['pdg_id']) == 13 

                        muon_traj_of_interaction = trajs_of_interactions[mask_muon]

                        start_muon, end_muon = muon_traj_of_interaction['xyz_start'][0], muon_traj_of_interaction['xyz_end'][0]

                        if self.intersects_aabb(start_muon, end_muon, min_bounds,max_bounds):
                            counts_of_true_rock_muons += 1
        
            except Exception as e:
                print(e)
                continue
        print(counts_of_true_rock_muons)
        return sum(n_rock_tracks)/counts_of_true_rock_muons


    def intersects_aabb(self, p1, p2, box_min, box_max):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        direction = p2 - p1
        tmin, tmax = 0.0, 1.0

        for i in range(3):  # x, y, z
            if direction[i] != 0:
                t1 = (box_min[i] - p1[i]) / direction[i]
                t2 = (box_max[i] - p1[i]) / direction[i]
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmax < tmin:
                    return False
            else:
                if p1[i] < box_min[i] or p1[i] > box_max[i]:
                    return False

        return tmin > 0.0 and tmax < 1.0 and tmin < tmax
    
x_boundaries = np.array([-63.931, -3.069, 3.069, 63.931])
y_boundaries = np.array([-42-19.8543, -42+103.8543]) 
z_boundaries = np.array([-64.3163,  -2.6837, 2.6837, 64.3163])

filelist = glob.glob(f"/global/cfs/cdirs/dune/users/demaross/MiniRun6.4/*.hdf5")

Purity_Eff = Purity(filelist, nFiles=2, hits_dset_name='calib_prompt_hits', x_boundaries=x_boundaries, y_boundaries=y_boundaries, z_boundaries=z_boundaries)

eff = Purity_Eff.detector_eff()

print(eff)
