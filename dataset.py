import os.path as osp
import glob
import multiprocessing as mp
from tqdm import tqdm
import random
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Dataset, DataLoader

#import config

class TrackMLParticleTrackingDataset(Dataset):
    def __init__(self, root, 
                 transform=None, 
                 n_events=0,
                 directed=False, 
                 layer_pairs_plus=False,
                 volume_layer_ids=[[8, 2], [8, 4], [8, 6], [8, 8]], #Layers Selecte
                 layer_pairs=[[7, 8], [8, 9], [9, 10]],             #Connected Layers
                 pt_min=2.0, 
                 eta_range=[-5, 5],                     
                 phi_slope_max=0.0006, 
                 z0_max=150,                  
                 n_phi_sections=1, 
                 n_eta_sections=1,  
                 augments = False,
                 tracking=False,                   
                 n_workers=mp.cpu_count(), 
                 n_tasks=1,               
                 download_full_dataset=False                        
             ):
        hits = glob.glob(osp.join(osp.join(root,'raw'), 'event*-hits.csv'))
        self.hits = sorted(hits)
        particles = glob.glob(osp.join(osp.join(root,'raw'), 'event*-particles.csv'))
        self.particles = sorted(particles)
        truth = glob.glob(osp.join(osp.join(root,'raw'), 'event*-truth.csv'))
        self.truth = sorted(truth)
        if (n_events > 0):
            self.hits = self.hits[:n_events]
            self.particles = self.particles[:n_events]
            self.truth = self.truth[:n_events]
        self.layer_pairs_plus = layer_pairs_plus
        self.volume_layer_ids = torch.tensor(volume_layer_ids)
        self.layer_pairs      = torch.tensor(layer_pairs)
        self.pt_min           = pt_min
        self.eta_range        = eta_range
        self.phi_slope_max    = phi_slope_max
        self.z0_max           = z0_max
        self.augments         = augments
        self.n_phi_sections   = n_phi_sections
        self.n_eta_sections   = n_eta_sections
        self.tracking         = tracking
        self.n_workers        = n_workers
        self.n_tasks          = n_tasks
        self.full_dataset     = download_full_dataset
        self.n_events         = n_events

        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)

    def len(self):
        N_events = len(self.hits)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments

    def __len__(self):
        N_events = len(self.hits)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments

    def read_events(self,idx):
        hits_filename = self.hits[idx]
        hits = pandas.read_csv(
            hits_filename, usecols=['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id'],
            dtype={
                'hit_id': np.int64,
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
                'module_id': np.int64
            })
        particles_filename = self.particles[idx]
        particles = pandas.read_csv(
            particles_filename, usecols=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits'],
            dtype={
                'particle_id': np.int64,
                'vx': np.float32,
                'vy': np.float32,
                'vz': np.float32,
                'px': np.float32,
                'py': np.float32,
                'pz': np.float32,
                'q': np.int64,
                'nhits': np.int64
            })
        truth_filename = self.truth[idx]
        truth = pandas.read_csv(
            truth_filename, usecols=['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight'],
            dtype={
                'hit_id': np.int64,
                'particle_id': np.int64,
                'tx': np.float32,
                'ty': np.float32,
                'tz': np.float32,
                'tpx': np.float32,
                'tpy': np.float32,
                'tpz': np.float32,
                'weight': np.float32
            })
        return hits,particles,truth
    
    def select_hits(self, hits, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        n_det_layers = len(valid_layer)

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        index = layer.unique(return_inverse=True)[1]
        hits = hits[['hit_id', 'x', 'y', 'z']].assign(layer=layer, index=index)

        particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)
        particles['pmag'] = np.sqrt(particles['pt']**2 + particles['pz']**2)
        particles['eta'] = 0.5*(np.log(particles['pmag'] + particles['pz']) - np.log(particles['pmag'] - particles['pz']))
        particles['phi'] = np.arctan2(particles['py'], particles['px'])
        
        pt = np.sqrt(particles['px'].values**2 + particles['py'].values**2)
        particles_mask = pt > self.pt_min
        particles_fail = particles[~particles_mask]
        particles = particles[particles_mask]

        valid_groups = hits.groupby(['layer'])
        hits = pandas.concat([valid_groups.get_group(valid_layer.numpy()[i]) for i in range(n_det_layers)])

        hits = (hits[['hit_id', 'x', 'y', 'z', 'index']].merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits['track_id'] = hits['hit_id'].astype(str) + "-" + hits['particle_id'].astype(str)

        hits['particle_id'].where(hits['particle_id'].isin(particles['particle_id']) | (hits['particle_id'] == 0), -1, inplace=True)
        pids_unique, pids_inverse, pids_counts = np.unique(hits['particle_id'].values, return_inverse=True, return_counts=True)        
        pids_unique = np.arange(pids_unique.size) # make it [not interested, noise, remapped pid]
        hits['remapped_pid'] = pids_unique[pids_inverse]

        hits = hits[(hits['remapped_pid'] > 0) & (hits['remapped_pid'] < (200 + hits.size%50))]
        hits['remapped_pid'] = hits['remapped_pid'] - 1
        r = np.sqrt(hits['x'].values**2 + hits['y'].values**2)
        phi = np.arctan2(hits['y'].values, hits['x'].values)
        theta = np.arctan2(r,hits['z'].values)
        eta = -1*np.log(np.tan(theta/2))
        hits = hits[['track_id','x','y','z', 'index', 'particle_id', 'remapped_pid']].assign(r=r, 
                                                                                     phi=phi, 
                                                                                     eta=eta, 
                                                                                     theta = theta)

        # Remove duplicate hits
        if not self.layer_pairs_plus:
            hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin()]
        particles = particles[['particle_id','q','pt','eta','phi']]
        if self.tracking:
            return hits,particles
        else:   
            x = torch.from_numpy(hits['x'].values)
            y = torch.from_numpy(hits['y'].values)
            r = torch.from_numpy(hits['r'].values)
            phi = torch.from_numpy(hits['phi'].values)
            z = torch.from_numpy(hits['z'].values)
            theta = torch.from_numpy(hits['theta'].values)
            eta = torch.from_numpy(hits['eta'].values)
            layer = torch.from_numpy(hits['index'].values)
            particle = torch.from_numpy(hits['particle_id'].values)
            plabel = torch.from_numpy(hits['remapped_pid'].values)
            pos = torch.stack([x, y, z, r, theta, phi], 1)

            return  pos, layer, particle, eta, plabel, particles

    def split_detector_sections(self, pos, layer, particle, eta, particle_label, phi_edges, eta_edges):
        pos_sect, layer_sect, particle_sect, particle_label_sect = [], [], [], []
        # Refer to the index of the column representing phi values in pos tensor
        phi_idx = -1
        for i in range(len(phi_edges) - 1):
            phi_mask1 = pos[:,phi_idx] > phi_edges[i]
            phi_mask2 = pos[:,phi_idx] < phi_edges[i+1]
            phi_mask  = phi_mask1 & phi_mask2
            phi_pos      = pos[phi_mask]
            phi_layer    = layer[phi_mask]
            phi_particle = particle[phi_mask]
            phi_eta      = eta[phi_mask]
            phi_particle_label = particle_label[phi_mask]

            for j in range(len(eta_edges) - 1):
                eta_mask1 = phi_eta > eta_edges[j]
                eta_mask2 = phi_eta < eta_edges[j+1]
                eta_mask  = eta_mask1 & eta_mask2
                phi_eta_pos = phi_pos[eta_mask]
                phi_eta_layer = phi_layer[eta_mask]
                phi_eta_particle = phi_particle[eta_mask]
                phi_eta_particle_label = phi_particle_label[eta_mask]
                pos_sect.append(phi_eta_pos)
                layer_sect.append(phi_eta_layer)
                particle_sect.append(phi_eta_particle)
                particle_label_sect.append(phi_eta_particle_label)

        return pos_sect, layer_sect, particle_sect, particle_label_sect

    def build_tracks(self,selected_hits,hits, particles, truth):

        tensors = []

        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id']]
                .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'particle_id']]
                .merge(particles[['particle_id', 'px', 'py', 'pz']], on='particle_id'))
        hits['track_id'] = hits['hit_id'].astype(str) + "-" + hits['particle_id'].astype(str)
        
        hits = hits.merge(selected_hits[['r','phi','theta','eta','index','track_id','remapped_pid']], 
                          on = 'track_id') 

        x = torch.from_numpy(hits['x'].values)
        y = torch.from_numpy(hits['y'].values)
        theta = torch.from_numpy(hits['theta'].values)
        r = torch.from_numpy(hits['r'].values)
        phi = torch.from_numpy(hits['phi'].values)
        z = torch.from_numpy(hits['z'].values)
        eta = torch.from_numpy(hits['eta'].values)
        layer = torch.from_numpy(hits['index'].values)
        particle = torch.from_numpy(hits['particle_id'].values)
        plabel = torch.from_numpy(hits['remapped_pid'].values)
        pos = torch.stack([x, y, z, r, theta, phi], 1)   
        tensors.extend([pos, layer, particle, eta, plabel])
        
        # Keeping the tracks coords to r, phi , z 
        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        pt = torch.from_numpy(np.sqrt(hits['px'].values**2 + hits['py'].values**2))
        particle = torch.from_numpy(hits['particle_id'].values)
        layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        pt_mask = pt > self.pt_min
        mask = pt_mask

        layer = layer.unique(return_inverse=True)[1]
        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        particle = particle[mask]
        layer = layer[mask]
        particle, indices = torch.sort(particle)
        particle = particle.unique(return_inverse=True)[1]
        pos = pos[indices]
        layer = layer[indices]
        tracks = torch.empty(0,5, dtype=torch.float)

        for i in range(particle.max()+1):
            track_pos   = pos[particle == i]
            track_layer = layer[particle == i]
            track_particle = particle[particle == i]
            track_layer, indices = torch.sort(track_layer)
            track_pos = track_pos[indices]
            track_layer = track_layer[:, None]
            track_particle = track_particle[:, None]
            track = torch.cat((track_pos, track_layer.type(torch.float)), 1)
            track = torch.cat((track, track_particle.type(torch.float)), 1)
            tracks = torch.cat((tracks, track), 0)  
        tensors.append(tracks)
        return tensors
    
    def get(self,idx):
        
        hits,particles,truth = self.read_events(idx)   
        
        if not self.tracking:
            pos, layer, particle, eta, particle_label, tps = self.select_hits(hits, particles, truth)
            tracks = torch.empty(0, 5, dtype=torch.long)
        else:
            selected_hits,tps = self.select_hits(hits, particles, truth)
            pos, layer, particle, eta, particle_label, tracks = self.build_tracks(selected_hits,
                                                                                  hits, 
                                                                                  particles, 
                                                                                  truth)      
        phi_edges = np.linspace(*(-np.pi, np.pi), num=self.n_phi_sections+1)
        eta_edges = np.linspace(*self.eta_range, num=self.n_eta_sections+1)
        pos_sect, layer_sect, particle_sect, particle_label_sect = self.split_detector_sections(pos, 
                                                                                                layer,
                                                                                                particle,
                                                                                                eta,
                                                                                                particle_label, 
                                                                                                phi_edges, 
                                                                                                eta_edges)
        for i in range(len(pos_sect)):
            y = particle_label_sect[0]
            return Data(x=pos_sect[0],
                        y=y,
                        tracks=tracks,
                        inpz = torch.Tensor([i]))

def fetch_dataloader(data_dir, 
                     batch_size, 
                     validation_split,
                     n_events = 100,
                     pt_min = 1.0,
                     n_workers = 1,
                     generate_tracks = True,
                     full_dataset = False,
                     shuffle=False):
    volume_layer_ids = [
        [8, 2], [8, 4], [8, 6], [8, 8], # barrel pixels
        [7, 2], [7, 4], [7, 6], [7, 8], [7, 10], [7, 12], [7, 14],# minus pixel endcap
        [9, 2], [9, 4], [9, 6], [9, 8], [9, 10], [9, 12], [9, 14], # plus pixel endcap
    ]
    dataset = TrackMLParticleTrackingDataset(root=data_dir,
                                             layer_pairs_plus=True, 
                                             pt_min= pt_min,
                                             volume_layer_ids=volume_layer_ids,
                                             n_events=n_events, 
                                             n_workers=n_workers, 
                                             tracking = generate_tracks,
                                             download_full_dataset=full_dataset)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    print(split)
    random_seed= 1001

    train_subset, val_subset = torch.utils.data.random_split(dataset, [dataset_size - split, split],
                                                             generator=torch.Generator().manual_seed(random_seed))
    print("train subset dim:", len(train_subset))
    print("validation subset dim", len(val_subset))
    dataloaders = {
        'train':  DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle),
        'val':   DataLoader(val_subset, batch_size=batch_size, shuffle=shuffle)
        }
    print("train_dataloader dim:", len(dataloaders['train']))
    print("val dataloader dim:", len(dataloaders['val']))
    return dataloaders
