'''
3DTeethLand Dataset for Point-BERT
Loads intraoral scans (IOS) from STL files
'''
import os
import numpy as np
import torch
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from pytorch3d.ops import sample_farthest_points

def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """Farthest Point Sampling using PyTorch3D for efficiency"""
    point = torch.from_numpy(point).unsqueeze(0)  # (1, N, 3)
    sampled_points, _ = sample_farthest_points(point, None, npoint)
    return sampled_points.squeeze(0).numpy()

@DATASETS.register_module()
class TeethLand3D(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.uniform = config.get('uniform', True)
        self.preload = config.get('preload', False)  # Preload all STL files into memory
        
        # Initialize cache for STL files
        self.stl_cache = {}
        
        print_log(f'[DATASET] Loading 3DTeethLand dataset from {self.data_root}', logger='3DTeethLand')
        print_log(f'[DATASET] Subset: {self.subset}', logger='3DTeethLand')
        print_log(f'[DATASET] Sample out {self.sample_points_num} points', logger='3DTeethLand')
        print_log(f'[DATASET] Preload mode: {self.preload}', logger='3DTeethLand')
        
        # Get all sample directories
        all_samples = []
        if os.path.exists(self.data_root):
            for sample_id in os.listdir(self.data_root):
                sample_path = os.path.join(self.data_root, sample_id)
                if os.path.isdir(sample_path):
                    stl_lower_file = os.path.join(sample_path, 'stl/ios_lower_processed.stl')
                    stl_upper_file = os.path.join(sample_path, 'stl/ios_upper_processed.stl')
                    if os.path.exists(stl_lower_file):
                        all_samples.append({
                            'sample_id': sample_id,
                            'file_path': stl_lower_file
                        })
                    if os.path.exists(stl_upper_file):
                        all_samples.append({
                            'sample_id': sample_id,
                            'file_path': stl_upper_file
                        })

        all_samples.sort(key=lambda x: x['sample_id'])
        
        # Split into train/test (80/20 split)
        n_total = len(all_samples)
        n_train = int(0.8 * n_total)
        
        if self.subset == 'train':
            self.file_list = all_samples[:n_train]
        elif self.subset == 'test':
            self.file_list = all_samples[n_train:]
        else:
            # Use all samples if subset not specified
            self.file_list = all_samples
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='3DTeethLand')
        
        # Preload all STL files into memory if enabled
        if self.preload:
            print_log('[DATASET] Preloading all STL files into memory...', logger='3DTeethLand')
            for i, sample in enumerate(self.file_list):
                file_path = sample['file_path']
                if i % 50 == 0:
                    print_log(f'[DATASET] Preloading {i}/{len(self.file_list)}...', logger='3DTeethLand')
                self.stl_cache[file_path] = IO.get(file_path).astype(np.float32)
            print_log(f'[DATASET] Preloading complete! Loaded {len(self.stl_cache)} files.', logger='3DTeethLand')
        
        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """Normalize point cloud: center and scale to unit sphere"""
        return pc_normalize(pc)

    def random_sample(self, pc, num):
        """Randomly sample points from point cloud"""
        if len(pc) < num:
            # If we have fewer points than needed, sample with replacement
            indices = np.random.choice(len(pc), num, replace=True)
            pc = pc[indices]
        else:
            # Otherwise sample without replacement
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        
        # Check cache first
        file_path = sample['file_path']
        if file_path in self.stl_cache:
            # Use cached data
            data = self.stl_cache[file_path].copy()
        else:
            # Load STL file and cache it
            data = IO.get(file_path).astype(np.float32)
            if "upper" in file_path:
                data[:, 2] = -data[:, 2]
            # self.stl_cache[file_path] = data.copy()
        
        # Sample points uniformly
        if self.uniform:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        
        # Normalize
        # data = self.pc_norm(data)
        data = data - np.mean(data, axis=0)
        
        # Shuffle points for training
        if self.subset == 'train':
            pt_idxs = np.arange(0, data.shape[0])
            np.random.shuffle(pt_idxs)
            data = data[pt_idxs].copy()
        
        data = torch.from_numpy(data).float()
        
        # Return format compatible with Point-BERT
        # (dataset_name, sample_id, data)
        return '3DTeethLand', sample['sample_id'], data

    def __len__(self):
        return len(self.file_list)
