'''
FerraraDump Dataset for Point-BERT
Loads intraoral scans (IOS) from STL files
Dataset structure: each patient folder contains ios/upper.stl and ios/lower.stl
'''
import os
import numpy as np
import torch
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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
class FerraraDump(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.uniform = config.get('uniform', True)
        self.preload = config.get('preload', True)  # Preload all STL files into memory
        self.arch_type = config.get('arch_type', 'both')  # 'upper', 'lower', or 'both'
        
        # Initialize cache for STL files
        self.stl_cache = {}
        
        print_log(f'[DATASET] Loading FerraraDump dataset from {self.data_root}', logger='FerraraDump')
        print_log(f'[DATASET] Subset: {self.subset}', logger='FerraraDump')
        print_log(f'[DATASET] Arch type: {self.arch_type}', logger='FerraraDump')
        print_log(f'[DATASET] Sample out {self.sample_points_num} points', logger='FerraraDump')
        print_log(f'[DATASET] Preload mode: {self.preload}', logger='FerraraDump')
        
        # Get all sample directories
        all_samples = []
        if os.path.exists(self.data_root):
            for sample_id in os.listdir(self.data_root):
                sample_path = os.path.join(self.data_root, sample_id)
                if os.path.isdir(sample_path):
                    ios_dir = os.path.join(sample_path, 'ios')
                    if os.path.exists(ios_dir):
                        # Add upper arch if needed
                        if self.arch_type in ['upper', 'both']:
                            upper_file = os.path.join(ios_dir, 'upper.stl')
                            if os.path.exists(upper_file):
                                all_samples.append({
                                    'sample_id': f"{sample_id}_upper",
                                    'file_path': upper_file
                                })
                        
                        # Add lower arch if needed
                        if self.arch_type in ['lower', 'both']:
                            lower_file = os.path.join(ios_dir, 'lower.stl')
                            if os.path.exists(lower_file):
                                all_samples.append({
                                    'sample_id': f"{sample_id}_lower",
                                    'file_path': lower_file
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
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='FerraraDump')
        
        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """Normalize point cloud: center and scale to unit sphere"""
        return pc_normalize(pc)
    
    def augmentation(self, pc):
        """Apply data augmentation: random rotation and jittering"""
        # Random rotation around z-axis from -pi/6 to pi/6
        # Random rotation around x and y axes from -pi/18 to pi/18
        z_rot = np.random.uniform(-np.pi/6, np.pi/6)
        y_rot = np.random.uniform(-np.pi/18, np.pi/18)
        x_rot = np.random.uniform(-np.pi/18, np.pi/18)
        rotation_matrix = np.array([
            [np.cos(z_rot)*np.cos(y_rot), 
             np.cos(z_rot)*np.sin(y_rot)*np.sin(x_rot)-np.sin(z_rot)*np.cos(x_rot), 
             np.cos(z_rot)*np.sin(y_rot)*np.cos(x_rot)+np.sin(z_rot)*np.sin(x_rot)],
            [np.sin(z_rot)*np.cos(y_rot), 
             np.sin(z_rot)*np.sin(y_rot)*np.sin(x_rot)+np.cos(z_rot)*np.cos(x_rot), 
             np.sin(z_rot)*np.sin(y_rot)*np.cos(x_rot)-np.cos(z_rot)*np.sin(x_rot)],
            [-np.sin(y_rot), 
             np.cos(y_rot)*np.sin(x_rot), 
             np.cos(y_rot)*np.cos(x_rot)]
        ])
        pc = pc.dot(rotation_matrix)
        
        # Random center displacement
        displacement = np.random.uniform(-10, 10, size=(1, 3))
        pc += displacement

        return pc

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
        arch_type = 'upper' if 'upper' in sample['sample_id'] else 'lower'
        if file_path in self.stl_cache:
            # Use cached data
            data = self.stl_cache[file_path].copy()
        else:
            # Load STL file and cache it
            data = IO.get(file_path).astype(np.float32)
            if arch_type == "upper":
                data[:, 2] = -data[:, 2]
            self.stl_cache[file_path] = data.copy()
        
        # Sample points uniformly
        if self.uniform:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        
        # Normalize
        # data = self.pc_norm(data)
        data = data - np.mean(data, axis=0)
        
        # Augmentation
        if self.subset == 'train':
            data = self.augmentation(data)
        
        # Shuffle points for training
        if self.subset == 'train':
            pt_idxs = np.arange(0, data.shape[0])
            np.random.shuffle(pt_idxs)
            data = data[pt_idxs].copy()
        
        data = torch.from_numpy(data).float()
        
        # Return format compatible with Point-BERT
        # (dataset_name, sample_id, data)
        return 'FerraraDump', sample['sample_id'], data

    def __len__(self):
        return len(self.file_list)
