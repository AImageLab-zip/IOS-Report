'''
PreprocessedIOS Dataset for Point-BERT
Loads preprocessed point clouds from .ply files
Dataset structure: flat directory with {prefix}_{id}_{arch}_{sample}.ply files
'''
import os
import numpy as np
import torch
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *
import open3d as o3d


@DATASETS.register_module()
class PreprocessedIOS(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.preload = config.get('preload', False)  # Preload all point clouds into memory
        self.arch_type = config.get('arch_type', 'both')  # 'upper', 'lower', or 'both'
        
        # Initialize cache for point clouds
        self.pc_cache = {}
        
        print_log(f'[DATASET] Loading PreprocessedIOS dataset from {self.data_root}', logger='PreprocessedIOS')
        print_log(f'[DATASET] Subset: {self.subset}', logger='PreprocessedIOS')
        print_log(f'[DATASET] Arch type: {self.arch_type}', logger='PreprocessedIOS')
        print_log(f'[DATASET] Sample out {self.sample_points_num} points', logger='PreprocessedIOS')
        print_log(f'[DATASET] Preload mode: {self.preload}', logger='PreprocessedIOS')
        
        # Get all .ply files from flat directory structure
        all_samples = []
        if os.path.exists(self.data_root):
            for ply_file in sorted(os.listdir(self.data_root)):
                if ply_file.endswith('.ply'):
                    # Parse filename: {prefix}_{id}_{arch}_{sample}.ply
                    # Examples: 3DTeeth_01328DDN_3747_upper_000.ply, Ferrara_201_1979_lower_002.ply
                    parts = ply_file[:-4].split('_')  # Remove .ply and split
                    if len(parts) >= 3:
                        # Find arch type (upper or lower)
                        arch = None
                        for part in parts:
                            if part in ['upper', 'lower']:
                                arch = part
                                break
                        
                        if arch and self.arch_type in [arch, 'both']:
                            file_path = os.path.join(self.data_root, ply_file)
                            all_samples.append({
                                'sample_id': ply_file[:-4],  # Remove .ply extension
                                'file_path': file_path,
                                'arch': arch
                            })
        
        all_samples.sort(key=lambda x: x['sample_id'])
        
        # Split into train/test (95/5 split)
        n_total = len(all_samples)
        n_train = int(0.95 * n_total)
        
        if self.subset == 'train':
            self.file_list = all_samples[:n_train]
        elif self.subset == 'test':
            self.file_list = all_samples[n_train:]
        else:
            # Use all samples if subset not specified
            self.file_list = all_samples
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='PreprocessedIOS')
        
        # Preload all point clouds into memory if enabled
        if self.preload:
            print_log('[DATASET] Preloading all point clouds into memory...', logger='PreprocessedIOS')
            for i, sample in enumerate(self.file_list):
                file_path = sample['file_path']
                if i % 100 == 0:
                    print_log(f'[DATASET] Preloading {i}/{len(self.file_list)}...', logger='PreprocessedIOS')
                # Load PLY file
                pcd = o3d.io.read_point_cloud(file_path)
                self.pc_cache[file_path] = np.asarray(pcd.points, dtype=np.float32)
            print_log(f'[DATASET] Preloading complete! Loaded {len(self.pc_cache)} files.', logger='PreprocessedIOS')
        
        self.permutation = np.arange(self.npoints)

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
    
    def augmentation(self, pc):
        """Apply data augmentation: random rotation and jittering"""
        z_rot = np.random.uniform(-np.pi/6, np.pi/6)
        y_rot = np.random.uniform(-np.pi/12, np.pi/12)
        x_rot = np.random.uniform(-np.pi/12, np.pi/12)
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

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        
        # Check cache first
        file_path = sample['file_path']
        if file_path in self.pc_cache:
            # Use cached data
            data = self.pc_cache[file_path].copy()
        else:
            # Load .ply file
            pcd = o3d.io.read_point_cloud(file_path)
            data = np.asarray(pcd.points, dtype=np.float32)
        
        # Subsample if needed
        if len(data) > self.sample_points_num:
            data = self.random_sample(data, self.sample_points_num)
        
        # Augmentation for training
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
        return 'PreprocessedIOS', sample['sample_id'], data

    def __len__(self):
        return len(self.file_list)
