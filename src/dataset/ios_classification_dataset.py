import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

class IOSClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        classification_path: str,
        schema_path: str,
        augment: bool = True,
        num_points: int = 32768,
        split: str = 'train',
        fold: int = 0,
        num_folds: int = 5
    ):
        self.dataset_path = Path(dataset_path)
        self.classification_path = Path(classification_path)
        self.augment = augment
        self.num_points = num_points
        self.split = split
        self.fold = fold
        self.num_folds = num_folds
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        self.class_to_idx = {
            task: {val: idx for idx, val in enumerate(values)}
            for task, values in self.schema.items()
        }
        self.tasks = list(self.schema.keys())
        self.num_classes_per_task = {task: len(values) for task, values in self.schema.items()}
        
        self.samples = []
        self._load_all_data()
    
    def _load_all_data(self):
        classification_files = sorted(self.classification_path.glob("*.json"))
        num_files = len(classification_files)
        
        # K-fold cross validation split
        fold_size = num_files // self.num_folds
        val_start = self.fold * fold_size
        val_end = val_start + fold_size if self.fold < self.num_folds - 1 else num_files
        
        if self.split == 'train':
            # Training: all files except the validation fold
            classification_files = classification_files[:val_start] + classification_files[val_end:]
        else:  # validation
            # Validation: only the current fold
            classification_files = classification_files[val_start:val_end]
        
        print(f"Fold {self.fold}/{self.num_folds}: Loading {len(classification_files)} {self.split} samples")
            
        for clf_file in tqdm(classification_files):
            patient_id = clf_file.stem
            patient_dir = self.dataset_path / patient_id
            
            upper_stl = patient_dir / "ios" / "upper.stl"
            lower_stl = patient_dir / "ios" / "lower.stl"
            
            if not upper_stl.exists() or not lower_stl.exists():
                continue
            
            with open(clf_file, 'r') as f:
                data = json.load(f)
            
            if "ios" not in data or "classification" not in data["ios"]:
                continue
            
            classification = data["ios"]["classification"]
            
            upper_mesh = trimesh.load(str(upper_stl), process=False, force='mesh')
            lower_mesh = trimesh.load(str(lower_stl), process=False, force='mesh')
            
            combined_points = np.vstack([
                np.asarray(upper_mesh.vertices),
                np.asarray(lower_mesh.vertices)
            ])
            
            labels = self._encode_classification(classification)
            
            self.samples.append({
                "patient_id": patient_id,
                "points": combined_points.astype(np.float32),
                "labels": labels
            })
    
    def _encode_classification(self, classification: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        labels = {}
        for task in self.tasks:
            num_classes = self.num_classes_per_task[task]
            
            if task in classification and len(classification[task]) > 0:
                label_vector = torch.zeros(num_classes, dtype=torch.float32)
                for value in classification[task]:
                    if value in self.class_to_idx[task]:
                        idx = self.class_to_idx[task][value]
                        label_vector[idx] = 1.0
                labels[task] = label_vector
            else:
                labels[task] = torch.full((num_classes,), -1.0, dtype=torch.float32)
        
        return labels
    
    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        replace = len(points) < self.num_points
        indices = np.random.choice(len(points), self.num_points, replace=replace)
        return points[indices]
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        angles = np.random.uniform(-45, 45, size=3)
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        points = rotation.apply(points)
        np.random.shuffle(points)
        return points
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        sample = self.samples[idx]
        points = sample["points"].copy()
        
        points = self._sample_points(points)
        
        if self.augment:
            points = self._augment_points(points)
        
        points = torch.from_numpy(points).float()
        
        return points, sample["labels"], sample["patient_id"]
