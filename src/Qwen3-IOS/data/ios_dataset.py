"""
Dataset loader for IOS point clouds with captions.
"""

import os
import json
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from stl import mesh as stl_mesh


class IOSPointCloudDataset(Dataset):
    """
    Dataset for intraoral scan point clouds with clinical descriptions.
    
    Loads point clouds from preprocessed dataset and pairs them with
    auto-generated captions.
    """
    
    def __init__(
        self,
        point_cloud_dir: str,
        captions_dir: str,
        num_points: int = 32768,
        normalize: bool = True,
        augment: bool = False,
        instruction_template_path: Optional[str] = None,
    ):
        """
        Args:
            point_cloud_dir: Directory containing patient folders with ios/upper.stl and ios/lower.stl
            captions_dir: Directory containing JSON files with patient descriptions (one JSON per patient)
            num_points: Number of points to sample from each point cloud
            normalize: Whether to normalize point clouds to unit sphere
            augment: Whether to apply data augmentation
            instruction_template: Template for instruction-following format
        """
        self.point_cloud_dir = Path(point_cloud_dir)
        self.captions_dir = Path(captions_dir)
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        
        # Default instruction template
        if instruction_template_path is None:
            self.instruction_template = (
                "You are a dental expert analyzing a 3D intraoral scan. "
                "Describe the patient's occlusion in detail.\n\n"
                "Description:"
            )
        else:
            with open(instruction_template_path, 'r') as f:
                self.instruction_template = f.read()
        
        # Identify positions of '{}' in the template
        self.loss_mask_template = [1 if '{}' in token else 0 for token in self.instruction_template.split()]
        
        # Find all valid samples
        self.samples = self._prepare_samples()
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare list of valid samples with paths.
        
        Scans the captions directory for JSON files and matches them
        with patient directories containing STL files.
        """
        samples = []
        
        # Get all JSON files from captions directory
        json_files = list(self.captions_dir.glob("*.json"))
        
        for json_file in json_files:
            patient_id = json_file.stem  # Filename without .json extension
            
            # Load description from JSON
            try:
                with open(json_file, 'r') as f:
                    caption_data = json.load(f)
                    if 'ios' in caption_data:
                        desc_field = 'ios'
                    elif 'intraoral-photo' in caption_data:
                        desc_field = 'intraoral-photo'
                    else:
                        raise ValueError(f"Warning: No 'ios' or 'intraoral-photo' field in {json_file}")

                    description = caption_data.get(desc_field).get('description').strip()
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue
            
            # Look for point cloud directory
            patient_dir = self.point_cloud_dir / patient_id
            ios_dir = patient_dir / "ios"
            upper_stl = ios_dir / "upper.stl"
            lower_stl = ios_dir / "lower.stl"
            
            if upper_stl.exists() and lower_stl.exists():
                samples.append({
                    'patient_id': patient_id,
                    'point_cloud_path': patient_dir,
                    'description': description
                })
            else:
                print(f"Warning: STL files not found for {patient_id} at {ios_dir}")
        
        return samples
    
    def _load_point_cloud(self, patient_dir: Path) -> np.ndarray:
        """
        Load point cloud from STL files.
        
        Loads both upper.stl and lower.stl from patient_dir/ios/
        and combines them into a single point cloud.
        """
        ios_dir = patient_dir / "ios"
        upper_stl_path = ios_dir / "upper.stl"
        lower_stl_path = ios_dir / "lower.stl"
        
        # Load upper STL
        upper_mesh = stl_mesh.Mesh.from_file(str(upper_stl_path))
        upper_points = upper_mesh.vectors.reshape(-1, 3)  # Flatten triangles to points
        
        # Load lower STL
        lower_mesh = stl_mesh.Mesh.from_file(str(lower_stl_path))
        lower_points = lower_mesh.vectors.reshape(-1, 3)
        
        # Combine upper and lower
        pc = np.vstack([upper_points, lower_points])
        
        # Remove duplicate vertices (STL files have duplicates at triangle edges)
        pc = np.unique(pc, axis=0)
        
        return pc
    
    def _normalize_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere."""
        # Center
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / max_dist
        
        return pc
    
    def _sample_points(self, pc: np.ndarray) -> np.ndarray:
        """Sample fixed number of points."""
        num_available = pc.shape[0]
        
        if num_available >= self.num_points:
            # Random sampling
            indices = np.random.choice(num_available, self.num_points, replace=False)
        else:
            # Upsample with replacement
            indices = np.random.choice(num_available, self.num_points, replace=True)
        
        return pc[indices]
    
    def _augment_point_cloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random rotation around Z Y and X
        angle_z = np.random.uniform(-np.pi//6, np.pi//6)
        angle_y = np.random.uniform(-np.pi//6, np.pi//6)
        angle_x = np.random.uniform(-np.pi//6, np.pi//6)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        rotation_matrix = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ]) @ np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ]) @ np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        pc = pc @ rotation_matrix.T
        
        # Random jitter
        jitter = np.random.normal(-0.01, 0.01, pc.shape)
        pc = pc + jitter
        
        # Random translation
        translation = np.random.uniform(-0.1, 0.1, (1, 3))
        pc = pc + translation
        
        # Random scale
        scale = np.random.uniform(0.95, 1.05)
        pc = pc * scale
        
        return pc
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Get a single sample.

        Returns:
            dict with keys:
                - point_cloud: (num_points, 3) tensor
                - description: str
                - patient_id: str
                - instruction: str (formatted instruction text)
        """
        sample = self.samples[idx]

        # Load point cloud (upper + lower STL files combined)
        pc = self._load_point_cloud(sample['point_cloud_path'])

        # Point cloud is already (N, 3) from STL loading
        if pc.shape[1] != 3:
            raise ValueError(f"Point cloud must have 3 dimensions (XYZ): {pc.shape}")

        # Normalize
        if self.normalize:
            pc = self._normalize_point_cloud(pc)

        # Sample points
        pc = self._sample_points(pc)

        # Augment
        if self.augment:
            pc = self._augment_point_cloud(pc)

        # Convert to tensor
        pc_tensor = torch.from_numpy(pc).float()

        instruction = self.instruction_template

        # Remove loss_mask from the dataset
        return {
            'point_cloud': pc_tensor,
            'description': sample['description'],
            'patient_id': sample['patient_id'],
            'instruction': instruction
        }


class IOSCollator:
    """
    Collate function for batching IOS samples.
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: Hugging Face processor for tokenization
        """
        self.processor = processor
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        Args:
            batch: List of dicts from IOSPointCloudDataset
        
        Returns:
            Batched dict with:
                - point_clouds: (B, N, 3) tensor
                - input_ids: (B, L) tensor
                - attention_mask: (B, L) tensor
                - labels: (B, L) tensor
                - patient_ids: List[str]
                - instructions: List[str]
                - descriptions: List[str]
        """
        # Stack point clouds
        point_clouds = torch.stack([item['point_cloud'] for item in batch])
        
        # Prepare text inputs
        instructions = [item['instruction'] for item in batch]
        descriptions = [item['description'] for item in batch]
        
        # Combine instruction (which now includes template) and description
        # Parse description to extract field: value pairs and shuffle as augmentation
        formatted_descriptions = []
        field_names_lists = []
        for desc in descriptions:
            desc_stripped = desc.strip()
            
            # Remove header if present in description
            if desc_stripped.startswith("Patient Intra-Oral Description:"):
                desc_content = desc_stripped.replace("Patient Intra-Oral Description:", "", 1).strip()
            else:
                desc_content = desc_stripped
            
            # Parse the description into field: value pairs
            # This handles the structured format from the ground truth
            field_value_pairs = []
            for line in desc_content.split('\n'):
                line = line.strip()
                if ':' in line and line and 'missing teeth' not in line.lower():
                    # Keep the entire "field: value" format
                    field_value_pairs.append(line)

            # random.shuffle(field_value_pairs)
            random_field_to_predict = random.choice(field_value_pairs)
            field_name = random_field_to_predict.split(":")[0].strip()
            field_value = random_field_to_predict.split(":")[1].strip()
            
            formatted_descriptions.append(f"{field_value}<|im_end|>")
        
        instructions = [
            inst.replace("<|field_to_predict|>", field_name)
            for inst in instructions
        ]
        
        # Tokenize
        instr_tokenized = self.processor(
            text=instructions,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True
        )
        
        response_tokenized = self.processor(
            text=formatted_descriptions,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True
        )
        
        input_ids = torch.cat(
            [instr_tokenized['input_ids'], response_tokenized['input_ids']], dim=1
        )
        attention_mask = torch.cat(
            [instr_tokenized['attention_mask'], response_tokenized['attention_mask']], dim=1
        )

        labels = input_ids.clone()
        for i in range(len(batch)):
            instr_len = instr_tokenized['input_ids'].shape[1]
            labels[i, :instr_len] = -100  # Mask instruction tokens
            labels[i, instr_len:] = input_ids[i, instr_len:]
        
        return {
            'point_clouds': point_clouds,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patient_ids': [item['patient_id'] for item in batch],
            'instructions': instructions,
            'descriptions': formatted_descriptions
        }


def get_dataloader(
    point_cloud_dir: str,
    captions_dir: str,
    processor,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for training.
    
    Args:
        point_cloud_dir: Directory with patient folders containing STL files
        captions_dir: Directory with JSON caption files
        processor: Hugging Face processor
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for IOSPointCloudDataset
    
    Returns:
        DataLoader
    """
    dataset = IOSPointCloudDataset(
        point_cloud_dir=point_cloud_dir,
        captions_dir=captions_dir,
        **dataset_kwargs
    )
    
    collator = IOSCollator(processor)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return dataloader
