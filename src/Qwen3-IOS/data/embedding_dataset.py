"""
Dataset loader for IOS point clouds with Qwen visual embeddings and captions.

This dataset loads:
- Intraoral scan point clouds (upper + lower STL merged)
- Five intraoral photo embeddings (center, down, left, right, up)
- Text captions describing the patient's occlusion
"""

import os
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from pathlib import Path
from stl import mesh as stl_mesh


class IOSEmbeddingDataset(Dataset):
    """
    Dataset for intraoral scans with Qwen visual embeddings and clinical descriptions.
    
    For each patient, loads:
    - Merged point cloud from upper.stl and lower.stl
    - Five pre-computed Qwen visual embeddings (center, down, left, right, up views)
    - Text caption describing the patient's dental condition
    """
    
    def __init__(
        self,
        point_cloud_dir: str = "/work/grana_maxillo/IOS-DraftReport/_data/ios-iop-text",
        captions_dir: str = "/work/grana_maxillo/IOS-DraftReport/src/auto-captioning/real_captions",
        embeddings_dir: str = "/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings",
        num_points: int = 32768,
        normalize: bool = True,
        augment: bool = False,
        require_all_views: bool = True,
    ):
        """
        Args:
            point_cloud_dir: Directory containing patient folders with ios/upper.stl and ios/lower.stl
            captions_dir: Directory containing JSON files with patient descriptions
            embeddings_dir: Directory containing patient folders with visual embeddings (.pt files)
            num_points: Number of points to sample from each point cloud
            normalize: Whether to normalize point clouds to unit sphere
            augment: Whether to apply data augmentation
            require_all_views: If True, only include patients with all 5 view embeddings
        """
        self.point_cloud_dir = Path(point_cloud_dir)
        self.captions_dir = Path(captions_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.require_all_views = require_all_views
        
        # View names in the order for concatenation: [center, up, down, left, right]
        self.view_names = ["center", "up", "down", "left", "right"]
        self.samples = self._prepare_samples()
        
        print(f"Loaded {len(self.samples)} samples with embeddings")
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare list of valid samples with paths.
        
        For each patient, checks that:
        1. Caption JSON exists
        2. STL files (upper.stl and lower.stl) exist
        3. Visual embeddings exist (all 5 views if require_all_views=True)
        """
        samples = []
        
        json_files = list(self.captions_dir.glob("*.json"))
        
        for json_file in json_files:
            patient_id = json_file.stem
            
            try:
                with open(json_file, 'r') as f:
                    caption_data = json.load(f)

                    if 'ios' in caption_data:
                        desc_field = 'ios'
                    elif 'intraoral-photo' in caption_data:
                        desc_field = 'intraoral-photo'
                    else:
                        print(f"Warning: No 'ios' or 'intraoral-photo' field in {json_file}")
                        continue
                    
                    description = caption_data.get(desc_field).get('description', '').strip()
                    if not description:
                        print(f"Warning: Empty description in {json_file}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue
            
            # Check for point cloud STL files
            patient_dir = self.point_cloud_dir / patient_id
            ios_dir = patient_dir / "ios"
            upper_stl = ios_dir / "upper.stl"
            lower_stl = ios_dir / "lower.stl"
            
            if not (upper_stl.exists() and lower_stl.exists()):
                print(f"Warning: STL files not found for {patient_id} at {ios_dir}")
                continue
            
            # Check for visual embeddings
            embeddings_patient_dir = self.embeddings_dir / patient_id
            if not embeddings_patient_dir.exists():
                print(f"Warning: Embeddings directory not found for {patient_id}")
                continue
            
            # Check if all view embeddings exist
            embedding_paths = {}
            missing_views = []
            for view in self.view_names:
                emb_path = embeddings_patient_dir / f"{view}.pt"
                if emb_path.exists():
                    embedding_paths[view] = emb_path
                else:
                    missing_views.append(view)
            
            if self.require_all_views and missing_views:
                print(f"Warning: Missing embeddings for {patient_id}: {missing_views}")
                continue
            
            # If we get here, the sample is valid
            samples.append({
                'patient_id': patient_id,
                'point_cloud_path': patient_dir,
                'description': description,
                'embedding_paths': embedding_paths
            })
        
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
        if max_dist > 0:
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
        # Random rotation around Z-axis Y-axis and X-axis
        angle_z = np.random.uniform(0, np.pi / 4)
        angle_y = np.random.uniform(0, np.pi / 4)
        angle_x = np.random.uniform(0, np.pi / 4)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        
        rotation_matrix_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        rotation_matrix_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        pc = pc @ rotation_matrix_z.T
        pc = pc @ rotation_matrix_y.T
        pc = pc @ rotation_matrix_x.T
        
        jitter = np.random.normal(0, 0.001, pc.shape)
        pc = pc + jitter
        
        # Random scale
        scale = np.random.uniform(0.95, 1.05)
        pc = pc * scale
        
        return pc
    
    def _load_embeddings(self, embedding_paths: Dict[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load visual embeddings for all available views.
        
        Args:
            embedding_paths: Dict mapping view name -> path to .pt file
            
        Returns:
            Dict mapping view name -> embedding tensor
        """
        embeddings = {}
        for view_name, emb_path in embedding_paths.items():
            try:
                emb = torch.load(emb_path, map_location='cpu')
                embeddings[view_name] = emb
            except Exception as e:
                print(f"Warning: Could not load embedding {emb_path}: {e}")
        
        return embeddings
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Get a single sample.

        Returns:
            dict with keys:
                - point_cloud: (num_points, 3) tensor
                - embeddings: dict with keys 'center', 'down', 'left', 'right', 'up'
                              each mapping to a tensor of shape (seq_len, hidden_dim)
                - description: str
                - patient_id: str
                - instruction: str (formatted instruction text)
        """
        sample = self.samples[idx]

        pc = self._load_point_cloud(sample['point_cloud_path'])

        if pc.shape[1] != 3:
            raise ValueError(f"Point cloud must have 3 dimensions (XYZ): {pc.shape}")

        if self.normalize:
            pc = self._normalize_point_cloud(pc)

        pc = self._sample_points(pc)

        if self.augment:
            pc = self._augment_point_cloud(pc)

        pc_tensor = torch.from_numpy(pc).float()
        embeddings = self._load_embeddings(sample['embedding_paths'])

        return {
            'point_cloud': pc_tensor,
            'embeddings': embeddings,  # Dict of tensors
            'description': sample['description'],
            'patient_id': sample['patient_id'],
        }


class IOSEmbeddingCollator:
    """
    Collate function for batching IOS samples with embeddings.
    """
    
    def __init__(self, processor, view_names: Optional[List[str]] = None):
        """
        Args:
            processor: Hugging Face processor for tokenization
            view_names: List of view names (default: ["center", "up", "down", "left", "right"])
        """
        self.processor = processor
        self.view_names = view_names or ["center", "up", "down", "left", "right"]
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        Args:
            batch: List of dicts from IOSEmbeddingDataset
        
        Returns:
            Batched dict with:
                - point_clouds: (B, N, 3) tensor
                - embeddings: dict with keys for each view, values are (B, seq_len, hidden_dim) tensors
                              (Note: seq_len may vary per view; may need padding)
                - input_ids: (B, L) tensor
                - attention_mask: (B, L) tensor
                - labels: (B, L) tensor
                - patient_ids: List[str]
                - instructions: List[str]
                - descriptions: List[str]
        """
        # Stack point clouds
        point_clouds = torch.stack([item['point_cloud'] for item in batch])
        
        # Collate embeddings for each view
        # Each view may have different sequence lengths, so we'll pad them
        embeddings_batched = {}
        for view_name in self.view_names:
            view_embeddings = []
            for item in batch:
                if view_name in item['embeddings']:
                    view_embeddings.append(item['embeddings'][view_name])
                else:
                    # If view is missing, use zero tensor as placeholder
                    # We'll need to know the hidden_dim from other samples
                    view_embeddings.append(None)
            
            # Filter out None values and pad
            valid_embeddings = [e for e in view_embeddings if e is not None]
            if valid_embeddings:
                # Pad to max seq_len in this batch for this view
                max_seq_len = max(e.shape[0] for e in valid_embeddings)
                hidden_dim = valid_embeddings[0].shape[1]
                
                padded_embeddings = []
                for e in view_embeddings:
                    if e is None:
                        # Create zero tensor
                        padded_embeddings.append(torch.zeros(max_seq_len, hidden_dim))
                    else:
                        if e.shape[0] < max_seq_len:
                            # Pad
                            padding = torch.zeros(max_seq_len - e.shape[0], hidden_dim)
                            padded_embeddings.append(torch.cat([e, padding], dim=0))
                        else:
                            padded_embeddings.append(e)
                
                embeddings_batched[view_name] = torch.stack(padded_embeddings)
            else:
                # All samples missing this view
                embeddings_batched[view_name] = None
        
        # Prepare text inputs
        instructions = [item['instruction'] for item in batch]
        descriptions = [item['description'] for item in batch]
        
        # Format descriptions (shuffle fields as augmentation)
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
            field_value_pairs = []
            for line in desc_content.split('\n'):
                line = line.strip()
                if ':' in line and line:
                    field_value_pairs.append(line)

            # Shuffle for augmentation
            random.shuffle(field_value_pairs)
            
            field_names = [x.split(":")[0].strip() for x in field_value_pairs]
            field_names_list = ', '.join(field_names) + "."
            
            field_names_lists.append(field_names_list)
            formatted_descriptions.append('\n'.join(field_value_pairs) + "<|im_end|>")
        
        # Replace placeholder in instructions
        instructions = [
            inst.replace("<|fields_name|>", field_names_lists[i])
            for i, inst in enumerate(instructions)
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

        # Create labels (mask instruction, keep description)
        labels = input_ids.clone()
        for i in range(len(batch)):
            instr_len = instr_tokenized['input_ids'].shape[1]
            labels[i, :instr_len] = -100  # Mask instruction tokens
            labels[i, instr_len:] = input_ids[i, instr_len:]
        
        return {
            'point_clouds': point_clouds,
            'embeddings': embeddings_batched,  # Dict of (B, seq_len, hidden_dim) tensors
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patient_ids': [item['patient_id'] for item in batch],
            'instructions': instructions,
            'descriptions': formatted_descriptions
        }


def get_embedding_dataloader(
    processor,
    point_cloud_dir: str = "/work/grana_maxillo/IOS-DraftReport/_data/ios-iop-text",
    captions_dir: str = "/work/grana_maxillo/IOS-DraftReport/src/auto-captioning/real_captions",
    embeddings_dir: str = "/work/grana_maxillo/IOS-DraftReport/_data/iop_qwen_embeddings",
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for training with embeddings.
    
    Args:
        processor: Hugging Face processor for tokenization
        point_cloud_dir: Directory with patient folders containing STL files
        captions_dir: Directory with JSON caption files
        embeddings_dir: Directory with patient folders containing embedding .pt files
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for IOSEmbeddingDataset
    
    Returns:
        DataLoader
    """
    dataset = IOSEmbeddingDataset(
        point_cloud_dir=point_cloud_dir,
        captions_dir=captions_dir,
        embeddings_dir=embeddings_dir,
        **dataset_kwargs
    )
    
    collator = IOSEmbeddingCollator(processor)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return dataloader
