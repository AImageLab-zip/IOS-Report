"""
Ferrara Dump Dataset Loader

Efficient dataset loader for the Ferrara Dump ToothFairy4M dataset.
Each patient has intraoral photos (5 JPGs) and IOS scans (upper.stl and lower.stl).
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import trimesh

def _load_stl_as_pointcloud_static(stl_path: Path, n_points: int) -> np.ndarray:
    """
    Static function to load an STL file and convert it to a point cloud.
    Used by multiprocessing worker.
    
    Args:
        stl_path: Path to the STL file
        n_points: Number of points to sample
        
    Returns:
        Point cloud as numpy array of shape (n_points, 3)
    """
    mesh = trimesh.load_mesh(str(stl_path))
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)


def _load_images_static(photos_dir: Path, image_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    Static function to load all intraoral photos from a patient's folder.
    Used by multiprocessing worker. Returns numpy arrays instead of PIL Images for serialization.
    
    Args:
        photos_dir: Path to the intraoral-photos directory
        image_size: Optional (width, height) tuple to resize images
        
    Returns:
        List of image numpy arrays
    """
    images = []
    
    for img_name in sorted(photos_dir.glob("*.jpg")):
        img = Image.open(img_name).convert('RGB')
        
        if image_size is not None:
            img = img.resize(image_size, Image.BILINEAR)
        
        # Convert to numpy for serialization
        img_array = np.array(img)
        images.append(img_array)
    
    return images


def _load_patient_data_worker(args: Tuple) -> Tuple[int, Dict]:
    """
    Worker function for multiprocessing loading.
    This needs to be a module-level function for pickling.
    
    Args:
        args: Tuple of (idx, root_dir, patient_id, n_points, load_meshes, load_images, image_size, load_captions, captions_dir)
        
    Returns:
        Tuple of (idx, data_dict)
    """
    idx, root_dir, patient_id, n_points, load_meshes, load_images, image_size, load_captions, captions_dir = args
    
    patient_dir = Path(root_dir) / patient_id
    
    data = {
        'patient_id': patient_id,
        'index': idx,
    }
    
    # Load meshes/point clouds
    if load_meshes:
        ios_dir = patient_dir / "ios"
        upper_points = _load_stl_as_pointcloud_static(ios_dir / "upper.stl", n_points)
        lower_points = _load_stl_as_pointcloud_static(ios_dir / "lower.stl", n_points)
        
        combined_points = np.vstack([upper_points, lower_points])
        
        data['upper_pointcloud'] = upper_points
        data['lower_pointcloud'] = lower_points
        data['combined_pointcloud'] = combined_points
    
    # Load images
    if load_images:
        photos_dir = patient_dir / "intraoral-photos"
        images = _load_images_static(photos_dir, image_size)
        data['images'] = images  # Store as numpy arrays
        data['num_images'] = len(images)
    
    # Load captions
    if load_captions and captions_dir is not None:
        caption_path = Path(captions_dir) / f"{patient_id}.json"
        if caption_path.exists():
            try:
                with open(caption_path, 'r') as f:
                    caption_data = json.load(f)
                    # Extract description from ios key
                    if 'ios' in caption_data and 'description' in caption_data['ios']:
                        data['report'] = caption_data['ios']['description']
                    else:
                        data['report'] = None
            except Exception as e:
                print(f"Warning: Could not load caption for {patient_id}: {e}")
                data['report'] = None
        else:
            data['report'] = None
    
    return (idx, data)


class FerraraDumpDataset(Dataset):
    """
    Dataset loader for Ferrara Dump ToothFairy4M dataset.
    
    Features:
    - Efficient RAM caching for fast repeated access
    - Loads STL files as point clouds
    - Loads all 5 intraoral photos per patient
    - Lazy loading or full preloading options
    
    Dataset Structure:
    - Root directory contains patient folders (e.g., 201_1979)
    - Each patient folder contains:
        - ios/upper.stl and ios/lower.stl
        - intraoral-photos/a.jpg, b.jpg, c.jpg, d.jpg, e.jpg
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        preload: bool = False,
        n_points: int = 10000,
        cache_in_ram: bool = False,
        load_images: bool = True,
        load_meshes: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
        return_tensor: bool = True,
        num_workers: int = 8,
        load_captions: bool = True,
        captions_dir: Optional[Union[str, Path]] = "_data/captions",
    ):
        """
        Initialize the Ferrara Dump dataset.
        
        Args:
            root_dir: Root directory of the dataset
            preload: If True, load all data into RAM on initialization
            n_points: Number of points to sample from STL meshes
            cache_in_ram: If True, cache loaded data in RAM for faster access
            load_images: If True, load intraoral photos
            load_meshes: If True, load IOS meshes
            image_size: Optional (width, height) tuple to resize images
            return_tensor: If True, return PyTorch tensors, else numpy arrays
            num_workers: Number of parallel workers for preloading (default: 4, use 0 for single-threaded)
            load_captions: If True, load captions/reports for each patient
            captions_dir: Directory containing caption JSON files (required if load_captions=True)
        """
        self.root_dir = Path(root_dir)
        self.preload = preload
        self.n_points = n_points
        self.cache_in_ram = cache_in_ram
        self.load_images = load_images
        self.load_meshes = load_meshes
        self.image_size = image_size
        self.return_tensor = return_tensor
        self.num_workers = max(0, min(num_workers, cpu_count()))
        self.load_captions = load_captions
        self.captions_dir = Path(captions_dir) if captions_dir is not None else None
        
        # Validate captions_dir if load_captions is True
        if self.load_captions and self.captions_dir is None:
            raise ValueError("captions_dir must be provided when load_captions=True")
        
        # Cache storage
        self._cache: Dict[int, Dict] = {}
        
        # Scan dataset and build patient list
        print(f"Scanning dataset at {self.root_dir}")
        self.patient_ids = self._scan_dataset()
        print(f"Found {len(self.patient_ids)} patients")
        
        # Preload all data if requested
        if self.preload:
            print("Preloading all data into RAM...")
            start_time = time.time()
            self._preload_all()
            elapsed = time.time() - start_time
            print(f"Preloading completed in {elapsed:.2f} seconds")
    
    def _scan_dataset(self) -> List[str]:
        """
        Scan the dataset directory and collect all valid patient IDs.
        
        Returns:
            List of patient IDs (folder names)
        """
        patient_ids = []
        
        if not self.root_dir.exists():
            raise ValueError(f"Dataset root directory does not exist: {self.root_dir}")
        
        for patient_folder in sorted(self.root_dir.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            # Check if required folders exist
            ios_dir = patient_folder / "ios"
            photos_dir = patient_folder / "intraoral-photos"
            
            if not ios_dir.exists() or not photos_dir.exists():
                print(f"Warning: Skipping {patient_folder.name} - missing required folders")
                continue
            
            # Check if required files exist
            upper_stl = ios_dir / "upper.stl"
            lower_stl = ios_dir / "lower.stl"
            
            if self.load_meshes and (not upper_stl.exists() or not lower_stl.exists()):
                print(f"Warning: Skipping {patient_folder.name} - missing STL files")
                continue
            
            patient_ids.append(patient_folder.name)
        
        return patient_ids
    
    def _load_stl_as_pointcloud(self, stl_path: Path) -> np.ndarray:
        """
        Load an STL file and convert it to a point cloud.
        
        Args:
            stl_path: Path to the STL file
            
        Returns:
            Point cloud as numpy array of shape (n_points, 3)
        """
        mesh = trimesh.load_mesh(str(stl_path))
        
        # Sample points from the mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, self.n_points)
        
        return points.astype(np.float32)
    
    def _load_images(self, photos_dir: Path) -> List[Image.Image]:
        """
        Load all intraoral photos from a patient's folder.
        
        Args:
            photos_dir: Path to the intraoral-photos directory
            
        Returns:
            List of PIL Images
        """
        images = []
        
        for img_name in ["left", "center", "right", "upper", "lower", "a", "b", "c", "d", "e"]:
            img_formats = [f"{img_name}.jpg", f"{img_name}.jpeg", f"{img_name}.png"]
            img_path = None
            for fmt in img_formats:
                candidate = photos_dir / fmt
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                # print(f"Warning: Image {img_name} not found in {photos_dir}")
                continue
            img = Image.open(img_path).convert('RGB')

            # Resize if requested
            if self.image_size is not None:
                img = img.resize(self.image_size, Image.BILINEAR)
            
            images.append(img)
        
        return images
    
    def _load_patient_data(self, idx: int) -> Dict:
        """
        Load all data for a specific patient.
        
        Args:
            idx: Patient index
            
        Returns:
            Dictionary containing patient data
        """
        patient_id = self.patient_ids[idx]
        patient_dir = self.root_dir / patient_id
        
        data = {
            'patient_id': patient_id,
            'index': idx,
        }
        
        # Load meshes/point clouds
        if self.load_meshes:
            ios_dir = patient_dir / "ios"
            upper_points = self._load_stl_as_pointcloud(ios_dir / "upper.stl")
            lower_points = self._load_stl_as_pointcloud(ios_dir / "lower.stl")
            
            # Combine upper and lower into single point cloud
            combined_points = np.vstack([upper_points, lower_points])
            
            data['upper_pointcloud'] = upper_points
            data['lower_pointcloud'] = lower_points
            data['combined_pointcloud'] = combined_points
        
        # Load images
        if self.load_images:
            photos_dir = patient_dir / "intraoral-photos"
            images = self._load_images(photos_dir)
            data['images'] = images
            data['num_images'] = len(images)
        
        # Load captions
        if self.load_captions and self.captions_dir is not None:
            caption_path = self.captions_dir / f"{patient_id}.json"
            if caption_path.exists():
                try:
                    with open(caption_path, 'r') as f:
                        caption_data = json.load(f)
                        # Extract description from ios key
                        if 'ios' in caption_data and 'description' in caption_data['ios']:
                            data['report'] = caption_data['ios']['description']
                        else:
                            data['report'] = None
                except Exception as e:
                    print(f"Warning: Could not load caption for {patient_id}: {e}")
                    data['report'] = None
            else:
                data['report'] = None
        
        return data
    
    def _preload_all(self):
        """
        Preload all patient data into RAM cache using multiprocessing.
        """
        total_patients = len(self.patient_ids)
        indices = list(range(total_patients))
        
        if self.num_workers == 0:
            # Single-threaded loading
            print("Loading data (single-threaded)...")
            for idx in indices:
                if idx % 20 == 0:
                    print(f"Preloading: {idx}/{total_patients}")
                self._cache[idx] = self._load_patient_data(idx)
        else:
            # Multiprocessing loading
            print(f"Loading data with {self.num_workers} workers...")
            
            # Create arguments for parallel processing
            args_list = [
                (idx, self.root_dir, self.patient_ids[idx], self.n_points, 
                 self.load_meshes, self.load_images, self.image_size,
                 self.load_captions, self.captions_dir)
                for idx in indices
            ]
            
            # Use multiprocessing pool
            with Pool(processes=self.num_workers) as pool:
                results = []
                for i, result in enumerate(pool.imap(_load_patient_data_worker, args_list)):
                    results.append(result)
                    if (i + 1) % 20 == 0 or (i + 1) == total_patients:
                        print(f"Preloading: {i + 1}/{total_patients}")
            
            # Store results in cache
            for idx, data in results:
                self._cache[idx] = data
    
    def _convert_to_tensor(self, data: Dict) -> Dict:
        """
        Convert numpy arrays to PyTorch tensors.
        
        Args:
            data: Dictionary with data
            
        Returns:
            Dictionary with tensors
        """
        tensor_data = data.copy()
        
        # Convert point clouds
        if 'upper_pointcloud' in tensor_data:
            tensor_data['upper_pointcloud'] = torch.from_numpy(data['upper_pointcloud'])
        if 'lower_pointcloud' in tensor_data:
            tensor_data['lower_pointcloud'] = torch.from_numpy(data['lower_pointcloud'])
        if 'combined_pointcloud' in tensor_data:
            tensor_data['combined_pointcloud'] = torch.from_numpy(data['combined_pointcloud'])
        
        # Convert images to tensors if needed
        if 'images' in tensor_data and self.return_tensor:
            # Convert images to tensors (C, H, W) format
            image_tensors = []
            for img in tensor_data['images']:
                # Handle both PIL Images and numpy arrays
                if isinstance(img, Image.Image):
                    img_array = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                elif isinstance(img, np.ndarray):
                    img_array = img.transpose(2, 0, 1)  # HWC to CHW
                else:
                    img_array = img
                
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                image_tensors.append(img_tensor)
            tensor_data['images'] = image_tensors
        
        return tensor_data
    
    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get data for a specific patient.
        
        Args:
            idx: Patient index
            
        Returns:
            Dictionary containing:
                - patient_id: Patient identifier string
                - index: Dataset index
                - upper_pointcloud: Upper jaw point cloud (n_points, 3)
                - lower_pointcloud: Lower jaw point cloud (n_points, 3)
                - combined_pointcloud: Combined point cloud (2*n_points, 3)
                - images: List of PIL Images or tensors
                - num_images: Number of images
                - report: Caption/description text (if load_captions=True)
        """
        # Check cache first
        if idx in self._cache:
            data = self._cache[idx]
        else:
            # Load data
            data = self._load_patient_data(idx)
            
            # Cache if enabled
            if self.cache_in_ram:
                self._cache[idx] = data
        
        # Convert to tensors if requested
        if self.return_tensor:
            data = self._convert_to_tensor(data)
        
        return data
    
    def get_patient_by_id(self, patient_id: str) -> Dict:
        """
        Get data for a specific patient by their ID.
        
        Args:
            patient_id: Patient identifier (e.g., "201_1979")
            
        Returns:
            Dictionary containing patient data
        """
        # if patient_id not in self.patient_ids:
        #     raise ValueError(f"Patient ID {patient_id} not found in dataset")
        
        idx = self.patient_ids.index(patient_id)
        return self[idx]
    
    def clear_cache(self):
        """Clear the RAM cache."""
        self._cache.clear()
        print("Cache cleared")
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_patients': len(self.patient_ids),
            'cached_patients': len(self._cache),
            'points_per_mesh': self.n_points,
            'total_points_per_patient': self.n_points * 2,  # upper + lower
        }
        
        if self.load_images and len(self._cache) > 0:
            # Get image info from first cached item
            first_item = next(iter(self._cache.values()))
            if 'images' in first_item and len(first_item['images']) > 0:
                img = first_item['images'][0]
                if isinstance(img, Image.Image):
                    stats['image_size'] = img.size
                elif isinstance(img, np.ndarray):
                    stats['image_size'] = (img.shape[1], img.shape[0])  # (width, height)
                stats['images_per_patient'] = first_item['num_images']
        
        return stats
