#!/usr/bin/env python3
"""
Quick test to verify image format detection works correctly.
"""

from pathlib import Path
import yaml

def find_image_with_extension(photos_dir: Path, view_name: str, extensions: list) -> Path:
    """Find an image file with any supported extension."""
    for ext in extensions:
        img_path = photos_dir / f"{view_name}{ext}"
        if img_path.exists():
            return img_path
    return None

def test_image_detection():
    """Test image detection on first few patients."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = Path(config['dataset']['root_path'])
    photos_subdir = config['dataset']['photos_subdir']
    views = config['dataset']['views']
    extensions = config['dataset']['image_extensions']
    
    print("="*60)
    print("IMAGE FORMAT DETECTION TEST")
    print("="*60)
    print(f"Supported extensions: {', '.join(extensions)}")
    print(f"Required views: {', '.join(views)}")
    print()
    
    # Test first 5 patients
    patient_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])[:5]
    
    for patient_dir in patient_dirs:
        print(f"Patient: {patient_dir.name}")
        photos_dir = patient_dir / photos_subdir
        
        if not photos_dir.exists():
            print(f"  ✗ No {photos_subdir} folder")
            continue
        
        all_found = True
        for view in views:
            img_path = find_image_with_extension(photos_dir, view, extensions)
            if img_path:
                ext = img_path.suffix
                print(f"  ✓ {view:8} → {img_path.name}")
            else:
                print(f"  ✗ {view:8} → NOT FOUND")
                all_found = False
        
        if all_found:
            print(f"  ✅ All images found!")
        else:
            print(f"  ❌ Some images missing")
        print()
    
    print("="*60)

if __name__ == "__main__":
    test_image_detection()
