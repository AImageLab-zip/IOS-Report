"""
Script to combine upper and lower teeth OBJ files into a single STL file.
Processes all matching patient folders from the 3DTeethLand dataset.
"""
import os
import gc
from pathlib import Path
import trimesh
from tqdm import tqdm


def find_matching_patients(upper_dir: Path, lower_dir: Path):
    """
    Find patient folders that exist in both upper and lower directories.
    
    Args:
        upper_dir: Path to upper teeth directory
        lower_dir: Path to lower teeth directory
    
    Returns:
        List of patient folder names that exist in both directories
    """
    upper_patients = set(p.name for p in upper_dir.iterdir() if p.is_dir())
    lower_patients = set(p.name for p in lower_dir.iterdir() if p.is_dir())
    
    matching_patients = sorted(upper_patients & lower_patients)
    
    print(f"Found {len(upper_patients)} upper patient folders")
    print(f"Found {len(lower_patients)} lower patient folders")
    print(f"Found {len(matching_patients)} matching patients")
    
    return matching_patients


def load_and_combine_meshes(upper_obj_path: Path, lower_obj_path: Path):
    """
    Load upper and lower OBJ files and combine them into a single mesh.
    
    Args:
        upper_obj_path: Path to upper teeth OBJ file
        lower_obj_path: Path to lower teeth OBJ file
    
    Returns:
        Combined trimesh object
    """
    # Load both meshes
    upper_mesh = trimesh.load(upper_obj_path, force='mesh')
    lower_mesh = trimesh.load(lower_obj_path, force='mesh')
    
    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([upper_mesh, lower_mesh])
    
    # Clean up to free memory
    del upper_mesh
    del lower_mesh
    
    return combined_mesh


def process_patient(patient_name: str, upper_dir: Path, lower_dir: Path, output_dir: Path):
    """
    Process a single patient: load upper and lower OBJ files, combine them, and save as STL.
    
    Args:
        patient_name: Name of the patient folder
        upper_dir: Path to upper teeth directory
        lower_dir: Path to lower teeth directory
        output_dir: Path to output directory
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct paths to OBJ files
        upper_obj = upper_dir / patient_name / f"{patient_name}_upper.obj"
        lower_obj = lower_dir / patient_name / f"{patient_name}_lower.obj"
        
        # Check if both files exist
        if not upper_obj.exists():
            print(f"  Warning: Upper OBJ not found for {patient_name}")
            return False
        if not lower_obj.exists():
            print(f"  Warning: Lower OBJ not found for {patient_name}")
            return False
        
        # Create output directory for this patient
        patient_output_dir = output_dir / patient_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists
        output_stl = patient_output_dir / "ios.stl"
        if output_stl.exists():
            return True  # Skip if already processed
        
        # Load and combine meshes
        combined_mesh = load_and_combine_meshes(upper_obj, lower_obj)
        
        # Save as STL
        combined_mesh.export(output_stl)
        
        # Clean up to free memory
        del combined_mesh
        
        return True
    
    except Exception as e:
        print(f"  Error processing {patient_name}: {str(e)}")
        return False


def main():
    """Main function to process all matching patients."""
    import sys
    
    # Define directories
    base_dir = Path("/work/grana_maxillo/BiteClassify/data/3dteethland_data")
    upper_dir = base_dir / "upper"
    lower_dir = base_dir / "lower"
    
    output_dir = Path("/work/grana_maxillo/IOS_DraftReport/_data/3DTeethLand")
    
    # Check if source directories exist
    if not upper_dir.exists():
        raise FileNotFoundError(f"Upper directory not found: {upper_dir}")
    if not lower_dir.exists():
        raise FileNotFoundError(f"Lower directory not found: {lower_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching patients
    matching_patients = find_matching_patients(upper_dir, lower_dir)
    
    if not matching_patients:
        print("No matching patients found!")
        return
    
    # Check command line arguments for batch processing
    start_idx = 0
    end_idx = len(matching_patients)
    
    if len(sys.argv) > 1:
        start_idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        end_idx = min(int(sys.argv[2]), len(matching_patients))
    
    patients_to_process = matching_patients[start_idx:end_idx]
    
    # Process each patient
    print(f"\nProcessing patients {start_idx} to {end_idx} (total: {len(patients_to_process)})...")
    successful = 0
    failed = 0
    skipped = 0
    
    for i, patient_name in enumerate(tqdm(patients_to_process, desc="Processing patients")):
        # Check if already exists
        output_stl = output_dir / patient_name / "ios.stl"
        if output_stl.exists():
            skipped += 1
            continue
            
        if process_patient(patient_name, upper_dir, lower_dir, output_dir):
            successful += 1
        else:
            failed += 1
        
        # Force garbage collection every 20 patients to free memory
        if (i + 1) % 20 == 0:
            gc.collect()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Total: {len(patients_to_process)}")
    print(f"{'='*60}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
