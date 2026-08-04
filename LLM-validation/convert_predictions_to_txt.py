#!/usr/bin/env python3
"""
Convert predictions.json format to individual text files.

This script converts a JSON file containing predictions and ground truth
into individual .txt files, one per patient, matching the format used by
other model outputs.
"""

import json
import argparse
from pathlib import Path


def convert_predictions_to_txt(json_file: Path, output_dir: Path):
    """
    Convert predictions.json to individual text files.
    
    Args:
        json_file: Path to predictions.json file
        output_dir: Directory to write individual .txt files
    """
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each entry to a text file
    for entry in data:
        patient_id = entry['patient_id']
        prediction = entry['prediction']
        
        # Clean up patient_id to create a valid filename
        # Replace problematic characters
        filename = patient_id.replace('/', '_').replace('\\', '_')
        
        # Ensure the filename ends with .txt
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        
        # Write prediction to file
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            f.write(prediction)
            # Add newline at end if not present
            if not prediction.endswith('\n'):
                f.write('\n')
    
    print(f"✓ Converted {len(data)} predictions to {output_dir}")
    print(f"  Created {len(list(output_dir.glob('*.txt')))} .txt files")


def main():
    parser = argparse.ArgumentParser(
        description='Convert predictions.json to individual .txt files'
    )
    parser.add_argument(
        'json_file',
        type=Path,
        help='Path to predictions.json file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        help='Output directory (default: same directory as json_file)'
    )
    
    args = parser.parse_args()
    
    # Default output directory is same as json_file
    if args.output_dir is None:
        args.output_dir = args.json_file.parent
    
    # Verify json_file exists
    if not args.json_file.exists():
        print(f"Error: {args.json_file} does not exist")
        return 1
    
    # Convert
    convert_predictions_to_txt(args.json_file, args.output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
