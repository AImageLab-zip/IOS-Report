"""
Main script for generating orthodontic reports using OpenAI API.

This script processes the Ferrara Dump dataset using OpenAI's vision models
(e.g., GPT-4o, GPT-4 Turbo) via their API instead of local models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generator_openai import OpenAIReportGenerator
from dataset.ferraradump import FerraraDumpDataset


DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4o-mini",
]


def get_model_folder_name(model_name):
    """Convert model name to a valid folder name"""
    return model_name.replace("/", "_").replace(":", "_")


def main():
    print("Starting Ferrara Dump Dataset Processing with OpenAI API...")
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Ferrara Dump Dataset Processing with OpenAI API")
    parser.add_argument("--root_dir", type=str, default="_data/Dataset_FerraraDump_400", 
                        help="Root directory of the dataset")
    parser.add_argument("--captions_dir", type=str, default="_data/captions", 
                        help="Directory containing caption JSON files")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save generated reports")
    parser.add_argument("--image_size", type=tuple, default=None, 
                        help="Size to resize images")
    parser.add_argument("--models", type=str, nargs='+', default=DEFAULT_MODELS, 
                        help="List of OpenAI model names to test (e.g., gpt-4o, gpt-4-turbo)")
    parser.add_argument("--system_prompt_path", type=str, default="templates/system.txt", 
                        help="Path to system prompt template")
    parser.add_argument("--user_prompt_path", type=str, default="templates/user.txt", 
                        help="Path to user prompt template")
    parser.add_argument("--skip_existing", action='store_true', 
                        help="Skip models whose output directories already exist")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {args.root_dir}...")
    dataset = FerraraDumpDataset(
        root_dir=args.root_dir,
        image_size=args.image_size,
        return_tensor=False,
        load_captions=True,
        captions_dir=args.captions_dir,
        load_images=True,
        load_meshes=False,
    )
    print(f"Dataset loaded: {len(dataset)} patients")
    
    # Process each model
    for model_name in args.models:
        model_folder_name = get_model_folder_name(model_name)
        model_output_dir = output_dir / model_folder_name
        
        if args.skip_existing and model_output_dir.exists():
            print(f"\n{'='*80}")
            print(f"Skipping model {model_name}")
            print(f"Output directory already exists: {model_output_dir}")
            print(f"{'='*80}\n")
            continue
        
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI report generator
        try:
            report_generator = OpenAIReportGenerator(
                model_name=model_name,
                system_prompt_path=args.system_prompt_path,
                user_prompt_path=args.user_prompt_path
            )
        except Exception as e:
            print(f"Error initializing OpenAI generator for {model_name}: {e}")
            continue
        
        # Process each patient
        for patient in dataset:
            patient_id = patient['patient_id']
            output_file = model_output_dir / f"{patient_id}.txt"
            
            # Skip if already processed
            if output_file.exists():
                print(f"Skipping {patient_id} (already processed)")
                continue
            
            print(f"\n{'='*80}")
            print(f"Model: {model_name}")
            print(f"Patient ID: {patient_id}")
            print(f"Number of images: {patient.get('num_images', 'N/A')}")
            
            # Print ground truth if available
            if 'report' in patient and patient['report']:
                print(f"\nGround Truth Report:")
                print("-" * 80)
                print(patient['report'])
                print("-" * 80)
            
            # Generate report using OpenAI API
            try:
                generated_report = report_generator.generate_report(patient)
            except Exception as e:
                print(f"Error generating report for patient {patient_id} with model {model_name}: {e}")
                # Save error message
                with open(output_file, 'w') as f:
                    f.write(f"ERROR: {str(e)}\n")
                continue
            
            print(f"\nGenerated Report:")
            print("-" * 80)
            print(generated_report)
            print("-" * 80)
            
            # Save generated report
            with open(output_file, 'w') as f:
                f.write(generated_report)
            
            print(f"\nSaved to: {output_file}")
            print(f"{'='*80}\n")
        
        print(f"\nCompleted processing for model: {model_name}")
        print(f"Output saved to: {model_output_dir}\n")


if __name__ == "__main__":
    main()
