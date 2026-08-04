"""
Main script for generating orthodontic reports using closed-source API models.

Supports OpenAI (GPT-5.2), Google Gemini (Gemini3), and DeepSeek (DeepSeek3.2).
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report_generator_api import APIReportGenerator
from dataset.ferraradump import FerraraDumpDataset


# Default closed-source models
DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-5.2",
    "gemini3",
    # "deepseek3.2",  # DeepSeek API doesn't support vision/image inputs
]


def get_model_folder_name(model_name: str) -> str:
    """Convert model name to a valid folder name."""
    return model_name.replace("/", "_").replace(":", "_").replace(".", "_")


def check_api_keys():
    """Check if required API keys are set."""
    missing_keys = []
    
    if not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.getenv('GEMINI_API_KEY'):
        missing_keys.append('GEMINI_API_KEY')
    if not os.getenv('DEEPSEEK_API_KEY'):
        missing_keys.append('DEEPSEEK_API_KEY')
    
    if missing_keys:
        print("Warning: The following API keys are not set:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nSome models may fail if their API key is not available.")
        print("Set them in .env file or as environment variables.\n")


def main():
    print("Starting Ferrara Dump Dataset Processing with Closed-Source APIs...")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check API keys
    check_api_keys()
    
    parser = argparse.ArgumentParser(
        description="Ferrara Dump Dataset Processing with Closed-Source APIs"
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="_data/Dataset_FerraraDump_400", 
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--captions_dir", 
        type=str, 
        default="_data/captions", 
        help="Directory containing caption JSON files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output", 
        help="Directory to save generated reports"
    )
    parser.add_argument(
        "--image_size", 
        type=tuple, 
        default=None, 
        help="Size to resize images"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs='+', 
        default=DEFAULT_MODELS, 
        help="List of API model names to test (e.g., gpt-5.2, gemini3, deepseek3.2)"
    )
    parser.add_argument(
        "--system_prompt_path", 
        type=str, 
        default="templates/system.txt", 
        help="Path to system prompt template"
    )
    parser.add_argument(
        "--user_prompt_path", 
        type=str, 
        default="templates/user.txt", 
        help="Path to user prompt template"
    )
    parser.add_argument(
        "--skip_existing", 
        action='store_true', 
        help="Skip models whose output directories already exist"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of concurrent workers (default: auto based on API rate limit)"
    )
    parser.add_argument(
        "--batch_mode",
        action='store_true',
        help="Use batch processing mode for faster generation"
    )
    
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
        
        # Initialize API report generator
        try:
            report_generator = APIReportGenerator(
                model_name=model_name,
                system_prompt_path=args.system_prompt_path,
                user_prompt_path=args.user_prompt_path
            )
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"Error initializing generator for {model_name}: {e}")
            print(f"{'='*80}\n")
            continue
        
        # Process patients
        print(f"\n{'='*80}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*80}")
        
        if args.batch_mode:
            # Batch processing mode (faster with concurrent API calls)
            print(f"Using batch processing mode (memory-efficient)...")
            
            # Collect indices of patients that need processing (not full data)
            indices_to_process = []
            for idx in range(len(dataset)):
                patient = dataset[idx]
                patient_id = patient['patient_id']
                output_file = model_output_dir / f"{patient_id}.txt"
                
                if not output_file.exists():
                    indices_to_process.append(idx)
            
            if not indices_to_process:
                print("No patients to process (all already completed)")
            else:
                print(f"Processing {len(indices_to_process)} patients...")
                
                # Generate reports in batch (loads data on-demand)
                results = report_generator.generate_reports_batch(
                    dataset,
                    indices_to_process,
                    max_workers=args.max_workers
                )
                
                # Save all results
                for patient_id, report in results.items():
                    output_file = model_output_dir / f"{patient_id}.txt"
                    with open(output_file, 'w') as f:
                        f.write(report)
        
        else:
            # Sequential processing mode (original behavior)
            print(f"Using sequential processing mode...")
            
            for idx, patient in enumerate(dataset):
                patient_id = patient['patient_id']
                output_file = model_output_dir / f"{patient_id}.txt"
                
                # Skip if already processed
                if output_file.exists():
                    print(f"[{idx+1}/{len(dataset)}] Skipping {patient_id} (already processed)")
                    continue
                
                print(f"\n[{idx+1}/{len(dataset)}] Processing Patient ID: {patient_id}")
                print(f"  Number of images: {patient.get('num_images', 'N/A')}")
                
                # Print ground truth if available
                if 'report' in patient and patient['report']:
                    print(f"  Ground Truth Report (first 100 chars):")
                    print(f"  {patient['report'][:100]}...")
                
                # Generate report using API
                try:
                    generated_report = report_generator.generate_report(patient)
                    
                    print(f"  Generated Report (first 100 chars):")
                    print(f"  {generated_report[:100]}...")
                    
                    # Save generated report
                    with open(output_file, 'w') as f:
                        f.write(generated_report)
                    
                    print(f"  ✓ Saved to: {output_file}")
                    
                except Exception as e:
                    error_str = str(e)
                    print(f"  ✗ Error generating report: {error_str}")
                    
                    # Check for critical errors
                    critical_keywords = ['quota', 'insufficient_quota', '429', 'billing', 'rate limit exceeded']
                    is_critical = any(keyword.lower() in error_str.lower() for keyword in critical_keywords)
                    
                    if is_critical:
                        print(f"\n{'='*80}")
                        print(f"CRITICAL ERROR: API quota exceeded or billing issue")
                        print(f"Stopping processing for model {model_name}")
                        print(f"Error: {error_str}")
                        print(f"{'='*80}\n")
                        break  # Stop processing this model
                    # Don't save error files - just skip to next patient
        
        print(f"\n{'='*80}")
        print(f"Completed processing for model: {model_name}")
        print(f"Output saved to: {model_output_dir}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("ALL MODELS PROCESSED")
    print("="*80)
    print(f"\nTo evaluate the results, run:")
    print(f"  python src/validation.py --output_dir {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
