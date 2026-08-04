import argparse
import os
from pathlib import Path

from report_generator import ReportGenerator
from dataset import FerraraDumpDataset


DEFAULT_MODELS = [
    "google/medgemma-4b-it",
    "Qwen/Qwen3-VL-2B-Instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def get_model_folder_name(model_name):
    """Convert model name to a valid folder name"""
    return model_name.replace("/", "_")


def main():
    print("Starting Ferrara Dump Dataset Processing...")
    parser = argparse.ArgumentParser(description="Ferrara Dump Dataset Processing")
    parser.add_argument("--root_dir", type=str, default="_data/Dataset_FerraraDump_400", help="Root directory of the dataset")
    parser.add_argument("--captions_dir", type=str, default="_data/captions", help="Directory containing caption JSON files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated reports")
    parser.add_argument("--image_size", type=tuple, default=None, help="Size to resize images")
    parser.add_argument("--models", type=str, nargs='+', default=DEFAULT_MODELS, help="List of model names to test")
    parser.add_argument("--system_prompt_path", type=str, default="templates/system.txt", help="Path to system prompt template")
    parser.add_argument("--user_prompt_path", type=str, default="templates/user.txt", help="Path to user prompt template")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dataset = FerraraDumpDataset(
        root_dir=args.root_dir,
        image_size=args.image_size,
        return_tensor=False,
        load_captions=True,
        captions_dir=args.captions_dir,
    )
    
    for model_name in args.models:
        model_folder_name = get_model_folder_name(model_name)
        model_output_dir = output_dir / model_folder_name
        
        if model_output_dir.exists():
            print(f"\n{'='*80}")
            print(f"Skipping model {model_name}")
            print(f"Output directory already exists: {model_output_dir}")
            print(f"{'='*80}\n")
            continue
        
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        report_generator = ReportGenerator(
            model_name=model_name,
            system_prompt_path=args.system_prompt_path,
            user_prompt_path=args.user_prompt_path
        )
        
        for patient in dataset:
            patient_id = patient['patient_id']
            output_file = model_output_dir / f"{patient_id}.txt"
            
            print(f"\n{'='*80}")
            print(f"Model: {model_name}")
            print(f"Patient ID: {patient_id}")
            print(f"Number of images: {patient.get('num_images', 'N/A')}")
            
            if 'report' in patient and patient['report']:
                print(f"\nGround Truth Report:")
                print("-" * 80)
                print(patient['report'])
                print("-" * 80)
            
            try:
                generated_report = report_generator.generate_report(patient)
            except Exception as e:
                print(f"Error generating report for patient {patient_id} with model {model_name}: {e}")
                continue
            print(f"\nGenerated Report:")
            print("-" * 80)
            print(generated_report)
            print("-" * 80)
            
            with open(output_file, 'w') as f:
                f.write(generated_report)
            
            print(f"\nSaved to: {output_file}")
            print(f"{'='*80}\n")

if __name__ == "__main__":
    main()