"""
Evaluation script for external models using intraoral photos.

This script evaluates external API-based models (OpenAI GPT5.2, Google Gemini3, DeepSeek)
on intraoral photo datasets and computes the same metrics as the main model.

Usage:
    python evaluate_external_models.py --config configs/evaluation/external_model_baselines.yaml \\
                                        --model openai \\
                                        [--num_samples N] [--output_dir DIR]
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import traceback

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from external_models import OpenAIModel, GeminiModel, DeepSeekModel
from validation import compute_all_metrics


def load_intraoral_dataset(
    dataset_dir: Path,
    captions_dir: Path,
    min_photos: int = 3
) -> List[Dict]:
    """
    Load intraoral photo dataset with captions.
    
    Args:
        dataset_dir: Directory containing patient folders with intraoral-photos/
        captions_dir: Directory containing JSON files with patient descriptions
        min_photos: Minimum number of photos required per patient
    
    Returns:
        List of sample dicts with keys: patient_id, photo_paths, description
    """
    samples = []
    
    # Get all JSON files from captions directory
    json_files = list(captions_dir.glob("*.json"))
    
    for json_file in json_files:
        patient_id = json_file.stem
        
        # Load description from JSON
        try:
            with open(json_file, 'r') as f:
                caption_data = json.load(f)
                
                # Try to get intraoral-photo description first, fallback to ios
                if 'intraoral-photo' in caption_data:
                    desc_field = 'intraoral-photo'
                elif 'ios' in caption_data:
                    desc_field = 'ios'
                else:
                    continue
                
                description = caption_data.get(desc_field, {}).get('description', '').strip()
                
                if not description:
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
        
        # Look for intraoral photos directory
        patient_dir = dataset_dir / patient_id
        photos_dir = patient_dir / "intraoral-photos"
        
        if not photos_dir.exists():
            continue
        
        # Find photo files
        photo_names = ["left", "center", "right", "upper", "lower", "a", "b", "c", "d", "e"]
        photo_paths = []
        for name in photo_names:
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                photo_path = photos_dir / f"{name}{ext}"
                if photo_path.exists():
                    photo_paths.append(photo_path)
                    break
        
        # Require minimum number of photos
        if len(photo_paths) >= min_photos:
            samples.append({
                'patient_id': patient_id,
                'photo_paths': photo_paths,
                'description': description
            })
    
    return samples


def create_model(model_type: str, config: Dict):
    """
    Create an external model client.
    
    Args:
        model_type: One of 'openai', 'gemini', 'deepseek'
        config: Configuration dict with model parameters
    
    Returns:
        Model client instance
    """
    model_config = config['models'][model_type]
    
    if model_type == 'openai':
        return OpenAIModel(
            api_key=model_config.get('api_key'),
            model_name=model_config['model_name'],
            system_prompt=model_config.get('system_prompt'),
            user_prompt_template=model_config.get('user_prompt_template'),
            temperature=model_config.get('temperature', 0.7),
            max_tokens=model_config.get('max_tokens', 512),
            rate_limit_delay=model_config.get('rate_limit_delay', 1.0),
        )
    elif model_type == 'gemini':
        return GeminiModel(
            api_key=model_config.get('api_key'),
            model_name=model_config['model_name'],
            system_prompt=model_config.get('system_prompt'),
            user_prompt_template=model_config.get('user_prompt_template'),
            temperature=model_config.get('temperature', 0.7),
            max_tokens=model_config.get('max_tokens', 512),
            rate_limit_delay=model_config.get('rate_limit_delay', 1.0),
        )
    elif model_type == 'deepseek':
        return DeepSeekModel(
            api_key=model_config.get('api_key'),
            model_name=model_config['model_name'],
            system_prompt=model_config.get('system_prompt'),
            user_prompt_template=model_config.get('user_prompt_template'),
            temperature=model_config.get('temperature', 0.7),
            max_tokens=model_config.get('max_tokens', 512),
            api_base=model_config.get('api_base', 'https://api.deepseek.com/v1'),
            rate_limit_delay=model_config.get('rate_limit_delay', 1.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(
    model,
    samples: List[Dict],
    num_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate a model on the dataset.
    
    Args:
        model: External model client
        samples: List of sample dicts
        num_samples: Number of samples to evaluate (None = all)
        output_dir: Directory to save predictions
    
    Returns:
        Dict with metrics and predictions
    """
    if num_samples is not None:
        samples = samples[:num_samples]
    
    predictions = []
    references = []
    
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Generating predictions"):
        try:
            # Extract field names from reference
            field_names = model.prepare_field_names(sample['description'])
            
            # Generate prediction
            prediction = model.generate(
                image_paths=sample['photo_paths'],
                field_names=field_names
            )
            
            # Store results
            predictions.append(prediction)
            references.append(sample['description'])
            
            # Save intermediate results
            if output_dir:
                pred_file = output_dir / f"{sample['patient_id']}_prediction.txt"
                with open(pred_file, 'w') as f:
                    f.write(f"Patient ID: {sample['patient_id']}\n\n")
                    f.write(f"Photos: {', '.join([p.name for p in sample['photo_paths']])}\n\n")
                    f.write(f"Prediction:\n{prediction}\n\n")
                    f.write(f"Reference:\n{sample['description']}\n")
        
        except Exception as e:
            print(f"\nError processing {sample['patient_id']}: {e}")
            traceback.print_exc()
            predictions.append(f"Error: {str(e)}")
            references.append(sample['description'])
    
    # Compute metrics
    print("\nComputing metrics...")
    
    metrics = compute_all_metrics(
        predictions=predictions,
        references=references,
        compute_sbert=True,
        return_per_field=True
    )
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references,
        'patient_ids': [s['patient_id'] for s in samples]
    }


def save_results(results: Dict, output_path: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dict from evaluate_model()
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate external models on intraoral photos")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['openai', 'gemini', 'deepseek'],
        help="Model to evaluate"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Output directory for results (default: outputs/{model}_eval)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('outputs') / f"{args.model}_eval"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"External Model Evaluation: {args.model.upper()}")
    print(f"{'='*80}\n")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_dir = Path(config['data']['dataset_dir'])
    captions_dir = Path(config['data']['captions_dir'])
    
    samples = load_intraoral_dataset(
        dataset_dir=dataset_dir,
        captions_dir=captions_dir,
        min_photos=config['data'].get('min_photos', 3)
    )
    
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("Error: No valid samples found!")
        return
    
    # Create model
    print(f"\nInitializing {args.model} model...")
    model = create_model(args.model, config)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        samples=samples,
        num_samples=args.num_samples,
        output_dir=output_dir
    )
    
    # Print metrics
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    metrics = results['metrics']
    
    print(f"Field-level Accuracy: {metrics['field_accuracy']:.2%}")
    print(f"BLEU-1: {metrics['bleu1']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    
    if 'sbert_similarity' in metrics:
        print(f"Sentence-BERT: {metrics['sbert_similarity']:.4f}")
    
    # Per-field accuracy if available
    if 'per_field_accuracy' in metrics:
        print("\nPer-Field Accuracy:")
        for field, acc in sorted(metrics['per_field_accuracy'].items()):
            print(f"  {field}: {acc:.2%}")
    
    # Save results
    results_file = output_dir / f"results_{args.model}.json"
    save_results(results, results_file)
    
    # Save metrics summary
    metrics_file = output_dir / f"metrics_{args.model}.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Results: {args.model.upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Dataset: {config['data']['dataset_dir']}\n")
        f.write(f"Samples: {len(results['predictions'])}\n\n")
        f.write(f"Field-level Accuracy: {metrics['field_accuracy']:.2%}\n")
        f.write(f"BLEU-1: {metrics['bleu1']:.4f}\n")
        f.write(f"ROUGE-L: {metrics['rouge_l']:.4f}\n")
        f.write(f"METEOR: {metrics['meteor']:.4f}\n")
        if 'sbert_similarity' in metrics:
            f.write(f"Sentence-BERT: {metrics['sbert_similarity']:.4f}\n")
        
        if 'per_field_accuracy' in metrics:
            f.write("\nPer-Field Accuracy:\n")
            for field, acc in sorted(metrics['per_field_accuracy'].items()):
                f.write(f"  {field}: {acc:.2%}\n")
    
    print(f"\nMetrics summary saved to: {metrics_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
