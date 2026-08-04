"""
Validation script for evaluating generated orthodontic reports.

This script compares model-generated reports with ground truth captions
using multiple evaluation metrics (field accuracy, BLEU, ROUGE, METEOR, SBERT).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import (
    compute_all_metrics,
    compute_field_accuracy,
    compute_bleu,
    compute_rouge,
    compute_meteor,
    compute_radfact_per_sample,
)


def load_ground_truth(captions_dir: Path) -> Dict[str, str]:
    """
    Load ground truth reports from caption JSON files.
    
    Args:
        captions_dir: Directory containing caption JSON files
    
    Returns:
        Dictionary mapping patient_id to report text
    """
    ground_truth = {}

    for json_file in captions_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            candidate = None

            # Known schema: { 'ios': { 'patient_id': ..., 'description': ... } }
            if isinstance(data, dict):
                if 'ios' in data and isinstance(data['ios'], dict):
                    candidate = data['ios']

                # Some files use other top-level keys (e.g. 'intraoral-photo')
                if candidate is None:
                    for v in data.values():
                        if isinstance(v, dict) and 'patient_id' in v and (
                            'description' in v or 'original_italian' in v
                        ):
                            candidate = v
                            break

                # Fallback: top-level itself may contain patient_id/description
                if candidate is None and 'patient_id' in data and (
                    'description' in data or 'original_italian' in data
                ):
                    candidate = data

            if candidate is not None:
                patient_id = candidate.get('patient_id')
                # prefer 'description' but fall back to 'original_italian'
                description = candidate.get('description') or candidate.get('original_italian')
                if patient_id and description:
                    ground_truth[str(patient_id)] = str(description)
                else:
                    # skip if required fields missing
                    pass

        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")

    return ground_truth


def load_model_predictions(model_dir: Path) -> Dict[str, str]:
    """
    Load predictions from a model's output directory.
    
    Args:
        model_dir: Directory containing model predictions as .txt files
    
    Returns:
        Dictionary mapping patient_id to predicted report text
    """
    predictions = {}
    
    for txt_file in model_dir.glob("*.txt"):
        patient_id = txt_file.stem  # filename without extension
        try:
            with open(txt_file, 'r') as f:
                predictions[patient_id] = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load {txt_file.name}: {e}")
    
    return predictions


def align_predictions_and_references(
    predictions: Dict[str, str],
    ground_truth: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Align predictions with ground truth, filtering to common patient IDs.
    
    Args:
        predictions: Dictionary of patient_id -> predicted text
        ground_truth: Dictionary of patient_id -> ground truth text
    
    Returns:
        Tuple of (patient_ids, pred_list, ref_list) with aligned samples
    """
    common_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    
    patient_ids = []
    pred_list = []
    ref_list = []
    
    for patient_id in common_ids:
        patient_ids.append(patient_id)
        pred_list.append(predictions[patient_id])
        ref_list.append(ground_truth[patient_id])
    
    return patient_ids, pred_list, ref_list


def evaluate_model(
    model_name: str,
    model_dir: Path,
    ground_truth: Dict[str, str],
    include_sbert: bool = True,
    sbert_device: str = None,
    include_radfact: bool = False,
    radfact_cache_dir: Optional[Path] = None,
    radfact_model: str = 'gpt-5-nano',
    ollama_url: Optional[str] = None,
    radfact_max_workers: int = 5,
    radfact_rate_limit: int = 400,
    bootstrap_iters: int = 5000,
    bootstrap_seed: int = 42,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluate a single model against ground truth.
    
    Args:
        model_name: Name of the model
        model_dir: Directory containing model predictions
        ground_truth: Ground truth reports
        include_sbert: Whether to compute Sentence-BERT similarity
        sbert_device: Device for SBERT computation
        include_radfact: Whether to compute RadFact metrics
        radfact_cache_dir: Directory for RadFact cache files
        radfact_model: Model to use for RadFact (default: gpt-5-nano)
        ollama_url: URL for Ollama API (for local models like llama70b)
        radfact_max_workers: Number of parallel workers for RadFact
        radfact_rate_limit: API rate limit in requests per minute
    
    Returns:
        Dictionary of metric scores
    """
    print(f"\nEvaluating {model_name}...")
    
    # Load predictions
    predictions = load_model_predictions(model_dir)
    
    if not predictions:
        print(f"  Warning: No predictions found in {model_dir}")
        return {}, []
    
    # Align with ground truth
    patient_ids, pred_list, ref_list = align_predictions_and_references(
        predictions, ground_truth
    )
    
    if not pred_list:
        print(f"  Warning: No matching patient IDs found")
        return {}, []
    
    print(f"  Evaluating {len(pred_list)} samples...")
    
    # Prepare RadFact cache file if needed
    radfact_cache_file = None
    if include_radfact and radfact_cache_dir:
        radfact_cache_file = str(radfact_cache_dir / f"{model_name}_radfact_cache.json")
    
    # Compute mean metrics (backward-compatible)
    metrics = compute_all_metrics(
        pred_list,
        ref_list,
        include_sbert=include_sbert,
        sbert_device=sbert_device,
        return_per_field=False,
        include_radfact=include_radfact,
        radfact_cache_file=radfact_cache_file,
        radfact_model=radfact_model,
        ollama_url=ollama_url,
        radfact_max_workers=radfact_max_workers,
        radfact_rate_limit=radfact_rate_limit
    )

    # Compute per-sample metrics for variance/CI reporting
    per_sample_rows: List[Dict[str, float]] = []
    for patient_id, pred, ref in zip(patient_ids, pred_list, ref_list):
        row = {
            "model": model_name,
            "patient_id": patient_id,
        }
        row.update(compute_field_accuracy([pred], [ref], return_per_field=False))
        row.update(compute_bleu([pred], [ref], n_gram=1))
        row.update(compute_bleu([pred], [ref], n_gram=4))
        row.update(compute_rouge([pred], [ref]))
        row.update(compute_meteor([pred], [ref]))
        per_sample_rows.append(row)

    # Batch SBERT once for speed
    if include_sbert and per_sample_rows:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer('all-MiniLM-L6-v2', device=sbert_device)
        pred_emb = model.encode(pred_list, convert_to_tensor=True, device=sbert_device, batch_size=64)
        ref_emb = model.encode(ref_list, convert_to_tensor=True, device=sbert_device, batch_size=64)
        sims = util.cos_sim(pred_emb, ref_emb).diagonal().detach().cpu().numpy()
        for row, sim in zip(per_sample_rows, sims):
            row['sbert-sim'] = float(sim)

    # Per-sample RadFact (strict per-sample stats)
    if include_radfact and per_sample_rows:
        rad_rows = compute_radfact_per_sample(
            pred_list,
            ref_list,
            radfact_model=radfact_model,
            cache_file=radfact_cache_file,
            ollama_url=ollama_url,
            max_workers=radfact_max_workers,
            rate_limit_rpm=radfact_rate_limit,
        )
        for row, rr in zip(per_sample_rows, rad_rows):
            row.update(rr)

    metric_keys = [
        'accuracy', 'coverage', 'num_fields',
        'bleu-1', 'bleu-4', 'rouge-l-p', 'rouge-l-r', 'rouge-l-f', 'meteor'
    ]
    if include_sbert:
        metric_keys.append('sbert-sim')
    if include_radfact:
        metric_keys.extend(['radfact-precision', 'radfact-recall', 'radfact-f1'])

    rng = np.random.default_rng(bootstrap_seed)
    for key in metric_keys:
        vals = np.array([float(r[key]) for r in per_sample_rows if key in r], dtype=float)
        if vals.size == 0:
            metrics[f'{key}_std'] = 0.0
            metrics[f'{key}_var'] = 0.0
            metrics[f'{key}_median'] = 0.0
            metrics[f'{key}_q25'] = 0.0
            metrics[f'{key}_q75'] = 0.0
            metrics[f'{key}_min'] = 0.0
            metrics[f'{key}_max'] = 0.0
            metrics[f'{key}_ci95_low'] = 0.0
            metrics[f'{key}_ci95_high'] = 0.0
            metrics[f'{key}_n'] = 0
            continue

        metrics[f'{key}_std'] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        metrics[f'{key}_var'] = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
        metrics[f'{key}_median'] = float(np.median(vals))
        metrics[f'{key}_q25'] = float(np.percentile(vals, 25))
        metrics[f'{key}_q75'] = float(np.percentile(vals, 75))
        metrics[f'{key}_min'] = float(np.min(vals))
        metrics[f'{key}_max'] = float(np.max(vals))
        metrics[f'{key}_n'] = int(vals.size)

        if vals.size == 1:
            metrics[f'{key}_ci95_low'] = float(vals[0])
            metrics[f'{key}_ci95_high'] = float(vals[0])
        else:
            boot_means = np.empty(bootstrap_iters, dtype=float)
            n = vals.size
            for i in range(bootstrap_iters):
                sample = rng.choice(vals, size=n, replace=True)
                boot_means[i] = float(np.mean(sample))
            metrics[f'{key}_ci95_low'] = float(np.percentile(boot_means, 2.5))
            metrics[f'{key}_ci95_high'] = float(np.percentile(boot_means, 97.5))
    
    # Add model name and sample count
    metrics['model'] = model_name
    metrics['num_samples'] = len(pred_list)
    
    return metrics, per_sample_rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated orthodontic reports against ground truth"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory containing model predictions"
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="_data/captions",
        help="Directory containing ground truth captions"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="validation_results.csv",
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Skip Sentence-BERT computation (faster)"
    )
    parser.add_argument(
        "--sbert-device",
        type=str,
        default="cuda",
        choices=['cuda', 'cpu'],
        help="Device for Sentence-BERT computation"
    )
    parser.add_argument(
        "--radfact",
        action="store_true",
        help="Include RadFact metrics (requires OpenAI API, slower)"
    )
    parser.add_argument(
        "--radfact-cache-dir",
        type=str,
        default="output/.radfact_cache",
        help="Directory for RadFact cache files (default: output/.radfact_cache)"
    )
    parser.add_argument(
        "--radfact-model",
        type=str,
        default="gpt-5-nano",
        help="Model to use for RadFact evaluation (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--radfact-rate-limit",
        type=int,
        default=400,
        help="Rate limit for RadFact API calls in requests per minute (default: 400, max: 500 for Tier 1)"
    )
    parser.add_argument(
        "--radfact-workers",
        type=int,
        default=5,
        help="Number of parallel workers for RadFact evaluation (default: 5)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="URL for Ollama API (e.g., http://hostname:11434) for local models like llama70b"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Specific model folders to evaluate (default: all)"
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=5000,
        help="Bootstrap iterations for 95% CI (default: 5000)"
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Bootstrap random seed (default: 42)"
    )
    parser.add_argument(
        "--per-sample-results-file",
        type=str,
        default=None,
        help="Optional CSV path to save per-sample metrics"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    captions_dir = Path(args.captions_dir)
    
    # Validate directories
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    if not captions_dir.exists():
        print(f"Error: Captions directory not found: {captions_dir}")
        return
    
    # Load ground truth
    print("Loading ground truth captions...")
    ground_truth = load_ground_truth(captions_dir)
    print(f"Loaded {len(ground_truth)} ground truth reports")
    
    # Find model directories to evaluate
    if args.models:
        model_dirs = [output_dir / model for model in args.models]
        model_dirs = [d for d in model_dirs if d.exists() and d.is_dir()]
    else:
        model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("Error: No model directories found to evaluate, available:")
        for d in output_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return
    
    print(f"\nFound {len(model_dirs)} model(s) to evaluate:")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")
    
    # Setup RadFact cache directory if needed
    radfact_cache_dir = None
    if args.radfact:
        radfact_cache_dir = Path(args.radfact_cache_dir)
        radfact_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRadFact cache directory: {radfact_cache_dir}")
    
    # Evaluate each model
    all_results = []
    all_per_sample_rows = []
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        try:
            metrics, per_sample_rows = evaluate_model(
                model_name,
                model_dir,
                ground_truth,
                include_sbert=not args.no_sbert,
                sbert_device=args.sbert_device,
                include_radfact=args.radfact,
                radfact_cache_dir=radfact_cache_dir,
                radfact_model=args.radfact_model,
                ollama_url=args.ollama_url,
                radfact_max_workers=args.radfact_workers,
                radfact_rate_limit=args.radfact_rate_limit,
                bootstrap_iters=args.bootstrap_iters,
                bootstrap_seed=args.bootstrap_seed,
            )
            
            if metrics:
                all_results.append(metrics)
                all_per_sample_rows.extend(per_sample_rows)
                
                # Print summary
                print(f"\n  Results for {model_name}:")
                print(f"    Samples: {metrics.get('num_samples', 0)}")
                print(f"    Field Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"    Coverage: {metrics.get('coverage', 0):.3f}")
                print(f"    BLEU-1: {metrics.get('bleu-1', 0):.3f}")
                print(f"    ROUGE-L F1: {metrics.get('rouge-l-f', 0):.3f}")
                print(f"    METEOR: {metrics.get('meteor', 0):.3f}")
                if not args.no_sbert:
                    print(f"    SBERT Similarity: {metrics.get('sbert-sim', 0):.3f}")
                if args.radfact:
                    print(f"    RadFact Precision: {metrics.get('radfact-precision', 0):.3f}")
                    print(f"    RadFact Recall: {metrics.get('radfact-recall', 0):.3f}")
                    print(f"    RadFact F1: {metrics.get('radfact-f1', 0):.3f}")
        
        except Exception as e:
            print(f"\n  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo results to save")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns: model first, then num_samples, then everything else alphabetically
    ordered_prefix = ['model', 'num_samples']
    remaining = [c for c in df.columns if c not in ordered_prefix]
    df = df[[c for c in ordered_prefix if c in df.columns] + sorted(remaining)]
    
    # Sort by accuracy (descending)
    if 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False)
    
    # Save to CSV
    results_file = Path(args.results_file)
    df.to_csv(results_file, index=False, float_format='%.4f')
    
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    if args.per_sample_results_file and all_per_sample_rows:
        per_sample_file = Path(args.per_sample_results_file)
        per_sample_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_per_sample_rows).to_csv(per_sample_file, index=False, float_format='%.6f')
        print(f"Per-sample metrics saved to: {per_sample_file}")


if __name__ == "__main__":
    main()
