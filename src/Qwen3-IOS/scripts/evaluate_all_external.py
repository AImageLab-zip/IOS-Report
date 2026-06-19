"""
Batch evaluation script for all external models.

This script evaluates all configured external models and generates a comparison report.

Usage:
    python evaluate_all_external.py --config configs/evaluation/external_model_baselines.yaml \\
                                     [--models openai gemini deepseek] \\
                                     [--num_samples N]
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate_external_models import (
    load_intraoral_dataset,
    create_model,
    evaluate_model,
    save_results
)


def compare_models(results: Dict[str, Dict], output_dir: Path):
    """
    Create comparison visualizations and reports.
    
    Args:
        results: Dict mapping model names to their results
        output_dir: Output directory
    """
    # Extract metrics for comparison
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name.upper(),
            'Field Accuracy': metrics['field_accuracy'] * 100,
            'BLEU-1': metrics['bleu1'] * 100,
            'ROUGE-L': metrics['rouge_l'] * 100,
            'METEOR': metrics['meteor'] * 100,
            'SBERT': metrics.get('sbert_similarity', 0) * 100
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('External Model Comparison on Intraoral Photos', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Field Accuracy', 'BLEU-1', 'ROUGE-L', 'METEOR', 'SBERT']
    colors = sns.color_palette('husl', len(df))
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bars = ax.bar(df['Model'], df[metric], color=colors)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel('Score (%)')
        ax.set_ylim([0, 100])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Per-field comparison if available
    per_field_data = {}
    for model_name, result in results.items():
        if 'per_field_accuracy' in result['metrics']:
            per_field_data[model_name] = result['metrics']['per_field_accuracy']
    
    if per_field_data:
        # Create per-field comparison
        field_df_data = []
        all_fields = set()
        for model_accs in per_field_data.values():
            all_fields.update(model_accs.keys())
        
        for field in sorted(all_fields):
            row = {'Field': field}
            for model_name in per_field_data.keys():
                row[model_name.upper()] = per_field_data[model_name].get(field, 0) * 100
            field_df_data.append(row)
        
        field_df = pd.DataFrame(field_df_data)
        
        # Save per-field comparison
        field_csv_path = output_dir / 'per_field_comparison.csv'
        field_df.to_csv(field_csv_path, index=False)
        print(f"\nPer-field comparison saved to: {field_csv_path}")
        
        # Create per-field heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        model_cols = [col for col in field_df.columns if col != 'Field']
        heatmap_data = field_df[model_cols].values
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(range(len(model_cols)))
        ax.set_yticks(range(len(field_df)))
        ax.set_xticklabels(model_cols)
        ax.set_yticklabels(field_df['Field'])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(len(field_df)):
            for j in range(len(model_cols)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Per-Field Accuracy Comparison', fontweight='bold', pad=20)
        
        plt.tight_layout()
        heatmap_path = output_dir / 'per_field_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Per-field heatmap saved to: {heatmap_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate all external models and compare")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['openai', 'gemini', 'deepseek'],
        choices=['openai', 'gemini', 'deepseek'],
        help="Models to evaluate (default: all)"
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
        default='outputs/external_models_comparison',
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EXTERNAL MODELS BATCH EVALUATION")
    print(f"{'='*80}\n")
    print(f"Config: {args.config}")
    print(f"Models: {', '.join([m.upper() for m in args.models])}")
    print(f"Output: {output_dir}")
    
    # Load dataset once
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
    
    # Evaluate each model
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Create model
            model = create_model(model_name, config)
            
            # Create model-specific output directory
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # Evaluate
            results = evaluate_model(
                model=model,
                samples=samples,
                num_samples=args.num_samples,
                output_dir=model_output_dir
            )
            
            # Save results
            results_file = model_output_dir / f"results_{model_name}.json"
            save_results(results, results_file)
            
            all_results[model_name] = results
            
            # Print metrics
            metrics = results['metrics']
            print(f"\nResults for {model_name.upper()}:")
            print(f"  Field Accuracy: {metrics['field_accuracy']:.2%}")
            print(f"  BLEU-1: {metrics['bleu1']:.4f}")
            print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
            print(f"  METEOR: {metrics['meteor']:.4f}")
            if 'sbert_similarity' in metrics:
                print(f"  Sentence-BERT: {metrics['sbert_similarity']:.4f}")
        
        except Exception as e:
            print(f"\nError evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison if we have results from multiple models
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("Creating comparison...")
        print(f"{'='*80}")
        compare_models(all_results, output_dir)
    
    print("\n" + "="*80)
    print("Batch evaluation complete!")
    print(f"All results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
