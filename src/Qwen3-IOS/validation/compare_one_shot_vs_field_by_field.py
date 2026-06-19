"""
Demo script for field-by-field decoder.

This script demonstrates how to use the FieldByFieldDecoder to generate
structured reports using ranking-based decoding.

Usage:
    python validation/compare_one_shot_vs_field_by_field.py --checkpoint <path> --config <path> \\
                                   --candidate_config <path> \\
                                   [--alpha 1.0] [--num_samples 5]
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import IOSPointCloudDataset
from validation import FieldByFieldDecoder, compare_decoders
from validation.validate import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Test field-by-field decoder")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='Path to training config file (optional if checkpoint has training_config.yaml)')
    
    # Decoder arguments
    parser.add_argument('--candidate_config', type=str, 
                        default='/work/grana_maxillo/IOS-DraftReport/_data/candidate_answers.json',
                        help='Path to candidate answers JSON')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Length normalization exponent (0=no norm, 1=mean log-prob)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to point cloud data directory')
    parser.add_argument('--captions_dir', type=str, default=None,
                        help='Path to captions directory')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to test (if set, ignores num_samples)')
    
    # Decoder parameters
    parser.add_argument('--scoring_batch_size', type=int, default=8,
                        help='Batch size for scoring candidates')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with one-shot generation')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config - either from checkpoint or from provided path
    checkpoint_config_path = os.path.join(args.checkpoint, 'training_config.yaml')
    
    if os.path.exists(checkpoint_config_path):
        print(f"\nLoading config from checkpoint: {checkpoint_config_path}")
        with open(checkpoint_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # If user also provided a config, warn them it will be ignored (except for overrides)
        if args.config:
            print(f"⚠️  Note: Checkpoint has training_config.yaml, ignoring --config {args.config}")
            print(f"         (Command-line overrides will still be applied)")
    elif args.config:
        print(f"\nLoading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(
            "No config file found! Either:\n"
            "  1. Checkpoint must contain training_config.yaml, OR\n"
            "  2. You must provide --config argument"
        )
    
    # Override data paths if provided
    if args.data_dir:
        config['data']['point_cloud_dir'] = args.data_dir
    if args.captions_dir:
        config['data']['captions_dir'] = args.captions_dir
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Point cloud dir: {config['data']['point_cloud_dir']}")
    print(f"Captions dir: {config['data']['captions_dir']}")
    
    dataset = IOSPointCloudDataset(
        point_cloud_dir=config['data']['point_cloud_dir'],
        captions_dir=config['data']['captions_dir'],
        num_points=config['data']['num_points'],
        normalize=config['data']['normalize'],
        augment=False,
        instruction_template_path=config['data'].get('instruction_template_path')
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print("="*70)
    
    # Initialize decoder
    print("\n" + "="*70)
    print("INITIALIZING FIELD-BY-FIELD DECODER")
    print("="*70)
    decoder = FieldByFieldDecoder(
        model=model,
        candidate_config_path=args.candidate_config,
        length_normalization_alpha=args.alpha,
    )
    print("="*70)
    
    # Test on samples
    if args.sample_idx is not None:
        # Test single specific sample
        sample_indices = [args.sample_idx]
    else:
        # Test random samples
        num_samples = min(args.num_samples, len(dataset))
        sample_indices = np.random.choice(len(dataset), num_samples, replace=False).tolist()
    
    print(f"\nTesting on {len(sample_indices)} sample(s)...")
    
    for i, idx in enumerate(sample_indices, 1):
        sample = dataset[idx]
        
        # Handle both numpy arrays and tensors
        point_cloud = sample['point_cloud']
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.from_numpy(point_cloud)
        point_cloud = point_cloud.unsqueeze(0).to(model.device)
        
        prompt = str(sample['instruction'])
        ground_truth = str(sample['description'])
        patient_id = str(sample['patient_id'])
        
        print("\n" + "="*70)
        print(f"SAMPLE {i}/{len(sample_indices)}: {patient_id}")
        print("="*70)
        
        if args.compare:
            # Compare both decoders
            one_shot, field_results = compare_decoders(
                model=model,
                point_cloud=point_cloud,
                prompt=prompt,
                candidate_config_path=args.candidate_config,
                alpha=args.alpha,
            )
            
            field_output = decoder.format_results(field_results)
            
            # Show ground truth
            print("\n3. GROUND TRUTH:")
            print("-" * 70)
            print(ground_truth.replace("<|im_end|>", "").strip())
            
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print("\nOne-shot output:")
            print(one_shot[:200] + "..." if len(one_shot) > 200 else one_shot)
            print("\nField-by-field output:")
            print(field_output)
            
        else:
            # Only field-by-field decoding
            print(f"\n📝 PROMPT:")
            print("-" * 70)
            print(prompt)
            
            print(f"\n✅ GROUND TRUTH:")
            print("-" * 70)
            print(ground_truth.replace("<|im_end|>", "").strip())
            
            print(f"\n🤖 FIELD-BY-FIELD GENERATION:")
            print("-" * 70)
            
            results = decoder.decode(
                point_cloud=point_cloud,
                base_prompt=prompt,
                verbose=True,
                scoring_batch_size=args.scoring_batch_size,
            )
            
            formatted_output = decoder.format_results(results)
            
            print("\n" + "-" * 70)
            print("FINAL OUTPUT:")
            print("-" * 70)
            print(formatted_output)
            print("-" * 70)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
