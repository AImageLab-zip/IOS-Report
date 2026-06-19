#!/usr/bin/env python3
"""
Test script for validation metrics.

This demonstrates the metrics on example predictions vs ground truth reports.
Run this to verify the metrics are working correctly before training.

Usage:
    python test_metrics.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from validation import compute_all_metrics


def test_example_1():
    """Test case 1: Partially correct prediction (from user's example)"""
    print("\n" + "="*80)
    print("TEST CASE 1: Partially Correct (General vs Specific)")
    print("="*80)
    
    predictions = [
        "Molar occlusion - Right side: Class I\nMolar occlusion - Left side: Class I"
    ]
    
    references = [
        "Molar occlusion - Left side: Class II head-to-head\nMolar occlusion - Right side: Class I"
    ]
    
    print("\nGround Truth:")
    print(references[0])
    print("\nPrediction:")
    print(predictions[0])
    
    metrics = compute_all_metrics(
        predictions,
        references,
        include_sbert=True,
        return_per_field=True
    )
    
    print("\n" + "-"*80)
    print("METRICS:")
    print("-"*80)
    for key, value in metrics.items():
        if key != 'per_field':
            if isinstance(value, float):
                print(f"{key:<25} {value:>10.3f}")
            else:
                print(f"{key:<25} {value:>10}")
        else:
            print(f"\nPer-field accuracy:")
            for field, score in value.items():
                print(f"  {field:<23} {score:>10.3f}")
    
    print("\n" + "="*80)


def test_example_2():
    """Test case 2: Completely accurate prediction"""
    print("\n" + "="*80)
    print("TEST CASE 2: Perfect Match")
    print("="*80)
    
    predictions = [
        "Overbite: Normal\nCrowding: Moderate in the upper arch\nMolar occlusion - Right side: Class I"
    ]
    
    references = [
        "Overbite: Normal\nCrowding: Moderate in the upper arch\nMolar occlusion - Right side: Class I"
    ]
    
    print("\nGround Truth:")
    print(references[0])
    print("\nPrediction:")
    print(predictions[0])
    
    metrics = compute_all_metrics(
        predictions,
        references,
        include_sbert=True,
        return_per_field=True
    )
    
    print("\n" + "-"*80)
    print("METRICS:")
    print("-"*80)
    for key, value in metrics.items():
        if key != 'per_field':
            if isinstance(value, float):
                print(f"{key:<25} {value:>10.3f}")
            else:
                print(f"{key:<25} {value:>10}")
        else:
            print(f"\nPer-field accuracy:")
            for field, score in value.items():
                print(f"  {field:<23} {score:>10.3f}")
    
    print("\n" + "="*80)


def test_example_3():
    """Test case 3: Completely wrong prediction"""
    print("\n" + "="*80)
    print("TEST CASE 3: Completely Wrong")
    print("="*80)
    
    predictions = [
        "Overbite: Increased\nCrowding: Severe in both arches\nMolar occlusion - Right side: Class III"
    ]
    
    references = [
        "Overbite: Normal\nCrowding: Absent\nMolar occlusion - Right side: Class I"
    ]
    
    print("\nGround Truth:")
    print(references[0])
    print("\nPrediction:")
    print(predictions[0])
    
    metrics = compute_all_metrics(
        predictions,
        references,
        include_sbert=True,
        return_per_field=True
    )
    
    print("\n" + "-"*80)
    print("METRICS:")
    print("-"*80)
    for key, value in metrics.items():
        if key != 'per_field':
            if isinstance(value, float):
                print(f"{key:<25} {value:>10.3f}")
            else:
                print(f"{key:<25} {value:>10}")
        else:
            print(f"\nPer-field accuracy:")
            for field, score in value.items():
                print(f"  {field:<23} {score:>10.3f}")
    
    print("\n" + "="*80)


def test_example_4():
    """Test case 4: Missing fields"""
    print("\n" + "="*80)
    print("TEST CASE 4: Missing Fields (Low Coverage)")
    print("="*80)
    
    predictions = [
        "Overbite: Normal"
    ]
    
    references = [
        "Overbite: Normal\nCrowding: Moderate in the upper arch\nMolar occlusion - Right side: Class I\nCanine occlusion - Left side: Class II"
    ]
    
    print("\nGround Truth:")
    print(references[0])
    print("\nPrediction:")
    print(predictions[0])
    
    metrics = compute_all_metrics(
        predictions,
        references,
        include_sbert=True,
        return_per_field=True
    )
    
    print("\n" + "-"*80)
    print("METRICS:")
    print("-"*80)
    for key, value in metrics.items():
        if key != 'per_field':
            if isinstance(value, float):
                print(f"{key:<25} {value:>10.3f}")
            else:
                print(f"{key:<25} {value:>10}")
        else:
            print(f"\nPer-field accuracy:")
            for field, score in value.items():
                print(f"  {field:<23} {score:>10.3f}")
    
    print("\n" + "="*80)


def test_batch():
    """Test case 5: Batch of predictions"""
    print("\n" + "="*80)
    print("TEST CASE 5: Batch Processing")
    print("="*80)
    
    predictions = [
        "Molar occlusion - Right side: Class I\nOverbite: Normal",
        "Crowding: Moderate in both arches\nMidlines: Centered",
        "Missing teeth: None\nCurve of Spee: Normal"
    ]
    
    references = [
        "Molar occlusion - Right side: Class II\nOverbite: Normal",
        "Crowding: Moderate in both arches\nMidlines: Slightly deviated",
        "Missing teeth: None\nCurve of Spee: Normal"
    ]
    
    print(f"\nProcessing {len(predictions)} samples...")
    
    metrics = compute_all_metrics(
        predictions,
        references,
        include_sbert=True,
        return_per_field=True
    )
    
    print("\n" + "-"*80)
    print("METRICS (AVERAGED OVER BATCH):")
    print("-"*80)
    for key, value in metrics.items():
        if key != 'per_field':
            if isinstance(value, float):
                print(f"{key:<25} {value:>10.3f}")
            else:
                print(f"{key:<25} {value:>10}")
        else:
            print(f"\nPer-field accuracy:")
            for field, score in value.items():
                print(f"  {field:<23} {score:>10.3f}")
    
    print("\n" + "="*80)


def main():
    """Run all test cases."""
    print("\n" + "="*80)
    print(" TESTING VALIDATION METRICS")
    print("="*80)
    print("\nThis script tests the validation metrics on example predictions.")
    print("It demonstrates how the metrics behave in different scenarios.")
    
    try:
        test_example_1()  # User's example: partially correct
        test_example_2()  # Perfect match
        test_example_3()  # Completely wrong
        test_example_4()  # Missing fields
        test_batch()      # Batch processing
        
        print("\n" + "="*80)
        print(" ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Insights:")
        print("- Field Accuracy: Measures correctness of individual medical fields")
        print("- Coverage: Percentage of required fields present in prediction")
        print("- BLEU-1: Word-level precision")
        print("- ROUGE-L: Sentence structure similarity")
        print("- METEOR: Morphology-aware matching")
        print("- Sentence-BERT: Semantic similarity")
        print("\nYou can now use these metrics in training by setting:")
        print("  compute_generation_metrics: true")
        print("in your config file.")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR DURING TESTING")
        print("="*80)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed the required packages:")
        print("  pip install -r validation/requirements.txt")
        print("="*80 + "\n")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
