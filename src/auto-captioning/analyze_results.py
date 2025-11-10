#!/usr/bin/env python3
"""
Utility script to analyze and inspect auto-captioning results.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_results(output_dir: str = "outputs") -> List[Dict]:
    """Load all individual result files."""
    output_path = Path(output_dir)
    results = []
    
    for json_file in output_path.glob("*.json"):
        if json_file.name == "all_descriptions.json":
            continue
        
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    
    return sorted(results, key=lambda x: x['patient_id'])


def show_statistics(results: List[Dict]):
    """Show statistics about the results."""
    print("\n" + "="*60)
    print("RESULTS STATISTICS")
    print("="*60)
    print(f"Total patients processed: {len(results)}")
    
    if not results:
        print("No results found.")
        return
    
    # Description lengths
    lengths = [len(r['description']) for r in results]
    print(f"\nDescription lengths:")
    print(f"  Average: {sum(lengths)/len(lengths):.0f} characters")
    print(f"  Min: {min(lengths)} characters")
    print(f"  Max: {max(lengths)} characters")
    
    # Models used
    models = {}
    for r in results:
        model = r.get('model', 'unknown')
        models[model] = models.get(model, 0) + 1
    
    print(f"\nModels used:")
    for model, count in models.items():
        print(f"  {model}: {count} patients")
    
    print("="*60 + "\n")


def show_patient(patient_id: str, output_dir: str = "outputs"):
    """Show details for a specific patient."""
    json_file = Path(output_dir) / f"{patient_id}.json"
    
    if not json_file.exists():
        print(f"Error: Patient {patient_id} not found in {output_dir}")
        return
    
    with open(json_file, 'r') as f:
        result = json.load(f)
    
    print("\n" + "="*60)
    print(f"PATIENT: {result['patient_id']}")
    print("="*60)
    print(f"Timestamp: {result['timestamp']}")
    print(f"Model: {result['model']}")
    print(f"\nDescription ({len(result['description'])} characters):")
    print("-"*60)
    print(result['description'])
    print("="*60 + "\n")


def list_patients(output_dir: str = "outputs", limit: int = None):
    """List all processed patients."""
    results = load_results(output_dir)
    
    print("\n" + "="*60)
    print("PROCESSED PATIENTS")
    print("="*60)
    
    for i, result in enumerate(results[:limit] if limit else results, 1):
        desc_len = len(result['description'])
        timestamp = result['timestamp'][:19]  # Remove microseconds
        print(f"{i:3}. {result['patient_id']:20} ({desc_len:5} chars, {timestamp})")
    
    if limit and len(results) > limit:
        print(f"\n... and {len(results) - limit} more")
    
    print("="*60 + "\n")


def search_description(keyword: str, output_dir: str = "outputs"):
    """Search for a keyword in descriptions."""
    results = load_results(output_dir)
    
    matches = []
    for result in results:
        if keyword.lower() in result['description'].lower():
            matches.append(result)
    
    print("\n" + "="*60)
    print(f"SEARCH RESULTS for '{keyword}'")
    print("="*60)
    print(f"Found {len(matches)} matches out of {len(results)} patients\n")
    
    for match in matches:
        print(f"Patient: {match['patient_id']}")
        # Find context around keyword
        desc_lower = match['description'].lower()
        keyword_lower = keyword.lower()
        idx = desc_lower.find(keyword_lower)
        
        start = max(0, idx - 50)
        end = min(len(desc_lower), idx + len(keyword) + 50)
        context = match['description'][start:end]
        
        if start > 0:
            context = "..." + context
        if end < len(match['description']):
            context = context + "..."
        
        print(f"  Context: {context}")
        print()
    
    print("="*60 + "\n")


def export_to_csv(output_dir: str = "outputs", output_file: str = "descriptions.csv"):
    """Export descriptions to a CSV file."""
    results = load_results(output_dir)
    
    if not results:
        print("No results to export")
        return
    
    df = pd.DataFrame([
        {
            'patient_id': r['patient_id'],
            'description': r['description'],
            'description_length': len(r['description']),
            'timestamp': r['timestamp'],
            'model': r['model']
        }
        for r in results
    ])
    
    df.to_csv(output_file, index=False)
    print(f"Exported {len(results)} descriptions to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze and inspect auto-captioning results"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory containing results'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all processed patients')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show details for a patient')
    show_parser.add_argument('patient_id', help='Patient ID to show')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for keyword in descriptions')
    search_parser.add_argument('keyword', help='Keyword to search for')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export to CSV')
    export_parser.add_argument('--output', default='descriptions.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'stats':
        results = load_results(args.output_dir)
        show_statistics(results)
    
    elif args.command == 'list':
        list_patients(args.output_dir, args.limit)
    
    elif args.command == 'show':
        show_patient(args.patient_id, args.output_dir)
    
    elif args.command == 'search':
        search_description(args.keyword, args.output_dir)
    
    elif args.command == 'export':
        export_to_csv(args.output_dir, args.output)


if __name__ == "__main__":
    main()
