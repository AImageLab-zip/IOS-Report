#!/usr/bin/env python3
"""
Utility to view and manage token usage.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_token_usage(usage_file: str = "token_usage.json"):
    """Load token usage data."""
    path = Path(usage_file)
    if not path.exists():
        print("No token usage data found.")
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def show_usage(usage_file: str = "token_usage.json"):
    """Show current token usage."""
    data = load_token_usage(usage_file)
    if not data:
        return
    
    print("\n" + "="*60)
    print("DAILY TOKEN USAGE")
    print("="*60)
    print(f"Date: {data['date']}")
    print(f"Total tokens used: {data['total_tokens']:,}")
    print(f"Patients processed: {len(data['patients'])}")
    
    if data['patients']:
        avg_tokens = data['total_tokens'] // len(data['patients'])
        print(f"Average per patient: {avg_tokens:,}")
    
    # Load config to show limit
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        daily_limit = config['token_limits']['daily_limit']
        percentage = (data['total_tokens'] / daily_limit) * 100
        remaining = daily_limit - data['total_tokens']
        
        print()
        print(f"Daily limit: {daily_limit:,}")
        print(f"Used: {percentage:.1f}%")
        print(f"Remaining: {remaining:,} tokens")
        
        # Estimate patients remaining
        if data['patients']:
            avg_per_patient = data['total_tokens'] / len(data['patients'])
            patients_remaining = int(remaining / avg_per_patient)
            print(f"Estimated patients remaining: ~{patients_remaining}")
    except:
        pass
    
    print("="*60 + "\n")


def show_details(usage_file: str = "token_usage.json", limit: int = None):
    """Show detailed per-patient token usage."""
    data = load_token_usage(usage_file)
    if not data:
        return
    
    print("\n" + "="*60)
    print("DETAILED TOKEN USAGE")
    print("="*60)
    
    patients = data['patients']
    if limit:
        patients = patients[:limit]
    
    for i, patient in enumerate(patients, 1):
        timestamp = patient['timestamp'][:19]  # Remove microseconds
        print(f"{i:3}. {patient['patient_id']:20} {patient['tokens']:6,} tokens  ({timestamp})")
    
    if limit and len(data['patients']) > limit:
        print(f"\n... and {len(data['patients']) - limit} more")
    
    print("="*60 + "\n")


def reset_usage(usage_file: str = "token_usage.json"):
    """Reset token usage (use with caution!)."""
    print("⚠️  WARNING: This will reset all token usage data!")
    confirm = input("Type 'RESET' to confirm: ")
    
    if confirm != "RESET":
        print("Cancelled.")
        return
    
    data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_tokens': 0,
        'patients': []
    }
    
    with open(usage_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ Token usage reset.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View and manage token usage")
    parser.add_argument(
        '--file',
        default='token_usage.json',
        help='Token usage file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Show command
    subparsers.add_parser('show', help='Show current usage summary')
    
    # Details command
    details_parser = subparsers.add_parser('details', help='Show detailed per-patient usage')
    details_parser.add_argument('--limit', type=int, help='Limit number of results')
    
    # Reset command
    subparsers.add_parser('reset', help='Reset token usage (use with caution!)')
    
    args = parser.parse_args()
    
    if not args.command or args.command == 'show':
        show_usage(args.file)
    elif args.command == 'details':
        show_details(args.file, args.limit)
    elif args.command == 'reset':
        reset_usage(args.file)


if __name__ == "__main__":
    main()
