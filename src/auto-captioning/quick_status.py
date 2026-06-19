#!/usr/bin/env python3
"""
Quick script to show token status at a glance.
"""

import json
from pathlib import Path

def main():
    usage_file = Path("token_usage.json")
    
    if not usage_file.exists():
        print("📊 No token usage data yet.")
        print("Run 'python auto_caption.py' to start processing.")
        return
    
    with open(usage_file, 'r') as f:
        data = json.load(f)
    
    # Try to load config for limit
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        daily_limit = config['token_limits']['daily_limit']
    except:
        daily_limit = 250000
    
    tokens_used = data['total_tokens']
    percentage = (tokens_used / daily_limit) * 100
    remaining = daily_limit - tokens_used
    
    # Simple visual bar
    bar_length = 40
    filled = int(bar_length * percentage / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n📅 Date: {data['date']}")
    print(f"🔢 Tokens: {tokens_used:,} / {daily_limit:,}")
    print(f"📊 [{bar}] {percentage:.1f}%")
    print(f"✅ Patients: {len(data['patients'])}")
    print(f"⏳ Remaining: {remaining:,} tokens")
    
    if data['patients']:
        avg = tokens_used // len(data['patients'])
        est_remaining = remaining // avg if avg > 0 else 0
        print(f"📈 Avg/patient: {avg:,} tokens")
        print(f"🎯 Can process: ~{est_remaining} more patients today")
    
    print()

if __name__ == "__main__":
    main()
