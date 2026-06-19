#!/usr/bin/env python3
"""
Test script to verify the setup and test on a single patient.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    required = ['openai', 'yaml', 'dotenv', 'pandas', 'tqdm', 'tenacity', 'PIL']
    missing = []
    
    for pkg in required:
        try:
            if pkg == 'yaml':
                import yaml
            elif pkg == 'dotenv':
                import dotenv
            elif pkg == 'PIL':
                import PIL
            else:
                __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print("\nMissing packages. Install with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("All dependencies installed!\n")
    return True

def check_env():
    """Check if .env file exists and has API key."""
    print("Checking environment configuration...")
    
    if not Path('.env').exists():
        print("  ✗ .env file not found")
        print("\nCreate .env file:")
        print("  cp .env.example .env")
        print("  # Then edit .env and add your OpenAI API key")
        return False
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("  ✗ OPENAI_API_KEY not set in .env")
        print("\nEdit .env and add:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("  ✗ OPENAI_API_KEY still has placeholder value")
        print("\nEdit .env and replace with your actual API key")
        return False
    
    print(f"  ✓ API key found (length: {len(api_key)})")
    print("Environment configured!\n")
    return True

def check_config():
    """Check if config file exists."""
    print("Checking configuration file...")
    
    if not Path('config.yaml').exists():
        print("  ✗ config.yaml not found")
        return False
    
    print("  ✓ config.yaml found")
    print("Configuration file ready!\n")
    return True

def check_dataset():
    """Check if dataset is accessible."""
    print("Checking dataset...")
    
    dataset_path = Path("/work/grana_maxillo/IOS-DraftReport/_data/Dataset_FerraraDump_ToothFairy4M")
    
    if not dataset_path.exists():
        print(f"  ✗ Dataset not found at {dataset_path}")
        return False
    
    # Check for at least one valid patient
    patient_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not patient_dirs:
        print("  ✗ No patient directories found")
        return False
    
    # Check first patient
    first_patient = patient_dirs[0]
    photos_dir = first_patient / "intraoral-photos"
    
    if not photos_dir.exists():
        print(f"  ✗ No intraoral-photos folder in {first_patient.name}")
        return False
    
    # Check for required image views (with multiple possible extensions)
    required_views = ['left', 'right', 'up', 'down', 'center']
    extensions = ['.jpg', '.jpeg', '.png']
    
    missing = []
    for view in required_views:
        found = False
        for ext in extensions:
            if (photos_dir / f"{view}{ext}").exists():
                found = True
                break
        if not found:
            missing.append(view)
    
    if missing:
        print(f"  ✗ Missing images in {first_patient.name}: {missing}")
        print(f"     (checked extensions: {', '.join(extensions)})")
        return False
    
    print(f"  ✓ Dataset accessible")
    print(f"  ✓ Found {len(patient_dirs)} patient directories")
    print(f"  ✓ Sample patient {first_patient.name} has all required images")
    print("Dataset ready!\n")
    return True

def check_prompt():
    """Check if prompt template exists."""
    print("Checking prompt template...")
    
    prompt_path = Path("/work/grana_maxillo/IOS-DraftReport/_data/prompt-template.txt")
    
    if not prompt_path.exists():
        print(f"  ✗ Prompt template not found at {prompt_path}")
        return False
    
    with open(prompt_path) as f:
        content = f.read()
    
    print(f"  ✓ Prompt template found ({len(content)} characters)")
    print("Prompt template ready!\n")
    return True

def main():
    """Run all checks."""
    print("="*60)
    print("AUTO-CAPTIONING SYSTEM - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Environment", check_env),
        ("Configuration", check_config),
        ("Prompt Template", check_prompt),
        ("Dataset", check_dataset),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20} {status}")
    
    print("="*60 + "\n")
    
    if all(results.values()):
        print("✓ All checks passed! Ready to run:")
        print("\n  # Test on single patient:")
        print("  python auto_caption.py --patient 201_1979")
        print("\n  # Process all patients:")
        print("  python auto_caption.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
