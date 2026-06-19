"""
Test script to verify external model setup.

This script tests the external model clients without making actual API calls.

Usage:
    python test_external_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from external_models import BaseExternalModel, OpenAIModel, GeminiModel, DeepSeekModel
from PIL import Image
import tempfile
import os


def test_base_model():
    """Test base model functionality."""
    print("\n" + "="*80)
    print("Testing BaseExternalModel...")
    print("="*80)
    
    # Create a dummy subclass
    class DummyModel(BaseExternalModel):
        def generate(self, image_paths, field_names=None):
            return "Test output"
    
    model = DummyModel(
        api_key="test_key",
        model_name="test_model",
        system_prompt="Test system",
        user_prompt_template="Test user {fields}"
    )
    
    # Test field name extraction
    description = """Overbite: Deep
Crowding: Moderate
Molar occlusion - Right side: Class I"""
    
    field_names = model.prepare_field_names(description)
    print(f"✓ Field extraction: {field_names}")
    
    # Test image loading and encoding
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f, 'JPEG')
        temp_path = Path(f.name)
    
    try:
        loaded_img = model.load_image(temp_path)
        print(f"✓ Image loading: {loaded_img.size}")
        
        base64_str = model.image_to_base64(loaded_img)
        print(f"✓ Image encoding: {len(base64_str)} chars")
    finally:
        temp_path.unlink()
    
    print("✓ BaseExternalModel tests passed!")


def test_client_initialization():
    """Test client initialization (without API calls)."""
    print("\n" + "="*80)
    print("Testing Client Initialization...")
    print("="*80)
    
    # Test OpenAI (will fail if no API key, which is expected)
    print("\nOpenAI Client:")
    try:
        model = OpenAIModel(
            api_key="test_key",
            model_name="gpt-4-vision-preview"
        )
        print(f"✓ OpenAI client initialized: {model.model_name}")
    except Exception as e:
        print(f"✗ OpenAI initialization error: {e}")
    
    # Test Gemini
    print("\nGemini Client:")
    try:
        # Set a dummy API key to avoid env var check
        os.environ['GOOGLE_API_KEY'] = 'test_key'
        model = GeminiModel(
            api_key="test_key",
            model_name="gemini-pro-vision"
        )
        print(f"✓ Gemini client initialized: {model.model_name}")
    except Exception as e:
        print(f"✗ Gemini initialization error: {e}")
    finally:
        # Clean up
        if 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
    
    # Test DeepSeek
    print("\nDeepSeek Client:")
    try:
        model = DeepSeekModel(
            api_key="test_key",
            model_name="deepseek-vl-7b-chat"
        )
        print(f"✓ DeepSeek client initialized: {model.model_name}")
    except Exception as e:
        print(f"✗ DeepSeek initialization error: {e}")


def test_config_loading():
    """Test configuration file loading."""
    print("\n" + "="*80)
    print("Testing Configuration Loading...")
    print("="*80)
    
    import yaml
    
    config_path = Path(__file__).parent.parent / 'configs' / 'external_models_eval.yaml'
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded successfully")
        print(f"  Dataset dir: {config['data']['dataset_dir']}")
        print(f"  Captions dir: {config['data']['captions_dir']}")
        print(f"  Models configured: {', '.join(config['models'].keys())}")
        
        # Check each model config
        for model_name, model_config in config['models'].items():
            print(f"\n  {model_name.upper()}:")
            print(f"    Model: {model_config['model_name']}")
            print(f"    Temp: {model_config['temperature']}")
            print(f"    Max tokens: {model_config['max_tokens']}")
        
        print("\n✓ Configuration validation passed!")
        
    except Exception as e:
        print(f"✗ Config loading error: {e}")


def test_dataset_loading():
    """Test dataset loading (first 3 samples)."""
    print("\n" + "="*80)
    print("Testing Dataset Loading...")
    print("="*80)
    
    import yaml
    from scripts.evaluate_external_models import load_intraoral_dataset
    
    config_path = Path(__file__).parent.parent / 'configs' / 'external_models_eval.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_dir = Path(config['data']['dataset_dir'])
        captions_dir = Path(config['data']['captions_dir'])
        
        if not dataset_dir.exists():
            print(f"✗ Dataset directory not found: {dataset_dir}")
            print("  Please update the path in configs/evaluation/external_model_baselines.yaml")
            return
        
        if not captions_dir.exists():
            print(f"✗ Captions directory not found: {captions_dir}")
            print("  Please update the path in configs/evaluation/external_model_baselines.yaml")
            return
        
        samples = load_intraoral_dataset(
            dataset_dir=dataset_dir,
            captions_dir=captions_dir,
            min_photos=3
        )
        
        print(f"✓ Loaded {len(samples)} samples")
        
        if len(samples) > 0:
            print(f"\nFirst 3 samples:")
            for i, sample in enumerate(samples[:3]):
                print(f"\n  Sample {i+1}:")
                print(f"    Patient: {sample['patient_id']}")
                print(f"    Photos: {len(sample['photo_paths'])}")
                print(f"    Description length: {len(sample['description'])} chars")
        else:
            print("\n✗ No samples found! Check your dataset paths.")
        
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*80)
    print("EXTERNAL MODEL SETUP TEST")
    print("="*80)
    
    test_base_model()
    test_client_initialization()
    test_config_loading()
    test_dataset_loading()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nSetup test complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_external.txt")
    print("2. Set API keys as environment variables:")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export GOOGLE_API_KEY='your-key'")
    print("   export DEEPSEEK_API_KEY='your-key'")
    print("3. Update model names in configs/evaluation/external_model_baselines.yaml if needed")
    print("4. Run evaluation:")
    print("   python scripts/evaluate_external_models.py --config configs/evaluation/external_model_baselines.yaml --model openai --num_samples 5")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
