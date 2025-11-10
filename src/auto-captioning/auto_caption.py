#!/usr/bin/env python3
"""
Auto-captioning system for dental intraoral photos using OpenAI Vision API.

This script processes patient intraoral photos and generates clinical descriptions
of their occlusion using GPT-4o's vision capabilities.
"""

import os
import sys
import json
import time
import logging
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import yaml
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image


class AutoCaptioner:
    """Main class for automatic caption generation of dental intraoral photos."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the AutoCaptioner.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        org_id = os.getenv('OPENAI_ORG_ID')
        if org_id:
            self.client = OpenAI(api_key=api_key, organization=org_id)
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Setup output directory
        self.output_dir = Path(self.config['output']['dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize token tracking
        self.token_usage_file = Path(self.config['token_limits']['usage_file'])
        self.daily_token_data = self._load_token_usage()
        
        # Initialize statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'tokens_used': 0,
            'start_time': datetime.now()
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        log_file = self.config['logging']['file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        template_path = Path(self.config['prompt']['template_path'])
        with open(template_path, 'r') as f:
            return f.read()
    
    def _load_token_usage(self) -> Dict:
        """Load token usage data from file."""
        if not self.token_usage_file.exists():
            return {'date': datetime.now().strftime('%Y-%m-%d'), 'total_tokens': 0, 'patients': []}
        
        with open(self.token_usage_file, 'r') as f:
            data = json.load(f)
        
        # Reset if it's a new day
        today = datetime.now().strftime('%Y-%m-%d')
        if data.get('date') != today:
            self.logger.info(f"New day detected. Previous usage: {data.get('total_tokens', 0)} tokens")
            return {'date': today, 'total_tokens': 0, 'patients': []}
        
        return data
    
    def _save_token_usage(self):
        """Save token usage data to file."""
        with open(self.token_usage_file, 'w') as f:
            json.dump(self.daily_token_data, f, indent=2)
    
    def _check_token_limit(self, estimated_tokens: int = 0) -> bool:
        """
        Check if we're within the daily token limit.
        
        Args:
            estimated_tokens: Estimated tokens for next request
            
        Returns:
            True if within limit, False if limit would be exceeded
        """
        daily_limit = self.config['token_limits']['daily_limit']
        safety_margin = self.config['token_limits']['safety_margin']
        safe_limit = int(daily_limit * safety_margin)
        
        current_usage = self.daily_token_data['total_tokens']
        projected_usage = current_usage + estimated_tokens
        
        if False and projected_usage >= safe_limit:
            self.logger.warning(
                f"Approaching daily token limit: {current_usage}/{daily_limit} tokens used "
                f"(safe limit: {safe_limit})"
            )
            return False
        
        return True
    
    def _update_token_usage(self, patient_id: str, tokens_used: int):
        """Update token usage tracking."""
        self.daily_token_data['total_tokens'] += tokens_used
        self.daily_token_data['patients'].append({
            'patient_id': patient_id,
            'tokens': tokens_used,
            'timestamp': datetime.now().isoformat()
        })
        self._save_token_usage()
        self.stats['tokens_used'] += tokens_used
    
    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _find_image_with_extension(self, photos_dir: Path, view_name: str) -> Optional[Path]:
        """
        Find an image file with any supported extension.
        
        Args:
            photos_dir: Directory containing photos
            view_name: Base name of the view (without extension)
            
        Returns:
            Path to the image file, or None if not found
        """
        extensions = self.config['dataset'].get('image_extensions', ['.jpg', '.jpeg', '.png'])
        
        for ext in extensions:
            img_path = photos_dir / f"{view_name}{ext}"
            if img_path.exists():
                return img_path
        
        return None
    
    def _validate_patient_images(self, patient_dir: Path) -> Tuple[bool, List[Path]]:
        """
        Validate that all required images exist for a patient.
        Searches for images with .jpg, .jpeg, or .png extensions.
        
        Args:
            patient_dir: Path to patient directory
            
        Returns:
            Tuple of (valid, list of image paths)
        """
        photos_dir = patient_dir / self.config['dataset']['photos_subdir']
        
        if not photos_dir.exists():
            return False, []
        
        image_paths = []
        for view in self.config['dataset']['views']:
            img_path = self._find_image_with_extension(photos_dir, view)
            if img_path is None:
                self.logger.warning(f"Missing {view} (tried .jpg, .jpeg, .png) in {patient_dir.name}")
                return False, []
            image_paths.append(img_path)
        
        return True, image_paths
    
    # def _prepare_messages(self, image_paths: List[Path]) -> List[Dict]:
    #     """
    #     Prepare messages for OpenAI API including images.
        
    #     Args:
    #         image_paths: List of paths to the 5 intraoral images
            
    #     Returns:
    #         List of message dictionaries for the API
    #     """
    #     content = [
    #         {
    #             "type": "text",
    #             "text": self.prompt_template
    #         }
    #     ]
        
    #     # Add images in order
    #     detail = self.config['api']['detail']
    #     for img_path in image_paths:
    #         # Read and encode image
    #         with open(img_path, "rb") as image_file:
    #             base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
    #         content.append({
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": f"data:image/jpeg;base64,{base64_image}",
    #                 "detail": detail
    #             }
    #         })
        
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": content
    #         }
    #     ]
        
    #     return messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def _call_openai_api(self, messages: List[Dict]) -> Tuple[str, int]:
        """
        Call OpenAI API with retry logic.
        
        Args:
            messages: Messages to send to the API
            
        Returns:
            Tuple of (generated description text, tokens used)
        """
        response = self.client.chat.completions.create(
            model=self.config['api']['model'],
            messages=messages,
            max_tokens=self.config['api']['max_tokens'],
            temperature=self.config['api']['temperature']
        )
        
        # Get token usage
        tokens_used = response.usage.total_tokens if response.usage else 0
        content = response.choices[0].message.content or ""
        
        return content, tokens_used
    
    def process_patient(self, patient_dir: Path) -> Optional[Dict]:
        """
        Process a single patient's images and generate description.
        
        Args:
            patient_dir: Path to patient directory
            
        Returns:
            Dictionary with patient ID and description, or None if failed
        """
        patient_id = patient_dir.name
        
        # Check if already processed and resume is enabled
        if self.config['output']['resume']:
            output_file = self.output_dir / f"{patient_id}.json"
            if output_file.exists():
                self.logger.info(f"Skipping {patient_id} (already processed)")
                self.stats['skipped'] += 1
                with open(output_file, 'r') as f:
                    return json.load(f)
        
        # Validate images
        # valid, image_paths = self._validate_patient_images(patient_dir)
        # if not valid:
        #     self.logger.error(f"Invalid or incomplete images for {patient_id}")
        #     self.stats['failed'] += 1
        #     return None
        
        # Check token limit (estimate ~10k tokens per patient)
        if False and not self._check_token_limit(estimated_tokens=10000):
            self.logger.warning(
                f"Daily token limit reached. Stopping processing. "
                f"Used: {self.daily_token_data['total_tokens']} tokens today."
            )
            return None
        
        try:
            # Prepare messages
            messages = self._prepare_messages(image_paths)
            
            # Call API
            self.logger.info(f"Processing {patient_id}...")
            description, tokens_used = self._call_openai_api(messages)
            
            # Update token tracking
            self._update_token_usage(patient_id, tokens_used)
            self.logger.info(f"Tokens used: {tokens_used} (total today: {self.daily_token_data['total_tokens']})")
            
            # Prepare result
            result = {
                'patient_id': patient_id,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'model': self.config['api']['model'],
                'tokens_used': tokens_used,
                'image_paths': [str(p) for p in image_paths]
            }
            
            # Save individual result
            if self.config['output']['save_individual']:
                output_file = self.output_dir / f"{patient_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            self.stats['success'] += 1
            self.logger.info(f"Successfully processed {patient_id}")
            
            # Rate limiting
            time.sleep(self.config['processing']['rate_limit_delay'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process {patient_id}: {str(e)}")
            self.stats['failed'] += 1
            return None
    
    def process_all_patients(self) -> List[Dict]:
        """
        Process all patients in the dataset.
        
        Returns:
            List of results for all patients
        """
        dataset_root = Path(self.config['dataset']['root_path'])
        
        # Get all patient directories
        patient_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
        self.stats['total'] = len(patient_dirs)
        
        self.logger.info(f"Found {len(patient_dirs)} patients to process")
        
        results = []
        
        # Process each patient with progress bar
        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            result = self.process_patient(patient_dir)
            if result:
                results.append(result)
        
        return results
    
    def save_consolidated_results(self, results: List[Dict]):
        """
        Save consolidated results to JSON and CSV files.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            self.logger.warning("No results to save")
            return
        
        # Save JSON
        if self.config['output']['save_json']:
            json_file = self.output_dir / "all_descriptions.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved consolidated JSON to {json_file}")
        
        # Save CSV
        if self.config['output']['save_csv']:
            csv_file = self.output_dir / "all_descriptions.csv"
            df = pd.DataFrame([
                {
                    'patient_id': r['patient_id'],
                    'description': r['description'],
                    'timestamp': r['timestamp'],
                    'model': r['model']
                }
                for r in results
            ])
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved consolidated CSV to {csv_file}")
    
    def print_statistics(self):
        """Print processing statistics."""
        duration = datetime.now() - self.stats['start_time']
        daily_limit = self.config['token_limits']['daily_limit']
        tokens_used = self.daily_token_data['total_tokens']
        percentage = (tokens_used / daily_limit) * 100
        
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total patients:         {self.stats['total']}")
        print(f"Successfully processed: {self.stats['success']}")
        print(f"Failed:                 {self.stats['failed']}")
        print(f"Skipped (already done): {self.stats['skipped']}")
        print(f"Duration:               {duration}")
        print()
        print(f"TOKEN USAGE:")
        print(f"  Session tokens:  {self.stats['tokens_used']:,}")
        print(f"  Daily tokens:    {tokens_used:,} / {daily_limit:,} ({percentage:.1f}%)")
        print(f"  Remaining today: {daily_limit - tokens_used:,}")
        if self.stats['success'] > 0:
            avg_tokens = tokens_used // len(self.daily_token_data['patients'])
            print(f"  Avg per patient: {avg_tokens:,}")
        print("="*60 + "\n")
    
    def run(self):
        """Main execution method."""
        self.logger.info("Starting auto-captioning process")
        self.logger.info(f"Using model: {self.config['api']['model']}")
        
        try:
            # Process all patients
            results = self.process_all_patients()
            
            # Save consolidated results
            self.save_consolidated_results(results)
            
            # Print statistics
            self.print_statistics()
            
            self.logger.info("Auto-captioning process completed")
            
        except KeyboardInterrupt:
            self.logger.warning("Process interrupted by user")
            self.print_statistics()
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}", exc_info=True)
            self.print_statistics()
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto-caption dental intraoral photos using OpenAI Vision API"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--patient',
        type=str,
        help='Process only a specific patient (for testing)'
    )
    
    args = parser.parse_args()
    
    # Initialize captioner
    captioner = AutoCaptioner(config_path=args.config)
    
    # Process single patient or all
    if args.patient:
        dataset_root = Path(captioner.config['dataset']['root_path'])
        patient_dir = dataset_root / args.patient
        
        if not patient_dir.exists():
            print(f"Error: Patient directory {patient_dir} not found")
            sys.exit(1)
        
        result = captioner.process_patient(patient_dir)
        if result:
            print("\n" + "="*60)
            print(f"Patient: {result['patient_id']}")
            print("="*60)
            print(result['description'])
            print("="*60 + "\n")
    else:
        captioner.run()


if __name__ == "__main__":
    main()
