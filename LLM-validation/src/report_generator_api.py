"""
Unified API-based Report Generator for closed-source models.

Supports OpenAI (GPT-5.2), Google Gemini (Gemini3), and DeepSeek (DeepSeek3.2).
Features:
- Concurrent API calls for faster processing
- Automatic rate limiting
- Batch processing support
"""

import os
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional, Callable
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Semaphore
from tqdm import tqdm


class APIReportGenerator:
    """
    Generate orthodontic reports using closed-source APIs.
    
    Supports multiple providers: OpenAI, Google Gemini, DeepSeek.
    """
    
    SUPPORTED_PROVIDERS = {
        'openai': ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-5.2', 'gpt-5'],
        'gemini': ['gemini-3-flash-preview', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini3', 'gemini3-flash'],
        'deepseek': ['deepseek-vl-7b-chat', 'deepseek3.2', 'deepseek-chat']
    }
    
    def __init__(
        self, 
        model_name: str,
        system_prompt_path: str, 
        user_prompt_path: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize the API report generator.
        
        Args:
            model_name: Model identifier (e.g., 'gpt-5.2', 'gemini3', 'deepseek3.2')
            system_prompt_path: Path to system prompt file
            user_prompt_path: Path to user prompt file
            api_key: API key (if None, uses environment variable)
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        
        # Determine provider
        self.provider = self._detect_provider(model_name)
        
        # Rate limiting semaphore (max concurrent requests)
        self.rate_limit = self._get_rate_limit(self.provider)
        self.semaphore = Semaphore(self.rate_limit)
        
        # Initialize client based on provider
        if self.provider == 'openai':
            from openai import OpenAI
            import httpx
            # Handle httpx version compatibility issues
            try:
                # Try creating a custom httpx client without proxies parameter
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True
                )
                self.client = OpenAI(
                    api_key=api_key or os.getenv('OPENAI_API_KEY'),
                    http_client=http_client
                )
            except TypeError as e:
                if 'proxies' in str(e) or 'proxy' in str(e):
                    # Fallback: try without custom http_client
                    self.client = OpenAI(
                        api_key=api_key or os.getenv('OPENAI_API_KEY'),
                        max_retries=3,
                        timeout=60.0
                    )
                else:
                    raise
        elif self.provider == 'gemini':
            try:
                # Try new package first
                from google import genai
                self.client = genai.Client(api_key=api_key or os.getenv('GEMINI_API_KEY'))
                self.gemini_new_api = True
            except ImportError:
                # Fall back to old package
                import google.generativeai as genai
                genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
                self.client = genai
                self.gemini_new_api = False
        elif self.provider == 'deepseek':
            from openai import OpenAI
            import httpx
            # DeepSeek uses OpenAI-compatible API
            # Handle httpx version compatibility issues
            try:
                # Try creating a custom httpx client without proxies parameter
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True
                )
                self.client = OpenAI(
                    api_key=api_key or os.getenv('DEEPSEEK_API_KEY'),
                    base_url="https://api.deepseek.com",
                    http_client=http_client
                )
            except TypeError as e:
                if 'proxies' in str(e) or 'proxy' in str(e):
                    # Fallback: try without custom http_client
                    self.client = OpenAI(
                        api_key=api_key or os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com",
                        max_retries=3,
                        timeout=60.0
                    )
                else:
                    raise
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        print(f"\n\nUsing {self.provider.upper()} model: {model_name}")
        print(f"Rate limit: {self.rate_limit} concurrent requests")
    
    def _get_rate_limit(self, provider: str) -> int:
        """Get rate limit for provider (max concurrent requests)."""
        rate_limits = {
            'openai': 10,    # OpenAI allows high concurrency
            'gemini': 5,     # Gemini 1.5 has stricter rate limits than 2.0-exp
            'deepseek': 5,   # DeepSeek more conservative
        }
        return rate_limits.get(provider, 5)
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect API provider from model name."""
        model_lower = model_name.lower()
        
        if any(m in model_lower for m in ['gpt', 'openai']):
            return 'openai'
        elif any(m in model_lower for m in ['gemini', 'google']):
            return 'gemini'
        elif any(m in model_lower for m in ['deepseek']):
            return 'deepseek'
        
        raise ValueError(f"Cannot detect provider for model: {model_name}")
    
    def _encode_image_to_base64(self, image: Image.Image, max_size: int = 1024, quality: int = 85) -> str:
        """Convert PIL Image to base64 string with resizing and compression.
        
        Args:
            image: PIL Image object
            max_size: Maximum dimension (width or height) for resizing
            quality: JPEG quality (1-100)
        
        Returns:
            Base64-encoded image string
        """
        # Resize if too large
        if max(image.size) > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        # Use JPEG for smaller file size (unless image has transparency)
        if image.mode == 'RGBA':
            image.save(buffered, format="PNG", optimize=True)
        else:
            # Convert to RGB if needed and save as JPEG
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
        
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _prepare_images(self, image_inputs: List) -> List[Image.Image]:
        """Convert images to PIL format if needed."""
        pil_images = []
        for img in image_inputs:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                # Assume numpy array
                pil_images.append(Image.fromarray(img))
        return pil_images
    
    def _generate_openai(self, pil_images: List[Image.Image], 
                        system_prompt: str, user_prompt: str) -> str:
        """Generate report using OpenAI API."""
        # Build message content with images
        content = []
        
        for img in pil_images:
            # Use higher quality for OpenAI (can handle larger payloads)
            img_base64 = self._encode_image_to_base64(img, max_size=1536, quality=90)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Call API
        # Use max_completion_tokens for newer models (GPT-5.x), fallback to max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_completion_tokens=4096,
                temperature=0.0,
            )
        except Exception as e:
            if 'max_completion_tokens' in str(e):
                # Fallback for older models that use max_tokens
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
            else:
                raise
        
        return response.choices[0].message.content.strip()
    
    def _generate_gemini(self, pil_images: List[Image.Image], 
                        system_prompt: str, user_prompt: str) -> str:
        """Generate report using Google Gemini API."""
        # Map model names - user specified gemini-3-flash-preview
        model_mapping = {
            'gemini3': 'gemini-3-flash-preview',
            'gemini-3': 'gemini-3-flash-preview',
            'gemini3-flash': 'gemini-3-flash-preview',
            'gemini-3-flash': 'gemini-3-flash-preview',
            'gemini3-flash-preview': 'gemini-3-flash-preview',
            'gemini-3-flash-preview': 'gemini-3-flash-preview',
        }
        actual_model = model_mapping.get(self.model_name.lower(), self.model_name)
        
        if self.gemini_new_api:
            # Use new google.genai API
            # Combine system and user prompts
            combined_prompt = system_prompt + "\n\n" + user_prompt
            
            # Prepare parts with images
            parts = []
            for img in pil_images:
                # Convert to bytes
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                parts.append({"inline_data": {"mime_type": "image/png", "data": img_bytes.getvalue()}})
            parts.append({"text": combined_prompt})
            
            response = self.client.models.generate_content(
                model=actual_model,
                contents=parts,
                config={
                    'temperature': 0.0,
                    'max_output_tokens': 1024,
                }
            )
            return response.text.strip()
        else:
            # Use old google.generativeai API
            model = self.client.GenerativeModel(actual_model)
            
            # Prepare content with system prompt, images, and user prompt
            content = [system_prompt + "\n\n" + user_prompt]
            content.extend(pil_images)
            
            # Generate
            response = model.generate_content(
                content,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 1024,
                }
            )
            
            return response.text.strip()
    
    def _generate_deepseek(self, pil_images: List[Image.Image], 
                          system_prompt: str, user_prompt: str) -> str:
        """Generate report using DeepSeek API."""
        # DeepSeek uses OpenAI-compatible API
        # Map model name
        model_mapping = {
            'deepseek3.2': 'deepseek-chat',
            'deepseek-3.2': 'deepseek-chat',
        }
        actual_model = model_mapping.get(self.model_name.lower(), self.model_name)
        
        # Build message content with images (smaller size for DeepSeek)
        content = []
        
        for img in pil_images:
            # DeepSeek has stricter size limits - use smaller images
            img_base64 = self._encode_image_to_base64(img, max_size=768, quality=75)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Call API
        # Use max_completion_tokens for newer models, fallback to max_tokens
        try:
            response = self.client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_completion_tokens=4096,
                temperature=0.0,
            )
        except Exception as e:
            if 'max_completion_tokens' in str(e):
                # Fallback for older models that use max_tokens
                response = self.client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
            else:
                raise
        
        return response.choices[0].message.content.strip()
    
    def generate_report(self, patient: Dict) -> str:
        """
        Generate an orthodontic report for a patient.
        
        Args:
            patient: Dictionary containing:
                - 'images': List of PIL Images or numpy arrays
                - 'report': (Optional) Ground truth report
        
        Returns:
            Generated report text
        """
        image_inputs = patient['images']
        
        # Convert to PIL images
        pil_images = self._prepare_images(image_inputs)
        
        # Load prompts
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
        with open(self.user_prompt_path, 'r') as f:
            user_prompt = f.read().strip()
        
        # Generate based on provider
        try:
            if self.provider == 'openai':
                return self._generate_openai(pil_images, system_prompt, user_prompt)
            elif self.provider == 'gemini':
                return self._generate_gemini(pil_images, system_prompt, user_prompt)
            elif self.provider == 'deepseek':
                return self._generate_deepseek(pil_images, system_prompt, user_prompt)
        except Exception as e:
            print(f"Error calling {self.provider.upper()} API: {e}")
            raise
    
    def _generate_report_with_retry(self, patient_data, patient_id: str, max_retries: int = 3) -> tuple:
        """
        Generate report with retry logic and rate limiting.
        
        Args:
            patient_data: Patient data dictionary or tuple (dataset, index)
            patient_id: Patient identifier
            max_retries: Maximum number of retry attempts
        
        Returns:
            Tuple of (patient_id, report_text or error_message, success_flag, is_critical_error)
        """
        with self.semaphore:  # Rate limiting
            for attempt in range(max_retries):
                try:
                    # If patient_data is a tuple, load on-demand
                    if isinstance(patient_data, tuple):
                        dataset, idx = patient_data
                        patient = dataset[idx]
                    else:
                        patient = patient_data
                    
                    report = self.generate_report(patient)
                    return (patient_id, report, True, False)
                except Exception as e:
                    error_str = str(e)
                    # Check for critical errors that should stop processing
                    critical_keywords = ['quota', 'insufficient_quota', '429', 'billing', 'rate limit exceeded']
                    is_critical = any(keyword.lower() in error_str.lower() for keyword in critical_keywords)
                    
                    if is_critical:
                        return (patient_id, f"CRITICAL ERROR: {error_str}", False, True)
                    
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        tqdm.write(f"  Retry {attempt + 1}/{max_retries} for {patient_id} after {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        return (patient_id, f"ERROR: {error_str}", False, False)
    
    def generate_reports_batch(self, dataset, patient_indices: List[int], 
                               max_workers: Optional[int] = None) -> Dict[str, str]:
        """
        Generate reports for multiple patients in parallel.
        
        Args:
            dataset: Dataset object to load patients from
            patient_indices: List of patient indices to process
            max_workers: Maximum number of concurrent workers (default: rate_limit)
        
        Returns:
            Dictionary mapping patient_id to generated report
        """
        if max_workers is None:
            max_workers = self.rate_limit
        
        results = {}
        total = len(patient_indices)
        success_count = 0
        error_count = 0
        
        print(f"\nProcessing {total} patients with {max_workers} workers...")
        print(f"Memory-efficient mode: Loading patients on-demand")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with dataset reference and index
            future_to_idx = {}
            for idx in patient_indices:
                patient_id = dataset[idx]['patient_id']
                future = executor.submit(
                    self._generate_report_with_retry,
                    (dataset, idx),  # Pass reference, not data
                    patient_id
                )
                future_to_idx[future] = idx
            
            # Process completed tasks with progress bar
            critical_error_encountered = False
            with tqdm(total=total, desc="Generating reports", unit="patient") as pbar:
                for future in as_completed(future_to_idx):
                    patient_id, report, success, is_critical = future.result()
                    
                    # Check for critical errors
                    if is_critical:
                        critical_error_encountered = True
                        tqdm.write(f"\n{'='*80}")
                        tqdm.write(f"CRITICAL ERROR DETECTED: Stopping all processing")
                        tqdm.write(f"Patient: {patient_id}")
                        tqdm.write(f"Error: {report}")
                        tqdm.write(f"{'='*80}\n")
                        # Cancel remaining futures
                        for f in future_to_idx:
                            f.cancel()
                        break
                    
                    # Only save successful reports, not errors
                    if success:
                        results[patient_id] = report
                        success_count += 1
                        pbar.set_postfix({"✓": success_count, "✗": error_count}, refresh=False)
                    else:
                        error_count += 1
                        pbar.set_postfix({"✓": success_count, "✗": error_count}, refresh=False)
                        tqdm.write(f"  ✗ Error for {patient_id}: {report[:80]}...")
                    
                    pbar.update(1)
            
            if critical_error_encountered:
                raise RuntimeError(f"Critical API error encountered. Processing stopped. Check your API quota and billing details.")
        
        total_time = time.time() - start_time
        print(f"\nCompleted {total} patients in {total_time/60:.1f} minutes "
              f"({total/total_time*60:.1f} patients/min average)")
        print(f"Success: {success_count}, Errors: {error_count}")
        
        return results


# Example usage
if __name__ == '__main__':
    import sys
    
    # Example configuration
    model_name = 'gpt-5.2'  # or 'gemini3', 'deepseek3.2'
    
    try:
        generator = APIReportGenerator(
            model_name=model_name,
            system_prompt_path='templates/system.txt',
            user_prompt_path='templates/user.txt'
        )
        
        # Test with dummy images
        dummy_image = Image.new('RGB', (224, 224), color='white')
        patient = {
            'images': [dummy_image] * 5,
            'report': 'Test ground truth'
        }
        
        report = generator.generate_report(patient)
        print("\nGenerated Report:")
        print(report)
        
    except Exception as e:
        print(f"Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
