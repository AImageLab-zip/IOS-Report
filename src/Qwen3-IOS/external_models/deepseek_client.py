"""
DeepSeek API client for evaluation.
"""

import os
from typing import List, Optional
from pathlib import Path
import time
import requests

from .base import BaseExternalModel


class DeepSeekModel(BaseExternalModel):
    """
    DeepSeek client for intraoral photo analysis.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "deepseek-vl",  # Update with actual model name
        system_prompt: str = "You are a dental expert analyzing intraoral photos. Provide accurate, structured orthodontic assessments.",
        user_prompt_template: str = "Analyze these intraoral photos and provide a structured orthodontic assessment using the following fields:\n{fields}\n\nProvide your response in the format 'Field name: Value', one per line.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_base: str = "https://api.deepseek.com/v1",
        **kwargs
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            model_name: Model name/ID
            system_prompt: System prompt for the model
            user_prompt_template: Template for user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_base: API base URL
        """
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DeepSeek API key not provided and DEEPSEEK_API_KEY not set")
        
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self.api_base = api_base
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 1.0)
    
    def generate(
        self,
        image_paths: List[Path],
        field_names: Optional[str] = None,
    ) -> str:
        """
        Generate description using DeepSeek.
        
        Args:
            image_paths: List of paths to intraoral photos
            field_names: Optional comma-separated list of field names
        
        Returns:
            Generated description text
        """
        # Prepare user prompt
        user_prompt = self.user_prompt_template
        if field_names:
            user_prompt = user_prompt.replace('{fields}', field_names)
        else:
            # Default fields if none provided
            user_prompt = user_prompt.replace('{fields}', 
                'Overbite, Crowding, Molar occlusion - Right side, Molar occlusion - Left side, '
                'Canine occlusion - Right side, Canine occlusion - Left side, Curve of Spee, '
                'Curve of Wilson, Midlines, Transverse relationship, Missing teeth')
        
        # Build message content with images
        content = []
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Add all images
        for image_path in image_paths:
            # Load and encode image
            image = self.load_image(image_path)
            base64_image = self.image_to_base64(image)
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Create messages (OpenAI-compatible format)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                generated_text = result['choices'][0]['message']['content'].strip()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return generated_text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"DeepSeek API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"DeepSeek API failed after {max_retries} attempts: {e}")
                    return f"Error: {str(e)}"
