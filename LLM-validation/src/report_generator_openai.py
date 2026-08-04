"""
OpenAI API-based Report Generator for orthodontic report generation.

This module uses OpenAI's API (via Responses API) instead of local models.
Supports vision models like GPT-4o and GPT-4 Turbo for analyzing intra-oral scans.
"""

import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from typing import List, Dict, Optional


class OpenAIReportGenerator:
    """
    Generate orthodontic reports using OpenAI's API.
    
    Uses the Responses API for structured text generation from images.
    """
    
    def __init__(
        self, 
        model_name: str,
        system_prompt_path: str, 
        user_prompt_path: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI report generator.
        
        Args:
            model_name: OpenAI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')
            system_prompt_path: Path to system prompt file
            user_prompt_path: Path to user prompt file
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
        print(f"\n\nUsing OpenAI model: {model_name}")
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64-encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def generate_report(self, patient: Dict) -> str:
        """
        Generate an orthodontic report for a patient using OpenAI API.
        
        Args:
            patient: Dictionary containing:
                - 'images': List of PIL Images or numpy arrays
                - 'report': (Optional) Ground truth report
        
        Returns:
            Generated report text
        """
        image_inputs = patient['images']
        
        # Convert images to PIL if needed
        pil_images = []
        for img in image_inputs:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                pil_images.append(Image.fromarray(img))
        
        # Load prompts
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
        with open(self.user_prompt_path, 'r') as f:
            user_prompt = f.read().strip()
        
        # Build input for API
        # For Responses API, we use the 'input' parameter with message-like structure
        input_content = []
        
        # Add images as base64-encoded data URLs
        for img in pil_images:
            img_base64 = self._encode_image_to_base64(img)
            input_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
        
        # Add text prompt
        input_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # Create the request using Responses API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": input_content
                    }
                ],
                max_tokens=1024,
                temperature=0.0,  # Use deterministic generation for medical reports
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise


# Example usage
if __name__ == '__main__':
    import sys
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Example configuration
    generator = OpenAIReportGenerator(
        model_name='gpt-4o',
        system_prompt_path='templates/system.txt',
        user_prompt_path='templates/user.txt'
    )
    
    # Test with dummy images
    dummy_image = Image.new('RGB', (224, 224), color='white')
    patient = {
        'images': [dummy_image] * 5,
        'report': 'Test ground truth'
    }
    
    try:
        report = generator.generate_report(patient)
        print("\nGenerated Report:")
        print(report)
    except Exception as e:
        print(f"Failed to generate report: {e}")
