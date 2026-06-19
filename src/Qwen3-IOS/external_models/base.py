"""
Base class for external model API clients.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO


class BaseExternalModel(ABC):
    """
    Base class for external model API clients.
    
    Provides common functionality for loading images, encoding them,
    and defining the interface for generating descriptions.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Args:
            api_key: API key for the service
            model_name: Name/identifier of the model
            system_prompt: System prompt for the model
            user_prompt_template: Template for user prompt (can contain {fields})
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def load_image(self, image_path: Path, max_size: tuple = (1024, 1024)) -> Image.Image:
        """
        Load and resize an image.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum size (width, height)
        
        Returns:
            PIL Image
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return img
    
    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            format: Image format (JPEG, PNG, etc.)
        
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def prepare_field_names(self, description: str) -> str:
        """
        Extract field names from a reference description.
        
        Args:
            description: Reference description with "Field: Value" format
        
        Returns:
            Comma-separated list of field names
        """
        field_names = []
        for line in description.split('\n'):
            line = line.strip()
            if ':' in line and line:
                field_name = line.split(':')[0].strip()
                field_names.append(field_name)
        
        return ', '.join(field_names) + '.' if field_names else ''
    
    @abstractmethod
    def generate(
        self,
        image_paths: List[Path],
        field_names: Optional[str] = None,
    ) -> str:
        """
        Generate a description based on intraoral photos.
        
        Args:
            image_paths: List of paths to intraoral photos
            field_names: Optional comma-separated list of field names to include
        
        Returns:
            Generated description text
        """
        pass
    
    def __call__(self, image_paths: List[Path], field_names: Optional[str] = None) -> str:
        """Convenience method to call generate()."""
        return self.generate(image_paths, field_names)
