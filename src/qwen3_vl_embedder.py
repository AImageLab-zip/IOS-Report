"""
Qwen3 VL Image Embedding Extractor

This module provides utilities to extract image embeddings using the Qwen3 VL vision encoder.
"""
import time
start_time = time.time()
import torch
print(f"Torch imports completed in {time.time() - start_time:.2f} seconds")

start_time = time.time()
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print(f"Transformers imports completed in {time.time() - start_time:.2f} seconds")

start_time = time.time()
from PIL import Image
import requests
from io import BytesIO
from typing import Union
from pathlib import Path
print(f"PIL and requests imports completed in {time.time() - start_time:.2f} seconds")


class Qwen3VLImageEmbedder:
    """
    A class to extract image embeddings using Qwen3 VL vision encoder.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "auto", dtype=torch.bfloat16):
        """
        Initialize the Qwen3 VL model and processor.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-VL-4B-Instruct")
            device: Device to load the model on ("auto", "cuda", "cpu")
            dtype: Data type for model weights (torch.float16, torch.float32, etc.)
        """
        print(f"Loading Qwen3 VL model: {model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded successfully!")
        
    def load_image(self, image_source: Union[str, Path]) -> Image.Image:
        """
        Load an image from URL or file path.
        
        Args:
            image_source: URL or file path to the image
            
        Returns:
            PIL Image object
        """
        if isinstance(image_source, str) and (image_source.startswith('http://') or image_source.startswith('https://')):
            # Load from URL
            response = requests.get(image_source)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Load from file path
            img = Image.open(image_source).convert('RGB')
        return img
    
    @torch.no_grad()
    def get_vision_encoder_embeddings(self, image_source: Union[str, Path, torch.Tensor]) -> Union[torch.Tensor, tuple]:
        """
        Extract raw embeddings directly from the vision encoder (before fusion with text).
        This gives you the pure visual features.
        
        Args:
            image_source: URL or file path to the image
            
        Returns:
            Tuple of (final_embeddings, list of intermediate layer embeddings)
                - final_embeddings: Tensor with shape (seq_len, hidden_size) - main vision features
                - deepstack_features: List of 3 tensors with shape (seq_len, hidden_size) each
                  representing features from different ViT layers (DeepStack architecture)
        """
        if isinstance(image_source, torch.Tensor):
            image = image_source
        else:
            image = self.load_image(image_source)
        
        image_inputs = self.processor.image_processor(
            images=[image],
            return_tensors="pt"
        )
        
        pixel_values = image_inputs['pixel_values'].to(self.model.device)
        image_grid_thw = image_inputs['image_grid_thw'].to(self.model.device)
        
        vision_outputs = self.model.model.visual(
            hidden_states=pixel_values,
            grid_thw=image_grid_thw
        )
        
        return vision_outputs
    
    @torch.no_grad()
    def get_deepstack_features(self, image_source: Union[str, Path]) -> dict:
        """
        Extract all DeepStack features (multi-level ViT features).
        
        DeepStack extracts features from multiple layers of the Vision Transformer
        for better visual understanding. These features are typically fed to different
        decoder layers in the full model.
        
        Args:
            image_source: URL or file path to the image
            
        Returns:
            Dictionary containing:
                - 'final': Final layer output, shape (seq_len, hidden_size)
                - 'intermediate': List of 3 intermediate layer features
        """
        final_output, deepstack_features = self.get_vision_encoder_embeddings(
            image_source
        )
        
        return {
            'final': final_output,
            'intermediate': deepstack_features,
        }
        
    def get_class_token(self, image_source: Union[str, Path]) -> torch.Tensor:
        """
        Extract the class token embedding from the vision encoder output.
        
        Args:
            image_source: URL or file path to the image
        Returns:
            Tensor representing the class token embedding
        """
        final_output, _ = self.get_vision_encoder_embeddings(image_source)
        class_token = final_output[0, :]
        return class_token


# Example usage
if __name__ == "__main__":
    # Initialize the embedder
    embedder = Qwen3VLImageEmbedder(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        device="auto",
        dtype=torch.bfloat16
    )
    
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    embeddings = embedder.get_deepstack_features(image_url)
    print("Final Embeddings Shape:", embeddings['final'].shape)
    for i, layer in enumerate(embeddings['intermediate']):
        print(f"Intermediate Layer {i} Shape:", layer.shape)