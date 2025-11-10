"""
Semantic Loss using sentence embeddings.

This module implements a semantic similarity loss using a pretrained sentence
transformer model to compare generated and ground truth text at the semantic level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel


class SemanticLoss(nn.Module):
    """
    Semantic loss using sentence embeddings and cosine similarity.
    
    This loss computes embeddings for both predicted and target text using a
    pretrained sentence transformer, then measures their cosine similarity.
    The loss is defined as 1 - cosine_similarity, so higher similarity gives lower loss.
    
    Args:
        model_name: HuggingFace model name for sentence embeddings
        device: Device to run the model on
        freeze: Whether to freeze the embedding model (recommended for efficiency)
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        freeze: bool = True,
        max_length: int = 512
    ):
        super().__init__()
        
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer and model
        print(f"Loading semantic loss model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Freeze model to save memory and computation
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        self.freeze = freeze
        
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings from token embeddings.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask for the input
            
        Returns:
            Sentence embeddings
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into sentence embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Normalized sentence embeddings of shape (batch_size, embedding_dim)
        """
        # Tokenize
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute embeddings
        with torch.set_grad_enabled(not self.freeze):
            model_output = self.model(**encoded_input)
        
        # Mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(
        self,
        predictions: List[str],
        targets: List[str],
        return_similarity: bool = False
    ) -> torch.Tensor:
        """
        Compute semantic loss between predictions and targets.
        
        Args:
            predictions: List of predicted text strings
            targets: List of target text strings
            return_similarity: If True, return (loss, similarity) tuple
            
        Returns:
            Semantic loss (and optionally similarity score)
        """
        # Encode predictions and targets
        pred_embeddings = self.encode(predictions)
        target_embeddings = self.encode(targets)
        
        # Compute cosine similarity
        # embeddings are already normalized, so dot product = cosine similarity
        cosine_sim = (pred_embeddings * target_embeddings).sum(dim=1)
        
        # Loss is 1 - similarity (range [0, 2], with 0 being perfect match)
        loss = (1.0 - cosine_sim).mean()
        
        if return_similarity:
            return loss, cosine_sim.mean()
        
        return loss
    
    def compute_similarity(self, predictions: List[str], targets: List[str]) -> torch.Tensor:
        """
        Compute cosine similarity between predictions and targets.
        
        Args:
            predictions: List of predicted text strings
            targets: List of target text strings
            
        Returns:
            Mean cosine similarity score
        """
        with torch.no_grad():
            pred_embeddings = self.encode(predictions)
            target_embeddings = self.encode(targets)
            cosine_sim = (pred_embeddings * target_embeddings).sum(dim=1)
            return cosine_sim.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with cross-entropy and semantic loss.
    
    This wrapper combines the standard language modeling cross-entropy loss
    with semantic similarity loss for improved training.
    
    Args:
        semantic_loss: SemanticLoss instance
        semantic_weight: Weight for semantic loss component
    """
    
    def __init__(
        self,
        semantic_loss: SemanticLoss,
        semantic_weight: float = 0.5
    ):
        super().__init__()
        self.semantic_loss = semantic_loss
        self.semantic_weight = semantic_weight
    
    def forward(
        self,
        ce_loss: torch.Tensor,
        predictions: List[str],
        targets: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            ce_loss: Cross-entropy loss from language model
            predictions: List of predicted text strings
            targets: List of target text strings
            
        Returns:
            Tuple of (total_loss, ce_loss, semantic_loss)
        """
        # Compute semantic loss
        sem_loss = self.semantic_loss(predictions, targets)
        
        # Combined loss
        total_loss = ce_loss + self.semantic_weight * sem_loss
        
        return total_loss, ce_loss, sem_loss
