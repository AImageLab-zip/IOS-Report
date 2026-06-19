"""
Evaluation metrics for medical report generation.

This module provides comprehensive metrics for evaluating generated orthodontic reports:
1. Field-level accuracy - Structured comparison of medical fields (e.g., "Molar occlusion - Left side")
2. BLEU-1 - Unigram precision
3. ROUGE-L - Longest common subsequence F1
4. METEOR - Morphology-aware matching
5. Sentence-BERT - Semantic similarity using embeddings
"""

import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class FieldExtractor:
    """
    Extract and parse structured fields from orthodontic reports.
    """
    
    # field names patterns
    FIELD_PATTERNS = {
        'overbite': r'Overbite',
        'crowding': r'Crowding',
        'molar_right': r'Molar\s+occlusion\s*[-–]\s*Right\s+side',
        'molar_left': r'Molar\s+occlusion\s*[-–]\s*Left\s+side',
        'canine_right': r'Canine\s+occlusion\s*[-–]\s*Right\s+side',
        'canine_left': r'Canine\s+occlusion\s*[-–]\s*Left\s+side',
        'curve_spee': r'Curve\s+of\s+Spee',
        'curve_wilson': r'Curve\s+of\s+Wilson',
        'midlines': r'Midlines',
        'transverse': r'Transverse\s+relationship',
        'missing_teeth': r'Missing\s+teeth'
    }
    
    @staticmethod
    def extract_fields(text: str) -> Dict[str, str]:
        """
        Extract field:value pairs from a report.
        
        Args:
            text: Report text with format "Field name: value"
        
        Returns:
            Dictionary mapping normalized field names to their values
        """
        fields = {}
        
        # Normalize text
        text = text.strip()
        
        # Split by newlines and process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Split on first colon optionally preceeded or followed by spaces
            parts = re.split(r'\s*:\s*', line, maxsplit=1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            field_value = parts[1].strip()
            
            # Match against known patterns
            for key, pattern in FieldExtractor.FIELD_PATTERNS.items():
                if re.match(pattern, field_name, re.IGNORECASE):
                    fields[key] = field_value
                    break
        
        return fields
    
    @staticmethod
    def normalize_value(value: str) -> str:
        """
        Normalize field value for comparison.
        
        - Lowercase
        - Remove extra whitespace
        - Normalize punctuation
        """
        value = value.lower().strip()
        # Normalize dashes/hyphens
        value = re.sub(r'[--—]', '-', value)
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value)
        return value
    
    @staticmethod
    def compare_values(pred_value: str, true_value: str, field_type: str = None) -> float:
        """
        Compare two field values and return a similarity score.
        
        Args:
            pred_value: Predicted value
            true_value: Ground truth value
            field_type: Type of field (for specialized comparison logic)
        
        Returns:
            Similarity score between 0 and 1
        """
        pred_norm = FieldExtractor.normalize_value(pred_value)
        true_norm = FieldExtractor.normalize_value(true_value)
        
        # Exact match
        if pred_norm == true_norm:
            return 1.0
        
        # Molar/Canine occlusion
        if field_type and ('molar' in field_type or 'canine' in field_type):
            # Extract class type (I, II, III)
            pred_class = FieldExtractor._extract_class(pred_norm)
            true_class = FieldExtractor._extract_class(true_norm)
            
            if pred_class and true_class:
                if pred_class == true_class:
                    # Same class - check if it's Class II with different modifiers
                    if pred_class == "ii":
                        # For Class II, give partial credit if modifiers differ
                        # (e.g., "Class II division 1" vs "Class II division 2")
                        return 0.75 if pred_norm != true_norm else 1.0
                    else:
                        # For Class I or III, same class means correct
                        return 1.0
                else:
                    # Different classes (e.g., Class I vs Class II) - incorrect
                    return 0.0
            # If class extraction failed for either, fall through to word overlap
        
        # missing teeth
        if 'missing teeth' in field_type:
            pred_teeth = set(re.findall(r'\b\d{2}\b', pred_norm))
            true_teeth = set(re.findall(r'\b\d{2}\b', true_norm))
            if not true_teeth:
                return 1.0 if not pred_teeth else 0.0
            overlap = len(pred_teeth & true_teeth)
            return overlap / len(true_teeth)
        
        # Other fields
        pred_words = set(pred_norm.split())
        true_words = set(true_norm.split())
        
        if not true_words:
            return 0.0
        
        # String IoU? Esiste? TODO: cercala su internet
        overlap = len(pred_words & true_words)
        return overlap / (len(true_words) + len(pred_words) - overlap)
    
    @staticmethod
    def _extract_class(text: str) -> Optional[str]:
        """Extract Angle's classification from occlusion text."""
        text = text.replace('1', 'I').replace('2', 'II').replace('3', 'III')
        match = re.search(r'class\s+(i{1,3})', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None
    
    @staticmethod
    def should_skip_field(field_name: str, field_value: str) -> bool:
        """
        Determine if a field should be skipped in accuracy computation.
        
        Skip criteria:
        - Occlusion fields (molar/canine) containing "Not assessable"
        - Any field containing "Unknown"
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
        
        Returns:
            True if field should be skipped, False otherwise
        """
        value_lower = field_value.lower().strip()
        
        # Skip if contains "unknown"
        if 'unknown' in value_lower:
            return True
        
        # Skip occlusion fields with "not assessable"
        if ('molar' in field_name or 'canine' in field_name):
            if 'not assessable' in value_lower:
                return True
        
        return False


def compute_field_accuracy(
    predictions: List[str],
    references: List[str],
    return_per_field: bool = False
) -> Dict[str, float]:
    """
    Compute field-level accuracy for structured medical reports.
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    all_scores = []
    field_scores = defaultdict(list)
    coverage_scores = []
    
    for pred, ref in zip(predictions, references):
        # Extract fields from both texts
        pred_fields = FieldExtractor.extract_fields(pred)
        ref_fields = FieldExtractor.extract_fields(ref)
        
        if not ref_fields: # Forse qua dovrei segnarmi qualcosa nelle metriche?
            continue
        
        # Filter out fields that should be skipped (Unknown, Not assessable)
        ref_fields_filtered = {
            k: v for k, v in ref_fields.items() 
            if not FieldExtractor.should_skip_field(k, v)
        }
        pred_fields_filtered = {
            k: v for k, v in pred_fields.items() 
            if not FieldExtractor.should_skip_field(k, v)
        }
        
        # Skip sample if all fields were filtered out
        if not ref_fields_filtered:
            continue
        
        # Coverage: how many reference fields are present in prediction
        # (only counting non-skipped fields)
        coverage = len(set(ref_fields_filtered.keys()) & set(pred_fields_filtered.keys())) / len(ref_fields_filtered)
        coverage_scores.append(coverage)
        
        # Compare each reference field (only non-skipped ones)
        sample_scores = []
        for field_name, ref_value in ref_fields_filtered.items():
            if field_name in pred_fields_filtered:
                score = FieldExtractor.compare_values(
                    pred_fields_filtered[field_name],
                    ref_value,
                    field_type=field_name
                )
                sample_scores.append(score)
                field_scores[field_name].append(score)
            else:
                # Field missing in prediction
                sample_scores.append(0.0)
                field_scores[field_name].append(0.0)
        
        all_scores.extend(sample_scores)
    
    # Compute overall metrics
    result = {
        'accuracy': np.mean(all_scores) if all_scores else 0.0,
        'coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
        'num_fields': len(all_scores)
    }
    
    # Add per-field statistics
    if return_per_field:
        result['per_field'] = {
            field: np.mean(scores) for field, scores in field_scores.items()
        }
    
    return result


def compute_bleu(
    predictions: List[str],
    references: List[str],
    n_gram: int = 1
) -> Dict[str, float]:
    """
    Compute BLEU score (unigram by default).
    
    BLEU measures n-gram precision with brevity penalty. BLEU-1 (unigram)
    is useful for capturing word-level accuracy.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        n_gram: Maximum n-gram size (1 for BLEU-1, 4 for BLEU-4)
    
    Returns:
        Dictionary with 'bleu' score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    smoothing = SmoothingFunction()
    scores = []
    
    # Set weights based on n_gram
    if n_gram == 1:
        weights = (1.0, 0, 0, 0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram == 3:
        weights = (0.33, 0.33, 0.33, 0)
    else:  # n_gram == 4 or higher
        weights = (0.25, 0.25, 0.25, 0.25)
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]  # BLEU expects list of references
        
        # Compute BLEU
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=weights,
            smoothing_function=smoothing.method1
        )
        scores.append(score)
    
    return {
        f'bleu-{n_gram}': np.mean(scores) if scores else 0.0
    }


def compute_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE-L score.
    
    ROUGE-L measures longest common subsequence, which captures sentence-level
    structure similarity.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with:
            - 'rouge-l-p': ROUGE-L precision
            - 'rouge-l-r': ROUGE-L recall
            - 'rouge-l-f': ROUGE-L F1 score
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        precision_scores.append(score['rougeL'].precision)
        recall_scores.append(score['rougeL'].recall)
        f1_scores.append(score['rougeL'].fmeasure)
    
    return {
        'rouge-l-p': np.mean(precision_scores) if precision_scores else 0.0,
        'rouge-l-r': np.mean(recall_scores) if recall_scores else 0.0,
        'rouge-l-f': np.mean(f1_scores) if f1_scores else 0.0
    }


def compute_meteor(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute METEOR score.
    
    METEOR considers synonyms, stemming, and paraphrasing. It aligns better
    with human judgment than BLEU in many cases.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with 'meteor' score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    scores = []
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        # Compute METEOR
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    
    return {
        'meteor': np.mean(scores) if scores else 0.0
    }


def compute_sentence_bert_similarity(
    predictions: List[str],
    references: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute semantic similarity using Sentence-BERT embeddings.
    
    This metric captures semantic meaning beyond surface-level word matching.
    Uses cosine similarity between sentence embeddings.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_name: Name of sentence-transformers model to use
        device: Device to run model on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with 'sbert-sim' score (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model (cached after first call)
    model = SentenceTransformer(model_name, device=device)
    
    # Encode all texts
    pred_embeddings = model.encode(predictions, convert_to_tensor=True, device=device)
    ref_embeddings = model.encode(references, convert_to_tensor=True, device=device)
    
    # Compute cosine similarities
    similarities = util.cos_sim(pred_embeddings, ref_embeddings)
    
    # Extract diagonal (pairwise similarities)
    scores = similarities.diagonal().cpu().numpy()
    
    return {
        'sbert-sim': np.mean(scores) if len(scores) > 0 else 0.0
    }


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    include_sbert: bool = True,
    sbert_device: Optional[str] = None,
    return_per_field: bool = False
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        include_sbert: Whether to compute Sentence-BERT (slower)
        sbert_device: Device for Sentence-BERT model
        return_per_field: Return per-field accuracy breakdown
    
    Returns:
        Dictionary with all metric scores
    """
    metrics = {}
    
    # Field-level accuracy
    field_metrics = compute_field_accuracy(
        predictions, references, return_per_field=return_per_field
    )
    metrics.update(field_metrics)
    
    # BLEU-1
    bleu_metrics = compute_bleu(predictions, references, n_gram=1)
    metrics.update(bleu_metrics)
    
    # ROUGE-L
    rouge_metrics = compute_rouge(predictions, references)
    metrics.update(rouge_metrics)
    
    # METEOR
    meteor_metrics = compute_meteor(predictions, references)
    metrics.update(meteor_metrics)
    
    # Sentence-BERT (optional, slower)
    if include_sbert:
        sbert_metrics = compute_sentence_bert_similarity(
            predictions, references, device=sbert_device
        )
        metrics.update(sbert_metrics)
    
    return metrics


# Example usage and testing
if __name__ == '__main__':
    # Test cases
    predictions = [
        "Molar occlusion - Left side: Class II head-to-head\nMolar occlusion - Right side: Class I",
        "Overbite: Normal\nCrowding: Moderate in the upper arch"
    ]
    
    references = [
        "Molar occlusion - Left side: Class I\nMolar occlusion - Right side: Class I",
        "Overbite: Increased\nCrowding: Moderate in the upper arch"
    ]
    
    print("Testing Field Accuracy:")
    result = compute_field_accuracy(predictions, references, return_per_field=True)
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Coverage: {result['coverage']:.3f}")
    print(f"  Per-field: {result.get('per_field', {})}")
    
    print("\nTesting BLEU-1:")
    result = compute_bleu(predictions, references, n_gram=1)
    print(f"  {result}")
    
    print("\nTesting ROUGE-L:")
    result = compute_rouge(predictions, references)
    print(f"  {result}")
    
    print("\nTesting METEOR:")
    result = compute_meteor(predictions, references)
    print(f"  {result}")
    
    print("\nTesting All Metrics:")
    result = compute_all_metrics(predictions, references, include_sbert=True, return_per_field=True)
    for key, value in result.items():
        if key != 'per_field':
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  per_field breakdown:")
            for field, score in value.items():
                print(f"    {field}: {score:.3f}")
