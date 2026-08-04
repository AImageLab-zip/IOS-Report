"""
Evaluation metrics for medical report generation.

This module provides comprehensive metrics for evaluating generated orthodontic reports:
1. Field-level accuracy - Structured comparison of medical fields (e.g., "Molar occlusion - Left side")
2. BLEU-1 - Unigram precision
3. ROUGE-L - Longest common subsequence F1
4. METEOR - Morphology-aware matching
5. Sentence-BERT - Semantic similarity using embeddings
6. RadFact - LLM-based entailment verification for factual correctness
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
        
        text = text.strip()
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue

            parts = re.split(r'\s*:\s*', line, maxsplit=1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            field_value = parts[1].strip()

            for key, pattern in FieldExtractor.FIELD_PATTERNS.items():
                if re.match(pattern, field_name, re.IGNORECASE):
                    fields[key] = field_value
                    break
        
        return fields
    
    @staticmethod
    def normalize_value(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r'[--—]', '-', value)
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

        if pred_norm == true_norm:
            return 1.0
        
        # Molar/Canine occlusion
        if field_type and ('molar' in field_type or 'canine' in field_type):
            # Extract class type (I, II, III)
            pred_class = FieldExtractor._extract_class(pred_norm)
            true_class = FieldExtractor._extract_class(true_norm)
            
            if pred_class and true_class:
                if pred_class == true_class:
                    if pred_class == "ii":
                        # For Class II, give partial credit if modifiers differ
                        return 0.75 if pred_norm != true_norm else 1.0
                    else:
                        return 1.0
                else:
                    return 0.0

        if 'missing teeth' in field_type:
            pred_teeth = set(re.findall(r'\b\d{2}\b', pred_norm))
            true_teeth = set(re.findall(r'\b\d{2}\b', true_norm))
            if not true_teeth:
                return 1.0 if not pred_teeth else 0.0
            overlap = len(pred_teeth & true_teeth)
            return overlap / len(true_teeth)

        pred_words = set(pred_norm.split())
        true_words = set(true_norm.split())
        
        if not true_words:
            return 0.0

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
        
        if 'unknown' in value_lower:
            return True
        
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
        pred_fields = FieldExtractor.extract_fields(pred)
        ref_fields = FieldExtractor.extract_fields(ref)
        
        if not ref_fields:
            continue

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
        coverage = len(set(ref_fields_filtered.keys()) & set(pred_fields_filtered.keys())) / len(ref_fields_filtered)
        coverage_scores.append(coverage)
        
        # Compare each reference field
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

    result = {
        'accuracy': np.mean(all_scores) if all_scores else 0.0,
        'coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
        'num_fields': len(all_scores)
    }

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
    
    if n_gram == 1:
        weights = (1.0, 0, 0, 0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram == 3:
        weights = (0.33, 0.33, 0.33, 0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
    
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
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
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

    model = SentenceTransformer(model_name, device=device)

    pred_embeddings = model.encode(predictions, convert_to_tensor=True, device=device)
    ref_embeddings = model.encode(references, convert_to_tensor=True, device=device)

    similarities = util.cos_sim(pred_embeddings, ref_embeddings)
    scores = similarities.diagonal().cpu().numpy()
    
    return {
        'sbert-sim': np.mean(scores) if len(scores) > 0 else 0.0
    }


def compute_radfact(
    predictions: List[str],
    references: List[str],
    radfact_model: str = 'gpt-5-nano',
    cache_file: Optional[str] = None,
    ollama_url: Optional[str] = None,
    max_workers: int = 5,
    rate_limit_rpm: int = 400
) -> Dict[str, float]:
    """
    Compute RadFact score using LLM-based entailment verification.
    
    RadFact measures factual correctness through logical precision, recall, and F1.
    Uses batched API calls (2 calls per sample: 1 for all precision checks, 1 for all recall checks)
    with parallel processing to stay under API rate limits while maximizing speed.
    It treats each line in the structured report as a "finding" and verifies entailment.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        radfact_model: Model to use for entailment verification (default: gpt-5-nano)
        cache_file: Path to cache file for storing results (optional)
        ollama_url: URL for Ollama API (required for local models like llama70b)
        max_workers: Number of parallel workers (default: 10, safe for 500 RPM limit)
    
    Returns:
        Dictionary with:
            - 'radfact-precision': Logical precision
            - 'radfact-recall': Logical recall
            - 'radfact-f1': Logical F1 score
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    try:
        import os
        import json
        import hashlib
        from openai import OpenAI
        import httpx
        from tqdm import tqdm
        
        # Load cache if exists - cache stores per-sample results
        cache = {}
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                # Check if cache format is the new per-sample format
                if cache and isinstance(list(cache.values())[0], dict):
                    print(f"  Loaded {len(cache)} cached sample results from {cache_file}")
                else:
                    # Old cache format, clear it
                    print(f"  Warning: Old cache format detected, resetting cache")
                    cache = {}
            except Exception as e:
                print(f"  Warning: Could not load cache: {e}")
        
        def get_sample_cache_key(pred: str, ref: str) -> str:
            """Generate cache key for a sample (prediction-reference pair)."""
            content = f"{pred}|{ref}|{radfact_model}"
            return hashlib.md5(content.encode()).hexdigest()
        
        def save_cache():
            """Save cache to file."""
            if cache_file:
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)
                except Exception as e:
                    print(f"  Warning: Could not save cache: {e}")
        
        # Determine if using Ollama or OpenAI
        use_ollama = ollama_url is not None
        
        if use_ollama:
            # Ollama setup - use requests directly
            print(f"  Using Ollama model '{radfact_model}' at {ollama_url}")
            client = "ollama"
        else:
            # Initialize OpenAI client with error handling for version conflicts
            try:
                # Try creating a custom httpx client without proxies parameter
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True
                )
                client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    http_client=http_client
                )
            except TypeError as e:
                if 'proxies' in str(e) or 'proxy' in str(e):
                    # Fallback: try without custom http_client
                    try:
                        client = OpenAI(
                            api_key=os.getenv('OPENAI_API_KEY'),
                            max_retries=3,
                            timeout=60.0
                        )
                    except TypeError:
                        # Ultimate fallback: use requests-based approach
                        print("    Warning: OpenAI client has compatibility issues")
                        print("    Using alternative API approach...")
                        client = "fallback"
                else:
                    raise
        
        # Helper function to make API call with rate limiting
        import time
        import threading
        api_call_times = []
        api_lock = threading.Lock()
        
        def make_api_call(prompt: str, max_retries: int = 5) -> str:
            """Make a single API call with rate limiting and retry logic."""
            # Rate limiting: ensure we don't exceed rate_limit_rpm
            with api_lock:
                current_time = time.time()
                # Remove calls older than 60 seconds
                api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
                
                # If we're at the limit, wait until we can make another call
                if len(api_call_times) >= rate_limit_rpm:
                    oldest_call = api_call_times[0]
                    wait_time = 60 - (current_time - oldest_call)
                    if wait_time > 0:
                        time.sleep(wait_time + 0.1)  # Add small buffer
                        current_time = time.time()
                        # Clean up old calls again after waiting
                        api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
                
                # Record this call
                api_call_times.append(current_time)
            
            # Make the actual API call with retry logic
            import requests
            
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    if client == "ollama":
                        data = {
                            "model": radfact_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False
                        }
                        resp = requests.post(
                            f"{ollama_url}/api/chat",
                            json=data,
                            timeout=120
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        content = result['message']['content'].strip()
                        
                        if not content:
                            raise ValueError(f"Empty response from Ollama API (model: {radfact_model})")
                        
                        return content
                    else:
                        # OpenAI API (including fallback)
                        api_key = os.getenv('OPENAI_API_KEY')
                        if not api_key:
                            raise ValueError("OPENAI_API_KEY environment variable not set")
                        
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        is_gpt5_plus = "gpt-5" in radfact_model.lower() or "gpt-6" in radfact_model.lower()
                        token_param = "max_completion_tokens" if is_gpt5_plus else "max_tokens"
                        data = {
                            "model": radfact_model,
                            "messages": [{"role": "user", "content": prompt}],
                            token_param: 4096
                        }
                        if not is_gpt5_plus:
                            data["temperature"] = 0.0
                        
                        resp = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=60
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        
                        # Check for API errors in response
                        if 'error' in result:
                            raise ValueError(f"API error: {result['error']}")
                        
                        if 'choices' not in result or not result['choices']:
                            raise ValueError(f"No choices in API response: {result}")
                        
                        content = result['choices'][0]['message']['content'].strip()
                        
                        if not content:
                            raise ValueError(f"Empty response from OpenAI API (model: {radfact_model})")
                        
                        return content
                        
                except requests.exceptions.Timeout as e:
                    last_error = TimeoutError(f"API request timed out: {e}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                        tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Timeout error, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    continue
                    
                except requests.exceptions.HTTPError as e:
                    # Handle rate limiting (429) with exponential backoff
                    if e.response.status_code == 429:
                        retry_count += 1
                        if retry_count <= max_retries:
                            # Check if API provides retry-after header
                            retry_after = e.response.headers.get('retry-after')
                            if retry_after:
                                try:
                                    wait_time = int(retry_after)
                                except:
                                    wait_time = min(2 ** retry_count * 5, 120)  # Exponential backoff
                            else:
                                wait_time = min(2 ** retry_count * 5, 120)  # Exponential backoff, max 120s
                            
                            tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Rate limit hit (429), waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Max retries exceeded for 429
                            try:
                                error_detail = e.response.json()
                                last_error = ValueError(f"HTTP 429 (Rate limit exceeded after {max_retries} retries): {error_detail}")
                            except:
                                last_error = ValueError(f"HTTP 429 (Rate limit exceeded after {max_retries} retries): {e}")
                            break
                    else:
                        # Other HTTP errors - try to extract error message
                        try:
                            error_detail = e.response.json()
                            last_error = ValueError(f"HTTP {e.response.status_code}: {error_detail}")
                        except:
                            last_error = ValueError(f"HTTP {e.response.status_code}: {e}")
                        
                        # Retry on 5xx errors
                        if 500 <= e.response.status_code < 600:
                            retry_count += 1
                            if retry_count <= max_retries:
                                wait_time = min(2 ** retry_count, 60)
                                tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Server error ({e.response.status_code}), waiting {wait_time}s before retry...")
                                time.sleep(wait_time)
                                continue
                        break
                        
                except requests.exceptions.RequestException as e:
                    last_error = ConnectionError(f"API request failed: {e}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = min(2 ** retry_count, 60)
                        tqdm.write(f"    [RETRY {retry_count}/{max_retries}] Connection error, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    continue
            
            # If we get here, all retries failed
            if last_error:
                raise last_error
            else:
                raise RuntimeError("API call failed after retries with unknown error")
        
        # Function to process a single sample
        def process_sample(idx, pred_text, ref_text):
            """Process one sample (prediction-reference pair) and return precision/recall."""
            try:
                # Check if this sample is already cached
                sample_key = get_sample_cache_key(pred_text, ref_text)
                if sample_key in cache:
                    # Use cached results, skip all API calls
                    cached_result = cache[sample_key]
                    return idx, cached_result['precision'], cached_result['recall'], None
                
                # Split reports into findings (one per line)
                pred_findings = [line.strip() for line in pred_text.strip().split('\n') if line.strip()]
                ref_findings = [line.strip() for line in ref_text.strip().split('\n') if line.strip()]
                
                if not ref_findings:
                    return idx, None, None, "empty_ref"
                
                # BATCHED PRECISION CHECK: Check all predicted findings in one API call
                if pred_findings:
                    pred_findings_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(pred_findings)])
                    precision_prompt = f"""You are a medical report evaluator. For each CLAIM below, determine if it is logically entailed by (supported by/consistent with) the REFERENCE.

REFERENCE:
{ref_text}

CLAIMS:
{pred_findings_text}

For each claim, answer with only "ENTAILMENT", "CONTRADICTION", or "NEUTRAL" on a separate line."""
                    
                    try:
                        precision_answer = make_api_call(precision_prompt)
                        
                        # Check for empty response
                        if not precision_answer or not precision_answer.strip():
                            error_msg = f"Empty response from API for precision check (sample {idx})"
                            tqdm.write(f"    [ERROR] {error_msg}")
                            entailed_count = 0
                        else:
                            # Count how many claims are entailed
                            entailed_count = precision_answer.upper().count("ENTAILMENT")
                    except Exception as e:
                        error_msg = f"Precision API call failed (sample {idx}): {type(e).__name__}: {e}"
                        tqdm.write(f"    [ERROR] {error_msg}")
                        entailed_count = 0
                else:
                    entailed_count = 0
                
                precision = entailed_count / len(pred_findings) if pred_findings else 0.0
                
                # BATCHED RECALL CHECK: Check all reference findings in one API call
                if ref_findings:
                    ref_findings_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(ref_findings)])
                    recall_prompt = f"""You are a medical report evaluator. For each CLAIM below, determine if it is logically entailed by (supported by/consistent with) the REFERENCE.

REFERENCE:
{pred_text}

CLAIMS:
{ref_findings_text}

For each claim, answer with only "ENTAILMENT", "CONTRADICTION", or "NEUTRAL" on a separate line."""
                    
                    try:
                        recall_answer = make_api_call(recall_prompt)
                        
                        # Check for empty response
                        if not recall_answer or not recall_answer.strip():
                            error_msg = f"Empty response from API for recall check (sample {idx})"
                            tqdm.write(f"    [ERROR] {error_msg}")
                            covered_count = 0
                        else:
                            # Count how many claims are entailed
                            covered_count = recall_answer.upper().count("ENTAILMENT")
                    except Exception as e:
                        error_msg = f"Recall API call failed (sample {idx}): {type(e).__name__}: {e}"
                        tqdm.write(f"    [ERROR] {error_msg}")
                        covered_count = 0
                else:
                    covered_count = 0
                
                recall = covered_count / len(ref_findings) if ref_findings else 0.0
                
                # Cache and return results
                sample_key = get_sample_cache_key(pred_text, ref_text)
                cache[sample_key] = {
                    'precision': precision,
                    'recall': recall
                }
                
                return idx, precision, recall, None
                
            except Exception as e:
                # Handle sample-level errors gracefully
                import traceback
                error_msg = f"Sample {idx} processing failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                tqdm.write(f"    [ERROR] {error_msg}")
                return idx, None, None, error_msg
        
        # Process samples in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Thread-safe cache saving
        cache_lock = threading.Lock()
        
        def save_cache_safe():
            with cache_lock:
                save_cache()
        
        all_precisions = []
        all_recalls = []
        failed_samples = 0
        results = {}  # Store results with original indices
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_sample, idx, pred, ref): idx 
                for idx, (pred, ref) in enumerate(zip(predictions, references))
            }
            
            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(futures), 
                             total=len(predictions),
                             desc="  RadFact evaluation",
                             unit="report"):
                idx, precision, recall, error = future.result()
                
                if error:
                    if error != "empty_ref":
                        tqdm.write(f"    Warning: Failed to process sample {idx}: {error}")
                        failed_samples += 1
                else:
                    results[idx] = (precision, recall)
                    # Save cache periodically (every 10 samples)
                    if len(results) % 10 == 0:
                        save_cache_safe()
        
        # Final cache save
        save_cache_safe()
        
        # Sort results by original index
        for idx in sorted(results.keys()):
            precision, recall = results[idx]
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        if failed_samples > 0:
            print(f"  Warning: {failed_samples} samples failed during RadFact computation")
        
        # Compute average precision and recall
        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        
        # Compute F1
        if avg_precision + avg_recall > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        print(f"  RadFact computed from {len(all_precisions)} samples:")
        print(f"    Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {f1:.4f}")
        
        return {
            'radfact-precision': avg_precision,
            'radfact-recall': avg_recall,
            'radfact-f1': f1
        }
        
    except ImportError as e:
        print(f"Warning: OpenAI package not available: {e}")
        return {
            'radfact-precision': 0.0,
            'radfact-recall': 0.0,
            'radfact-f1': 0.0
        }


def compute_radfact_per_sample(
    predictions: List[str],
    references: List[str],
    radfact_model: str = 'gpt-5-nano',
    cache_file: Optional[str] = None,
    ollama_url: Optional[str] = None,
    max_workers: int = 5,
    rate_limit_rpm: int = 400,
) -> List[Dict[str, float]]:
    """
    Compute RadFact precision/recall/F1 per sample.

    Returns a list aligned with inputs, each item containing:
      - radfact-precision
      - radfact-recall
      - radfact-f1
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")

    import os
    import json
    import hashlib
    import time
    import threading
    import requests
    import httpx
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from tqdm import tqdm

    def safe_zero() -> Dict[str, float]:
        return {
            'radfact-precision': 0.0,
            'radfact-recall': 0.0,
            'radfact-f1': 0.0,
        }

    cache = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load RadFact cache: {e}")
            cache = {}

    def get_sample_cache_key(pred: str, ref: str) -> str:
        content = f"{pred}|{ref}|{radfact_model}"
        return hashlib.md5(content.encode()).hexdigest()

    cache_lock = threading.Lock()

    def save_cache() -> None:
        if not cache_file:
            return
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not save RadFact cache: {e}")

    use_ollama = ollama_url is not None

    try:
        if use_ollama:
            client = "ollama"
            print(f"  Using Ollama model '{radfact_model}' at {ollama_url}")
        else:
            try:
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True,
                )
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), http_client=http_client)
            except TypeError:
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), max_retries=3, timeout=60.0)
    except Exception as e:
        print(f"  Warning: Failed to initialize RadFact client: {e}")
        return [safe_zero() for _ in predictions]

    api_call_times = []
    api_lock = threading.Lock()

    def make_api_call(prompt: str, max_retries: int = 5) -> str:
        with api_lock:
            current_time = time.time()
            api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
            if len(api_call_times) >= rate_limit_rpm:
                oldest_call = api_call_times[0]
                wait_time = 60 - (current_time - oldest_call)
                if wait_time > 0:
                    time.sleep(wait_time + 0.1)
                    current_time = time.time()
                    api_call_times[:] = [t for t in api_call_times if current_time - t < 60]
            api_call_times.append(current_time)

        retry_count = 0
        last_error = None
        while retry_count <= max_retries:
            try:
                if client == "ollama":
                    data = {
                        "model": radfact_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    }
                    resp = requests.post(f"{ollama_url}/api/chat", json=data, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()
                    content = result['message']['content'].strip()
                    if not content:
                        raise ValueError("Empty response from Ollama API")
                    return content

                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                is_gpt5_plus = "gpt-5" in radfact_model.lower() or "gpt-6" in radfact_model.lower()
                token_param = "max_completion_tokens" if is_gpt5_plus else "max_tokens"
                data = {
                    "model": radfact_model,
                    "messages": [{"role": "user", "content": prompt}],
                    token_param: 4096,
                }
                if not is_gpt5_plus:
                    data["temperature"] = 0.0

                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                if 'choices' not in result or not result['choices']:
                    raise ValueError(f"No choices in API response: {result}")
                content = result['choices'][0]['message']['content'].strip()
                if not content:
                    raise ValueError("Empty response from OpenAI API")
                return content

            except requests.exceptions.RequestException as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(min(2 ** retry_count, 60))
                continue
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(min(2 ** retry_count, 60))
                continue

        raise RuntimeError(f"API call failed after retries: {last_error}")

    def process_sample(idx: int, pred_text: str, ref_text: str):
        try:
            sample_key = get_sample_cache_key(pred_text, ref_text)
            if sample_key in cache:
                c = cache[sample_key]
                p = float(c.get('precision', 0.0))
                r = float(c.get('recall', 0.0))
                f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                return idx, p, r, f1, None

            pred_findings = [line.strip() for line in pred_text.strip().split('\n') if line.strip()]
            ref_findings = [line.strip() for line in ref_text.strip().split('\n') if line.strip()]
            if not ref_findings:
                return idx, 0.0, 0.0, 0.0, "empty_ref"

            if pred_findings:
                pred_findings_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(pred_findings)])
                precision_prompt = f"""You are a medical report evaluator. For each CLAIM below, determine if it is logically entailed by (supported by/consistent with) the REFERENCE.

REFERENCE:
{ref_text}

CLAIMS:
{pred_findings_text}

For each claim, answer with only \"ENTAILMENT\", \"CONTRADICTION\", or \"NEUTRAL\" on a separate line."""
                precision_answer = make_api_call(precision_prompt)
                entailed_count = precision_answer.upper().count("ENTAILMENT")
            else:
                entailed_count = 0
            precision = entailed_count / len(pred_findings) if pred_findings else 0.0

            if ref_findings:
                ref_findings_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(ref_findings)])
                recall_prompt = f"""You are a medical report evaluator. For each CLAIM below, determine if it is logically entailed by (supported by/consistent with) the REFERENCE.

REFERENCE:
{pred_text}

CLAIMS:
{ref_findings_text}

For each claim, answer with only \"ENTAILMENT\", \"CONTRADICTION\", or \"NEUTRAL\" on a separate line."""
                recall_answer = make_api_call(recall_prompt)
                covered_count = recall_answer.upper().count("ENTAILMENT")
            else:
                covered_count = 0
            recall = covered_count / len(ref_findings) if ref_findings else 0.0

            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            with cache_lock:
                cache[sample_key] = {
                    'precision': precision,
                    'recall': recall,
                }

            return idx, precision, recall, f1, None
        except Exception as e:
            return idx, 0.0, 0.0, 0.0, str(e)

    per_sample = [safe_zero() for _ in predictions]
    failed_samples = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sample, idx, pred, ref): idx
            for idx, (pred, ref) in enumerate(zip(predictions, references))
        }
        for future in tqdm(as_completed(futures), total=len(predictions), desc="  RadFact per-sample", unit="sample"):
            idx, precision, recall, f1, error = future.result()
            if error and error != "empty_ref":
                failed_samples += 1
            per_sample[idx] = {
                'radfact-precision': float(precision),
                'radfact-recall': float(recall),
                'radfact-f1': float(f1),
            }

            if (idx + 1) % 25 == 0:
                with cache_lock:
                    save_cache()

    with cache_lock:
        save_cache()

    if failed_samples > 0:
        print(f"  Warning: {failed_samples} samples failed during per-sample RadFact computation")

    return per_sample


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    include_sbert: bool = True,
    sbert_device: Optional[str] = None,
    return_per_field: bool = False,
    include_radfact: bool = False,
    radfact_cache_file: Optional[str] = None,
    radfact_model: str = 'gpt-5-nano',
    ollama_url: Optional[str] = None,
    radfact_max_workers: int = 5,
    radfact_rate_limit: int = 400
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        include_sbert: Whether to compute Sentence-BERT (slower)
        sbert_device: Device for Sentence-BERT model
        return_per_field: Return per-field accuracy breakdown
        include_radfact: Whether to compute RadFact (requires OpenAI API, slower)
        radfact_cache_file: Path to cache file for RadFact results
        radfact_model: Model to use for RadFact (default: gpt-5-nano)
        ollama_url: URL for Ollama API (for local models)
        radfact_max_workers: Number of parallel workers for RadFact
        radfact_rate_limit: API rate limit in requests per minute
    
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
    
    # BLEU-4
    bleu4_metrics = compute_bleu(predictions, references, n_gram=4)
    metrics.update(bleu4_metrics)
    
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
    
    # RadFact (optional, requires OpenAI API, slower)
    if include_radfact:
        radfact_metrics = compute_radfact(
            predictions, references, 
            radfact_model=radfact_model,
            cache_file=radfact_cache_file,
            ollama_url=ollama_url,
            max_workers=radfact_max_workers,
            rate_limit_rpm=radfact_rate_limit
        )
        metrics.update(radfact_metrics)
    
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
