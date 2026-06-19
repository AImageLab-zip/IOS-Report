"""
Field-by-field decoder for structured report generation.

This module implements a ranking-based decoding approach where each field
is generated independently by scoring all candidate values and selecting
the one with highest log-probability.
"""

import json
import itertools
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


class FieldByFieldDecoder:
    """
    Decoder that generates structured reports field-by-field using
    ranking/scoring of predefined candidate values.
    
    This decoder:
    1. Iterates through each field in the report
    2. For each field, generates all possible candidates (with cartesian product for multi-part fields)
    3. Scores all candidates in batch by computing log-probabilities
    4. Selects the candidate with highest score (argmax)
    5. Appends the selected value to the growing report and continues
    """
    
    def __init__(
        self,
        model,
        candidate_config_path: str,
        length_normalization_alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the field-by-field decoder.
        
        Args:
            model: The underlying PointQwen model (must have generate/forward methods)
            candidate_config_path: Path to JSON file defining field candidates
            length_normalization_alpha: Exponent for length normalization (0=no norm, 1=mean log-prob)
            device: Device to run scoring on (default: model.device)
        """
        self.model = model
        self.alpha = length_normalization_alpha
        self.device = device if device is not None else model.device
        
        # Load candidate configuration
        with open(candidate_config_path, 'r') as f:
            self.field_candidates = json.load(f)
        
        # Field ordering (from the config file)
        self.field_order = list(self.field_candidates.keys())
        
        print(f"\n{'='*70}")
        print("FIELD-BY-FIELD DECODER INITIALIZED")
        print(f"{'='*70}")
        print(f"Number of fields: {len(self.field_order)}")
        print(f"Length normalization alpha: {self.alpha}")
        print(f"Fields: {', '.join(self.field_order)}")
        print(f"{'='*70}\n")
    
    def _generate_field_candidates(self, field_name: str) -> List[str]:
        """
        Generate all possible candidate strings for a given field.
        
        For fields with multiple components (lists of lists), this performs
        a cartesian product to generate all combinations.
        
        Args:
            field_name: Name of the field
        
        Returns:
            List of candidate strings
        
        Example:
            For "Crowding": [["Mild", "Severe"], ["in the upper arch", "in the lower arch"]]
            Returns: ["Mild in the upper arch", "Mild in the lower arch", 
                     "Severe in the upper arch", "Severe in the lower arch"]
        """
        components = self.field_candidates[field_name]
        
        # Separate lists from strings
        list_components = []
        string_components = []
        
        for comp in components:
            if isinstance(comp, list):
                list_components.append(comp)
            elif isinstance(comp, str):
                string_components.append(comp)
        
        # If no lists, only strings - return them as-is
        if not list_components:
            return string_components if string_components else ["Unknown"]
        
        # Generate cartesian product of all list components
        product = itertools.product(*list_components)
        
        # Build candidate strings
        candidates = []
        for combo in product:
            # Combine tuple items with strings
            parts = list(combo)
            
            # Insert string components at their original positions
            # For simplicity, append string components
            full_parts = list(parts) + string_components
            
            # Join with spaces
            candidate = " ".join(full_parts)
            candidates.append(candidate)
        
        return candidates
    
    def _score_candidates(
        self,
        point_cloud: torch.Tensor,
        prompt_prefix: str,
        field_name: str,
        candidates: List[str],
        batch_size: int = 8,
    ) -> Tuple[List[float], List[int]]:
        """
        Score all candidates for a given field using log-probabilities.
        
        This creates a batch where each item has:
            prompt_prefix + field_name + ": " + candidate
        
        Then computes the log-probability of the candidate tokens given the context,
        using length normalization.
        
        Args:
            point_cloud: (1, N, 3) point cloud tensor
            prompt_prefix: Text before this field (prompt + previously generated fields)
            field_name: Name of the current field
            candidates: List of candidate strings to score
            batch_size: Batch size for scoring (process multiple candidates at once)
        
        Returns:
            scores: List of scores (one per candidate)
            token_counts: List of token counts (for debugging)
        """
        scores = []
        token_counts = []
        
        # Process candidates in batches
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            batch_size_actual = len(batch_candidates)
            
            # Replicate point cloud for batch
            point_clouds_batch = point_cloud.repeat(batch_size_actual, 1, 1).to(torch.bfloat16)
            
            # Build prompts for this batch
            prompts = []
            candidate_texts = []
            for candidate in batch_candidates:
                # Full prompt: prefix + field_name + ": " + candidate
                full_prompt = f"{prompt_prefix}{field_name}: {candidate}"
                prompts.append(full_prompt)
                candidate_texts.append(candidate)
            
            # Tokenize all prompts
            image_pad_token_id = 151655
            inputs = self.model.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            input_ids = inputs.input_ids  # (B, L)
            attention_mask = inputs.attention_mask
            
            # Also tokenize just the prefix to know where candidates start
            # IMPORTANT: Don't include ": " in prefix - it gets tokenized with the candidate
            prefix_prompts = [f"{prompt_prefix}{field_name}:" for _ in batch_candidates]
            prefix_inputs = self.model.processor(
                text=prefix_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Count non-padding tokens in prefix
            prefix_lengths = (prefix_inputs.attention_mask).sum(dim=1)
            
            # Encode point cloud
            visual_tokens = self.model.encode_point_cloud(point_clouds_batch)
            num_visual_tokens = visual_tokens.shape[1]
            
            # Find image pad token positions
            image_pad_token_index = (input_ids == image_pad_token_id).nonzero(as_tuple=True)
            
            # Get text embeddings
            embed_tokens = self.model.qwen.model.language_model.embed_tokens
            text_embeds = embed_tokens(input_ids)
            
            # CRITICAL: Ensure everything is in bfloat16 to match model's dtype
            target_dtype = torch.bfloat16
            text_embeds = text_embeds.to(dtype=target_dtype)
            visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=target_dtype)
            
            # Inject visual tokens
            inputs_embeds = text_embeds.clone()
            
            new_embeds_list = []
            new_attention_mask_list = []
            new_input_ids_list = []
            
            for batch_idx in range(batch_size_actual):
                batch_mask = image_pad_token_index[0] == batch_idx
                if batch_mask.any():
                    token_positions = image_pad_token_index[1][batch_mask]
                    if len(token_positions) > 0:
                        pos = token_positions[0].item()
                        
                        # Embeddings - ensure concatenation preserves bfloat16
                        new_embeds = torch.cat([
                            inputs_embeds[batch_idx, :pos, :],
                            visual_tokens[batch_idx],
                            inputs_embeds[batch_idx, pos+1:, :]
                        ], dim=0)
                        new_embeds_list.append(new_embeds)
                        
                        # Attention mask
                        visual_attn = torch.ones(num_visual_tokens, dtype=attention_mask.dtype,
                                                device=attention_mask.device)
                        new_attn = torch.cat([
                            attention_mask[batch_idx, :pos],
                            visual_attn,
                            attention_mask[batch_idx, pos+1:]
                        ], dim=0)
                        new_attention_mask_list.append(new_attn)
                        
                        # Input IDs (for candidate token identification)
                        # Replace image pad token with a dummy token (we'll mask it anyway)
                        new_ids = torch.cat([
                            input_ids[batch_idx, :pos],
                            torch.full((num_visual_tokens,), fill_value=-100, 
                                      dtype=input_ids.dtype, device=input_ids.device),
                            input_ids[batch_idx, pos+1:]
                        ], dim=0)
                        new_input_ids_list.append(new_ids)
                    else:
                        new_embeds_list.append(inputs_embeds[batch_idx])
                        new_attention_mask_list.append(attention_mask[batch_idx])
                        new_input_ids_list.append(input_ids[batch_idx])
                else:
                    new_embeds_list.append(inputs_embeds[batch_idx])
                    new_attention_mask_list.append(attention_mask[batch_idx])
                    new_input_ids_list.append(input_ids[batch_idx])
            
            # Stack into batch
            inputs_embeds_batch = torch.stack(new_embeds_list, dim=0)
            extended_attention_mask = torch.stack(new_attention_mask_list, dim=0)
            extended_input_ids = torch.stack(new_input_ids_list, dim=0)
            
            # Verify dtype is bfloat16
            assert inputs_embeds_batch.dtype == torch.bfloat16, f"Expected bfloat16, got {inputs_embeds_batch.dtype}"
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model.qwen(
                    inputs_embeds=inputs_embeds_batch,
                    attention_mask=extended_attention_mask,
                )
                logits = outputs.logits  # (B, L, vocab_size)
            
            # Compute log-probabilities for each candidate
            for batch_idx in range(batch_size_actual):
                # Get the token IDs for this sequence
                seq_ids = extended_input_ids[batch_idx]
                seq_logits = logits[batch_idx]  # (L, vocab_size)
                seq_attn = extended_attention_mask[batch_idx]
                
                # Find where the candidate tokens start
                # We tokenized prefix as: prompt_prefix + field_name + ":"  (NO space)
                # And full as: prompt_prefix + field_name + ": " + candidate
                # This way, the " " + candidate gets tokenized together
                
                prefix_len = prefix_lengths[batch_idx].item()
                
                # After visual injection:
                # candidate_start = prefix_len - 1 (for image_pad) + num_visual_tokens
                candidate_start = prefix_len - 1 + num_visual_tokens
                
                # Get candidate tokens (from candidate_start to end, excluding padding)
                seq_len = seq_attn.sum().item()
                candidate_end = int(seq_len)
                
                if candidate_start >= candidate_end:
                    # Edge case: candidate has no tokens (shouldn't happen)
                    scores.append(float('-inf'))
                    token_counts.append(0)
                    continue
                
                # Extract candidate token IDs and logits
                # Logits are shifted: logit[i] predicts token[i+1]
                candidate_token_ids = seq_ids[candidate_start:candidate_end]
                candidate_logits = seq_logits[candidate_start-1:candidate_end-1]  # Shifted by 1
                
                # Filter out special tokens (-100)
                valid_mask = candidate_token_ids != -100
                candidate_token_ids = candidate_token_ids[valid_mask]
                candidate_logits = candidate_logits[valid_mask]
                
                num_tokens = len(candidate_token_ids)
                if num_tokens == 0:
                    # Edge case: candidate has no valid tokens
                    scores.append(float('-inf'))
                    token_counts.append(0)
                    continue
                
                # Compute log-probabilities
                log_probs = F.log_softmax(candidate_logits, dim=-1)
                
                # Get log-prob of actual tokens
                token_log_probs = log_probs[torch.arange(num_tokens), candidate_token_ids]
                
                # Sum log-probs
                total_log_prob = token_log_probs.sum().item()
                
                # Length normalization: score = sum(log_probs) / (num_tokens ** alpha)
                if self.alpha > 0:
                    normalized_score = total_log_prob / (num_tokens ** self.alpha)
                else:
                    normalized_score = total_log_prob
                
                scores.append(normalized_score)
                token_counts.append(num_tokens)
        
        return scores, token_counts
    
    def decode(
        self,
        point_cloud: torch.Tensor,
        base_prompt: str,
        verbose: bool = True,
        scoring_batch_size: int = 8,
    ) -> Dict[str, str]:
        """
        Generate structured report field-by-field.
        
        Args:
            point_cloud: (1, N, 3) point cloud tensor
            base_prompt: Base instruction prompt (e.g., "Analyze the intraoral scan...")
            verbose: Whether to print progress
            scoring_batch_size: Batch size for scoring candidates
        
        Returns:
            Dictionary mapping field_name -> selected_value
        """
        self.model.eval()
        
        # Start with base prompt
        prompt_prefix = base_prompt
        if not prompt_prefix.endswith("\n"):
            prompt_prefix += "\n"
        
        results = {}
        
        if verbose:
            print("\n" + "="*70)
            print("FIELD-BY-FIELD DECODING")
            print("="*70)
        
        # Iterate through fields
        iterator = tqdm(self.field_order, desc="Generating fields") if verbose else self.field_order
        
        for field_name in iterator:
            # Generate candidates for this field
            candidates = self._generate_field_candidates(field_name)
            
            if verbose:
                print(f"\n{field_name}:")
                print(f"  Candidates: {len(candidates)}")
            
            # Score all candidates
            scores, token_counts = self._score_candidates(
                point_cloud=point_cloud,
                prompt_prefix=prompt_prefix,
                field_name=field_name,
                candidates=candidates,
                batch_size=scoring_batch_size,
            )
            
            # Select best candidate (argmax)
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_candidate = candidates[best_idx]
            best_score = scores[best_idx]
            
            if verbose:
                print(f"  Selected: {best_candidate}")
                print(f"  Score: {best_score:.4f} ({token_counts[best_idx]} tokens)")
                
                # Show top-3 candidates
                if len(candidates) > 1:
                    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    print(f"  Top candidates:")
                    for rank, idx in enumerate(sorted_indices[:3], 1):
                        print(f"    {rank}. {candidates[idx]:40s} (score: {scores[idx]:.4f})")
            
            # Save result
            results[field_name] = best_candidate
            
            # Update prompt prefix for next field
            prompt_prefix += f"{field_name}: {best_candidate}\n"
        
        if verbose:
            print("\n" + "="*70)
            print("DECODING COMPLETE")
            print("="*70)
        
        return results
    
    def format_results(self, results: Dict[str, str]) -> str:
        """
        Format results dictionary into a readable report string.
        
        Args:
            results: Dictionary of field_name -> value
        
        Returns:
            Formatted report string
        """
        lines = []
        for field_name in self.field_order:
            if field_name in results:
                lines.append(f"{field_name}: {results[field_name]}")
        return "\n".join(lines)
    
    def decode_batch(
        self,
        point_clouds: torch.Tensor,
        base_prompts: List[str],
        verbose: bool = False,
        scoring_batch_size: int = 8,
    ) -> List[Dict[str, str]]:
        """
        Decode multiple samples (sequentially, one at a time).
        
        Args:
            point_clouds: (B, N, 3) point cloud tensors
            base_prompts: List of base prompts (one per sample)
            verbose: Whether to print progress
            scoring_batch_size: Batch size for scoring candidates
        
        Returns:
            List of result dictionaries
        """
        batch_size = point_clouds.shape[0]
        all_results = []
        
        for i in tqdm(range(batch_size), desc="Decoding batch", disable=not verbose):
            point_cloud = point_clouds[i:i+1]
            prompt = base_prompts[i]
            
            results = self.decode(
                point_cloud=point_cloud,
                base_prompt=prompt,
                verbose=False,  # Disable per-sample verbose
                scoring_batch_size=scoring_batch_size,
            )
            all_results.append(results)
        
        return all_results


def compare_decoders(
    model,
    point_cloud: torch.Tensor,
    prompt: str,
    candidate_config_path: str,
    alpha: float = 1.0,
) -> Tuple[str, Dict[str, str]]:
    """
    Utility function to compare one-shot generation vs field-by-field decoding.
    
    Args:
        model: PointQwen model
        point_cloud: (1, N, 3) point cloud
        prompt: Base prompt
        candidate_config_path: Path to candidate config JSON
        alpha: Length normalization exponent
    
    Returns:
        one_shot_output: String from one-shot generation
        field_by_field_output: Dict from field-by-field decoding
    """
    print("\n" + "="*70)
    print("DECODER COMPARISON")
    print("="*70)
    
    # One-shot generation
    print("\n1. ONE-SHOT GENERATION:")
    print("-" * 70)
    with torch.no_grad():
        one_shot = model.generate(
            point_cloud=point_cloud,
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
    one_shot_output = one_shot[0]
    
    # Extract generated part
    if prompt in one_shot_output:
        one_shot_output = one_shot_output.split(prompt)[-1].strip()
    
    print(one_shot_output)
    
    # Field-by-field decoding
    print("\n2. FIELD-BY-FIELD DECODING:")
    print("-" * 70)
    decoder = FieldByFieldDecoder(
        model=model,
        candidate_config_path=candidate_config_path,
        length_normalization_alpha=alpha,
    )
    
    field_results = decoder.decode(
        point_cloud=point_cloud,
        base_prompt=prompt,
        verbose=True,
    )
    
    field_output = decoder.format_results(field_results)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    return one_shot_output, field_results
