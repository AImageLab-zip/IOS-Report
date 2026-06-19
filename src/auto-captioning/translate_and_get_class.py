#!/usr/bin/env python3
"""
Script to classify Italian medical captions into structured multi-task orthodontic classes.
Reads captions from _data/{dataset}/*/text/*.txt
Outputs classification JSON files to src/auto-captioning/task_classes/{patient_name}.json
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Paths
BASE_DIR = Path("/work/grana_maxillo/IOS-DraftReport")
CLASSIFICATION_SCHEMA_PATH = BASE_DIR / "_data/multi_task_classification.json"
OUTPUT_DIR = BASE_DIR / "src/auto-captioning/task_classes"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the classification schema
with open(CLASSIFICATION_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    CLASSIFICATION_SCHEMA = json.load(f)

SYSTEM_PROMPT = """You are a medical language expert specializing in orthodontics. 
You must analyze short Italian clinical captions describing dental occlusion findings and convert them into a structured JSON object with predefined classes and values.

Follow these rules strictly:
1. Use ONLY the allowed class names and possible values provided below.
2. If a class is not mentioned or cannot be determined, DO NOT invent a value — omit that key.
3. If the text explicitly negates a finding (e.g., "assenza di affollamento"), assign the appropriate negative label (like "absent" or "none").
4. Never use any term outside the provided possible values.
5. Do not include explanations or additional text — output only the final JSON object.
6. Multiple values per task are allowed ONLY if they are both present and compatible (e.g., "left" and "right" for Scissor_bite).
7. Never combine mutually exclusive values (e.g., "none" and "present").
8. For classes indicating only "present" or "absent"/"none", use "absent"/"none" as default if not specified in the caption.
9. Ensure to include all relevant findings mentioned in the caption which are part of the schema.
Allowed classes and values (JSON schema):
{classification_schema}

Examples:

Example 1:
Input Caption: "Classe I molare e canina bilaterale, overjet nella norma, overbite aumentato, linee medie coincidenti, lieve affollamento inferiore, assenza di contrazione mascellare."
Output JSON:
{{
  "Molar_class_right": ["I"],
  "Molar_class_left": ["I"],
  "Canine_class_right": ["I"],
  "Canine_class_left": ["I"],
  "Overjet": ["normal"],
  "Overbite": ["increased"],
  "Midline_deviation": ["coincident"],
  "Crowding_lower": ["mild"],
  "Transverse_constriction": ["absent"]
}}

Example 2:
Input Caption: "II classe molare destra testa a testa, I classe sinistra, morso profondo, overjet aumentato, affollamento superiore moderato-severo, linee medie lievemente deviate, curva di Spee accentuata, crossbite posteriore presente da entrambi i lati"
Output JSON:
{{
  "Molar_class_right": ["II edge_to_edge"],
  "Molar_class_left": ["I"],
  "Crossbite_presence": ["posterior"],
  "Crossbite_laterality": ["left", "right"],
  "Overjet": ["increased"],
  "Overbite": ["increased"],
  "Vertical_pattern": ["deepbite"],
  "Crowding_upper": ["moderate", "severe"],
  "Midline_deviation": ["deviated"],
  "Curve_of_Spee": ["increased"]
}}
"""

USER_PROMPT_TEMPLATE = """Caption (in Italian):
"{caption_text}"

Return the structured JSON containing only the relevant class:value pairs, 
using exactly the keys and values defined in the schema above.

If a class is not mentioned, omit it.
Output only a valid JSON."""


def validate_classification(classification: dict) -> tuple[bool, str]:
    """
    Validate that the classification output follows the schema.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(classification, dict):
        return False, "Output is not a dictionary"
    
    for task, values in classification.items():
        # Check if task exists in schema
        if task not in CLASSIFICATION_SCHEMA:
            return False, f"Unknown task: {task}"
        
        # Check if values is a list
        if not isinstance(values, list):
            return False, f"Values for task '{task}' must be a list, got {type(values)}"
        
        # Check if all values are valid
        allowed_values = CLASSIFICATION_SCHEMA[task]
        for value in values:
            if value not in allowed_values:
                return False, f"Invalid value '{value}' for task '{task}'. Allowed: {allowed_values}"
        
        # Check for mutually exclusive values
        if len(values) > 1:
            # Check for "none"/"absent" with other values
            negative_indicators = ["none", "absent"]
            has_negative = any(v in negative_indicators for v in values)
            has_positive = any(v not in negative_indicators for v in values)
            
            if has_negative and has_positive:
                return False, f"Task '{task}' has mutually exclusive values: {values}"
    
    return True, ""


def classify_caption(caption_text: str, model: str = "gpt-4o-mini", max_retries: int = 2) -> tuple[dict, int, int]:
    """
    Classify Italian caption into structured orthodontic classes.
    
    Args:
        caption_text: Italian clinical caption to classify
        model: OpenAI model to use
        max_retries: Maximum number of retries if validation fails
    
    Returns:
        tuple: (classification_dict, tokens_used, retry_count)
    """
    system_prompt = SYSTEM_PROMPT.format(
        classification_schema=json.dumps(CLASSIFICATION_SCHEMA, indent=2, ensure_ascii=False)
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(caption_text=caption_text)
    
    total_tokens = 0
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic output for structured classification
                max_tokens=2000,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            output_text = response.choices[0].message.content.strip()
            total_tokens += response.usage.total_tokens
            
            # Parse JSON
            classification = json.loads(output_text)
            
            # Validate
            is_valid, error_msg = validate_classification(classification)
            
            if is_valid:
                return classification, total_tokens, attempt
            else:
                print(f"  ⚠ Validation failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                if attempt < max_retries:
                    # Add error feedback to retry
                    user_prompt += f"\n\nPrevious attempt failed validation: {error_msg}\nPlease correct and try again."
                else:
                    raise ValueError(f"Classification validation failed after {max_retries + 1} attempts: {error_msg}")
        
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                user_prompt += f"\n\nPrevious attempt produced invalid JSON: {e}\nPlease output only valid JSON."
            else:
                raise ValueError(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
        
        except Exception as e:
            print(f"  ✗ Error during classification (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt >= max_retries:
                raise
    
    raise ValueError(f"Classification failed after {max_retries + 1} attempts")


def process_patient(patient_path: Path) -> dict:
    """
    Process a single patient's caption files.
    
    Returns:
        dict: Patient classification data, or None if no text files found
    """
    text_dir = patient_path / "text"
    if not text_dir.exists():
        return None
    
    patient_id = patient_path.name
    
    # Find text files (ONLY ios.txt or intraoral-photo.txt or intraoral-photos.txt)
    text_files = list(text_dir.glob("ios.txt")) + list(text_dir.glob("intraoral-photo.txt")) + list(text_dir.glob("intraoral-photos.txt"))

    if not text_files:
        return None
    
    results = {}
    
    for text_file in text_files:
        caption_type = text_file.stem  # ios or intraoral-photos/intraoral-photo
        
        # Read Italian caption
        with open(text_file, 'r', encoding='utf-8') as f:
            italian_text = f.read().strip()
        
        if not italian_text:
            continue
        
        print(f"  Processing {caption_type}...")
        
        # Classify caption
        classification, tokens_used, retry_count = classify_caption(italian_text)
        
        # Create result dictionary
        result = {
            "patient_id": patient_id,
            "caption_type": caption_type,
            "original_caption": italian_text,
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "tokens_used": tokens_used,
            "retry_count": retry_count,
            "source_file": str(text_file.relative_to(BASE_DIR))
        }
        
        results[caption_type] = result
    
    return results if results else None


def process_all_patients(dataset_path: Path, force: bool = False):
    """Process all patients in a dataset folder."""
    # Find all patient directories
    patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(patient_dirs)} patient directories in {dataset_path}")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("-" * 80)
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_tokens = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients", unit="patient"):
        patient_id = patient_dir.name
        
        # Check if already processed
        output_file = OUTPUT_DIR / f"{patient_id}.json"
        if output_file.exists() and not force:
            tqdm.write(f"⊘ Skipping {patient_id} (already processed, use --force to reprocess)")
            skipped_count += 1
            continue
        
        tqdm.write(f"Processing {patient_id}...")
        
        try:
            results = process_patient(patient_dir)
            
            if results:
                # Save to JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                processed_count += 1
                for caption_type, result in results.items():
                    total_tokens += result['tokens_used']
                    retry_info = f" ({result['retry_count']} retries)" if result['retry_count'] > 0 else ""
                    tqdm.write(f"  ✓ Saved {caption_type} - {result['tokens_used']} tokens{retry_info}")
            else:
                skipped_count += 1
                tqdm.write(f"  ⊘ No text files found")
        
        except Exception as e:
            tqdm.write(f"  ✗ Error: {e}")
            error_count += 1
    
    print("\n" + "-" * 80)
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Estimated cost (gpt-4o-mini): ${total_tokens * 0.0015 / 1000:.2f}")


def process_single_patient(patient_name: str, dataset_path: Path, force: bool = False):
    """Process a single patient by name."""
    patient_dir = dataset_path / patient_name
    
    if not patient_dir.exists():
        print(f"Error: Patient directory not found: {patient_dir}")
        return
    
    output_file = OUTPUT_DIR / f"{patient_name}.json"
    
    if output_file.exists() and not force:
        print(f"Patient {patient_name} already processed. Use --force to reprocess.")
        return
    
    print(f"Processing patient: {patient_name}")
    print("-" * 80)
    
    try:
        results = process_patient(patient_dir)
        
        if results:
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Successfully processed {patient_name}")
            total_tokens = sum(r['tokens_used'] for r in results.values())
            print(f"Total tokens used: {total_tokens:,}")
            print(f"Estimated cost (gpt-4o-mini): ${total_tokens * 0.0015 / 1000:.2f}")
            print(f"\nOutput saved to: {output_file}")
            
            # Print classification summary
            print("\nClassification Summary:")
            for caption_type, result in results.items():
                print(f"\n{caption_type}:")
                for task, values in result['classification'].items():
                    print(f"  {task}: {values}")
        else:
            print(f"No text files found for patient {patient_name}")
    
    except Exception as e:
        print(f"✗ Error processing {patient_name}: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Classify Italian medical captions into structured orthodontic classes"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="/work/grana_maxillo/IOS-DraftReport/_data/Dataset_FerraraDump_400_and_Bits2Bites",
        help="Path to dataset folder containing patient directories"
    )
    
    parser.add_argument(
        "--patient",
        type=str,
        help="Process only this patient (by name)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed patients"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    if args.patient:
        # Process single patient
        process_single_patient(args.patient, dataset_path, args.force)
    else:
        # Process all patients
        process_all_patients(dataset_path, args.force)


if __name__ == "__main__":
    main()
