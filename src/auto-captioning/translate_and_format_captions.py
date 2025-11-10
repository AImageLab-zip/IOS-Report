#!/usr/bin/env python3
"""
Script to translate Italian medical captions to English and format them according to the template.
Reads captions from _data/ios-iop-text/{patient}/text/{ios|intraoral-photos}.txt
Outputs formatted JSON files to src/auto-captioning/real_captions/
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Paths
BASE_DIR = Path("/work/grana_maxillo/IOS-DraftReport")
INPUT_DIR = BASE_DIR / "_data/Dataset_Bits2Bites"
OUTPUT_DIR = BASE_DIR / "src/auto-captioning/real_captions"
TEMPLATE_PATH = BASE_DIR / "_data/autocaption-template.txt"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the template
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    TEMPLATE = f.read()

TRANSLATION_PROMPT = """You are a professional medical translator and certified orthodontist.

Your task:
1. Translate the following *Italian intra-oral scan description* into English.
2. Convert it into a structured caption following the provided standardized template.
3. Preserve all clinical meaning and orthodontic terminology.
4. Output only the formatted English description, strictly following the template’s structure and labels.

Input information:
- The Italian text may be short, informal, or use abbreviations (e.g., "aff." for "affected").
- Expand abbreviations and normalize phrasing while maintaining clinical accuracy.
- Use FDI numbering (11-18 UR, 21-28 UL, 31-38 LL, 41-48 LR).
- If a piece of information is missing or cannot be inferred, write “Unknown.”
- Do **not** invent findings.
- Do **not** add commentary, definitions, or treatment suggestions.
- The template options are enclosed in curly braces separated by pipes |. Placeholders are described inside square brackets. Choose the most appropriate option based on the Italian text.
- Do not put any curly braces or pipes | or other unrelated symbols in your final output.
- Do not put any whitespace before newlines.

TEMPLATE:
{template}

ITALIAN DESCRIPTION TO TRANSLATE AND FORMAT:
{italian_text}

OUTPUT INSTRUCTIONS:
- Then list only the fields and corresponding values in the same order and formatting as the template.
- Do not include any sections not requested in the template.
- Write all field names exactly as in the template.
- Ensure capitalization, punctuation, and terminology are consistent and professional.
- Do not include any introductory or closing remarks.

Your final answer must contain **only** the translated and formatted intra-oral description.
"""

def translate_and_format_caption(italian_text: str, model: str = "gpt-4o-mini") -> tuple[str, int]:
    """
    Translate Italian caption to English and format according to template.
    
    Returns:
        tuple: (formatted_description, tokens_used)
    """
    prompt = TRANSLATION_PROMPT.format(template=TEMPLATE, italian_text=italian_text)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional medical translator and orthodontist specializing in dental descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent medical translations
            max_tokens=5000
        )
        
        description = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        return description, tokens_used
    
    except Exception as e:
        print(f"Error during translation: {e}")
        raise

def process_patient(patient_path: Path) -> dict:
    """
    Process a single patient's caption files.
    
    Returns:
        dict: Patient data with translated captions, or None if no text files found
    """
    text_dir = patient_path / "text"
    if not text_dir.exists():
        return None
    
    patient_id = patient_path.name
    
    # Find text files (ios.txt or intraoral-photos.txt or intraoral-photo.txt)
    text_files = list(text_dir.glob("*.txt"))
    
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
        
        print(f"Processing {patient_id} - {caption_type}...")
        
        # Translate and format
        description, tokens_used = translate_and_format_caption(italian_text)
        
        # Create result dictionary
        result = {
            "patient_id": patient_id,
            "caption_type": caption_type,
            "original_italian": italian_text,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "tokens_used": tokens_used,
            "source_file": str(text_file.relative_to(BASE_DIR))
        }
        
        results[caption_type] = result
    
    return results if results else None

def main():
    """Main processing function."""
    # Find all patient directories
    patient_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("-" * 80)
    
    processed_count = 0
    skipped_count = 0
    total_tokens = 0
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        # Check if already processed
        output_file = OUTPUT_DIR / f"{patient_id}.json"
        if output_file.exists():
            print(f"Skipping {patient_id} (already processed)")
            skipped_count += 1
            continue
        
        try:
            results = process_patient(patient_dir)
            
            if results:
                # Save to JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                processed_count += 1
                for caption_type, result in results.items():
                    total_tokens += result['tokens_used']
                    print(f"✓ Saved {patient_id} ({caption_type}) - {result['tokens_used']} tokens")
            else:
                skipped_count += 1
                print(f"⊘ No text files for {patient_id}")
        
        except Exception as e:
            print(f"✗ Error processing {patient_id}: {e}")
            skipped_count += 1
    
    print("-" * 80)
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Estimated cost (GPT-4o-mini): ${total_tokens * 0.0015 / 1000:.2f}")

if __name__ == "__main__":
    main()
