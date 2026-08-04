#!/usr/bin/env python3
"""
Script to translate Italian medical captions to English and format them according to the template.
Reads IOS reports from {input_dir}/{patient}/ios/reports/{annotator_id}_{voice_caption_id}.txt
Outputs formatted JSON files to the specified output directory.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


TRANSLATION_PROMPT = """You are a professional medical translator and certified orthodontist.

Your task:
1. Translate the following *Italian intra-oral scan description* into English.
2. Convert it into a structured caption following the provided standardized template.
3. Preserve all clinical meaning and orthodontic terminology.
4. Output only the formatted English description, strictly following the template's structure and labels.

Input information:
- The Italian text may be short, informal, or use abbreviations (e.g., "aff." for "affected").
- Expand abbreviations and normalize phrasing while maintaining clinical accuracy.
- Use FDI numbering (11-18 UR, 21-28 UL, 31-38 LL, 41-48 LR).
- If a piece of information is missing or cannot be inferred, write "Unknown."
- Do not invent findings.
- Do not add commentary, definitions, or treatment suggestions.
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

Your final answer must contain only the translated and formatted intra-oral description.
"""


def translate_and_format_caption(client: OpenAI, italian_text: str, template: str, model: str) -> tuple[str, int]:
    """
    Translate Italian caption to English and format according to template.

    Returns:
        tuple: (formatted_description, tokens_used)
    """
    prompt = TRANSLATION_PROMPT.format(template=template, italian_text=italian_text)

    request_common = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional medical translator and orthodontist specializing in dental descriptions.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    request_with_temp = dict(request_common)
    request_with_temp["temperature"] = 0.1

    try:
        response = client.chat.completions.create(
            **request_with_temp,
            max_completion_tokens=5000,
        )
    except Exception as e:
        err = str(e)
        if "temperature" in err and "Only the default (1) value is supported" in err:
            try:
                response = client.chat.completions.create(
                    **request_common,
                    max_completion_tokens=5000,
                )
            except Exception as e2:
                if "max_completion_tokens" in str(e2):
                    response = client.chat.completions.create(
                        **request_common,
                        max_tokens=5000,
                    )
                else:
                    raise
        elif "max_completion_tokens" in err:
            response = client.chat.completions.create(
                **request_with_temp,
                max_tokens=5000,
            )
        else:
            raise

    description = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens
    return description, tokens_used


def process_patient(
    client: OpenAI,
    patient_path: Path,
    template: str,
    model: str,
    input_dir: Path,
) -> dict | None:
    """
    Process a single patient's report files.

    Returns:
        dict: Patient data with translated reports, or None if no report files found
    """
    reports_dir = patient_path / "ios" / "reports"
    if not reports_dir.exists():
        return None

    patient_id = patient_path.name
    report_files = sorted(reports_dir.glob("*.txt"))
    if not report_files:
        return None

    results = {}

    for report_file in report_files:
        caption_type = report_file.stem
        match = re.match(r"^(?P<annotator_id>\d+)_(?P<voice_caption_id>\d+)$", caption_type)
        if not match:
            print(f"Skipping malformed report filename: {report_file.name}")
            continue

        modality_slug = "ios"
        annotator_id = match.group("annotator_id")
        voice_caption_id = match.group("voice_caption_id")

        with open(report_file, "r", encoding="utf-8") as f:
            italian_text = f.read().strip()

        if not italian_text:
            continue

        print(f"Processing {patient_id} - {caption_type}...")
        description, tokens_used = translate_and_format_caption(client, italian_text, template, model)

        results[caption_type] = {
            "patient_id": patient_id,
            "caption_type": caption_type,
            "modality_slug": modality_slug,
            "annotator_id": annotator_id,
            "voice_caption_id": voice_caption_id,
            "original_italian": italian_text,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens_used": tokens_used,
            "source_file": str(report_file.relative_to(input_dir)),
        }

    return results if results else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate Italian captions and format into templated English JSON captions"
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with patient subfolders")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where JSON captions are saved")
    parser.add_argument("--template_path", type=Path, required=True, help="Path to template text file")
    parser.add_argument("--model", type=str, default="gpt-5.4-nano", help="OpenAI model name")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.template_path.exists():
        raise FileNotFoundError(f"Template file not found: {args.template_path}")

    with open(args.template_path, "r", encoding="utf-8") as f:
        template = f.read()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)

    patient_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])

    print(f"Found {len(patient_dirs)} patient directories")
    print(f"Output will be saved to: {args.output_dir}")
    print("-" * 80)

    processed_count = 0
    skipped_count = 0
    updated_count = 0
    total_tokens = 0
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        output_file = args.output_dir / f"{patient_id}.json"

        try:
            current_results = process_patient(client, patient_dir, template, args.model, args.input_dir)
            if current_results:
                if output_file.exists():
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                    merged_results = dict(existing_results)
                    merged_results.update(current_results)
                    is_update = True
                else:
                    merged_results = current_results
                    is_update = False

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(merged_results, f, indent=2, ensure_ascii=False)

                if is_update:
                    updated_count += 1
                else:
                    processed_count += 1

                for caption_type, result in current_results.items():
                    total_tokens += result["tokens_used"]
                    print(f"Saved {patient_id} ({caption_type}) - {result['tokens_used']} tokens")
            else:
                skipped_count += 1
                print(f"No report files for {patient_id}")

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            skipped_count += 1

    print("-" * 80)
    print("\nProcessing complete!")
    print(f"Processed (new files): {processed_count}")
    print(f"Updated (existing files): {updated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Estimated cost (approx): ${total_tokens * 0.0015 / 1000:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
