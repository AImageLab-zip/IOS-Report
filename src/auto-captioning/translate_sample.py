#!/usr/bin/env python3
"""
Test script to translate and format a few sample captions to verify the approach.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

BASE_DIR = Path("/work/grana_maxillo/IOS-DraftReport")
TEMPLATE_PATH = BASE_DIR / "_data/prompt-template.txt"

with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    TEMPLATE = f.read()

TRANSLATION_PROMPT = """You are a professional medical translator and orthodontist. 

Your task is to:
1. Translate the following Italian dental/orthodontic description to English
2. Format it according to the provided standardized template structure
3. Maintain all clinical accuracy and terminology

The Italian description may be informal or abbreviated. You should expand it into a complete, structured clinical description following the template format exactly.

TEMPLATE:
{template}

ITALIAN DESCRIPTION TO TRANSLATE AND FORMAT:
{italian_text}

OUTPUT INSTRUCTIONS:
- Provide ONLY the formatted English description following the template structure
- Be thorough and professional
- If information for a section is not provided in the Italian text, write "Not clearly discernible from the provided clinical notes."
- Maintain third person perspective ("The patient presents...")
"""

# Sample Italian texts to test
SAMPLES = [
    {
        "patient_id": "201_1979",
        "text": "il paziente presenta a destra un rapporto di Classe II testa a testa molare e canina, mentre a sinistra è presente un rapporto di II classe piena molare e canina. a livello trasversale è evidenziabile una contrazione mascellare (i torque dei denti posteriori sono positivi) in assenza tuttavia di crossbite posteriore. l'overjet è aumentato così come l'overbite. le linee mediane sono lievemente deviate. la curva di Spee è aumentata. è presente un lieve affollamento inferiormente, mentre risulta moderato-severo superiormente."
    },
    {
        "patient_id": "296_2070",
        "text": "Il paziente presenta una contrazione del mascellare superiore cross bite sugli elementi 16-15-14-24-25-26-27. Dal punto di vista verticale presenta un open bite anteriore. Sagittalmente è una terza classe piena molare e prima canina bilaterale con un cross bite anteriore sugli elementi 12 e 22. Le linee mediane non sono coincidenti. Curva di spee nella norma, curva di Wilson molto aumentata in arcata inferiore. Presenta affollamento severo in arcata superiore ed inferiore con una recessione vestibolare su 31."
    }
]

def translate_and_format_caption(italian_text: str, model: str = "gpt-4o-mini") -> tuple[str, int]:
    """
    Translate Italian caption to English and format according to template.
    """
    prompt = TRANSLATION_PROMPT.format(template=TEMPLATE, italian_text=italian_text)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional medical translator and orthodontist specializing in dental descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    
    description = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens
    
    return description, tokens_used

def main():
    print("Testing translation and formatting on sample captions...")
    print("=" * 80)
    
    for sample in SAMPLES:
        print(f"\nPatient ID: {sample['patient_id']}")
        print(f"\nOriginal Italian:")
        print(sample['text'])
        print("\n" + "-" * 80)
        
        description, tokens = translate_and_format_caption(sample['text'])
        
        print(f"\nFormatted English Translation:")
        print(description)
        print(f"\nTokens used: {tokens}")
        print("=" * 80)

if __name__ == "__main__":
    main()
