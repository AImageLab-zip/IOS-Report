#!/usr/bin/env python3
"""Quick script to check available OpenAI models."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("Fetching available models...\n")

models = client.models.list()

# Filter for GPT models with vision capabilities
gpt_models = [m for m in models.data if 'gpt' in m.id.lower()]

print("Available GPT models:")
print("-" * 60)
for model in sorted(gpt_models, key=lambda x: x.id):
    print(f"  {model.id}")

print("-" * 60)
print(f"\nTotal GPT models: {len(gpt_models)}")
