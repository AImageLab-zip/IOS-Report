"""
External model API clients for evaluation.

This module provides unified interfaces for evaluating external models:
- OpenAI GPT5.2
- Google Gemini3
- DeepSeek

All models use only intraoral photos as input.
"""

from .base import BaseExternalModel
from .openai_client import OpenAIModel
from .gemini_client import GeminiModel
from .deepseek_client import DeepSeekModel

__all__ = [
    'BaseExternalModel',
    'OpenAIModel',
    'GeminiModel',
    'DeepSeekModel',
]
