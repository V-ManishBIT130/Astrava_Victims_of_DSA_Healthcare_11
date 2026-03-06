"""
ASTRAVA Preprocessing Module
============================

Text cleaning, crisis detection, tokenization, and embedding generation
for the AI-Powered Mental Health Chatbot.

Quick Start:
    from preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline()

    # Full pipeline (loads models)
    result = pipeline.process("I can't do this anymore")

    # Lightweight mode (no model loading)
    result = pipeline.process_text_only("I feel stressed about work")
    print(result.cleaned_text)
    print(result.crisis_result.severity)
    print(result.crisis_result.psycholinguistic)

Individual Components:
    from preprocessing import TextCleaner, CrisisDetector, EmotionalStopwordFilter
    from preprocessing import ModelTokenizer, ModelEmbedder

Models Used:
    - Emotion:    mental/mental-roberta-base       (RoBERTa)
    - Stress:     jnyx74/stress-prediction          (DistilBERT)
    - Depression: poudel/Depression_and_Non-Depression_Classifier (BERT)
"""

from .cleaner import TextCleaner
from .config import (
    ALL_MODEL_NAMES,
    DEPRESSION_MODEL_NAME,
    EMOTION_MODEL_NAME,
    MAX_SEQ_LENGTH,
    PUNCT_TAGS,
    SLANG_MAP,
    STRESS_MODEL_NAME,
)
from .crisis_detector import CrisisDetector, CrisisResult
from .embedder import ModelEmbedder, get_all_embedders, get_embedder
from .keywords import (
    ABSOLUTE_LANGUAGE,
    ANXIETY_KEYWORDS,
    CRISIS_KEYWORDS,
    CRISIS_PATTERNS,
    DEPRESSION_KEYWORDS,
    DISSOCIATION_MARKERS,
    EMOTION_INTENSIFIERS,
    FIRST_PERSON,
    NEGATION_WORDS,
    RUMINATION_MARKERS,
    STRESS_KEYWORDS,
)
from .pipeline import PreprocessingPipeline, PreprocessingResult
from .stopwords import EmotionalStopwordFilter
from .tokenizer import ModelTokenizer, get_all_tokenizers, get_tokenizer

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessingResult",
    # Components
    "TextCleaner",
    "CrisisDetector",
    "CrisisResult",
    "EmotionalStopwordFilter",
    "ModelTokenizer",
    "ModelEmbedder",
    # Factory functions
    "get_tokenizer",
    "get_all_tokenizers",
    "get_embedder",
    "get_all_embedders",
    # Config
    "EMOTION_MODEL_NAME",
    "STRESS_MODEL_NAME",
    "DEPRESSION_MODEL_NAME",
    "ALL_MODEL_NAMES",
    "MAX_SEQ_LENGTH",
    # Keywords
    "CRISIS_KEYWORDS",
    "DEPRESSION_KEYWORDS",
    "ANXIETY_KEYWORDS",
    "STRESS_KEYWORDS",
    "NEGATION_WORDS",
    "EMOTION_INTENSIFIERS",
]
