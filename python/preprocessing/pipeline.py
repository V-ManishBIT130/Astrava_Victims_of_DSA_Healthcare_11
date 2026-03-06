"""
pipeline.py — Unified preprocessing pipeline orchestrator for ASTRAVA.

Transforms raw user text into clean, crisis-assessed output ready for ML inference.

Usage:
    pipeline = PreprocessingPipeline()
    result = pipeline.process_text_only("I can't sleep, everything feels hopeless")

    result.cleaned_text   → pass directly to classification models
    result.filtered_text  → pass to RAG / FAISS keyword search
    result.crisis_result  → check BEFORE model inference
    result.was_truncated  → True if input exceeded 4000 chars

NOTE: Tokenization and embedding are intentionally NOT done here.
  Each model module (depression_classifier, emotion_detector, stress_detector)
  handles its own tokenization internally. Doing it here too would run every
  tokenizer and forward pass TWICE, doubling inference time.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .cleaner import TextCleaner
from .crisis_detector import CrisisDetector, CrisisResult
from .stopwords import EmotionalStopwordFilter

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """
    Result of the preprocessing pipeline.

    Fields:
        original_text:  Raw input exactly as provided.
        cleaned_text:   15-step cleaned text — pass this to ML models.
        filtered_text:  Stopword-filtered version — for RAG / FAISS only, NOT for models.
                        Transformers are pre-trained on full sentences; stopword-filtered
                        text is ungrammatical input they weren't trained on.
        crisis_result:  Severity-graded safety check — always read this BEFORE inference.
        was_truncated:  True if input exceeded MAX_RAW_CHARS and was cut off.
    """

    original_text: str = ""
    cleaned_text: str = ""
    filtered_text: str = ""
    crisis_result: Optional[CrisisResult] = None
    was_truncated: bool = False

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "filtered_text": self.filtered_text,
            "crisis_result": self.crisis_result.to_dict() if self.crisis_result else None,
            "was_truncated": self.was_truncated,
        }


class PreprocessingPipeline:
    """
    Preprocessing pipeline for the ASTRAVA chatbot.

    Orchestrates:
        1. TextCleaner — 15-step text normalization (shared across all models)
        2. CrisisDetector — safety check with severity routing (runs before inference)
        3. EmotionalStopwordFilter — removes noise words (output used for RAG, not models)

    This class does NOT load any ML models. Classification models are separate
    and receive cleaned_text directly. This keeps the pipeline fast and stateless.
    """

    # Cap raw input at 4000 chars before cleaning.
    # 4000 chars ≈ ~1000 tokens — well above the 512-token model limit.
    # Prevents the 15-step cleaner wasting time on 10k-char pastes.
    MAX_RAW_CHARS = 4000

    def __init__(self):
        """Initialize the pipeline. No ML models are loaded."""
        self.cleaner = TextCleaner()
        self.crisis_detector = CrisisDetector()
        self.stopword_filter = EmotionalStopwordFilter()

    def process_text_only(self, raw_text: str) -> PreprocessingResult:
        """
        Preprocess text: clean → crisis detect → stopword filter.

        This is the main method to call before running inference.
        No ML models are loaded by this method.

        Args:
            raw_text: Raw user input.

        Returns:
            PreprocessingResult — pass result.cleaned_text to classification models,
            result.filtered_text to RAG/FAISS, result.crisis_result to decide
            whether to short-circuit before inference.
        """
        if not raw_text or not isinstance(raw_text, str):
            return PreprocessingResult(original_text=raw_text or "")

        # Guard: cap at MAX_RAW_CHARS to protect cleaner from huge pastes.
        was_truncated = len(raw_text) > self.MAX_RAW_CHARS
        text = raw_text[:self.MAX_RAW_CHARS] if was_truncated else raw_text

        # Step 1: Normalise the text (emojis → unicode → tags → lowercase → ...)
        cleaned = self.cleaner.clean(text)

        # Step 2: Crisis detection on cleaned text — BEFORE model inference
        crisis_result = self.crisis_detector.detect(cleaned)

        # Step 3: Stopword filtering — for RAG/FAISS keyword search only
        # Do NOT feed filtered_text to transformer models — they expect full sentences.
        filtered = self.stopword_filter.filter_text(cleaned)

        return PreprocessingResult(
            original_text=raw_text,
            cleaned_text=cleaned,
            filtered_text=filtered,
            crisis_result=crisis_result,
            was_truncated=was_truncated,
        )

    def process(self, raw_text: str) -> PreprocessingResult:
        """Alias for process_text_only(). Prefer calling that directly for clarity."""
        return self.process_text_only(raw_text)

    def __repr__(self) -> str:
        return "PreprocessingPipeline(cleaner, crisis_detector, stopword_filter)"
