"""
pipeline.py — Unified preprocessing pipeline orchestrator for ASTRAVA.

Combines all preprocessing steps into a single, easy-to-use pipeline:
    Raw text → Clean → Crisis detect → Stopword filter → Tokenize → Embed

Two modes:
    1. process()          — Full pipeline (text cleaning + crisis detection + tokenization + embedding)
    2. process_text_only() — Lightweight mode (text cleaning + crisis detection only, no model loading)

The pipeline shares text cleaning across all 3 models but produces
model-specific tokens and embeddings for each downstream classifier.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .cleaner import TextCleaner
from .config import ALL_MODEL_NAMES
from .crisis_detector import CrisisDetector, CrisisResult
from .stopwords import EmotionalStopwordFilter

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Complete result of the preprocessing pipeline."""

    # --- Shared across all models ---
    original_text: str = ""
    cleaned_text: str = ""
    filtered_text: str = ""  # After stopword removal
    crisis_result: Optional[CrisisResult] = None

    # --- Model-specific (only populated in full mode) ---
    # Each key is a model key: "emotion", "stress", "depression"
    tokens: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "filtered_text": self.filtered_text,
            "crisis_result": self.crisis_result.to_dict() if self.crisis_result else None,
        }

        # Tokens: convert tensors to lists for JSON serialization
        if self.tokens:
            result["tokens"] = {}
            for model_key, token_dict in self.tokens.items():
                result["tokens"][model_key] = {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in token_dict.items()
                }

        # Embeddings: convert tensors to lists
        if self.embeddings:
            result["embeddings"] = {}
            for model_key, emb in self.embeddings.items():
                result["embeddings"][model_key] = (
                    emb.tolist() if hasattr(emb, "tolist") else emb
                )

        return result


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for the ASTRAVA chatbot.

    Orchestrates:
        1. TextCleaner — normalize raw text (shared)
        2. CrisisDetector — immediate safety check (shared)
        3. EmotionalStopwordFilter — remove irrelevant stopwords (shared)
        4. ModelTokenizer — model-specific tokenization (per-model)
        5. ModelEmbedder — model-specific embedding generation (per-model)

    Usage:
        pipeline = PreprocessingPipeline()

        # Full pipeline (loads models — heavier)
        result = pipeline.process("I can't do this anymore, I want to end it all")

        # Text-only mode (no model loading — lightweight)
        result = pipeline.process_text_only("I feel stressed about exams")
    """

    def __init__(
        self,
        model_keys: Optional[list] = None,
        load_models: bool = False,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            model_keys: List of model keys to load tokenizers/embedders for.
                        Defaults to all 3: ["emotion", "stress", "depression"].
                        Pass a subset to save memory, e.g. ["emotion"].
            load_models: If True, load models immediately. If False (default),
                         models are loaded lazily on first process() call.
        """
        self.model_keys = model_keys or list(ALL_MODEL_NAMES.keys())

        # Validate model keys
        for key in self.model_keys:
            if key not in ALL_MODEL_NAMES:
                raise ValueError(
                    f"Unknown model key '{key}'. "
                    f"Available: {list(ALL_MODEL_NAMES.keys())}"
                )

        # Initialize shared components (lightweight, no model loading)
        self.cleaner = TextCleaner()
        self.crisis_detector = CrisisDetector()
        self.stopword_filter = EmotionalStopwordFilter()

        # Model-specific components (lazy loaded)
        self._tokenizers = {}
        self._embedders = {}

        if load_models:
            self._load_all_models()

    def _load_all_models(self):
        """Load all tokenizers and embedders for configured model keys."""
        from .embedder import ModelEmbedder
        from .tokenizer import ModelTokenizer

        for key in self.model_keys:
            model_name = ALL_MODEL_NAMES[key]
            if key not in self._tokenizers:
                logger.info(f"Loading tokenizer for [{key}]: {model_name}")
                self._tokenizers[key] = ModelTokenizer(model_name)
            if key not in self._embedders:
                logger.info(f"Loading embedder for [{key}]: {model_name}")
                self._embedders[key] = ModelEmbedder(model_name)

    def _ensure_models_loaded(self):
        """Ensure all models are loaded (lazy loading)."""
        if not self._tokenizers or not self._embedders:
            self._load_all_models()

    def process_text_only(self, raw_text: str) -> PreprocessingResult:
        """
        Lightweight preprocessing — text cleaning and crisis detection only.

        Does NOT load any ML models. Use this when you only need cleaned text
        and crisis detection (e.g., for logging, quick safety checks, or
        when models will be called separately).

        Args:
            raw_text: Raw user input text.

        Returns:
            PreprocessingResult with cleaned_text, filtered_text, and crisis_result.
        """
        if not raw_text or not isinstance(raw_text, str):
            return PreprocessingResult(original_text=raw_text or "")

        # Step 1: Clean text (shared pipeline)
        cleaned = self.cleaner.clean(raw_text)

        # Step 2: Crisis detection (on cleaned text)
        crisis_result = self.crisis_detector.detect(cleaned)

        # Step 3: Stopword filtering (for downstream use / display)
        filtered = self.stopword_filter.filter_text(cleaned)

        return PreprocessingResult(
            original_text=raw_text,
            cleaned_text=cleaned,
            filtered_text=filtered,
            crisis_result=crisis_result,
        )

    def process(self, raw_text: str) -> PreprocessingResult:
        """
        Full preprocessing pipeline — clean, detect crisis, tokenize, embed.

        Produces model-specific tokens and embeddings for each configured model.

        Args:
            raw_text: Raw user input text.

        Returns:
            PreprocessingResult with all fields populated:
                - cleaned_text (shared)
                - crisis_result (shared)
                - tokens[model_key] (per-model)
                - embeddings[model_key] (per-model)
        """
        # Start with text-only processing
        result = self.process_text_only(raw_text)

        if not result.cleaned_text:
            return result

        # Ensure models are loaded
        self._ensure_models_loaded()

        # Step 4: Model-specific tokenization
        for key in self.model_keys:
            tokenizer = self._tokenizers[key]
            result.tokens[key] = tokenizer.tokenize(result.cleaned_text)

        # Step 5: Model-specific embedding generation
        for key in self.model_keys:
            embedder = self._embedders[key]
            result.embeddings[key] = embedder.generate_embedding(result.cleaned_text)

        return result

    def process_for_model(self, raw_text: str, model_key: str) -> PreprocessingResult:
        """
        Process text for a SINGLE specific model.

        Useful when you want to run inference on just one model at a time.

        Args:
            raw_text: Raw user input text.
            model_key: One of "emotion", "stress", "depression".

        Returns:
            PreprocessingResult with tokens and embedding for the specified model only.
        """
        if model_key not in ALL_MODEL_NAMES:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(ALL_MODEL_NAMES.keys())}"
            )

        # Text-only first
        result = self.process_text_only(raw_text)

        if not result.cleaned_text:
            return result

        # Load only the needed tokenizer/embedder
        from .embedder import ModelEmbedder
        from .tokenizer import ModelTokenizer

        model_name = ALL_MODEL_NAMES[model_key]

        if model_key not in self._tokenizers:
            self._tokenizers[model_key] = ModelTokenizer(model_name)
        if model_key not in self._embedders:
            self._embedders[model_key] = ModelEmbedder(model_name)

        result.tokens[model_key] = self._tokenizers[model_key].tokenize(result.cleaned_text)
        result.embeddings[model_key] = self._embedders[model_key].generate_embedding(result.cleaned_text)

        return result

    def __repr__(self) -> str:
        return (
            f"PreprocessingPipeline(models={self.model_keys}, "
            f"tokenizers_loaded={len(self._tokenizers)}, "
            f"embedders_loaded={len(self._embedders)})"
        )
