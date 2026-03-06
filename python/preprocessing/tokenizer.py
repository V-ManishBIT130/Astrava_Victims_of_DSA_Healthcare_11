"""
tokenizer.py — Model-aware tokenizer wrapper for the ASTRAVA preprocessing module.

Wraps HuggingFace's AutoTokenizer to provide consistent tokenization across
the three different model architectures:
  - mental/mental-roberta-base  (RoBERTa BPE tokenizer)
  - jnyx74/stress-prediction    (DistilBERT WordPiece tokenizer)
  - poudel/Depression_and_Non-Depression_Classifier (BERT WordPiece tokenizer)

Each model requires its OWN tokenizer since the vocabulary and tokenization
algorithm differ. The text cleaning pipeline is shared; tokenization is not.
"""

import logging
from typing import Dict, Optional

from .config import (
    ALL_MODEL_NAMES,
    MAX_SEQ_LENGTH,
    RETURN_TENSORS,
)

logger = logging.getLogger(__name__)


class ModelTokenizer:
    """
    Model-aware tokenizer using HuggingFace AutoTokenizer.

    Lazily loads the tokenizer on first use to avoid startup overhead.
    Each instance wraps a SINGLE model's tokenizer.

    Usage:
        tokenizer = ModelTokenizer("mental/mental-roberta-base")
        result = tokenizer.tokenize("I feel really sad today")
        # result = {"input_ids": tensor, "attention_mask": tensor}
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = MAX_SEQ_LENGTH,
        return_tensors: str = RETURN_TENSORS,
    ):
        """
        Initialize the ModelTokenizer.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum sequence length (padding/truncation target).
            return_tensors: Tensor format ("pt" for PyTorch).
        """
        self.model_name = model_name
        self.max_length = max_length
        self.return_tensors = return_tensors
        self._tokenizer = None  # Lazy loaded

    def _load_tokenizer(self):
        """Lazily load the tokenizer from HuggingFace."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logger.info(f"Loading tokenizer for: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded: {self._tokenizer.__class__.__name__}")

    @property
    def tokenizer(self):
        """Access the underlying HuggingFace tokenizer (lazy-loaded)."""
        self._load_tokenizer()
        return self._tokenizer

    def tokenize(self, text: str) -> dict:
        """
        Tokenize text for the specific model.

        Args:
            text: Cleaned text string.

        Returns:
            Dictionary with keys:
                - input_ids: Token ID tensor [1, seq_len]
                - attention_mask: Attention mask tensor [1, seq_len]
                (and token_type_ids for BERT-based models)
        """
        self._load_tokenizer()

        encoded = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        return dict(encoded)

    def tokenize_batch(self, texts: list) -> dict:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of cleaned text strings.

        Returns:
            Dictionary with batched tensors [batch_size, seq_len].
        """
        self._load_tokenizer()

        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        return dict(encoded)

    def decode(self, token_ids) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Tensor or list of token IDs.

        Returns:
            Decoded text string.
        """
        self._load_tokenizer()
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        self._load_tokenizer()
        return len(self._tokenizer)

    def __repr__(self) -> str:
        return f"ModelTokenizer(model='{self.model_name}', max_length={self.max_length})"


# =============================================================================
# Factory functions
# =============================================================================

def get_tokenizer(model_key: str) -> ModelTokenizer:
    """
    Get a tokenizer by model key.

    Args:
        model_key: One of "emotion", "stress", "depression".

    Returns:
        ModelTokenizer for the specified model.
    """
    if model_key not in ALL_MODEL_NAMES:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available keys: {list(ALL_MODEL_NAMES.keys())}"
        )
    return ModelTokenizer(ALL_MODEL_NAMES[model_key])


def get_all_tokenizers() -> Dict[str, ModelTokenizer]:
    """
    Get tokenizers for ALL three models.

    Returns:
        Dict mapping model key → ModelTokenizer:
            {"emotion": ..., "stress": ..., "depression": ...}
    """
    return {key: ModelTokenizer(name) for key, name in ALL_MODEL_NAMES.items()}
