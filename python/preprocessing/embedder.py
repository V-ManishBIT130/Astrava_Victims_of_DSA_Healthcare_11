"""
embedder.py — Model-aware embedding generator for the ASTRAVA preprocessing module.

Generates dense vector embeddings (768-dim CLS token output) using the
base encoder of each model. These embeddings are used for:
  - Downstream classification heads (emotion, stress, depression)
  - RAG query embedding for FAISS retrieval

Each model produces embeddings from its own encoder:
  - mental/mental-roberta-base  → RoBERTa encoder
  - jnyx74/stress-prediction    → DistilBERT encoder
  - poudel/Depression_and_Non-Depression_Classifier → BERT encoder
"""

import logging
from typing import Dict, Optional

from .config import (
    ALL_MODEL_NAMES,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    RETURN_TENSORS,
)

logger = logging.getLogger(__name__)


class ModelEmbedder:
    """
    Model-aware embedding generator using HuggingFace AutoModel.

    Lazily loads both the tokenizer and model on first use to avoid
    startup overhead. Generates CLS token embeddings (768-dim).

    Usage:
        embedder = ModelEmbedder("mental/mental-roberta-base")
        embedding = embedder.generate_embedding("I feel really sad today")
        # embedding.shape → torch.Size([768])
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = MAX_SEQ_LENGTH,
        return_tensors: str = RETURN_TENSORS,
    ):
        """
        Initialize the ModelEmbedder.

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum sequence length for tokenization.
            return_tensors: Tensor format ("pt" for PyTorch).
        """
        self.model_name = model_name
        self.max_length = max_length
        self.return_tensors = return_tensors
        self._tokenizer = None  # Lazy loaded
        self._model = None      # Lazy loaded

    def _load(self):
        """Lazily load the model and tokenizer from HuggingFace."""
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading model and tokenizer for embedding: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

            # Set to eval mode (inference only)
            self._model.eval()

            logger.info(
                f"Embedder loaded: {self._model.__class__.__name__} "
                f"(hidden_size={self._model.config.hidden_size})"
            )

    @property
    def model(self):
        """Access the underlying HuggingFace model (lazy-loaded)."""
        self._load()
        return self._model

    @property
    def tokenizer(self):
        """Access the underlying HuggingFace tokenizer (lazy-loaded)."""
        self._load()
        return self._tokenizer

    def generate_embedding(self, text: str):
        """
        Generate a dense embedding vector for a single text.

        Uses the CLS token output from the last hidden layer as the
        sentence-level representation.

        Args:
            text: Cleaned text string.

        Returns:
            torch.Tensor of shape [768] — the CLS token embedding.
        """
        import torch

        self._load()

        # Tokenize
        encoded = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        # Run inference with no gradient computation
        with torch.no_grad():
            outputs = self._model(**encoded)

        # Extract CLS token embedding (first token of last hidden state)
        # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Remove batch dimension → [hidden_size]
        return cls_embedding.squeeze(0)

    def generate_embedding_batch(self, texts: list):
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of cleaned text strings.

        Returns:
            torch.Tensor of shape [batch_size, 768].
        """
        import torch

        self._load()

        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        with torch.no_grad():
            outputs = self._model(**encoded)

        # CLS token embeddings for the full batch
        return outputs.last_hidden_state[:, 0, :]

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        self._load()
        return self._model.config.hidden_size

    def __repr__(self) -> str:
        return f"ModelEmbedder(model='{self.model_name}')"


# =============================================================================
# Factory functions
# =============================================================================

def get_embedder(model_key: str) -> ModelEmbedder:
    """
    Get an embedder by model key.

    Args:
        model_key: One of "emotion", "stress", "depression".

    Returns:
        ModelEmbedder for the specified model.
    """
    if model_key not in ALL_MODEL_NAMES:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available keys: {list(ALL_MODEL_NAMES.keys())}"
        )
    return ModelEmbedder(ALL_MODEL_NAMES[model_key])


def get_all_embedders() -> Dict[str, ModelEmbedder]:
    """
    Get embedders for ALL three models.

    Returns:
        Dict mapping model key → ModelEmbedder:
            {"emotion": ..., "stress": ..., "depression": ...}
    """
    return {key: ModelEmbedder(name) for key, name in ALL_MODEL_NAMES.items()}
