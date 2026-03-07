"""
retriever.py — FAISS-based retrieval for MentalChat16K therapeutic responses.

Loads the pre-built FAISS index and metadata, embeds incoming queries
with the same sentence-transformer model, and returns the top-K most
similar patient scenarios with their counselor answers.

Usage (standalone test):
  .venv\\Scripts\\python python/rag/retriever.py "I feel so hopeless"

Integration:
  from rag.retriever import MentalHealthRetriever
  retriever = MentalHealthRetriever()
  results = retriever.retrieve("I can't stop crying", top_k=3)
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "indices"


class MentalHealthRetriever:
    """Retrieve relevant therapeutic Q&A pairs from MentalChat16K via FAISS."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        config_path = index_dir / "config.json"
        index_path  = index_dir / "faiss_index.bin"
        meta_path   = index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `python python/rag/build_index.py` first."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = SentenceTransformer(self.config["model_name"])
        print(f"[RAG] Loaded FAISS index: {self.index.ntotal} vectors, "
              f"dim={self.config['dimension']}, model={self.config['model_name']}")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Embed the query and return the top-K most similar Q&A pairs.

        Returns list of dicts:
          [{"question": str, "answer": str, "score": float}, ...]
        Scores are cosine similarity (0–1, higher = more similar).
        """
        query_vec = self.model.encode(
            [query], normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            results.append({
                "question": entry["question"],
                "answer":   entry["answer"],
                "score":    round(float(score), 4),
            })
        return results

    def format_for_llm(self, results: list[dict], max_answer_len: int = 400) -> str:
        """
        Format retrieved results as a compact text block for LLM injection.
        Truncates long answers to save context window tokens.
        """
        if not results:
            return ""

        lines = []
        for i, r in enumerate(results, 1):
            answer_trunc = r["answer"][:max_answer_len]
            if len(r["answer"]) > max_answer_len:
                answer_trunc += "..."
            lines.append(
                f"--- Reference {i} (similarity: {r['score']:.2f}) ---\n"
                f"Patient situation: {r['question'][:200]}\n"
                f"Therapeutic approach: {answer_trunc}"
            )
        return "\n\n".join(lines)


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "I feel so hopeless and alone"
    print(f"\nQuery: {query}\n")

    retriever = MentalHealthRetriever()
    results = retriever.retrieve(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Result {i}  (score: {r['score']:.4f})")
        print(f"Q: {r['question'][:150]}")
        print(f"A: {r['answer'][:300]}")

    print(f"\n{'='*60}")
    print("\nFormatted for LLM:")
    print(retriever.format_for_llm(results))
