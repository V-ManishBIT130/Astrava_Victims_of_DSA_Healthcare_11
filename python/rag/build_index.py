"""
build_index.py — Build a FAISS vector index from the MentalChat16K dataset.

Embeds the *questions* (patient messages) so that incoming user messages
can be matched against similar patient scenarios.  The corresponding
counselor *answers* become the grounding context for the LLM.

Generates:
  python/rag/indices/faiss_index.bin   — FAISS inner-product index
  python/rag/indices/metadata.json     — parallel array of {question, answer}
  python/rag/indices/config.json       — embedding model info

Usage:
  cd <project_root>
  .venv\\Scripts\\python python/rag/build_index.py
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "all-MiniLM-L6-v2"       # 384-dim, fast, good quality
BATCH_SIZE  = 256
INDEX_DIR   = Path(__file__).parent / "indices"

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("[RAG BUILD] Loading MentalChat16K dataset from HuggingFace...")
    ds = load_dataset("ShenLab/MentalChat16K")

    # Dataset has a single "train" split with "input" and "output" fields
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    print(f"[RAG BUILD] Dataset loaded — {len(split)} rows")

    # Detect field names
    sample = split[0]
    if "input" in sample:
        q_field, a_field = "input", "output"
    elif "question" in sample:
        q_field, a_field = "question", "answer"
    else:
        print(f"[RAG BUILD] Unknown fields: {list(sample.keys())}")
        sys.exit(1)

    questions = split[q_field]
    answers   = split[a_field]

    # Filter out empty/None entries
    pairs = [
        (q.strip(), a.strip())
        for q, a in zip(questions, answers)
        if q and a and q.strip() and a.strip()
    ]
    print(f"[RAG BUILD] {len(pairs)} valid Q&A pairs after filtering")

    questions_clean = [p[0] for p in pairs]
    answers_clean   = [p[1] for p in pairs]

    # ── Embed questions ──────────────────────────────────────────────────────
    print(f"[RAG BUILD] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[RAG BUILD] Encoding {len(questions_clean)} questions (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        questions_clean,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    print(f"[RAG BUILD] Embedding shape: {embeddings.shape}  (dim={dim})")

    # ── Build FAISS index ────────────────────────────────────────────────────
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normed = cosine
    index.add(embeddings)
    print(f"[RAG BUILD] FAISS index built — {index.ntotal} vectors")

    # ── Save ─────────────────────────────────────────────────────────────────
    index_path = INDEX_DIR / "faiss_index.bin"
    meta_path  = INDEX_DIR / "metadata.json"
    conf_path  = INDEX_DIR / "config.json"

    faiss.write_index(index, str(index_path))
    print(f"[RAG BUILD] Index saved → {index_path}")

    metadata = [{"question": q, "answer": a} for q, a in zip(questions_clean, answers_clean)]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"[RAG BUILD] Metadata saved → {meta_path}  ({len(metadata)} entries)")

    config = {
        "model_name": MODEL_NAME,
        "dimension": dim,
        "total_vectors": len(metadata),
        "embed_field": "question",
        "dataset": "ShenLab/MentalChat16K",
        "normalize": True,
        "metric": "cosine (inner product on L2-normalised vectors)",
    }
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[RAG BUILD] Config saved → {conf_path}")

    print("\n[RAG BUILD] Done! Index is ready for retrieval.")


if __name__ == "__main__":
    main()
