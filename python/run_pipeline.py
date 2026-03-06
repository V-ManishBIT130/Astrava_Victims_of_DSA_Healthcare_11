"""
run_pipeline.py — Interactive runner for the ASTRAVA preprocessing pipeline.

Takes raw user input text and runs it through the full preprocessing pipeline:

    Raw text
        ↓ Step 1 — TextCleaner        (normalize, expand contractions, remove noise)
        ↓ Step 2 — CrisisDetector     (scan for crisis / mental health keywords)
        ↓ Step 3 — StopwordFilter     (remove irrelevant stopwords, keep emotional words)
        ↓
    Preprocessed text  ← final usable output

Runs in lightweight mode (no ML model loading) by default.
Pass --full to also tokenize and generate embeddings for all 3 models.

Usage:
    python run_pipeline.py
    python run_pipeline.py --full
"""

import argparse
import sys
import os

# Make sure the python/ directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import PreprocessingPipeline


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

DIVIDER = "─" * 60


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        ASTRAVA — Preprocessing Pipeline Runner           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_result(result, full_mode: bool = False):
    """Pretty-print the PreprocessingResult to the terminal."""

    print(DIVIDER)
    print("📥  ORIGINAL TEXT")
    print(DIVIDER)
    print(result.original_text)
    print()

    print(DIVIDER)
    print("🧹  STEP 1 — Cleaned Text  (TextCleaner)")
    print(DIVIDER)
    print(result.cleaned_text or "(empty after cleaning)")
    print()

    print(DIVIDER)
    print("🔍  STEP 2 — Crisis Detection  (CrisisDetector)")
    print(DIVIDER)
    cr = result.crisis_result
    if cr:
        crisis_icon = "CRISIS" if cr.is_crisis else "SAFE"
        review_icon = " | NEEDS REVIEW" if cr.needs_review else ""
        print(f"  Status           : [{crisis_icon}{review_icon}]")
        print(f"  Severity         : {cr.severity}")
        print(f"  Has Intensifiers : {cr.has_intensifiers}")
        print(f"  Has Rumination   : {cr.has_rumination}")
        print(f"  Has Dissociation : {cr.has_dissociation}")
        if cr.matched_pattern_categories:
            print(f"  Pattern Cats     : {', '.join(cr.matched_pattern_categories)}")
        if cr.matched_crisis_keywords:
            print(f"  Crisis Keywords  : {', '.join(cr.matched_crisis_keywords)}")
        if cr.matched_depression_keywords:
            print(f"  Depression Words : {', '.join(cr.matched_depression_keywords)}")
        if cr.matched_anxiety_keywords:
            print(f"  Anxiety Words    : {', '.join(cr.matched_anxiety_keywords)}")
        if cr.matched_stress_keywords:
            print(f"  Stress Words     : {', '.join(cr.matched_stress_keywords)}")
        if cr.psycholinguistic:
            p = cr.psycholinguistic
            print(f"  Psycholinguistic :")
            print(f"    i_ratio          = {p.get('i_ratio', 0):.4f}   (Pennebaker depression marker)")
            print(f"    absolute_ratio   = {p.get('absolute_ratio', 0):.4f}   (black-and-white thinking)")
            print(f"    negation_count   = {p.get('negation_count', 0)}")
            print(f"    intensifier_count= {p.get('intensifier_count', 0)}")
        if cr.confidence_note:
            print(f"  Note             : {cr.confidence_note}")
    else:
        print("  (no crisis result)")
    print()

    print(DIVIDER)
    print("🗑️   STEP 3 — Filtered Text  (EmotionalStopwordFilter)")
    print(DIVIDER)
    print(result.filtered_text or "(empty after filtering)")
    print()

    if full_mode:
        if result.tokens:
            print(DIVIDER)
            print("🔢  STEP 4 — Tokenization  (per model)")
            print(DIVIDER)
            for model_key, token_dict in result.tokens.items():
                ids = token_dict.get("input_ids", [])
                count = len(ids[0]) if hasattr(ids, "__len__") and len(ids) > 0 else "?"
                print(f"  [{model_key:>10}]  {count} tokens")
            print()

        if result.embeddings:
            print(DIVIDER)
            print("🧠  STEP 5 — Embeddings  (per model)")
            print(DIVIDER)
            for model_key, emb in result.embeddings.items():
                shape = emb.shape if hasattr(emb, "shape") else f"len={len(emb)}"
                print(f"  [{model_key:>10}]  shape = {shape}")
            print()

    print(DIVIDER)
    print("✅  FINAL PREPROCESSED TEXT  (ready for ML models)")
    print(DIVIDER)
    print(result.filtered_text or result.cleaned_text)
    print(DIVIDER)
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASTRAVA Preprocessing Pipeline — interactive runner"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline including tokenization and embedding generation (loads ML models)",
    )
    args = parser.parse_args()

    print_header()

    mode_label = "FULL (tokenize + embed)" if args.full else "TEXT-ONLY (clean + crisis detect)"
    print(f"  Mode : {mode_label}")
    print(f"  Type 'quit' or 'exit' to stop.\n")

    # Initialise pipeline once — lazy model loading
    pipeline = PreprocessingPipeline()

    while True:
        try:
            raw_text = input("Enter text ❯  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not raw_text:
            print("  ⚠  Please enter some text.\n")
            continue

        if raw_text.lower() in {"quit", "exit"}:
            print("\nGoodbye!")
            break

        print()

        try:
            if args.full:
                result = pipeline.process(raw_text)
            else:
                result = pipeline.process_text_only(raw_text)

            print_result(result, full_mode=args.full)

        except Exception as exc:
            print(f"  ❌  Pipeline error: {exc}\n")


if __name__ == "__main__":
    main()
