"""
run_inference.py — Unified inference runner for ASTRAVA.

Flow:
    1. Raw text → PreprocessingPipeline.process_text_only()
       - 15-step text cleaning
       - Crisis detection (CRITICAL/HIGH → immediate short-circuit)
    2. Cleaned text → 3 models run SEQUENTIALLY (see reason below)
       - Depression Classifier  (BERT)
       - Stress Detector        (DistilBERT)
       - Emotion Detector       (RoBERTa, 28 labels)
    3. Print combined results

WHY SEQUENTIAL (not parallel):
    PyTorch uses OpenMP thread pools internally for each forward pass.
    Running 3 models simultaneously in threads causes those thread pools to
    compete for the same CPUs, making total inference SLOWER.
    Python's GIL also blocks true CPU-bound parallelism in threads.
    Sequential = each model gets 100% of CPU resources for its pass.
    All 3 models are loaded at startup once — no re-loading overhead.

Usage:
    python python/run_inference.py
    python python/run_inference.py --text "I feel overwhelmed and hopeless"
"""

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PYTHON_DIR = Path(__file__).parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "depression classifier model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "go_emotion model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "Stress detection model"))

# ── Imports ───────────────────────────────────────────────────────────────────
from preprocessing.pipeline import PreprocessingPipeline
from depression_classifier import DepressionClassifier
from emotion_detector import detect_emotions, get_all_scores
from stress_detector import detect_stress, get_stress_level

# ── Severity colours (ANSI) ───────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Crisis severities that require immediate response before model inference
CRITICAL_SEVERITIES = {"CRITICAL", "HIGH"}


class AstravaInference:
    """
    Loads all three models once at startup and runs them sequentially
    on preprocessed input text.
    """

    def __init__(self):
        print(f"{CYAN}{BOLD}[ASTRAVA] Loading pipeline...{RESET}")

        self.pipeline = PreprocessingPipeline()

        print(f"{CYAN}[ASTRAVA] Loading Depression Classifier (BERT)...{RESET}")
        t = time.time()
        self.depression_clf = DepressionClassifier()
        print(f"          Loaded in {time.time() - t:.1f}s")

        print(f"{CYAN}[ASTRAVA] Loading Stress Detector (DistilBERT)...{RESET}")
        t = time.time()
        # Trigger lazy load by running a warm-up pass
        from stress_detector import get_classifier as _load_stress
        _load_stress()
        print(f"          Loaded in {time.time() - t:.1f}s")

        print(f"{CYAN}[ASTRAVA] Loading Emotion Detector (RoBERTa)...{RESET}")
        t = time.time()
        from emotion_detector import get_classifier as _load_emotion
        _load_emotion()
        print(f"          Loaded in {time.time() - t:.1f}s")

        print(f"{GREEN}{BOLD}[ASTRAVA] All models ready.\n{RESET}")

    def run(self, raw_text: str) -> dict:
        """
        Full pipeline: preprocess → crisis check → run 3 models → return results.

        Args:
            raw_text: Raw user input

        Returns:
            dict with keys: preprocessing, crisis, depression, stress, emotions, total_ms
        """
        total_start = time.time()

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        t = time.time()
        prep_result = self.pipeline.process_text_only(raw_text)
        prep_ms = (time.time() - t) * 1000

        crisis = prep_result.crisis_result
        cleaned = prep_result.cleaned_text

        results = {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "filtered_text": prep_result.filtered_text,
            "crisis": crisis.to_dict() if crisis else None,
            "preprocessing_ms": round(prep_ms, 1),
            "depression": None,
            "stress": None,
            "emotions": None,
        }

        # ── Step 2: Crisis short-circuit ──────────────────────────────────────
        # If CRITICAL or HIGH severity, return immediately.
        # Do not wait for model inference — user needs help NOW.
        if crisis and crisis.severity in CRITICAL_SEVERITIES:
            results["total_ms"] = round((time.time() - total_start) * 1000, 1)
            results["short_circuited"] = True
            return results

        results["short_circuited"] = False

        if not cleaned:
            results["total_ms"] = round((time.time() - total_start) * 1000, 1)
            return results

        # ── Step 3: Run models sequentially on cleaned text ───────────────────

        # 3a. Depression classifier (BERT)
        t = time.time()
        dep_result = self.depression_clf.predict(cleaned)
        dep_result["inference_ms"] = round((time.time() - t) * 1000, 1)
        results["depression"] = dep_result

        # 3b. Stress detector (DistilBERT)
        t = time.time()
        stress_result = detect_stress(cleaned)
        stress_result["stress_level"] = get_stress_level(cleaned)
        stress_result["inference_ms"] = round((time.time() - t) * 1000, 1)
        results["stress"] = stress_result

        # 3c. Emotion detector (RoBERTa, 28 labels) — top 5 emotions
        t = time.time()
        all_scores = get_all_scores(cleaned)
        top5 = sorted(all_scores, key=lambda x: x["score"], reverse=True)[:5]
        active_emotions = detect_emotions(cleaned)  # above per-label threshold
        results["emotions"] = {
            "top_5": top5,
            "active_emotions": active_emotions,  # emotions that crossed their threshold
            "inference_ms": round((time.time() - t) * 1000, 1),
        }

        results["total_ms"] = round((time.time() - total_start) * 1000, 1)
        return results


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_results(results: dict):
    sep = "-" * 60

    print(f"\n{BOLD}{sep}")
    print(f"  ASTRAVA INFERENCE RESULTS")
    print(f"{sep}{RESET}")

    print(f"\n{BOLD}INPUT{RESET}")
    print(f"  Raw:     {results['raw_text']}")
    print(f"  Cleaned: {results['cleaned_text']}")
    print(f"  Prep time: {results['preprocessing_ms']}ms")

    # Crisis
    crisis = results.get("crisis")
    if crisis:
        sev = crisis["severity"]
        colour = RED if sev in CRITICAL_SEVERITIES else (YELLOW if sev in {"ELEVATED", "MEDIUM"} else GREEN)
        print(f"\n{BOLD}CRISIS DETECTION{RESET}")
        print(f"  Severity:     {colour}{BOLD}{sev}{RESET}")
        print(f"  Is Crisis:    {crisis['is_crisis']}")
        print(f"  Needs Review: {crisis['needs_review']}")
        if crisis["matched_pattern_categories"]:
            print(f"  Patterns:     {', '.join(crisis['matched_pattern_categories'])}")
        if crisis["matched_crisis_keywords"]:
            print(f"  Keywords:     {', '.join(crisis['matched_crisis_keywords'][:5])}")
        psy = crisis.get("psycholinguistic", {})
        if psy:
            print(f"  I-ratio:      {psy.get('i_ratio', 0):.3f}  |  Abs-ratio: {psy.get('absolute_ratio', 0):.3f}  |  Negations: {psy.get('negation_count', 0)}")

    if results.get("short_circuited"):
        print(f"\n{RED}{BOLD}  [!] SHORT-CIRCUITED -- Crisis response triggered before model inference.{RESET}")
        print(f"  Total time: {results['total_ms']}ms\n")
        return

    # Depression
    dep = results.get("depression")
    if dep:
        colour = RED if dep["label"] == "Depression" else GREEN
        print(f"\n{BOLD}DEPRESSION CLASSIFIER  {RESET}({dep['inference_ms']}ms)")
        print(f"  Label:      {colour}{BOLD}{dep['label']}{RESET}")
        print(f"  Confidence: {dep['confidence'] * 100:.2f}%")
        probs = dep.get("probabilities", {})
        if probs:
            print(f"  Depression: {probs.get('depression', 0) * 100:.2f}%  |  Non-Depression: {probs.get('non_depression', 0) * 100:.2f}%")

    # Stress
    stress = results.get("stress")
    if stress:
        colour = RED if stress["is_stressed"] else GREEN
        level_colour = RED if stress["stress_level"] == "high" else (YELLOW if stress["stress_level"] == "moderate" else GREEN)
        print(f"\n{BOLD}STRESS DETECTOR  {RESET}({stress['inference_ms']}ms)")
        print(f"  Label:      {colour}{BOLD}{stress['readable_label']}{RESET}")
        print(f"  Level:      {level_colour}{BOLD}{stress['stress_level'].upper()}{RESET}")
        print(f"  Confidence: {stress['confidence'] * 100:.2f}%")

    # Emotions
    emo = results.get("emotions")
    if emo:
        print(f"\n{BOLD}EMOTION DETECTOR (28 labels)  {RESET}({emo['inference_ms']}ms)")
        print(f"  Top 5 scores:")
        for e in emo["top_5"]:
            bar = "█" * int(e["score"] * 20)
            print(f"    {e['label']:<16} {e['score']:.4f}  {bar}")
        active = emo.get("active_emotions", [])
        if active:
            active_names = [f"{e['label']} ({e['score']:.2f})" for e in active]
            print(f"  Active (above threshold): {', '.join(active_names)}")
        else:
            print(f"  Active (above threshold): none")

    print(f"\n{BOLD}  Total time: {results['total_ms']}ms{RESET}\n")
    print(sep)


# ── Interactive / CLI mode ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASTRAVA unified inference runner")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze (skip interactive mode)")
    args = parser.parse_args()

    engine = AstravaInference()

    if args.text:
        results = engine.run(args.text)
        print_results(results)
        return

    # Interactive loop
    print(f"{BOLD}Interactive mode — type text and press Enter. Type 'quit' to exit.{RESET}\n")
    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            break

        results = engine.run(raw)
        print_results(results)


if __name__ == "__main__":
    main()
