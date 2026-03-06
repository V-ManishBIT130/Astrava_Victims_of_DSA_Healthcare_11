"""
Emotion Detection Module
========================
Uses SamLowe/roberta-base-go_emotions for 28-label emotion classification.
Ready-to-use standalone module — import into any project.

Usage:
    from emotion_detector import detect_emotions, get_top_emotion

    emotions = detect_emotions("I'm so happy!")
    top = get_top_emotion("I'm so happy!")
"""

from transformers import pipeline
import threading

_classifier = None
_lock = threading.Lock()

# Per-label optimized thresholds from the model author's evaluation.
# These maximize F1 per label (better than a flat 0.5 or 0.25 threshold).
THRESHOLDS = {
    "admiration": 0.25, "amusement": 0.45, "anger": 0.15,
    "annoyance": 0.10, "approval": 0.30, "caring": 0.40,
    "confusion": 0.55, "curiosity": 0.25, "desire": 0.25,
    "disappointment": 0.40, "disapproval": 0.30, "disgust": 0.20,
    "embarrassment": 0.10, "excitement": 0.35, "fear": 0.40,
    "gratitude": 0.45, "grief": 0.05, "joy": 0.40,
    "love": 0.25, "nervousness": 0.25, "optimism": 0.20,
    "pride": 0.10, "realization": 0.15, "relief": 0.05,
    "remorse": 0.10, "sadness": 0.40, "surprise": 0.15,
    "neutral": 0.25,
}


def get_classifier():
    """Thread-safe lazy-load (double-checked locking)."""
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:  # second check inside lock
                _classifier = pipeline(
                    task="text-classification",
                    model="SamLowe/roberta-base-go_emotions",
                    top_k=None,
                )
    return _classifier


def detect_emotions(text, use_optimized_thresholds=True):
    """
    Detect emotions in text.

    Args:
        text: Input string.
        use_optimized_thresholds: If True, use per-label thresholds.
                                  If False, use flat 0.25 threshold.

    Returns:
        List of dicts: [{"label": str, "score": float}, ...]
        Sorted by score descending. Only includes emotions above threshold.
    """
    clf = get_classifier()
    results = clf(text)[0]

    detected = []
    for r in results:
        if use_optimized_thresholds:
            threshold = THRESHOLDS.get(r["label"], 0.25)
        else:
            threshold = 0.25
        if r["score"] >= threshold:
            detected.append(r)

    return detected


def detect_emotions_batch(texts, use_optimized_thresholds=True):
    """Process multiple texts at once. Returns list of results per text."""
    return [detect_emotions(t, use_optimized_thresholds) for t in texts]


def get_top_emotion(text):
    """Get the single highest-scoring emotion for a text."""
    clf = get_classifier()
    results = clf(text)[0]
    return results[0]  # already sorted by score descending


def get_all_scores(text):
    """Get raw scores for all 28 labels (no thresholding)."""
    clf = get_classifier()
    return clf(text)[0]


# Quick self-test when run directly
if __name__ == "__main__":
    print("Testing emotion_detector module...\n")

    test = "I'm so excited and grateful to be here!"
    print(f'Input: "{test}"')

    print("\nDetected emotions (optimized thresholds):")
    for e in detect_emotions(test):
        print(f"  {e['label']:20s} {e['score']:.4f}")

    print(f"\nTop emotion: {get_top_emotion(test)}")
    print(f"\nAll 28 scores: {len(get_all_scores(test))} labels returned")
    print("\nModule is working correctly!")
