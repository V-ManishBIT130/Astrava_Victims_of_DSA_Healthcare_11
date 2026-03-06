"""
Stress Detection Module
========================
Uses jnyx74/stress-prediction (DistilBERT fine-tuned on Dreaddit dataset)
for binary stress classification in social media text.

Usage:
    from stress_detector import detect_stress, get_stress_level

    result = detect_stress("I'm so overwhelmed with deadlines!")
    # {'is_stressed': True, 'confidence': 0.9859, 'label': 'LABEL_1'}

    level = get_stress_level("Having a great day!")
    # 'low'
"""

from transformers import pipeline
import threading

_classifier = None
_lock = threading.Lock()

MODEL_NAME = "jnyx74/stress-prediction"

# Label mapping
LABELS = {
    "LABEL_0": "No Stress",
    "LABEL_1": "Stressed"
}


def get_classifier():
    """Thread-safe lazy-load (double-checked locking)."""
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:  # second check inside lock
                _classifier = pipeline(
                    task="text-classification",
                    model=MODEL_NAME
                )
    return _classifier


def detect_stress(text: str, truncation: bool = True, max_length: int = 512) -> dict:
    """
    Detect stress level in text.

    Args:
        text: Input text string (social media post, message, etc.)
        truncation: Whether to truncate text longer than max_length
        max_length: Maximum token length (default: 512)

    Returns:
        dict with keys:
            - is_stressed: bool (True if stressed detected)
            - confidence: float (0.0 to 1.0)
            - label: str (raw model label, e.g., 'LABEL_1')
            - readable_label: str (human-readable, e.g., 'Stressed')
    """
    clf = get_classifier()
    result = clf(text, truncation=truncation, max_length=max_length)[0]
    
    label = result['label']
    score = result['score']
    
    return {
        "is_stressed": label == "LABEL_1",
        "confidence": score,
        "label": label,
        "readable_label": LABELS.get(label, label)
    }


def detect_stress_batch(texts: list, truncation: bool = True, max_length: int = 512) -> list:
    """
    Process multiple texts at once.

    Args:
        texts: List of text strings
        truncation: Whether to truncate text longer than max_length
        max_length: Maximum token length

    Returns:
        List of dicts (one per input text)
    """
    return [detect_stress(text, truncation, max_length) for text in texts]


def get_stress_level(text: str, high_threshold: float = 0.85) -> str:
    """
    Categorize stress into levels based on confidence.

    Args:
        text: Input text
        high_threshold: Confidence threshold for "high" stress (default: 0.85)

    Returns:
        str: 'high', 'moderate', or 'low'
    """
    result = detect_stress(text)
    
    if not result['is_stressed']:
        return 'low'
    elif result['confidence'] >= high_threshold:
        return 'high'
    else:
        return 'moderate'


def should_trigger_alert(text: str, threshold: float = 0.85) -> bool:
    """
    Determine if stress level warrants an alert/intervention.

    Args:
        text: Input text
        threshold: Minimum confidence to trigger alert (default: 0.85)

    Returns:
        bool: True if high-confidence stress detected
    """
    result = detect_stress(text)
    return result['is_stressed'] and result['confidence'] >= threshold
