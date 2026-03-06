# GoEmotions Model — Complete Setup & Integration Guide

> **Model:** [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)  
> **Base Architecture:** RoBERTa-base (125M parameters)  
> **Task:** Multi-label emotion classification (28 labels)  
> **Dataset:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (58k Reddit comments)

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Model Details](#model-details)
4. [Input / Output Specification](#input--output-specification)
5. [Usage Examples](#usage-examples)
6. [All 28 Emotion Labels](#all-28-emotion-labels)
7. [Threshold Tuning](#threshold-tuning)
8. [Integration Patterns](#integration-patterns)
9. [Performance & Limitations](#performance--limitations)
10. [ONNX (Lightweight / Faster Inference)](#onnx-lightweight--faster-inference)
11. [Pickle (.pkl) File — Can I Use It?](#pickle-pkl-file--can-i-use-it)

---

## Overview

This is a **pre-trained, ready-to-use** model. No fine-tuning or training is needed. You pass in text and it returns probability scores for 28 emotion labels.

- **Multi-label:** A single text can trigger multiple emotions simultaneously (e.g., "joy" + "gratitude").
- **Based on Reddit data:** Trained on the Google GoEmotions dataset of ~58k Reddit comments.
- **Architecture:** `roberta-base` fine-tuned with `AutoModelForSequenceClassification` using `problem_type="multi_label_classification"`.

---

## Environment Setup

### 1. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install transformers torch
```

> **Minimal install (CPU-only, smaller download):**
> ```bash
> pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. First Run — Model Download

On the first run, the model (~500MB) is automatically downloaded from HuggingFace and cached at:

| OS | Cache Path |
|---|---|
| Windows | `C:\Users\<you>\.cache\huggingface\hub\` |
| Linux/macOS | `~/.cache/huggingface/hub/` |

Subsequent runs load instantly from cache.

---

## Model Details

| Property | Value |
|---|---|
| **Model ID** | `SamLowe/roberta-base-go_emotions` |
| **Base model** | `roberta-base` |
| **Parameters** | ~125 million |
| **Max input length** | 512 tokens (~300-400 words) |
| **Output dimensions** | 28 (one per emotion label) |
| **Output type** | Independent sigmoid probabilities (0.0 – 1.0) |
| **Classification type** | Multi-label (NOT mutually exclusive) |
| **Training** | 3 epochs, lr=2e-5, weight_decay=0.01 |
| **Framework** | PyTorch / HuggingFace Transformers |

---

## Input / Output Specification

### Input

| Field | Type | Description |
|---|---|---|
| **text** | `str` or `List[str]` | Raw text string(s). Max 512 tokens. |

- Plain text, no preprocessing needed — the tokenizer handles everything.
- Supports English text (trained on English Reddit data).
- Texts longer than 512 tokens are **truncated** (not split).

### Output

The model returns a **list of 28 dictionaries**, one per emotion label, each containing:

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Emotion name (e.g., `"joy"`, `"anger"`) |
| `score` | `float` | Probability between 0.0 and 1.0 |

**Example output structure:**
```python
[
    {"label": "joy",          "score": 0.7616},
    {"label": "gratitude",    "score": 0.2641},
    {"label": "admiration",   "score": 0.2282},
    {"label": "approval",     "score": 0.0323},
    # ... 24 more labels with decreasing scores
]
```

> **Key:** Scores are **independent sigmoid probabilities**, not softmax. They do NOT sum to 1.0. Multiple emotions can be "active" (above threshold) simultaneously.

---

## Usage Examples

### Basic Usage (Pipeline API)

```python
from transformers import pipeline

# Load model (downloads on first run, cached after)
classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,  # return ALL 28 labels
)

# Single text
results = classifier("I am so happy today!")[0]
# results = [{"label": "joy", "score": 0.95}, {"label": "optimism", "score": 0.42}, ...]

# Batch of texts
texts = ["I love this!", "I'm scared.", "This is boring."]
batch_results = classifier(texts)
# batch_results[0] = results for "I love this!"
# batch_results[1] = results for "I'm scared."
# ...
```

### Get Only Top Emotions (With Threshold)

```python
THRESHOLD = 0.25

text = "Thank you so much, this is amazing!"
results = classifier(text)[0]

detected = [r for r in results if r["score"] >= THRESHOLD]
for r in detected:
    print(f"  {r['label']}: {r['score']:.4f}")
# Output:
#   gratitude: 0.9848
#   admiration: 0.4402
```

### Get Single Top Emotion

```python
results = classifier("I'm furious!", top_k=1)
print(results[0][0])
# {"label": "anger", "score": 0.89}
```

### Direct Model Usage (Without Pipeline)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love this product!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

# Raw logits -> probabilities via sigmoid (NOT softmax)
probabilities = torch.sigmoid(outputs.logits)[0]

# Map to label names
labels = model.config.id2label
for idx, prob in enumerate(probabilities):
    if prob.item() > 0.25:
        print(f"  {labels[idx]}: {prob.item():.4f}")
```

### Batch Processing for Large Datasets

```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0,       # use GPU if available (set to -1 for CPU)
    batch_size=16,   # process 16 texts at once
)

texts = ["text1", "text2", ...]  # your dataset
all_results = classifier(texts)
```

---

## All 28 Emotion Labels

The model outputs scores for these 28 labels (27 emotions + neutral):

| # | Label | Category | Training Support |
|---|---|---|---|
| 0 | admiration | Positive | High (504) |
| 1 | amusement | Positive | Medium (264) |
| 2 | anger | Negative | Medium (198) |
| 3 | annoyance | Negative | High (320) |
| 4 | approval | Positive | High (351) |
| 5 | caring | Positive | Low (135) |
| 6 | confusion | Ambiguous | Low (153) |
| 7 | curiosity | Ambiguous | Medium (284) |
| 8 | desire | Positive | Low (83) |
| 9 | disappointment | Negative | Low (151) |
| 10 | disapproval | Negative | Medium (267) |
| 11 | disgust | Negative | Low (123) |
| 12 | embarrassment | Negative | Very Low (37) |
| 13 | excitement | Positive | Low (103) |
| 14 | fear | Negative | Low (78) |
| 15 | gratitude | Positive | High (352) |
| 16 | grief | Negative | Very Low (6) |
| 17 | joy | Positive | Low (161) |
| 18 | love | Positive | Medium (238) |
| 19 | nervousness | Negative | Very Low (23) |
| 20 | optimism | Positive | Medium (186) |
| 21 | pride | Positive | Very Low (16) |
| 22 | realization | Ambiguous | Low (145) |
| 23 | relief | Positive | Very Low (11) |
| 24 | remorse | Negative | Low (56) |
| 25 | sadness | Negative | Low (156) |
| 26 | surprise | Ambiguous | Low (141) |
| 27 | neutral | Neutral | Very High (1787) |

> **"Training Support"** = number of examples in test split. Labels with Very Low support (grief, pride, relief, embarrassment, nervousness) have weaker performance.

---

## Threshold Tuning

### Default Threshold: 0.5

Standard binary classification threshold. Higher precision, lower recall.

### Optimized Thresholds: Per-label (Recommended)

The model author found that per-label optimized thresholds improve F1 significantly:

```python
OPTIMIZED_THRESHOLDS = {
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

def classify_with_optimized_thresholds(text):
    results = classifier(text)[0]
    detected = []
    for r in results:
        threshold = OPTIMIZED_THRESHOLDS.get(r["label"], 0.25)
        if r["score"] >= threshold:
            detected.append(r)
    return detected
```

### Quick Threshold: 0.25 (Good Default)

A flat 0.25 threshold is a solid middle ground — better recall than 0.5, reasonable precision.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.575 | 0.396 | 0.450 |
| 0.25 (flat) | ~0.54 | ~0.58 | ~0.54 |
| Per-label optimized | 0.572 | 0.677 | 0.611 |

---

## Integration Patterns

### As a Standalone Module in Your Project

Create a file `emotion_detector.py`:

```python
"""
Emotion Detection Module
Uses SamLowe/roberta-base-go_emotions for 28-label emotion classification.
"""

from transformers import pipeline

_classifier = None

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
    """Lazy-load the model (singleton pattern)."""
    global _classifier
    if _classifier is None:
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
```

**Use it anywhere in your project:**

```python
from emotion_detector import detect_emotions, get_top_emotion

# Get detected emotions
emotions = detect_emotions("I'm so excited about this hackathon!")
# [{"label": "excitement", "score": 0.82}, {"label": "joy", "score": 0.45}]

# Get just the top emotion
top = get_top_emotion("I'm so excited about this hackathon!")
# {"label": "excitement", "score": 0.82}
```

### Flask / FastAPI Integration

```python
from fastapi import FastAPI
from emotion_detector import detect_emotions, get_all_scores

app = FastAPI()

@app.post("/emotions")
async def analyze_emotions(text: str):
    return {
        "text": text,
        "emotions": detect_emotions(text),
    }

@app.post("/emotions/all")
async def all_scores(text: str):
    return {
        "text": text,
        "scores": get_all_scores(text),
    }
```

---

## Performance & Limitations

### Strengths
- **High-performing labels:** gratitude (F1=0.92), amusement (F1=0.83), love (F1=0.81)
- **No training needed** — works out of the box
- **Multi-label** — captures complex emotional states

### Limitations
- **English only** — trained on English Reddit data
- **Reddit bias** — informal, internet-style text; may underperform on formal writing
- **Weak labels:** grief (F1≈0.33), relief (F1≈0.25), realization (F1≈0.27) due to very few training examples
- **512 token limit** — longer texts are truncated
- **No sarcasm detection** — sarcastic text may be misclassified
- **Ambiguity in labels** — some emotions overlap (annoyance vs. anger, sadness vs. disappointment)

### Inference Speed (Approximate)

| Hardware | Single Text | Batch of 16 |
|---|---|---|
| CPU (modern) | ~100-200ms | ~600-800ms |
| GPU (CUDA) | ~10-20ms | ~30-50ms |

---

## ONNX (Lightweight / Faster Inference)

For production or resource-constrained environments, use the ONNX version:

```bash
pip install optimum[onnxruntime]
```

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_name = "SamLowe/roberta-base-go_emotions-onnx"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ORTModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
)

results = classifier("I love this!")
```

**ONNX benefits:**
- **75% smaller** model file (INT8 quantized version)
- **Faster inference** especially for small batch sizes
- **No PyTorch dependency** needed — just `onnxruntime`
- **Cross-platform** — runs on any platform with ONNX runtime

---

## Pickle (.pkl) File — Can I Use It?

### What is a `.pkl` file?

A `.pkl` (pickle) file is Python's native binary serialization format. It uses `pickle.dump()` to save any Python object to a file and `pickle.load()` to restore it. It's commonly used for **traditional ML models** (scikit-learn, XGBoost, etc.).

### We saved one for you: `go_emotions_model.pkl` (~478 MB)

It works — but with major caveats. Here's how to use it:

**Saving (already done):**
```python
import pickle
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
)

with open("go_emotions_model.pkl", "wb") as f:
    pickle.dump(classifier, f)
```

**Loading and using:**
```python
import pickle

with open("go_emotions_model.pkl", "rb") as f:
    classifier = pickle.load(f)

results = classifier("I am so happy!")[0]
for r in results[:5]:
    print(f"  {r['label']}: {r['score']:.4f}")
```

### ⚠️ IMPORTANT: Why `.pkl` is NOT recommended for this model

| Issue | Details |
|---|---|
| **Version lock-in** | The `.pkl` file is tied to your EXACT versions of Python, PyTorch, and Transformers. If any version changes, it will likely **fail to load**. |
| **Security risk** | Pickle files can execute arbitrary code when loaded. Never load a `.pkl` from an untrusted source. |
| **Not portable** | Won't work if your friend has a different OS, Python version, or library versions. |
| **Large file size** | ~478 MB — same as the native format, no size benefit. |
| **No ecosystem support** | Can't be uploaded to HuggingFace Hub, can't be converted to ONNX, etc. |

### ✅ The BEST way: `save_pretrained` (recommended)

This is the industry-standard way to save and share transformer models:

**Saving:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to a folder
model.save_pretrained("./go_emotions_saved")
tokenizer.save_pretrained("./go_emotions_saved")
```

**Loading (on ANY machine with `transformers` installed):**
```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="./go_emotions_saved",   # just point to the folder
    top_k=None,
)

results = classifier("I am so happy!")[0]
```

**We already saved this for you too:** the `go_emotions_saved/` folder is ready.

### Comparison: All three approaches

| Feature | `.pkl` (Pickle) | `save_pretrained` | ONNX |
|---|---|---|---|
| **File size** | ~478 MB | ~479 MB | ~120 MB (quantized) |
| **Cross-version** | ❌ Breaks | ✅ Works | ✅ Works |
| **Cross-platform** | ❌ Risky | ✅ Works | ✅ Works |
| **Security** | ❌ Unsafe | ✅ Safe | ✅ Safe |
| **Needs PyTorch** | ✅ Yes | ✅ Yes | ❌ No |
| **Industry standard** | ❌ No | ✅ Yes | ✅ Yes |
| **Speed** | Normal | Normal | Faster |
| **Best for** | Quick hack only | Sharing & deploying | Production |

### Bottom Line

> **If your friend insists on `.pkl`:** Give them `go_emotions_model.pkl` — it works, but tell them they MUST use the same Python + PyTorch + Transformers versions.
>
> **If they want it done right:** Give them the `go_emotions_saved/` folder. They just need `pip install transformers torch` and it works everywhere.

---

## Project File Structure

```
go emotions model/
├── venv/                    # Virtual environment
├── requirements.txt         # Dependencies
├── test_model.py            # Test script (run to verify setup)
├── emotion_detector.py      # Standalone module for your project (copy this)
├── save_model.py            # Script to save model in pkl + native formats
├── go_emotions_model.pkl    # Pickle file (~478 MB) — for your friend
├── go_emotions_saved/       # HuggingFace native save (recommended)
└── SETUP_GUIDE.md           # This file
```

## Quick Start Checklist

- [ ] Create & activate virtual environment
- [ ] `pip install transformers torch`
- [ ] Run `python test_model.py` to verify everything works
- [ ] Copy `emotion_detector.py` into your project
- [ ] Import and use: `from emotion_detector import detect_emotions`
