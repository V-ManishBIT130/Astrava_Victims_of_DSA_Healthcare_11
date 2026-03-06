# Depression & Non-Depression Text Classifier — Setup & Integration Guide

> A plug-and-play guide for using the **poudel/Depression_and_Non-Depression_Classifier** model in any Python project.

---

## 📋 Table of Contents

1. [Model Overview](#model-overview)
2. [Architecture & Training Details](#architecture--training-details)
3. [Environment Setup](#environment-setup)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Input Specification](#input-specification)
7. [Output Specification](#output-specification)
8. [Pluggable Integration Patterns](#pluggable-integration-patterns)
9. [Using the Pickle (.pkl) File](#using-the-pickle-pkl-file)
10. [API / REST Service Example](#api--rest-service-example)
11. [Performance & Benchmarks](#performance--benchmarks)
12. [Limitations & Disclaimers](#limitations--disclaimers)
13. [License & Citation](#license--citation)

---

## Model Overview

| Property | Value |
|---|---|
| **Model Name** | `poudel/Depression_and_Non-Depression_Classifier` |
| **Task** | Binary Text Classification |
| **Base Model** | `google-bert/bert-base-uncased` |
| **Language** | English |
| **Model Size** | ~110M parameters (438 MB, FP32) |
| **Format** | Safetensors |
| **License** | Apache 2.0 |
| **HuggingFace** | [Link](https://huggingface.co/poudel/Depression_and_Non-Depression_Classifier) |

### Labels

| Label ID | Label Name | Description |
|----------|------------|-------------|
| `0` | **Depression** | Text indicates depressive sentiment |
| `1` | **Non-Depression** | Text indicates neutral/positive sentiment |

---

## Architecture & Training Details

| Detail | Value |
|---|---|
| **Architecture** | BERT (`bert-base-uncased`) fine-tuned for sequence classification |
| **Tokenizer** | `BertTokenizer` (WordPiece, vocab size 30,522) |
| **Max Sequence Length** | 512 tokens |
| **Training Data** | Custom dataset of tweets labeled as depression / non-depression |
| **Preprocessing** | Tokenization + removal of special characters |
| **Training Hardware** | NVIDIA T4 GPU (Google Colab) |
| **Epochs** | 3 |
| **Batch Size** | 16 |
| **Learning Rate** | 5e-5 |
| **Training Time** | ~1 hour |
| **Eval Split** | 20% holdout set |

---

## Environment Setup

### Prerequisites

- **Python** >= 3.8
- **pip** (package manager)
- (Recommended) A virtual environment

### Create Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate
```

---

## Installation

### Required Packages

```bash
pip install transformers torch
```

### Optional (for saving as pickle)

```bash
pip install pickle5  # Only needed for Python < 3.8
```

### Optional (for REST API)

```bash
pip install fastapi uvicorn
```

### Full requirements.txt

```txt
transformers>=4.30.0
torch>=2.0.0
```

---

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("poudel/Depression_and_Non-Depression_Classifier")
model = AutoModelForSequenceClassification.from_pretrained("poudel/Depression_and_Non-Depression_Classifier")
model.eval()

# 2. Prepare input
text = "I feel so hopeless and empty inside"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# 3. Run inference
with torch.no_grad():
    outputs = model(**inputs)

# 4. Get prediction
probs = torch.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probs, dim=-1).item()

label = "Depression" if predicted_class == 0 else "Non-Depression"
confidence = probs[0][predicted_class].item()

print(f"Prediction: {label} ({confidence:.2%})")
# Output: Prediction: Depression (100.00%)
```

---

## Input Specification

### Input Format

| Field | Type | Description |
|---|---|---|
| `text` | `str` | A single English text string (sentence, paragraph, tweet, etc.) |

### Tokenization Details

```python
inputs = tokenizer(
    text,
    return_tensors="pt",      # Return PyTorch tensors
    truncation=True,           # Truncate to max_length
    padding=True,              # Pad shorter sequences
    max_length=512             # Max tokens (BERT limit)
)
```

### Tokenizer Output (Model Input)

| Key | Shape | Type | Description |
|---|---|---|---|
| `input_ids` | `(1, seq_len)` | `torch.LongTensor` | Token IDs from vocabulary |
| `attention_mask` | `(1, seq_len)` | `torch.LongTensor` | 1 for real tokens, 0 for padding |
| `token_type_ids` | `(1, seq_len)` | `torch.LongTensor` | Segment IDs (all 0 for single sentence) |

### Batch Input

```python
texts = ["I feel sad", "What a great day!", "I'm so tired of everything"]
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
# input_ids shape: (3, max_seq_len_in_batch)
```

---

## Output Specification

### Raw Model Output

```python
outputs = model(**inputs)
```

| Field | Shape | Type | Description |
|---|---|---|---|
| `outputs.logits` | `(batch_size, 2)` | `torch.FloatTensor` | Raw unnormalized scores for each class |

### Logits → Probabilities

```python
probs = torch.softmax(outputs.logits, dim=-1)
# Shape: (batch_size, 2)
# probs[i][0] = P(Depression)
# probs[i][1] = P(Non-Depression)
```

### Example Output

```
Input:  "I feel so hopeless and empty inside"
Logits: tensor([[ 4.9617, -5.1866]])
Probs:  tensor([[0.9999, 0.0001]])
→ Depression (99.99%)

Input:  "I had a wonderful day at the park"
Logits: tensor([[-5.0732,  4.8955]])
Probs:  tensor([[0.0001, 0.9999]])
→ Non-Depression (99.99%)
```

### Structured Response Format

```python
def predict(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

    return {
        "text": text,
        "label": "Depression" if predicted_class == 0 else "Non-Depression",
        "label_id": predicted_class,
        "confidence": round(probs[0][predicted_class].item(), 4),
        "probabilities": {
            "depression": round(probs[0][0].item(), 4),
            "non_depression": round(probs[0][1].item(), 4),
        }
    }
```

**Sample JSON output:**

```json
{
    "text": "I can't sleep at night, I keep thinking about how worthless I am.",
    "label": "Depression",
    "label_id": 0,
    "confidence": 1.0,
    "probabilities": {
        "depression": 1.0,
        "non_depression": 0.0
    }
}
```

---

## Pluggable Integration Patterns

### Pattern 1: Standalone Module

Create a file `depression_classifier.py` in your project:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DepressionClassifier:
    """Pluggable depression text classifier using BERT."""

    MODEL_NAME = "poudel/Depression_and_Non-Depression_Classifier"
    LABELS = {0: "Depression", 1: "Non-Depression"}

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Classify a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu()
        predicted_class = torch.argmax(probs, dim=-1).item()

        return {
            "label": self.LABELS[predicted_class],
            "label_id": predicted_class,
            "confidence": round(probs[0][predicted_class].item(), 4),
            "probabilities": {
                "depression": round(probs[0][0].item(), 4),
                "non_depression": round(probs[0][1].item(), 4),
            }
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu()
        results = []
        for i in range(len(texts)):
            predicted_class = torch.argmax(probs[i]).item()
            results.append({
                "text": texts[i],
                "label": self.LABELS[predicted_class],
                "label_id": predicted_class,
                "confidence": round(probs[i][predicted_class].item(), 4),
                "probabilities": {
                    "depression": round(probs[i][0].item(), 4),
                    "non_depression": round(probs[i][1].item(), 4),
                }
            })
        return results
```

**Usage in any project:**

```python
from depression_classifier import DepressionClassifier

clf = DepressionClassifier()
result = clf.predict("I don't want to get out of bed")
print(result)
# {'label': 'Depression', 'label_id': 0, 'confidence': 1.0, ...}
```

### Pattern 2: Load from Pickle (.pkl)

```python
import pickle

# Load
with open("depression_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

result = clf.predict("I'm feeling great today!")
print(result)
```

---

## Using the Pickle (.pkl) File

A `depression_classifier.pkl` file is provided for convenience. It bundles the model, tokenizer, and prediction logic into a single file.

### How it was created

```python
import pickle
from depression_classifier import DepressionClassifier

clf = DepressionClassifier(device="cpu")

with open("depression_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
```

### How to use it

```python
import pickle

# Load the classifier (~438 MB)
with open("depression_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# Single prediction
result = clf.predict("I feel empty inside")
print(result["label"])       # "Depression"
print(result["confidence"])  # 1.0

# Batch prediction
results = clf.predict_batch(["I'm happy!", "Nothing matters anymore"])
for r in results:
    print(f"{r['text']} → {r['label']} ({r['confidence']:.2%})")
```

### Requirements to load the pickle

Even when loading from `.pkl`, you still need these packages installed:

```bash
pip install transformers torch
```

> ⚠️ **Note**: The `.pkl` file is large (~438 MB) since it contains the full BERT model weights. For production, consider saving only the model files using `model.save_pretrained()`.

---

## API / REST Service Example

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from depression_classifier import DepressionClassifier

app = FastAPI(title="Depression Classifier API")
clf = DepressionClassifier()

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(req: TextRequest):
    return clf.predict(req.text)

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return clf.predict_batch(req.texts)

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so alone and worthless"}'
```

---

## Performance & Benchmarks

### Evaluation Metrics (on custom tweet dataset)

| Metric | Score |
|---|---|
| **Accuracy** | 99.87% |
| **Precision** | 99.91% |
| **Recall** | 99.81% |
| **F1 Score** | 99.86% |

### Inference Speed (approximate)

| Hardware | Single Text | Batch (32 texts) |
|---|---|---|
| CPU (Intel i7) | ~50-100ms | ~500-800ms |
| GPU (T4) | ~5-10ms | ~30-50ms |
| GPU (RTX 3090) | ~3-5ms | ~15-25ms |

---

## Limitations & Disclaimers

- **NOT a clinical tool**: This model is for research/educational purposes only. It should NOT be used for medical diagnosis.
- **Training data bias**: Trained on tweets — may not generalize well to long-form text, clinical notes, or non-English text.
- **Binary classification only**: Does not detect severity levels or specific mental health conditions.
- **Context-free**: Each text is classified independently without conversational context.
- **Over-confident**: The model tends to give very high confidence scores (99%+), which may not reflect true uncertainty.

---

## License & Citation

**License**: Apache 2.0

```bibtex
@misc{poudel2024sentimentclassifier,
  author = {Poudel, Ashish},
  title  = {Sentiment Classifier for Depression},
  year   = {2024},
  url    = {https://huggingface.co/poudel/Depression_and_Non-Depression_Classifier},
}
```

---

## File Structure (Recommended)

```
your-project/
├── depression_classifier.py      # Pluggable classifier class
├── depression_classifier.pkl     # Pickled model (optional)
├── MODEL_SETUP_GUIDE.md          # This guide
├── requirements.txt              # transformers, torch
├── api.py                        # REST API (optional)
└── testing.py                    # Test / demo script
```
