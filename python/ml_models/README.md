# ML Models - Standalone Projects

This folder contains two emotion/mental health classification models that work as **standalone, plug-and-play** modules. Model weights are **NOT** included in this repository (they're 400-500MB each). Instead, models auto-download from HuggingFace on first use.

---

## 📦 Available Models

### 1. Depression Classifier
**Location:** `depression classifier model/`  
**Model:** [`poudel/Depression_and_Non-Depression_Classifier`](https://huggingface.co/poudel/Depression_and_Non-Depression_Classifier)  
**Task:** Binary text classification (Depression / Non-Depression)  
**Base:** BERT-base-uncased  
**Accuracy:** 99.87%

### 2. GoEmotions Detector
**Location:** `go_emotion model/`  
**Model:** [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)  
**Task:** Multi-label emotion classification (28 emotions)  
**Base:** RoBERTa-base  
**Labels:** joy, anger, sadness, fear, gratitude, admiration, and 22 more

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install transformers torch
```

### Use Depression Classifier
```python
from depression_classifier import DepressionClassifier

clf = DepressionClassifier()
result = clf.predict("I feel empty and hopeless")
print(result)
# {'label': 'Depression', 'confidence': 0.9995, ...}
```

### Use Emotion Detector
```python
from emotion_detector import detect_emotions, get_top_emotion

emotions = detect_emotions("I'm so excited about this!")
print(emotions)
# [{'label': 'excitement', 'score': 0.82}, {'label': 'joy', 'score': 0.45}]

top = get_top_emotion("Thank you so much!")
print(top)
# {'label': 'gratitude', 'score': 0.98}
```

---

## 📖 Documentation

Each model folder contains:
- **Setup Guide** - Complete docs with usage examples, API specs, integration patterns
- **Python module** - Ready-to-import classifier class
- **Test script** - Run to verify everything works
- **Export script** - Optional: save models locally

| Model | Setup Guide | Module |
|---|---|---|
| Depression Classifier | [MODEL_SETUP_GUIDE.md](depression%20classifier%20model/MODEL_SETUP_GUIDE.md) | `depression_classifier.py` |
| GoEmotions | [SETUP_GUIDE.md](go_emotion%20model/SETUP_GUIDE.md) | `emotion_detector.py` |

---

## 🔧 First Run Setup

**Both models auto-download on first use** (~500MB each, one-time):

1. Install dependencies: `pip install transformers torch`
2. Import and use the model - it downloads automatically
3. Subsequent runs load from cache instantly

**Cache location:**
- Windows: `C:\Users\<you>\.cache\huggingface\hub\`
- Linux/macOS: `~/.cache/huggingface/hub/`

---

## 📁 Why No Model Files in Git?

Model weights (`.pkl`, `.safetensors`) are **400-500MB each** - too large for GitHub (100MB limit). They're gitignored and users download them automatically from HuggingFace Hub.

**Benefits:**
- ✅ Fast repo cloning
- ✅ Always get latest model version
- ✅ Cross-platform compatible
- ✅ No version lock-in issues

---

## 🏗️ Using as Standalone Projects

Each model folder is **self-contained** and can be copied anywhere:

```bash
# Copy just the depression classifier
cp -r "depression classifier model" /path/to/your/project/

# Install deps and use
cd /path/to/your/project/depression classifier model/
pip install transformers torch
python testing.py  # verify it works
```

Then import in your code:
```python
from depression_classifier import DepressionClassifier
```

---

## 🌐 API / REST Service

Both setup guides include FastAPI examples for creating REST endpoints. Quick example:

```python
from fastapi import FastAPI
from emotion_detector import detect_emotions

app = FastAPI()

@app.post("/analyze")
async def analyze(text: str):
    return {"emotions": detect_emotions(text)}

# Run: uvicorn api:app --port 8000
```

---

## ⚡ Performance

| Model | Size | CPU Inference | GPU Inference |
|---|---|---|---|
| Depression Classifier | ~438MB | ~50-100ms | ~5-10ms |
| GoEmotions | ~479MB | ~100-200ms | ~10-20ms |

For production, consider ONNX (quantized, faster) - see setup guides.

---

## 🎯 Integration Examples

### Batch Processing
```python
texts = ["text1", "text2", "text3", ...]
results = clf.predict_batch(texts)
```

### Jupyter Notebook
```python
# Just import and use - works out of the box
from emotion_detector import detect_emotions

emotions = detect_emotions("I love this!")
```

### Django/Flask Backend
```python
# In views.py or routes.py
from emotion_detector import detect_emotions

def analyze_text_endpoint(request):
    text = request.POST.get('text')
    emotions = detect_emotions(text)
    return JsonResponse({"emotions": emotions})
```

---

## 🛠️ Troubleshooting

**"Model not downloading?"**
- Check internet connection
- Verify `transformers` and `torch` are installed
- Clear cache: `rm -rf ~/.cache/huggingface/hub/`

**"Import error?"**
- Run from the model folder OR add it to Python path
- `pip install transformers torch` in your environment

**"Out of memory?"**
- Use CPU mode explicitly: `DepressionClassifier(device="cpu")`
- Process smaller batches
- Consider ONNX version (see setup guides)

---

## 📚 Further Reading

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [GoEmotions Dataset Paper](https://arxiv.org/abs/2005.00547)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

## 📄 License

Both models are Apache 2.0 licensed. See individual setup guides for details.

---

## 🤝 Contributing

Each model is self-contained. To improve:
1. Read the setup guide in the model folder
2. Test your changes with the test script
3. Update documentation if needed
4. Submit PR with clear description

---

**Need help?** Check the detailed setup guides in each model folder!
