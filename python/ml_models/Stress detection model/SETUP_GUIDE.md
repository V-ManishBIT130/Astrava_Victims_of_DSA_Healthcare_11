# 🧠 Dreaddit Stress Detection - Quick Setup Guide

This guide ensures anyone on your hackathon team can instantly use the `jnyx74/stress-prediction` model. 

This model is a DistilBERT text-classification model specifically fine-tuned on the **Kaggle Dreaddit (Stress Analysis in Social Media)** dataset by `ruchi798`.

---

## ⚙️ 1. Requirements & Installation

You only need **three** Python packages to run this model as a standalone block.

```bash
# Create a virtual environment (optional but recommended)
python -m venv stress_env

# Activate it (Windows)
.\stress_env\Scripts\activate

# Install the required libraries
pip install transformers torch scikit-learn
```

---

## 💻 2. Core Python Setup (The Exact Code to Run)

Here is the exact boilerplate you need to embed this directly into your backend (like Flask, FastAPI, or just a CLI script):

```python
from transformers import pipeline

# 1. Initialize the pipeline (First run takes ~20s to download weights)
# It downloads automatically from HuggingFace, no manual file downloads needed.
print("Loading model...")
stress_analyzer = pipeline("text-classification", model="jnyx74/stress-prediction")
print("Model ready!")

# 2. Define your Input Data
text_input = "I am so overwhelmed with this hackathon deadline. My code keeps crashing and I haven't slept!"

# 3. Get the Output
result = stress_analyzer(text_input)[0]

# 4. Parse the output for your app
label = result['label']       # Returns 'LABEL_1' (Stress) or 'LABEL_0' (No Stress)
confidence = result['score']  # Returns float between 0.0 and 1.0

# Human-readable output formatting
is_stressed = True if label == "LABEL_1" else False

print(f"Stressed: {is_stressed}")
print(f"Confidence: {confidence * 100:.2f}%")
```

---

## 📥 3. Inputs & Outputs Explained

### **The Input**
*   **Format:** A standard Python `string` (`str`).
*   **Max Length:** Up to 512 tokens (roughly 300-400 words). If your text is too long, the pipeline automatically handles it by truncating (cutting off) the extra words or you can explicitly tell it to truncate:
    ```python
    stress_analyzer(long_text, truncation=True, max_length=512)
    ```
*   **Best use case:** Social media posts, chat messages, or journal entries (since it was trained on Reddit data).

### **The Output**
*   **Format:** A Python dictionary inside a list: `[{'label': 'LABEL_X', 'score': 0.YYYY}]`
*   `LABEL_0` = **No Stress** (Neutral, happy, calm, relaxed).
*   `LABEL_1` = **Stressed** (Anxious, overwhelmed, pressured, panicking).
*   `score` = The model's confidence level (e.g., `0.9859` means 98.5% sure).

---

## 🚀 4. Hackathon Pro-Tips (How to Win)

1. **The "Confidence Check" Logic:** 
   Don't just rely on the label. Only trigger a "Stress Alert" in your app if the model is highly confident. 
   ```python
   if is_stressed and confidence > 0.85:
       trigger_alert("High Stress Detected!")
   elif is_stressed and confidence <= 0.85:
       trigger_alert("User might be stressed, keep monitoring.")
   ```

2. **Combine with Emotion Detection (The Ensemble Method):**
   Run the output of this model alongside the GoEmotions model (`SamLowe/roberta-base-go_emotions`). 
   *   *Why?* If this model says "Stressed" and the GoEmotions model says "Fear" or "Nervousness", your algorithm looks incredibly smart and robust to judges.

3. **Inference Speed:**
   This DistilBERT model is blazing fast. It can process **17-20 sentences per second** on a standard CPU. You can easily use it in real-time chat applications or batch process hundreds of database entries without needing a GPU!

4. **Kaggle Validation for Judges:**
   If a judge asks where you got the model, cite: *"We used a Hugging Face DistilBERT architecture specifically fine-tuned on the Dreaddit Kaggle Dataset (Turcan & McKeown, 2019) to ensure high-accuracy categorization of social media stress."*
