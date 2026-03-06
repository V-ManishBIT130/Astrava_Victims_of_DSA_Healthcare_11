# ASTRAVA — ML Models

This document covers all three machine learning models used in ASTRAVA's inference pipeline: what they detect, where they come from, how they work, what they output, and the key engineering decisions around them.

---

## Overview: Three Models, Three Jobs

ASTRAVA uses three separate, specialized classifier models rather than one general-purpose model for a deliberate reason. Each model was trained on a specific dataset targeting a specific psychological signal. A single model would need to be a jack of all trades and would likely perform worse on each individual task than a dedicated model trained exactly for that purpose.

The three signals they target are genuinely different:
- Depression is a medium-to-long term pattern of hopelessness, loss of meaning, and withdrawal
- Stress is a near-term state of overwhelm, overload, and strain from pressure
- Emotions are the specific felt qualities of a moment (sadness, anger, joy, shame, etc.)

Each model sees the same cleaned input text and produces its output independently.

---

## Model 1: Depression Classifier

**HuggingFace Checkpoint:** `poudel/Depression_and_Non-Depression_Classifier`
**Base Architecture:** BERT-base-uncased
**Task:** Binary classification
**Labels:** Depression, Non-Depression
**Reported Accuracy:** 99.87%

**Where it comes from:**
This model was fine-tuned on a cleaned version of the Depression Reddit Dataset — a collection of Reddit posts where users explicitly expressed depressive thoughts or symptoms, plus a matched set of non-depressive posts. Reddit is a useful source for this kind of training data because people often share raw, unfiltered emotional content in subreddits like r/depression or in their own posts across various communities.

**What it detects:**
The model has learned to recognize the linguistic patterns that correlate with clinical depression: language about hopelessness, meaninglessness, persistent sadness, withdrawal from activities, sleep disruption, inability to function, and feelings of worthlessness. It is not diagnosing clinical depression — it is detecting whether the language in the text matches the patterns found in self-reported depressive content.

**What it returns:**
It returns two probability scores that sum to 1.0: the probability of Depression and the probability of Non-Depression. The Depression probability is used downstream in the risk aggregator. Inference typically takes 400–600ms on CPU.

**Known limitations:**
Because it was trained primarily on Reddit posts, it performs very well on informal, first-person reflective text (exactly what a chatbot conversation produces). However, it may over-classify burnout or acute stress as depression because those experiences also produce hopeless-sounding language. This is a known limitation and is why all three model scores are combined rather than treating any single model's output as definitive.

**Module location:** `python/ml_models/depression classifier model/depression_classifier.py`

---

## Model 2: Stress Detector

**HuggingFace Checkpoint:** `jnyx74/stress-prediction`
**Base Architecture:** DistilBERT
**Task:** Binary classification
**Labels:** Stressed, No Stress
**Dataset:** Dreaddit (Stress Analysis in Social Media)
**Speed:** 17–20 sentences per second on CPU

**Where it comes from:**
Fine-tuned on the Dreaddit dataset — a carefully annotated corpus of Reddit posts from five subreddits known to produce both stress and non-stress content: r/Anxiety, r/survivorsofabuse, r/domesticviolence, r/stress, and r/PTSD (stress posts) versus r/relationships, r/LegalAdvice, r/financial, r/almosthomeless, r/homeless (non-stress posts). Annotators marked each post as stressed or not.

**What it detects:**
Stress in the psychological sense: feelings of overwhelm from external demands, strain from trying to cope with too much, the subjective experience of being under pressure. This is distinct from depression (which is more about internal hopelessness) and is also distinct from specific emotions (which are more granular feeling states).

**What it returns:**
A binary label (Stressed or No Stress) plus a confidence probability. The raw probability is also converted to a three-tier stress level: LOW (probability below 0.40), MEDIUM (0.40 to 0.70), HIGH (above 0.70). There is also a convenience function `should_trigger_alert()` that returns True if the stress confidence exceeds 0.85. Inference takes 20–30ms (DistilBERT is approximately 40% smaller and faster than full BERT).

**Why DistilBERT here?**
DistilBERT is a knowledge-distilled version of BERT that retains about 97% of BERT's performance while being 40% smaller and 60% faster. For binary classification on medium-length text, the performance difference is negligible, and the speed advantage is meaningful when the model is one of three running in sequence.

**Module location:** `python/ml_models/Stress detection model/stress_detector.py`

---

## Model 3: Emotion Classifier (GoEmotions)

**HuggingFace Checkpoint:** `SamLowe/roberta-base-go_emotions`
**Base Architecture:** RoBERTa-base
**Task:** Multi-label classification (multiple emotions can be active simultaneously)
**Number of Labels:** 28 (27 emotion categories + neutral)
**Dataset:** GoEmotions (Google Research, 58,000 examples)

**Where it comes from:**
Google Research released GoEmotions in 2020 — the largest fine-grained emotion dataset for English text, containing 58,000 Reddit comments each annotated with one or more of 27 emotion labels plus neutral. The `SamLowe/roberta-base-go_emotions` checkpoint on HuggingFace is a community fine-tune of RoBERTa-base on this dataset that has become the standard choice for this task.

**What it detects:**
28 fine-grained emotional states. For mental health context, the most relevant labels are:
- High-concern emotions: sadness, grief, remorse, disappointment, fear, nervousness, disgust (especially when self-directed in context with other signals), confusion
- Protective emotions: joy, optimism, love, gratitude, admiration, caring — the presence of these signals resilience and works against the risk score
- Ambiguous emotions: surprise, curiosity, excitement — these require contextual interpretation

**What makes it multi-label:**
Unlike binary classifiers, this model assigns independent probabilities to all 28 labels simultaneously. A person can be both disappointed AND annoyed AND neutral in the same message. The model reflects that by returning a full vector of 28 probabilities. Labels above a per-label threshold are considered "active" emotions.

**What it returns:**
- A dictionary of all 28 label scores (scores between 0.0 and 1.0)
- A list of "active" emotions — those that crossed their respective threshold
- The top emotions by raw score (used in risk scoring and LLM context)

Inference takes 50–70ms on CPU.

**Why RoBERTa instead of BERT for this model?**
RoBERTa (Robustly Optimized BERT Pretraining Approach) was trained with more data, longer sequences, and with dynamic masking rather than static masking. For multi-label fine-grained emotion classification — a harder task than binary classification — the better pretraining gives it a measurable edge over BERT-base. The Dreaddit stress checkpoint uses DistilBERT because binary classification on shorter posts doesn't need the extra capacity. GoEmotions is harder (28 labels, subtle distinctions) so the more capable base model is justified.

**Module location:** `python/ml_models/go_emotion model/emotion_detector.py`

---

## How All Three Are Loaded and Managed

**Lazy loading with thread-safe singletons:**
Each model module uses a singleton pattern — the model is loaded once when first called and reused for all subsequent calls. The loading is protected by a threading lock using double-checked locking, which ensures that even in a multi-threaded server environment (FastAPI with multiple workers, for example), the model is never loaded twice simultaneously.

**Auto-downloading from HuggingFace:**
None of the model weight files are stored in the repository. They are too large (200–500MB each) and would exceed GitHub's file size limits. Instead, HuggingFace's Transformers library automatically downloads the model weights on first use and caches them locally. On subsequent runs they load from cache instantly.

The local cache location on Windows is:
`C:\Users\<username>\.cache\huggingface\hub\`

**Model file exclusions (.gitignore):**
The `.gitignore` at the repository root excludes all file types used for model weights: `.pkl`, `.safetensors`, `.bin`, `.pth`, `.onnx`. This prevents large model files from ever being accidentally committed.

**Startup warm-up:**
When `run_inference.py` starts, all three models are loaded immediately as part of the `AstravaInference` class initialization. A warm-up forward pass is run on the stress and emotion models during loading so that the first real inference call does not carry any cold-start overhead.

---

## Observed Performance (Real Test Results)

These results come from actual runs on the development machine (PyTorch 2.6.0, CUDA 11.8, Windows 11):

**Test 1 — Hopeless/low-mood input:**
"I haven't been able to sleep for days, everything feels pointless"
- Depression: 99.99% confidence (Depression label)
- Stress: HIGH, 96.78% confidence
- Emotions: disappointment (active), annoyance (active)
- Total pipeline time: 645ms

**Test 2 — Positive input:**
"just got promoted!! so happy rn, feeling amazing honestly"
- Depression: 99.74% confidence (Non-Depression label)
- Stress: LOW, 98.41% confidence (No Stress)
- Emotions: joy (active, 86.66%)
- Total pipeline time: 343ms

**Test 3 — Crisis input (short-circuited):**
"i cant take it anymore i want to end it all goodbye"
- Crisis severity: HIGH — short-circuit triggered
- Models did NOT run — response in 2.7ms
- Matched crisis keywords: "want to end it", "end it all"

**Test 4 — Burnout input:**
"work has been insane lately, my boss keeps adding more tasks, i feel like im burning out and cant focus on anything"
- Depression: 99.97% confidence (Depression label) — note: BERT over-classifies burnout as depression
- Stress: HIGH, 97.90% confidence
- Emotions: annoyance (active, 0.39), disappointment (0.33)
- Total pipeline time: 489ms
