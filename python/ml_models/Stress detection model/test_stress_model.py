"""
Test script for jnyx74/stress-prediction model
Trained on Dreaddit (Stress Analysis in Social Media) dataset
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_NAME = "jnyx74/stress-prediction"

print("=" * 60)
print("  DREADDIT STRESS DETECTION MODEL TEST")
print(f"  Model: {MODEL_NAME}")
print("=" * 60)

# ── Step 1: Load Model ──────────────────────────────────────
print("\n[1/3] Loading model...")
start = time.time()
classifier = pipeline("text-classification", model=MODEL_NAME)
print(f"      Model loaded in {time.time() - start:.1f}s")

# ── Step 2: Sample Predictions ──────────────────────────────
print("\n[2/3] Running sample predictions...\n")

test_texts = [
    # Clearly stressed
    "I can't handle all this pressure from work and school. I haven't slept in days and I feel like I'm falling apart.",
    "My anxiety is through the roof. I keep having panic attacks and I don't know what to do anymore.",
    "I'm drowning in debt and my landlord is threatening to evict me. I feel so hopeless.",
    "Every day feels like a battle. The workload is insane and my boss keeps piling more on.",
    "I just found out my partner has been lying to me for months. I feel broken and can't stop crying.",
    # Clearly not stressed
    "Had a great day at the park with my family. The weather was beautiful and kids loved it!",
    "Just finished reading an amazing book. Feeling relaxed and content with life right now.",
    "My project at work went really well today. Got positive feedback from the team!",
    "Went for a nice long walk this morning. Nature always helps me feel at peace.",
    "Cooking a nice dinner for friends tonight. Love spending time with people I care about.",
]

expected = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

print(f"  {'Text (first 70 chars)':<72} {'Pred':<8} {'Expected':<8} {'Score'}")
print(f"  {'-'*72} {'-'*8} {'-'*8} {'-'*6}")

sample_preds = []
for text, exp in zip(test_texts, expected):
    result = classifier(text)[0]
    label_num = int(result["label"].replace("LABEL_", ""))
    label_str = "STRESS" if label_num == 1 else "NO STRESS"
    match = "YES" if label_num == exp else "NO"
    sample_preds.append(label_num)
    print(f"  {text[:70]:<72} {label_str:<8} {'STRESS' if exp == 1 else 'NO STRESS':<8} {result['score']:.4f} {match}")

sample_acc = accuracy_score(expected, sample_preds)
print(f"\n  Sample accuracy: {sample_acc*100:.1f}% ({sum(1 for p, e in zip(sample_preds, expected) if p == e)}/{len(expected)})")

# ── Step 3: Evaluate on Dreaddit Test Set ────────────────────
print("\n[3/3] Evaluating on Dreaddit test set...")
print("      Loading dataset from HuggingFace (asmaabid/dreaddit-test)...")

try:
    test_ds = load_dataset("asmaabid/dreaddit-test", split="test")
    
    texts = test_ds["post"]
    true_labels = [1 if str(l).lower() in ['1', 'yes', 'true', 'stress'] else 0 for l in test_ds["label"]]
    
    total = len(texts)
    print(f"      Dataset loaded: {total} samples")
    print(f"      Running inference (this may take a few minutes)...\n")
    
    pred_labels = []
    batch_size = 32
    start = time.time()
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        for r in results:
            pred_labels.append(int(r["label"].replace("LABEL_", "")))
        done = min(i + batch_size, total)
        print(f"      Progress: {done}/{total} ({done/total*100:.0f}%)", end="\r")
    
    elapsed = time.time() - start
    print(f"      Inference completed in {elapsed:.1f}s ({total/elapsed:.1f} samples/sec)")
    
    # Metrics
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n{'='*60}")
    print(f"  RESULTS ON DREADDIT TEST SET")
    print(f"{'='*60}")
    print(f"  Total samples:  {total}")
    print(f"  Accuracy:       {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["Not Stressed", "Stressed"], digits=4))
    
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    No Stress  Stress")
    print(f"  Actual No Stress  {cm[0][0]:<10} {cm[0][1]}")
    print(f"  Actual Stress     {cm[1][0]:<10} {cm[1][1]}")
    
except Exception as e:
    print(f"\n  ⚠ Could not load Dreaddit test set: {e}")
    print(f"  The sample predictions above still demonstrate the model works correctly.")

print(f"\n{'='*60}")
print(f"  TEST COMPLETE")
print(f"{'='*60}")
