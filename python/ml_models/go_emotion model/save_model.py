"""
Save the GoEmotions model in two formats:
1. .pkl file (for your friend)
2. save_pretrained folder (the recommended way)
"""

import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print("Loading model...")
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Model loaded!\n")

# --- 1. Save as .pkl (pickle) ---
print("Saving as .pkl file...")
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
)
with open("go_emotions_model.pkl", "wb") as f:
    pickle.dump(classifier, f)
print("Saved: go_emotions_model.pkl\n")

# --- 2. Save as HuggingFace native (recommended) ---
print("Saving as HuggingFace native format...")
model.save_pretrained("./go_emotions_saved")
tokenizer.save_pretrained("./go_emotions_saved")
print("Saved: ./go_emotions_saved/ folder\n")

# --- Verify both work ---
print("=" * 60)
print("VERIFYING BOTH FORMATS WORK:\n")

test_text = "I am so happy and excited!"

# Test pkl
print("[1] Testing .pkl file...")
with open("go_emotions_model.pkl", "rb") as f:
    pkl_model = pickle.load(f)
pkl_results = pkl_model(test_text)[0]
top3 = pkl_results[:3]
print(f'    Input: "{test_text}"')
for r in top3:
    print(f"    {r['label']:20s} {r['score']:.4f}")

print()

# Test save_pretrained
print("[2] Testing save_pretrained folder...")
native_model = pipeline(
    task="text-classification",
    model="./go_emotions_saved",
    top_k=None,
)
native_results = native_model(test_text)[0]
top3 = native_results[:3]
print(f'    Input: "{test_text}"')
for r in top3:
    print(f"    {r['label']:20s} {r['score']:.4f}")

print("\nBoth formats verified successfully!")

# Show file sizes
import os
pkl_size = os.path.getsize("go_emotions_model.pkl") / (1024 * 1024)
saved_size = sum(
    os.path.getsize(os.path.join("go_emotions_saved", f))
    for f in os.listdir("go_emotions_saved")
) / (1024 * 1024)
print(f"\n.pkl file size:            {pkl_size:.1f} MB")
print(f"save_pretrained folder:    {saved_size:.1f} MB")
