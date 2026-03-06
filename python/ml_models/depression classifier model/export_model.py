"""
Script to export the DepressionClassifier as a .pkl file.
This bundles the model, tokenizer, and prediction logic into one portable file.
"""

import pickle
from depression_classifier import DepressionClassifier

print("=" * 60)
print("  Depression Classifier — Export to .pkl")
print("=" * 60)

# Step 1: Initialize classifier (downloads model if not cached)
print("\n[1/3] Loading model and tokenizer...")
clf = DepressionClassifier(device="cpu")
print(f"  ✓ Model loaded: {clf}")

# Step 2: Verify it works before saving
print("\n[2/3] Running verification tests...")
test_cases = [
    ("I feel so hopeless and empty inside", "Depression"),
    ("I had a wonderful day at the park!", "Non-Depression"),
    ("I can't stop crying, everything hurts", "Depression"),
    ("Just got promoted, feeling amazing!", "Non-Depression"),
]

all_passed = True
for text, expected in test_cases:
    result = clf.predict(text)
    status = "✓" if result["label"] == expected else "✗"
    if result["label"] != expected:
        all_passed = False
    print(f"  {status} '{text[:50]}...' → {result['label']} ({result['confidence']:.2%})")

if not all_passed:
    print("\n  ⚠ Some tests failed! Saving anyway...")

# Step 3: Save as pickle
pkl_path = "depression_classifier.pkl"
print(f"\n[3/3] Saving to {pkl_path}...")
with open(pkl_path, "wb") as f:
    pickle.dump(clf, f)

import os
size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
print(f"  ✓ Saved successfully! File size: {size_mb:.1f} MB")

# Free memory before verification
del clf
import gc
gc.collect()

# Step 4: Verify the pickle loads correctly
print("\n[Verify] Loading back from .pkl...")
with open(pkl_path, "rb") as f:
    clf_loaded = pickle.load(f)

result = clf_loaded.predict("I feel terrible about myself")
print(f"  ✓ Loaded and working: '{result['label']}' ({result['confidence']:.2%})")

print("\n" + "=" * 60)
print("  Done! You can now use 'depression_classifier.pkl' in any project.")
print("=" * 60)
print("""
Usage:
    import pickle

    with open("depression_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    result = clf.predict("your text here")
    print(result)
""")
