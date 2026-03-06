"""
GoEmotions Model Test Script
=============================
Uses the pre-trained SamLowe/roberta-base-go_emotions model for
multi-label emotion classification (27 emotions + neutral = 28 labels).

No training required - ready to use out-of-the-box.
"""

import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from transformers import pipeline

# -- Load the model -------------------------------------------------------
print("Loading SamLowe/roberta-base-go_emotions model...")
classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,  # return scores for ALL 28 labels
)
print("Model loaded successfully!\n")

# -- Test sentences covering a range of emotions --------------------------
test_sentences = [
    "I am not having a great day",
    "I am so happy and grateful for everything you've done!",
    "This is absolutely disgusting, I can't believe you did that.",
    "I'm really nervous about the presentation tomorrow.",
    "Wow, I never expected that! What a surprise!",
    "I feel so lonely and sad right now.",
    "That joke was hilarious, I can't stop laughing!",
    "I'm curious about how this model works under the hood.",
    "Thank you so much for helping me, I really appreciate it.",
    "I'm angry that they cancelled the event without notice.",
]

# -- Run inference and display results ------------------------------------
THRESHOLD = 0.25  # optimized threshold from the model card

for sentence in test_sentences:
    print("-" * 70)
    print(f'Input: "{sentence}"\n')

    results = classifier(sentence)[0]  # list of {label, score} dicts

    # Show top emotions (above threshold)
    top_emotions = [r for r in results if r["score"] >= THRESHOLD]
    if top_emotions:
        print("  Detected emotions (threshold >= 0.25):")
        for r in top_emotions:
            bar = "#" * int(r["score"] * 30)
            print(f"     {r['label']:20s}  {r['score']:.4f}  {bar}")
    else:
        print("  (No emotions above threshold)")

    # Also show top-5 regardless of threshold
    print("\n  Top-5 scores:")
    for r in results[:5]:
        bar = "#" * int(r["score"] * 30)
        print(f"     {r['label']:20s}  {r['score']:.4f}  {bar}")
    print()

print("-" * 70)
print("All tests completed!")
print(f"\nModel: SamLowe/roberta-base-go_emotions")
print(f"Labels: {len(classifier(test_sentences[0])[0])} (27 emotions + neutral)")
print(f"Threshold used: {THRESHOLD}")
