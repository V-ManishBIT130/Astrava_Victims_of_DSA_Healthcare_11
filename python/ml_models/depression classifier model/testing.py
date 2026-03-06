from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("poudel/Depression_and_Non-Depression_Classifier")
model = AutoModelForSequenceClassification.from_pretrained("poudel/Depression_and_Non-Depression_Classifier")
model.eval()
print("Model loaded successfully!\n")

# Print model info
print(f"Model labels: {model.config.id2label}")
print(f"Number of labels: {model.config.num_labels}\n")

# Test with sample inputs
test_texts = [
    "I feel so hopeless and empty inside, nothing brings me joy anymore.",
    "I had a wonderful day at the park with my friends and family!",
    "I can't sleep at night, I keep thinking about how worthless I am.",
    "Just got a promotion at work, feeling really excited about the future!",
    "I don't want to get out of bed, everything feels pointless.",
    "The weather is nice today, I'm going for a walk.",
]

print("=" * 80)
print(f"{'TEXT':<60} {'PREDICTION':<15} {'CONFIDENCE'}")
print("=" * 80)

for text in test_texts:
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()

    label = model.config.id2label[predicted_class]
    print(f"{text[:58]:<60} {label:<15} {confidence:.2%}")

print("=" * 80)

# Interactive mode
print("\n--- Interactive Mode (type 'quit' to exit) ---\n")
while True:
    text = input("Enter text: ").strip()
    if text.lower() == "quit":
        print("Goodbye!")
        break
    if not text:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    label = "Depression" if predicted_class == 0 else "Non-Depression"

    print(f"  → Prediction: {label} ({confidence:.2%})")
    print(f"    Depression: {probs[0][0].item():.2%} | Non-Depression: {probs[0][1].item():.2%}\n")
