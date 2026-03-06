import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DepressionClassifier:
    """Pluggable depression text classifier using BERT.

    Model: poudel/Depression_and_Non-Depression_Classifier
    Base:  bert-base-uncased (fine-tuned)
    Labels: 0 = Depression, 1 = Non-Depression
    """

    MODEL_NAME = "poudel/Depression_and_Non-Depression_Classifier"
    LABELS = {0: "Depression", 1: "Non-Depression"}

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Classify a single text string.

        Args:
            text: Input text to classify.

        Returns:
            dict with keys: label, label_id, confidence, probabilities
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
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
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Classify a list of text strings.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of dicts, each with keys: text, label, label_id, confidence, probabilities
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu()
        results = []
        for i in range(len(texts)):
            predicted_class = torch.argmax(probs[i]).item()
            results.append(
                {
                    "text": texts[i],
                    "label": self.LABELS[predicted_class],
                    "label_id": predicted_class,
                    "confidence": round(probs[i][predicted_class].item(), 4),
                    "probabilities": {
                        "depression": round(probs[i][0].item(), 4),
                        "non_depression": round(probs[i][1].item(), 4),
                    },
                }
            )
        return results

    def __repr__(self):
        return f"DepressionClassifier(model='{self.MODEL_NAME}', device='{self.device}')"
