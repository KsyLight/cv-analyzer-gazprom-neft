import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

# Загружаем модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_vector(text, threshold=0.5):
    """
    Возвращает бинарный вектор и вероятности для каждой компетенции.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_vector = (probs > threshold).astype(int)
    return binary_vector, probs