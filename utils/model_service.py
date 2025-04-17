import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_vector(text):
    """
    Возвращает бинарный вектор и вероятности для каждой компетенции,
    используя threshold, заданный в config модели.
    """
    # Получаем threshold из config (если нет — по умолчанию 0.5)
    threshold = getattr(model.config, "threshold", 0.5)

    # Токенизация и предсказание
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # Бинаризация по threshold
    binary_vector = (probs > threshold).astype(int)
    return binary_vector, probs