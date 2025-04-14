import torch
import numpy as np
from lime.lime_text import LimeTextExplainer


def get_lime_explanation(text, model, tokenizer, class_names, label_id, num_features=10):
    """
    Возвращает интерпретацию LIME для выбранной метки (label_id).
    """
    def predict(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict,
        labels=[label_id],
        num_features=num_features
    )
    return explanation


def get_attention_weights(model, tokenizer, text):
    """
    Возвращает attention веса модели (на всех слоях и головах).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions  # Tuple из attention-матриц для каждого слоя