import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

from utils.cv_reader import read_resume_from_file
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.resume_processor import preprocess_text
from utils.constants import competency_list, profession_matrix, profession_names


# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# Предсказание компетенций
def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs


# Интерфейс Streamlit
st.set_page_config(page_title="AI Резюме Анализ", layout="wide")
st.title("💼 AI Анализ Резюме и Компетенций")

uploaded_file = st.file_uploader("📤 Загрузите ваше резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    # Сохраняем временный файл
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.read())

    # Обработка резюме + GitHub
    base_text = read_resume_from_file("temp_file")
    gh_links = extract_github_links_from_text(base_text)
    github_text = " ".join([collect_github_text(link) for link in gh_links]) if gh_links else ""
    full_text = preprocess_text(base_text + " " + github_text)

    # 🔮 Предсказание
    st.subheader("🧠 Обнаруженные компетенции:")
    pred_vector, prob_vector = predict_competencies(full_text)
    detected_competencies = [competency_list[i] for i, val in enumerate(pred_vector) if val == 1]

    for comp in detected_competencies:
        st.markdown(f"- ✅ {comp}")

    st.markdown("---")
    st.subheader("📈 Укажите свой грейд (0–3) по каждой компетенции:")

    user_grades = []
    for i, comp in enumerate(competency_list):
        if pred_vector[i] == 1:
            grade = st.selectbox(f"{comp}", options=[0, 1, 2, 3], index=1, key=f"grade_{i}")
        else:
            grade = 0
        user_grades.append(grade)

    st.markdown("---")
    st.subheader("🧩 Соответствие профессиям:")

    user_vector = np.array(user_grades)
    for prof_idx in range(len(profession_names)):
        prof_required = profession_matrix[:, prof_idx]
        total_required = np.sum(prof_required > 0)
        matched = np.sum((user_vector >= prof_required) & (prof_required > 0))
        percent = (matched / total_required) * 100 if total_required > 0 else 0
        st.write(f"🔹 **{profession_names[prof_idx]}**: {percent:.1f}% соответствия")

    st.markdown("---")
    st.subheader("🔮 Рекомендации (будет доступно позже)")
    st.info("Здесь в будущем появятся рекомендации по карьерным трекам и подходящим вакансиям на основе ваших компетенций.")

    # Просмотр исходного текста
    with st.expander("🧾 Посмотреть полный текст резюме"):
        st.text(full_text)

    # Удаляем временный файл
    os.remove("temp_file")