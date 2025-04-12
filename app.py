import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import numpy as np
import os

from utils.cv_reader import read_resume_from_file
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.resume_processor import preprocess_text
from utils.constants import competency_list, profession_matrix, profession_names

# Настройки страницы
st.set_page_config(page_title="AI Резюме Анализ", layout="wide")
st.title("💼 AI Анализ Резюме и Компетенций")

# Загрузка модели с приватного Hugging Face Hub
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])  # Авторизация через токен
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, use_auth_token=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Функция предсказания
def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs

# Загрузка резюме
uploaded_file = st.file_uploader("📤 Загрузите ваше резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("⏳ Обработка файла и извлечение текста..."):
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.read())

        base_text = read_resume_from_file("temp_file")
        gh_links = extract_github_links_from_text(base_text)
        github_text = " ".join([collect_github_text(link) for link in gh_links]) if gh_links else ""
        full_text = preprocess_text(base_text + " " + github_text)

    with st.spinner("🤖 Запуск модели и анализ..."):
        pred_vector, prob_vector = predict_competencies(full_text)

    # Вывод предсказанных компетенций
    st.subheader("🧠 Обнаруженные компетенции с вероятностями:")
    for i, prob in enumerate(prob_vector):
        if pred_vector[i] == 1:
            comp = competency_list[i]
            st.markdown(f"- ✅ {comp} — **{prob:.2f}**")

    # Грейды пользователя
    st.markdown("---")
    st.subheader("📈 Укажите свой грейд (0–3) по каждой компетенции:")

    user_grades = []
    for i, comp in enumerate(competency_list):
        default = 1 if pred_vector[i] == 1 else 0
        grade = st.selectbox(f"{comp}", options=[0, 1, 2, 3], index=default, key=f"grade_{i}")
        user_grades.append(grade)

    # Анализ соответствия профессиям
    st.markdown("---")
    st.subheader("🧩 Соответствие профессиям:")

    user_vector = np.array(user_grades)
    for prof_idx, prof_name in enumerate(profession_names):
        prof_required = profession_matrix[:, prof_idx]
        total_required = np.sum(prof_required > 0)
        matched = np.sum((user_vector >= prof_required) & (prof_required > 0))
        percent = (matched / total_required) * 100 if total_required > 0 else 0
        st.write(f"🔹 **{prof_name}**: {percent:.1f}% соответствия")

    # Заглушка для будущих рекомендаций
    st.markdown("---")
    st.subheader("🔮 Рекомендации (будет доступно позже)")
    st.info("Здесь появятся рекомендации по карьерным трекам и подходящим вакансиям.")

    # Полный текст резюме
    with st.expander("🧾 Посмотреть полный текст резюме"):
        st.text(full_text)

    os.remove("temp_file")