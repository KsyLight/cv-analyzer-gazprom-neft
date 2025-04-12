import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import numpy as np
import os
import io
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from utils.cv_reader import read_resume_from_file, preprocess_text
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.constants import competency_list, profession_matrix, profession_names

# 🔧 Настройки страницы
st.set_page_config(page_title="AI Резюме Анализ", layout="wide")

# 📘 Логирование
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/errors.log", level=logging.ERROR,
                    format="%(asctime)s — %(levelname)s — %(message)s")

# 🔐 Загрузка модели
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, use_auth_token=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# 🤖 Предсказание
def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs

# 📂 Интерфейс
st.title("💼 AI Анализ Резюме и Компетенций")
uploaded_file = st.file_uploader("📤 Загрузите ваше резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)

        with st.spinner("⏳ Извлечение текста..."):
            text = read_resume_from_file(uploaded_file.name, file_buffer)
            if not text or not isinstance(text, str):
                st.error("❌ Не удалось извлечь текст из файла.")
                st.stop()

        # GitHub
        gh_links = extract_github_links_from_text(text)
        gh_text = ""
        if gh_links:
            st.markdown("🔗 <b>Обнаружены GitHub-ссылки:</b>", unsafe_allow_html=True)
            for link in gh_links:
                st.markdown(f"- [{link}]({link})")
                try:
                    gh_text += " " + preprocess_text(collect_github_text(link))
                except Exception as e:
                    st.warning(f"⚠️ Ошибка с {link}")
                    logging.error(f"GitHub error: {e}")

        full_text = preprocess_text(text + " " + gh_text)

        with st.spinner("🤖 Анализ..."):
            pred_vector, prob_vector = predict_competencies(full_text)

        # 🧠 Компетенции
        st.subheader("🧠 Найденные компетенции:")
        for i, prob in enumerate(prob_vector):
            if pred_vector[i] == 1:
                st.markdown(f"- ✅ {competency_list[i]} — **{prob:.2f}**")

        # 📊 Грейды
        st.markdown("---")
        st.subheader("📈 Оцените свои компетенции:")
        user_grades = []
        for i, comp in enumerate(competency_list):
            default = 1 if pred_vector[i] == 1 else 0
            grade = st.radio(comp, [0, 1, 2, 3], index=default, horizontal=True, key=f"grade_{i}")
            user_grades.append(grade)

        user_vector = np.array(user_grades)

        # 🧩 Профессии
        st.markdown("---")
        st.subheader("👔 Соответствие профессиям:")
        results = []
        for i, prof in enumerate(profession_names):
            prof_req = profession_matrix[:, i]
            matched = np.sum((user_vector >= prof_req) & (prof_req > 0))
            total = np.sum(prof_req > 0)
            percent = (matched / total) * 100 if total > 0 else 0
            st.write(f"🔹 **{prof}** — {percent:.1f}% соответствия")
            results.append(percent)

        # 🎨 Тепловая карта
        st.markdown("### 📊 Визуализация")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        sns.heatmap([results], annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=profession_names, yticklabels=["%"])
        st.pyplot(fig)

        # 🔮 Рекомендации
        st.markdown("---")
        st.subheader("🔮 Рекомендации (в разработке)")
        st.info("В будущем здесь появятся персональные предложения по развитию.")

        # 📃 Полный текст
        with st.expander("📄 Посмотреть полный текст резюме"):
            st.text(full_text)

    except Exception as e:
        st.error("Произошла ошибка при обработке.")
        logging.error(f"Critical error: {e}")