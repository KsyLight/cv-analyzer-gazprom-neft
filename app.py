import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import numpy as np
import tempfile
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from utils.cv_reader import read_resume_from_file
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.resume_processor import preprocess_text
from utils.constants import competency_list, profession_matrix, profession_names

# 🌞 Настройка светлой темы
st.set_page_config(page_title="AI Резюме Анализ", layout="wide", initial_sidebar_state="expanded")

# 📝 Логирование
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/errors.log", level=logging.ERROR,
                    format="%(asctime)s — %(levelname)s — %(message)s")

st.title("💼 AI Анализ Резюме и Компетенций")

# 🔐 Авторизация и загрузка модели
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, use_auth_token=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# 🤖 Предсказание компетенций
def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs

# 📄 Загрузка файла
uploaded_file = st.file_uploader("📤 Загрузите ваше резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("⏳ Обработка и анализ резюме..."):
            base_text = read_resume_from_file(tmp_file_path)
            if not base_text:
                st.error("❌ Не удалось извлечь текст. Попробуйте другой файл.")
                st.stop()

            gh_links = extract_github_links_from_text(base_text)
            github_text = ""
            if gh_links:
                st.markdown("🔗 <b>Найденные GitHub-ссылки:</b>", unsafe_allow_html=True)
                for link in gh_links:
                    st.markdown(f"- [{link}]({link})")
                    try:
                        github_text += " " + preprocess_text(collect_github_text(link))
                    except Exception as e:
                        st.warning(f"⚠️ Не удалось загрузить содержимое {link}")
                        logging.error(f"GitHub fetch error ({link}): {e}")

            full_text = preprocess_text(base_text + " " + github_text)
            pred_vector, prob_vector = predict_competencies(full_text)

        # 🧠 Компетенции
        st.subheader("🧠 Обнаруженные компетенции с вероятностями:")
        for i, prob in enumerate(prob_vector):
            if pred_vector[i] == 1:
                st.markdown(f"- ✅ {competency_list[i]} — **{prob:.2f}**")

        # 🧪 Грейды (radio-кнопки)
        st.markdown("---")
        st.subheader("📈 Укажите ваш грейд по каждой компетенции:")
        user_grades = []
        for i, comp in enumerate(competency_list):
            default = 1 if pred_vector[i] == 1 else 0
            grade = st.radio(comp, [0, 1, 2, 3], index=default, horizontal=True, key=f"grade_{i}")
            user_grades.append(grade)

        user_vector = np.array(user_grades)

        # 👔 Соответствие профессиям
        st.markdown("---")
        st.subheader("🧩 Соответствие профессиям:")
        results = []
        for prof_idx, prof_name in enumerate(profession_names):
            prof_required = profession_matrix[:, prof_idx]
            total_required = np.sum(prof_required > 0)
            matched = np.sum((user_vector >= prof_required) & (prof_required > 0))
            percent = (matched / total_required) * 100 if total_required > 0 else 0
            st.write(f"🔹 **{prof_name}**: {percent:.1f}% соответствия")
            results.append(percent)

        # 🎨 Визуализация соответствия (heatmap)
        st.markdown("### 📊 Визуализация соответствия")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        sns.heatmap([results], annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=profession_names, yticklabels=["%"])
        st.pyplot(fig)

        # 💡 Рекомендации (будущее)
        st.markdown("---")
        st.subheader("🔮 Рекомендации (в разработке)")
        st.info("Скоро здесь появятся индивидуальные карьерные советы и предложения.")

        # 🧾 Полный текст
        with st.expander("📃 Посмотреть текст резюме и GitHub:"):
            st.text(full_text)

    except Exception as e:
        st.error("🚫 Ошибка при анализе. Проверьте файл или повторите попытку.")
        logging.error(f"Unexpected error: {e}")
    finally:
        os.remove(tmp_file_path)