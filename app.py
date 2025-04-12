import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from utils.cv_reader import read_resume_from_file, preprocess_text
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.constants import competency_list, profession_matrix, profession_names

# Конфигурация страницы
st.set_page_config(page_title="AI Резюме Анализ", layout="wide")

# Кастомная светлая тема
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .stRadio > div { flex-direction: row; }
        h1, h2, h3 { color: #0d3b66; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("💼 AI Анализ Резюме и Компетенций")

# Логирование
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/errors.log", level=logging.ERROR, 
                    format="%(asctime)s — %(levelname)s — %(message)s")

# Загрузка модели
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=st.secrets["HUGGINGFACE_TOKEN"])
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=st.secrets["HUGGINGFACE_TOKEN"])
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Предсказание
def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs

# Интерфейс с вкладками
tab1, tab2, tab3 = st.tabs(["📄 Загрузка резюме", "🧠 Компетенции", "👔 Профессии"])

with tab1:
    uploaded_file = st.file_uploader("Загрузите резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        os.makedirs("temp", exist_ok=True)
        tmp_file_path = os.path.join("temp", uploaded_file.name)

        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            with st.spinner("Извлечение текста..."):
                base_text = read_resume_from_file(tmp_file_path)
                if not base_text:
                    st.error("❌ Не удалось извлечь текст.")
                    st.stop()

                gh_links = extract_github_links_from_text(base_text)
                github_text = ""
                if gh_links:
                    st.markdown("🔗 <b>GitHub-ссылки:</b>", unsafe_allow_html=True)
                    for link in gh_links:
                        st.markdown(f"- [{link}]({link})")
                        try:
                            github_text += " " + preprocess_text(collect_github_text(link))
                        except Exception as e:
                            st.warning(f"⚠️ Ошибка при загрузке {link}")
                            logging.error(f"GitHub fetch error ({link}): {e}")

                full_text = preprocess_text(base_text + " " + github_text)

            st.session_state["resume_text"] = full_text
            st.success("✅ Файл успешно обработан. Перейдите во вкладку 🧠 Компетенции")

        except Exception as e:
            st.error("🚫 Ошибка при обработке файла.")
            logging.error(f"Ошибка: {e}")

        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

with tab2:
    if "resume_text" not in st.session_state:
        st.info("⬅️ Пожалуйста, сначала загрузите резюме во вкладке '📄 Загрузка резюме'")
        st.stop()

    text = st.session_state["resume_text"]
    pred_vector, prob_vector = predict_competencies(text)

    st.subheader("🧠 Обнаруженные компетенции:")
    for i, prob in enumerate(prob_vector):
        if pred_vector[i]:
            st.markdown(f"- ✅ **{competency_list[i]}** — вероятность: **{prob:.2f}**")

    st.subheader("📈 Укажите ваш уровень (0–3):")
    user_grades = []
    for i, comp in enumerate(competency_list):
        default = 1 if pred_vector[i] else 0
        try:
            grade = st.radio(comp, [0, 1, 2, 3], index=default, horizontal=True, key=f"grade_{i}")
            user_grades.append(grade)
        except Exception as e:
            logging.error(f"Ошибка при выборе грейда: {e}")
            st.error(f"Ошибка для: {comp}")
            user_grades.append(0)

    if len(user_grades) != profession_matrix.shape[0]:
        st.error("❌ Количество компетенций не совпадает с матрицей.")
        st.stop()

    st.session_state["user_vector"] = np.array(user_grades)
    st.success("✅ Грейды сохранены! Перейдите во вкладку 👔 Профессии")

with tab3:
    if "user_vector" not in st.session_state:
        st.info("⬅️ Пожалуйста, укажите грейды во вкладке '🧠 Компетенции'")
        st.stop()

    st.subheader("👔 Соответствие профессиям")
    percentages = []
    user_vector = st.session_state["user_vector"]

    for i, prof in enumerate(profession_names):
        required = profession_matrix[:, i]
        matched = np.sum((user_vector >= required) & (required > 0))
        total = np.sum(required > 0)
        percent = (matched / total) * 100 if total else 0
        percentages.append(percent)
        st.write(f"🔹 **{prof}** — {percent:.1f}% соответствия")

    # Тепловая карта
    st.markdown("### 📊 Визуализация соответствия")
    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.heatmap([percentages], annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=profession_names, yticklabels=["% соответствия"])
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔮 Рекомендации (будет позже)")
    st.info("Здесь появятся индивидуальные предложения по обучению и карьерному росту.")