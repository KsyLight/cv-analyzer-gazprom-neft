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

# Настройки страницы
st.set_page_config(
    page_title="AI Резюме Анализ",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("💼 AI Анализ Резюме и Компетенций")

# Логирование
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

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

def predict_competencies(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    binary_preds = (probs > 0.5).astype(int)
    return binary_preds, probs

# Загрузка файла
uploaded_file = st.file_uploader("📤 Загрузите резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    tmp_file_path = os.path.join("temp", uploaded_file.name)

    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        with st.spinner("⏳ Извлечение текста..."):
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

        with st.spinner("🤖 Анализ..."):
            pred_vector, prob_vector = predict_competencies(full_text)

        # Вкладки
        tab1, tab2, tab3 = st.tabs(["📋 Опрос", "📊 Профессии", "📄 Резюме"])

        # Вкладка Опрос
        with tab1:
            st.subheader("📈 Ваш уровень владения по компетенциям (0–3):")
            user_grades = []
            for i, comp in enumerate(competency_list):
                default = 1 if pred_vector[i] else 0
                grade = st.radio(comp, [0, 1, 2, 3], index=default, horizontal=True, key=f"grade_{i}")
                user_grades.append(grade)

            st.session_state.user_grades = user_grades
            st.success("✅ Грейды сохранены! Перейдите во вкладку 'Профессии'")

        # Вкладка Профессии
        with tab2:
            if "user_grades" not in st.session_state:
                st.warning("⚠️ Сначала заполните грейды во вкладке 'Опрос'")
                st.stop()

            user_vector = np.array(st.session_state.user_grades)

            if len(user_vector) != profession_matrix.shape[0]:
                st.error("⚠️ Количество компетенций не совпадает с матрицей профессий.")
                logging.error(f"user_vector={len(user_vector)}, matrix_rows={profession_matrix.shape[0]}")
                st.stop()

            col1, col2 = st.columns(2)

            # Левая колонка: Компетенции с грейдами
            with col1:
                st.markdown("### 🧠 Ваши компетенции и грейды:")
                for comp, grade in zip(competency_list, user_vector):
                    st.markdown(f"- **{comp}**: {grade}")

            # Правая колонка: Соответствие профессиям + визуализация
            with col2:
                st.markdown("### 👔 Соответствие профессиям")
                percentages = []
                for i, prof in enumerate(profession_names):
                    required = profession_matrix[:, i]
                    matched = np.sum((user_vector >= required) & (required > 0))
                    total = np.sum(required > 0)
                    percent = (matched / total) * 100 if total else 0
                    percentages.append(percent)
                    st.markdown(f"🔹 **{prof}** — {percent:.1f}% соответствия")

                st.markdown("### 📊 Визуализация в виде круговой диаграммы")
                fig, ax = plt.subplots()
                colors = sns.color_palette("pastel")[0:len(profession_names)]
                ax.pie(percentages, labels=profession_names, autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis("equal")
                st.pyplot(fig)

            # Описание профессий
            st.markdown("### 📘 Описание профессий")
            profession_descriptions = {
                "Аналитик данных": "Изучает и визуализирует данные, применяет ML для анализа.",
                "Инженер данных": "Отвечает за хранение, очистку, подготовку и передачу данных.",
                "Технический аналитик в ИИ": "Связывает бизнес и технологии ИИ, отвечает за требования.",
                "Менеджер в ИИ": "Определяет стратегии внедрения ИИ и координирует команду."
            }
            for prof, desc in profession_descriptions.items():
                st.markdown(f"**{prof}** — {desc}")

        # Вкладка Текст резюме
        with tab3:
            st.markdown("### 📄 Извлечённый текст резюме")
            st.text(full_text)

    except Exception as e:
        st.error("🚫 Не удалось обработать файл.")
        logging.error(f"Общая ошибка: {e}")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)