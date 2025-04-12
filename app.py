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

# Логирование
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/errors.log", level=logging.ERROR,
                    format="%(asctime)s — %(levelname)s — %(message)s")

# Настройки Streamlit
st.set_page_config(page_title="AI Резюме Анализ", layout="wide")
st.title("💼 AI Анализ Резюме и Компетенций")

# Загрузка модели с Hugging Face
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])  # Приватный токен
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, use_auth_token=True)
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

# Загрузка файла
uploaded_file = st.file_uploader("📤 Загрузите ваше резюме (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        with st.spinner("⏳ Извлечение текста..."):
            base_text = read_resume_from_file(tmp_file_path)
            if not base_text or not isinstance(base_text, str):
                st.error("❌ Не удалось извлечь текст. Загрузите другой файл.")
                st.stop()
            st.success("📄 Текст успешно извлечён.")

            # GitHub-ссылки
            gh_links = []
            try:
                gh_links = extract_github_links_from_text(base_text)
                if gh_links:
                    st.markdown("🔗 <b>Обнаружены GitHub-ссылки:</b>", unsafe_allow_html=True)
                    for link in gh_links:
                        st.markdown(f"- [{link}]({link})")
            except Exception as e:
                logging.error(f"GitHub link error: {e}")

            github_text = ""
            for link in gh_links:
                try:
                    github_text += " " + preprocess_text(collect_github_text(link))
                except Exception as e:
                    st.warning(f"⚠️ Ошибка с {link}")
                    logging.error(f"GitHub fetch error: {e}")

            full_text = preprocess_text(base_text + " " + github_text)

        with st.spinner("🤖 Запуск модели..."):
            pred_vector, prob_vector = predict_competencies(full_text)

        # Компетенции
        st.subheader("🧠 Обнаруженные компетенции:")
        for i, prob in enumerate(prob_vector):
            if pred_vector[i] == 1:
                st.markdown(f"- ✅ {competency_list[i]} — **{prob:.2f}**")

        # Грейды
        st.markdown("---")
        st.subheader("📈 Укажите свой грейд (0–3):")
        user_grades = []
        for i, comp in enumerate(competency_list):
            grade = st.selectbox(f"{comp}", [0, 1, 2, 3], index=1 if pred_vector[i] else 0, key=f"grade_{i}")
            user_grades.append(grade)

        # Профессии
        st.markdown("---")
        st.subheader("🧩 Соответствие профессиям:")
        user_vector = np.array(user_grades)
        for idx, name in enumerate(profession_names):
            req = profession_matrix[:, idx]
            total = np.sum(req > 0)
            matched = np.sum((user_vector >= req) & (req > 0))
            score = (matched / total) * 100 if total > 0 else 0
            st.write(f"🔹 **{name}**: {score:.1f}% соответствия")

        # 🔥 Визуализация heatmap
        st.markdown("---")
        st.subheader("📊 Визуализация соответствия")

        heatmap_data = (user_vector[:, None] >= profession_matrix).astype(int)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=profession_matrix,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=profession_names,
            yticklabels=competency_list,
            cbar=False,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Грейды пользователя vs Требования профессий", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(fontsize=7)
        st.pyplot(fig)

        # Рекомендации (позже)
        st.markdown("---")
        st.subheader("🔮 Рекомендации (будет доступно позже)")
        st.info("Здесь появятся предложения по карьерному росту и обучению.")

        # Просмотр текста
        with st.expander("🧾 Посмотреть полный текст резюме"):
            st.text(full_text)

    except Exception as e:
        st.error("🚫 Ошибка при обработке.")
        logging.error(f"App error: {e}")

    finally:
        os.remove(tmp_file_path)