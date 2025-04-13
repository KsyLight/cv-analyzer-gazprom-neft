import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk

plt.style.use('cyberpunk')

from utils.cv_reader import read_resume_from_file, preprocess_text
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.constants import competency_list, profession_matrix, profession_names

# Настройки страницы
st.set_page_config(
    page_title="Анализ резюме по матрице Альянса ИИ",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Анализ резюме по матрице Альянса ИИ")

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

# Предсказание компетенций
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
            github_text_raw = ""
            if gh_links:
                st.markdown("🔗 <b>GitHub-ссылки:</b>", unsafe_allow_html=True)
                for link in gh_links:
                    st.markdown(f"- [{link}]({link})")
                    try:
                        github_text_raw += " " + collect_github_text(link)
                    except Exception as e:
                        st.warning(f"⚠️ Ошибка при загрузке {link}")
                        logging.error(f"GitHub fetch error ({link}): {e}")

            full_text = preprocess_text(base_text + " " + github_text_raw)

        with st.spinner("🤖 Анализ..."):
            pred_vector, prob_vector = predict_competencies(full_text)

        # Вкладки
        tab1, tab2, tab3 = st.tabs(["Опрос", "Профессии", "Резюме"])

        # Вкладка Опрос (две колонки)
        with tab1:
            st.subheader("Ваш уровень владения по компетенциям (0–3):")
            user_grades = []
            col1, col2 = st.columns(2)

            for i, comp in enumerate(competency_list):
                default = 1 if pred_vector[i] else 0
                with col1 if i % 2 == 0 else col2:
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

            with col1:
                st.markdown("### Ваши компетенции и грейды:")

                # Легенда
                st.markdown("""
                <div style='font-size: 15px; margin-bottom: 10px;'>
                    <b>🟩 — грейд 3</b> (высокий уровень)<br>
                    <b>🟨 — грейд 2</b> (уверенный уровень)<br>
                    <b>🟦 — грейд 1</b> (начальный уровень)<br>
                    <b>⬜️ — грейд 0</b> (отсутствует)
                </div>
                """, unsafe_allow_html=True)

                # Сортировка по грейду
                sorted_competencies = sorted(zip(competency_list, user_vector), key=lambda x: -x[1])
                for comp, grade in sorted_competencies:
                    color = {3: "🟩", 2: "🟨", 1: "🟦", 0: "⬜️"}.get(grade, "⬜️")
                    st.markdown(f"{color} **{comp}** — грейд: **{grade}**")

            with col2:
                st.markdown("### Соответствие профессиям")

                # Расчёт процентов и сортировка
                percentages = []
                for i, prof in enumerate(profession_names):
                    required = profession_matrix[:, i]
                    matched = np.sum((user_vector >= required) & (required > 0))
                    total = np.sum(required > 0)
                    percent = (matched / total) * 100 if total else 0
                    percentages.append((prof, percent))

                sorted_percentages = sorted(percentages, key=lambda x: x[1], reverse=True)

                for prof, percent in sorted_percentages:
                    st.markdown(f"🔹 **{prof}** — {percent:.1f}% соответствия")

                # Круговая диаграмма
                st.markdown("### Круговая диаграмма")
                fig, ax = plt.subplots()
                labels = [prof for prof, _ in sorted_percentages]
                values = [percent for _, percent in sorted_percentages]
                colors = sns.color_palette("pastel")[0:len(sorted_percentages)]

                ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
                ax.axis("equal")
                mplcyberpunk.add_glow_effects()
                st.pyplot(fig)

                # Описание профессий в таблице
                st.markdown("### Описание профессий")
                descriptions = {
                    "Аналитик данных (Data scientist, ML engineer)": """Специалист, который работает с данными компании, анализирует их и разрабатывает решения на основе ИИ. Совместно с техническими аналитиками формирует технические метрики, которые зависят от бизнес-метрик.

                • Определяет лучший метод машинного обучения и способ его адаптации к специфике задачи  
                • Разрабатывает новые признаки (feature-engineering)  
                • Реализует общий пайплайн решения  
                • Формирует техническую часть документации проекта""",

                    "Менеджер в ИИ (Manager in AI)": """Специалист, который обеспечивает общее выполнение проекта, работу по бюджету, ресурсам, срокам.  
                Отвечает за конверсию и вывод решений в продуктив на организационном уровне. Также в круг его обязанностей может входить обработка пользовательских отзывов и части документирования продукта.""",

                    "Технический аналитик в ИИ (Technical analyst in AI)": """Специалист, который обеспечивает эффективное взаимодействие между аналитиком данных и заказчиком.  
                Анализирует потребности бизнеса, подтверждает и уточняет проблематику, анализирует бизнес-процессы и выявляет ключевые артефакты данных в них.  
                Также специалист оценивает техническую реализуемость запроса, формализует техническое задание и в дальнейшем может участвовать в документировании результатов экспериментов и итогового тестирования.""",

                    "Инженер данных (Data engineer)": """Специалист, который отвечает за сбор, анализ, очистку и подготовку данных для последующего использования.  
                Работает с системами хранения и анализа данных, обеспечивает их эффективное функционирование, а также поддержку систем версионирования данных."""
                }

                # Упорядочиваем профессии по убыванию процента соответствия
                sorted_professions = sorted(
                    zip(profession_names, percentages),
                    key=lambda x: -x[1]
                )

                # Сопоставляем названия профессий с полными именами (с англ. эквивалентом)
                prof_name_mapping = {
                    "Аналитик данных": "Аналитик данных (Data scientist, ML engineer)",
                    "Менеджер в ИИ": "Менеджер в ИИ (Manager in AI)",
                    "Технический аналитик в ИИ": "Технический аналитик в ИИ (Technical analyst in AI)",
                    "Инженер данных": "Инженер данных (Data engineer)"
                }

                # Формируем таблицу
                table_data = {
                    "Профессия": [],
                    "Описание": []
                }

                for prof, _ in sorted_professions:
                    full_name = prof_name_mapping.get(prof, prof)
                    table_data["Профессия"].append(full_name)
                    table_data["Описание"].append(descriptions.get(full_name, "—"))

                st.dataframe(table_data, use_container_width=True)
                
        # Вкладка Резюме
        with tab3:
            st.markdown("### Извлечённый текст резюме")
            with st.expander("📝 Текст из файла резюме"):
                st.text(base_text)

            if github_text_raw.strip():
                with st.expander("🧑‍💻 Текст, собранный с GitHub"):
                    st.text(github_text_raw)
            else:
                st.info("GitHub-ссылки не найдены или не удалось получить содержимое.")

    except Exception as e:
        st.error("🚫 Не удалось обработать файл.")
        logging.error(f"Общая ошибка: {e}")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
