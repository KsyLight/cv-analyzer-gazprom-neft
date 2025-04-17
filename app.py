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

# ─── Настройки страницы ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Анализ резюме по матрице Альянса ИИ",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Анализ резюме по матрице Альянса ИИ")

# ─── Логирование ────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# ─── Загрузка модели ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_competencies(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs

# ─── Загрузка резюме ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Загрузите резюме (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    tmp_file_path = os.path.join("temp", uploaded_file.name)

    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Извлечение текста
        with st.spinner("⏳ Извлечение текста..."):
            base_text = read_resume_from_file(tmp_file_path)
            if not base_text:
                st.error("❌ Не удалось извлечь текст резюме.")
                st.stop()

            # GitHub
            gh_links = extract_github_links_from_text(base_text)
            github_text_raw = ""
            if gh_links:
                st.markdown("🔗 **GitHub‑ссылки:**")
                for link in gh_links:
                    st.markdown(f"- [{link}]({link})")
                    try:
                        github_text_raw += " " + collect_github_text(link)
                    except Exception as e:
                        st.warning(f"⚠️ Ошибка при загрузке {link}")
                        logging.error(f"GitHub fetch error ({link}): {e}")
            else:
                st.info("GitHub‑ссылки не найдены.")

            st.session_state["github_text_raw"] = github_text_raw
            full_text = preprocess_text(base_text + " " + github_text_raw)

        # Предсказание компетенций
        with st.spinner("🤖 Анализ компетенций..."):
            pred_vector, prob_vector = predict_competencies(full_text)

        # Табы
        tab1, tab2, tab3 = st.tabs(["Опрос", "Профессии", "Резюме"])

        # ─── Таб 1: Опрос ────────────────────────────────────────────────────────────
        with tab1:
            st.subheader("Ваш уровень владения по компетенциям (0–3):")
            user_grades = []
            col1, col2 = st.columns(2)
            for i, comp in enumerate(competency_list):
                default = 1 if pred_vector[i] else 0
                with col1 if (i % 2 == 0) else col2:
                    grade = st.radio(
                        comp,
                        [0, 1, 2, 3],
                        index=default,
                        horizontal=True,
                        key=f"grade_{i}"
                    )
                    user_grades.append(grade)
            st.session_state.user_grades = user_grades
            st.success("✅ Грейды сохранены! Перейдите во вкладку 'Профессии'")

        # ─── Таб 2: Профессии ───────────────────────────────────────────────────────
        with tab2:
            if "user_grades" not in st.session_state:
                st.warning("⚠️ Сначала заполните грейды во вкладке 'Опрос'")
                st.stop()

            user_vector = np.array(st.session_state.user_grades)
            if user_vector.shape[0] != profession_matrix.shape[0]:
                st.error("⚠️ Число компетенций не совпадает с размерностью матрицы.")
                st.stop()

            col1, col2 = st.columns(2)

            # ——— Левый столбец: Компетенции и грейды с отступом ———
            with col1:
                st.markdown("### Ваши компетенции и грейды:")
                st.markdown("""
                    <div style="
                        border:1px solid #ddd;
                        border-radius:8px;
                        padding:10px;
                        margin-bottom:10px;
                        width:60%;
                    ">
                      <p style="margin:0; line-height:1.4em; padding-left:10px;">
                        <b>🟩 — грейд 3</b> (высокий уровень)<br>
                        <b>🟨 — грейд 2</b> (уверенный уровень)<br>
                        <b>🟦 — грейд 1</b> (начальный уровень)<br>
                        <b>⬜️ — грейд 0</b> (отсутствует)
                      </p>
                    </div>
                """, unsafe_allow_html=True)

                sorted_comps = sorted(
                    zip(competency_list, user_vector),
                    key=lambda x: -x[1]
                )
                for comp, grade in sorted_comps:
                    emoji = {3: "🟩", 2: "🟨", 1: "🟦", 0: "⬜️"}[grade]
                    st.markdown(
                        f"<div style='margin-left:20px;'>{emoji} **{comp}** — грейд: **{grade}**</div>",
                        unsafe_allow_html=True
                    )

            # ——— Правый столбец: графики и описания профессий ———
            with col2:
                # Расчёт относительного соответствия
                percentages = []
                for i, prof in enumerate(profession_names):
                    required = profession_matrix[:, i]
                    total = np.count_nonzero(required)
                    matched = np.count_nonzero((user_vector >= required) & (required > 0))
                    pct = matched / total * 100 if total else 0.0
                    percentages.append((prof, pct))

                sorted_percentages = sorted(percentages, key=lambda x: x[1], reverse=True)
                labels = [p for p, _ in sorted_percentages]
                values = [v for _, v in sorted_percentages]

                # Круговая диаграмма
                fig, ax = plt.subplots(figsize=(6, 6))
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#0d1117')
                palette = sns.color_palette("pastel", len(labels))
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=palette,
                    wedgeprops={'edgecolor':'#0d1117','linewidth':1}
                )
                for txt in texts + autotexts:
                    txt.set_color('white')
                    txt.set_fontsize(11)
                ax.axis('equal')
                mplcyberpunk.add_glow_effects()
                st.markdown("### Относительное соответствие по профессиям")
                st.pyplot(fig)

                # Столбчатая диаграмма
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                fig_bar.patch.set_facecolor('#0d1117')
                ax_bar.set_facecolor('#0d1117')
                bars = ax_bar.barh(
                    labels,
                    values,
                    color=sns.color_palette("dark", len(labels)),
                    edgecolor='white',
                    linewidth=0.8
                )
                ax_bar.set_xlim(0, 100)
                ax_bar.invert_yaxis()
                ax_bar.set_xlabel("Процент соответствия", color='white')
                ax_bar.grid(axis='x', linestyle='--', alpha=0.3)
                for bar in bars:
                    w = bar.get_width()
                    ax_bar.text(
                        w + 1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}%",
                        va='center',
                        color='white',
                        fontsize=10
                    )
                mplcyberpunk.add_glow_effects()
                st.markdown("### Абсолютное соответствие по профессиям")
                st.pyplot(fig_bar)

                # Блок описаний профессий
                st.markdown("### Описание профессий")
                descriptions = {
                    "Аналитик данных (Data scientist, ML engineer)": """Специалист, который работает с данными компании, анализирует их и разрабатывает решения на основе ИИ. Совместно с техническими аналитиками формирует технические метрики, которые зависят от бизнес-метрик.
• Определяет метод ML и адаптирует его к задаче  
• Разрабатывает признаки  
• Строит пайплайн  
• Ведёт документацию""",
                    "Менеджер в ИИ (Manager in AI)": """Руководит проектом, контролирует сроки и ресурсы. Отвечает за внедрение решения в продуктив, может участвовать в документации и анализе фидбека.""",
                    "Технический аналитик в ИИ (Technical analyst in AI)": """Связывает заказчика и ML-команду. Анализирует бизнес-процессы, готовит ТЗ и участвует в оценке реализуемости и тестировании.""",
                    "Инженер данных (Data engineer)": """Готовит данные: собирает, очищает, передаёт. Поддерживает хранилища и пайплайны данных."""
                }
                prof_name_mapping = {
                    "Аналитик данных": "Аналитик данных (Data scientist, ML engineer)",
                    "Менеджер в ИИ": "Менеджер в ИИ (Manager in AI)",
                    "Технический аналитик в ИИ": "Технический аналитик в ИИ (Technical analyst in AI)",
                    "Инженер данных": "Инженер данных (Data engineer)"
                }
                for prof, _ in sorted_percentages:
                    full_name = prof_name_mapping.get(prof, prof)
                    desc = descriptions.get(full_name, "—")
                    st.markdown(f"""
                        <div style="
                            border:1px solid #ddd;
                            border-radius:8px;
                            padding:10px;
                            margin-bottom:10px;
                        ">
                            <h4 style="margin:0 0 5px 0;">{full_name}</h4>
                            <p style="margin:0;">{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # ─── Таб 3: Резюме ────────────────────────────────────────────────────────────
        with tab3:
            st.markdown("### Извлечённый текст резюме")
            with st.expander("📝 Текст из файла резюме"):
                st.text(base_text)

            github_text_final = st.session_state.get("github_text_raw", "")
            if github_text_final.strip():
                with st.expander("🧑‍💻 Текст, собранный с GitHub"):
                    st.text(github_text_final)
            else:
                st.info("GitHub‑ссылки не найдены или не удалось получить содержимое.")

    except Exception as e:
        st.error("🚫 Не удалось обработать файл.")
        logging.error(f"Общая ошибка: {e}", exc_info=True)
    finally:
        try:
            os.remove(tmp_file_path)
        except OSError:
            pass