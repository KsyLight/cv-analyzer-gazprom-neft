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
from utils.constants import competency_list, profession_matrix, profession_names, recommendations

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
        repo_id,
        token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        token=st.secrets["HUGGINGFACE_TOKEN"]
    )
    model.eval()
    return tokenizer, model

# Загружаем модель и токенизатор
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

        # Табы: Опрос, Профессии, Рекомендации, Резюме
        tab1, tab2, tab3, tab4 = st.tabs([
            "Опрос", "Профессии", "Рекомендации", "Резюме"
        ])

        # ─── Таб 1: Опрос ────────────────────────────────────────────────────────────
        with tab1:
            st.subheader("Ваш уровень владения по компетенциям (0–3):")
            user_grades = []
            col1, col2 = st.columns(2)
            for i, comp in enumerate(competency_list):
                default = 1 if pred_vector[i] else 0
                with col1 if i % 2 == 0 else col2:
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

            # Левый столбец: Компетенции и грейды с отступом
            with col1:
                st.markdown("### Ваши компетенции и грейды:")
                st.markdown(
                    """
                    <div style="border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:10px; width:60%; background:#1a1a1a;">
                      <p style="margin:0; padding-left:12px; color:white; line-height:1.4em;">
                        <strong style="color:#4caf50;">🟩 — грейд 3</strong> (высокий уровень)<br>
                        <strong style="color:#ffeb3b;">🟨 — грейд 2</strong> (уверенный уровень)<br>
                        <strong style="color:#2196f3;">🟦 — грейд 1</strong> (начальный уровень)<br>
                        <strong style="color:#ffffff;">⬜️ — грейд 0</strong> (отсутствует)
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                sorted_comps = sorted(
                    zip(competency_list, user_vector),
                    key=lambda x: -x[1]
                )
                for comp, grade in sorted_comps:
                    emoji = {3: "🟩", 2: "🟨", 1: "🟦", 0: "⬜️"}[grade]
                    st.markdown(
                        f"<div style='margin-left:20px; color:white;'>{emoji} <strong>{comp}</strong> — грейд: <strong>{grade}</strong></div>",
                        unsafe_allow_html=True
                    )

            # Правый столбец: графики и таблица описаний профессий
            with col2:
                # Расчёт соответствия
                percentages = []
                for i, prof in enumerate(profession_names):
                    req = profession_matrix[:, i]
                    tot = np.count_nonzero(req)
                    match = np.count_nonzero((user_vector >= req) & (req > 0))
                    pct = match / tot * 100 if tot else 0.0
                    percentages.append((prof, pct))

                sorted_pct = sorted(percentages, key=lambda x: x[1], reverse=True)
                labels = [p for p, _ in sorted_pct]
                values = [v for _, v in sorted_pct]

                # Круговая диаграмма (единый стиль)
                fig, ax = plt.subplots(figsize=(6, 6))
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#0d1117')
                dark_colors = sns.color_palette("dark", len(labels))
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=dark_colors,
                    wedgeprops={'edgecolor':'white','linewidth':0.8}
                )
                for t in texts + autotexts:
                    t.set_color('white')
                    t.set_fontsize(10)
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
                    color=dark_colors,
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
                        va='center', color='white', fontsize=10
                    )
                mplcyberpunk.add_glow_effects()
                st.markdown("### Абсолютное соответствие по профессиям")
                st.pyplot(fig_bar)

                # Таблица описаний профессий
                st.markdown("### Описание профессий")
                descriptions = {
                    "Аналитик данных (Data scientist, ML engineer)":
                        "Специалист, который работает с данными компании, анализирует их и разрабатывает решения на основе ИИ. • Определяет метод ML и адаптирует его к задаче • Разрабатывает признаки • Строит пайплайн • Ведёт документацию",
                    "Менеджер в ИИ (Manager in AI)":
                        "Руководит проектом, контролирует сроки и ресурсы. Отвечает за внедрение решения в продуктив, может участвовать в документации и анализе фидбека.",
                    "Технический аналитик в ИИ (Technical analyst in AI)":
                        "Связывает заказчика и ML-команду. Анализирует бизнес-процессы, готовит ТЗ и участвует в оценке реализуемости и тестировании.",
                    "Инженер данных (Data engineer)":
                        "Готовит данные: собирает, очищает, передаёт. Поддерживает хранилища и пайплайны данных."
                }
                prof_name_mapping = {
                    "Аналитик данных": "Аналитик данных (Data scientist, ML engineer)",
                    "Менеджер в ИИ": "Менеджер в ИИ (Manager in AI)",
                    "Технический аналитик в ИИ": "Технический аналитик в ИИ (Technical analyst in AI)",
                    "Инженер данных": "Инженер данных (Data engineer)"
                }
                table_rows = ""
                for prof, _ in sorted_pct:
                    full_name = prof_name_mapping.get(prof, prof)
                    desc = descriptions.get(full_name, "—")
                    parts = desc.split(" • ")
                    if len(parts) > 1:
                        intro = parts[0].strip()
                        items = parts[1:]
                        desc_html = f"<p style='margin:0 0 4px 0;'>{intro}</p><ul style='margin:0; padding-left:20px;'>"
                        for it in items:
                            desc_html += f"<li style='margin-bottom:2px;'>{it.strip()}</li>"
                        desc_html += "</ul>"
                    else:
                        desc_html = f"<p style='margin:0;'>{desc}</p>"
                    table_rows += f"""
<tr>
  <td style="border:1px solid #444; padding:8px; color:white; vertical-align:top;">{full_name}</td>
  <td style="border:1px solid #444; padding:8px; color:white; vertical-align:top;">{desc_html}</td>
</tr>"""
                table_html = f"""
<table style="width:100%; border-collapse:collapse;">
  <thead>
    <tr style="background-color:#1f1f1f;">
      <th style="border:1px solid #555; padding:8px; color:white; text-align:left;">Профессия</th>
      <th style="border:1px solid #555; padding:8px; color:white; text-align:left;">Описание</th>
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>
"""
                st.markdown(table_html, unsafe_allow_html=True)

        # ─── Таб 3: Рекомендации по слабым компетенциям ───────────────────────────────
        with tab3:
            st.subheader("Рекомендации по слабым компетенциям для выбранной профессии")
            prof_choice = st.selectbox("Выберите профессию", profession_names)
            idx = profession_names.index(prof_choice)
            req = profession_matrix[:, idx]
            user_vec = np.array(st.session_state.user_grades)
            weak = [i for i, (u, r) in enumerate(zip(user_vec, req)) if u < r]
            if weak:
                for i in weak:
                    comp = competency_list[i]
                    st.markdown(f"**{comp}**: ваш грейд {user_vec[i]}, требуется {req[i]}")
                    links = recommendations.get(comp, [])
                    if links:
                        for url in links:
                            st.markdown(f"- [{url}]({url})")
                    else:
                        st.markdown("- Рекомендаций нет")
            else:
                st.success("Все компетенции соответствуют требованиям!")

        # ─── Таб 4: Резюме ────────────────────────────────────────────────────────────
        with tab4:
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