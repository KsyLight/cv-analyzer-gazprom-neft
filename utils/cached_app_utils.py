import re
import uuid
import logging
import numpy as np
import psycopg2
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

from utils.constants import (
    competency_list,
    profession_names,
    profession_matrix,
    recommendations,
)

# ─── Кэшируем тяжёлые функции ─────────────────────────────────────────────────
@st.cache_data
def preprocess_cached(text: str) -> str:
    return preprocess_text(text)

@st.cache_data
def collect_github_text_cached(link: str) -> str:
    return collect_github_text(link)

@st.cache_resource
def _load_model():
    login(token=st.secrets["HUGGINGFACE_TOKEN"])
    repo_id = "KsyLight/resume-ai-competency-model"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=st.secrets["HUGGINGFACE_TOKEN"])
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=st.secrets["HUGGINGFACE_TOKEN"])
    model.eval()
    return tokenizer, model

def load_model_safe():
    try:
        return _load_model()
    except Exception:
        st.error("Не удалось загрузить модель. Проверьте токен или соединение.")
        logging.error("Ошибка загрузки модели", exc_info=True)
        st.stop()

def validate_candidate_form(surname, name, email, professions, telegram_handle, phone, consent):
    if not all([surname, name, email, professions, telegram_handle, phone]):
        return "Пожалуйста, заполните все поля формы."
    if not 1 <= len(professions) <= 2:
        return "Нужно выбрать хотя бы одну и не более двух профессий."
    if not re.match(r"^\+7\d{10}$", phone):
        return "Телефон должен быть в формате +7XXXXXXXXXX."
    if not consent:
        return "Необходимо согласиться на обработку персональных данных."
    return None

def save_application_to_db():
    # Составляем grade0…grade3
    grades = st.session_state.user_grades
    grade_lists = {i: [] for i in range(4)}
    for comp, g in zip(competency_list, grades):
        grade_lists[g].append(comp)

    # Расчёт процентов соответствия
    user_vector = np.array(st.session_state.user_grades)
    percentages = []
    for i, prof in enumerate(profession_names):
        req = profession_matrix[:, i]
        tot = np.count_nonzero(req)
        match = np.count_nonzero((user_vector >= req) & (req > 0))
        pct = match / tot * 100 if tot else 0.0
        percentages.append(pct)

    datan_score      = percentages[profession_names.index("Аналитик данных")]
    ai_manager_score = percentages[profession_names.index("Менеджер в ИИ")]
    techan_score     = percentages[profession_names.index("Технический аналитик в ИИ")]
    daten_score      = percentages[profession_names.index("Инженер данных")]

    # Файл резюме
    uploaded = st.session_state.uploaded_file
    file_bytes = uploaded.getvalue()
    filename   = uploaded.name

    # GitHub-ссылки
    links = st.session_state.gh_links
    git_available = bool(links)
    url_github = links[0] if links else None

    fields = dict(
        original_filename    = filename,
        cv_file              = psycopg2.Binary(file_bytes),
        grade0               = grade_lists[0],
        grade1               = grade_lists[1],
        grade2               = grade_lists[2],
        grade3               = grade_lists[3],
        sender_email         = st.session_state.email,
        code                 = uuid.uuid4().int & 0x7FFFFFFF,
        ai_manager_score     = ai_manager_score,
        techan_score         = techan_score,
        datan_score          = datan_score,
        daten_score          = daten_score,
        git_available        = git_available,
        name                 = st.session_state.name,
        surname              = st.session_state.surname,
        patronymic           = st.session_state.patronymic,
        url_github           = url_github,
        telegram_handle      = st.session_state.telegram_handle,
        phone                = st.session_state.phone,
        consent              = st.session_state.consent,
        selected_professions = st.session_state.selected_professions,
        form_submitted_at    = st.session_state.form_submitted_at
    )

    conn = psycopg2.connect(
        host="localhost",
        dbname="resumes",
        user="appuser",
        password="duduki"
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO resume_records
          (original_filename, cv_file, grade0, grade1, grade2, grade3,
           sender_email, code,
           ai_manager_score, techan_score, datan_score, daten_score,
           git_available, name, surname, patronymic, url_github,
           telegram_handle, phone, consent, selected_professions, form_submitted_at)
        VALUES (
          %(original_filename)s, %(cv_file)s, %(grade0)s, %(grade1)s,
          %(grade2)s, %(grade3)s, %(sender_email)s, %(code)s,
          %(ai_manager_score)s, %(techan_score)s, %(datan_score)s, %(daten_score)s,
          %(git_available)s, %(name)s, %(surname)s, %(patronymic)s, %(url_github)s,
          %(telegram_handle)s, %(phone)s, %(consent)s, %(selected_professions)s,
          %(form_submitted_at)s
        )
        RETURNING id;
    """, fields)
    rec_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return rec_id