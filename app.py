import streamlit as st
import pandas as pd
import re
import logging
import os
import uuid
import psycopg2
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import mplcyberpunk
import random

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

from utils.cv_reader import read_resume_from_file, preprocess_text
from utils.github_reader import extract_github_links_from_text, collect_github_text
from utils.constants import (
    competency_list,
    profession_matrix,
    profession_names,
    recommendations,)

import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import psycopg2.errors
from st_aggrid import AgGrid, GridOptionsBuilder

# ─── Константы ────────────────────────────────────────────────────────────────
THRESHOLD = 0.46269254347612143  # порог для бинаризации меток модели
# Файл учётных данных OAuth
CREDENTIALS_FILE = "client_secret_2_496304292584-focgmts10r0pc3cplngprpkiqshp5d2j.apps.googleusercontent.com.json"
# Файл для хранения токена доступа
TOKEN_FILE = "token.json"
# Область прав: отправка почты
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# ─── Gmail API setup ────────────────────────────────────────────────────────
def get_gmail_service():
    creds = None
    # Попробуем загрузить существующий токен, если он есть и не пустой
    if os.path.exists(TOKEN_FILE) and os.path.getsize(TOKEN_FILE) > 0:
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            # файл повреждён или не JSON — забываем о нём
            creds = None

    # Если токена нет или он просрочен — начинаем OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Сохраняем свежие токены в файл
        with open(TOKEN_FILE, "w") as token_f:
            token_f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

# Массовая авторасслыка
def send_email_custom(to_email: str, subject: str, body_text: str) -> bool:
    """
    Отправляет письмо через Gmail API и возвращает True/False.
    """
    # валидируем адрес
    to_email = to_email.strip()

    msg = MIMEText(body_text)
    msg["To"]      = to_email
    msg["From"]    = "hacaton.gpn@gmail.com"
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    service = get_gmail_service()
    try:
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True
    except HttpError as e:
        if e.resp.status == 400 and "Invalid To header" in str(e):
            st.error(f"⚠️ Некорректный email **{to_email}**, рассылка пропущена.")
            return False
        logging.error("Ошибка при отправке письма через Gmail API", exc_info=True)
        st.error("⚠️ Не удалось отправить письмо. Попробуйте позже.")
        return False

# ─── Массовая авторассылка ───────────────────────────────────────────────────────
def send_bulk_mail(row: pd.Series, prof: str, threshold: float, above: bool) -> bool:
    """
    row: pd.Series с полями grade0…grade3, sender_email, name, code, score
    prof: профессия, по которой идёт рассылка
    threshold: значение T, которое выбрал HR
    above: True если row['score'] >= threshold
    """
    email = str(row["sender_email"]).strip()
    name  = row["name"]
    score = row["score"]
    code  = row["code"]

    # 0) явная валидация e-mail
    if "@" not in email or "." not in email.split("@")[-1]:
        logging.warning(f"BulkMail: некорректный email {email}, пропущен.")
        return False

    # 1) idx и вектор требований
    idx = profession_names.index(prof)
    req_vec = profession_matrix[:, idx]

    # 2) вектор кандидата
    user_vec = [
        0 if comp in row["grade0"]
        else 1 if comp in row["grade1"]
        else 2 if comp in row["grade2"]
        else 3 if comp in row["grade3"]
        else 0
        for comp in competency_list
    ]

    # 3) слабые компетенции
    weak = [i for i,(u,r) in enumerate(zip(user_vec,req_vec)) if u<r]

    # 4) рекомендации
    rec_texts = []
    for i in weak:
        comp = competency_list[i]
        recs = recommendations.get(comp, [])
        if recs:
            rec_texts.append(f"- {comp}: {random.choice(recs)}")

    # 5) выбираем шаблоны
    if above:
        subject = "Поздравляем! Вы прошли первый этап"
        intro = (
            f"Здравствуйте, {name}!\n\n"
            f"Вы отправили заявку на вакансию «{prof}» и набрали {score:.1f}% соответствия.\n"
            "Поздравляем — вы прошли первый этап отбора!"
        )
        if rec_texts:
            intro += "\n\nРекомендуем перед техническим собеседованием освежить в памяти следующие материалы:\n"
        else:
            # если нет слабых компетенций, добавляем приглашение на интервью
            intro += "\n\nСкоро вас ожидает техническое интервью!"
    else:
        subject = "Спасибо за участие в конкурсе"
        intro = (
            f"Здравствуйте, {name}!\n\n"
            f"Благодарим вас за интерес к вакансии «{prof}».\n"
            f"На данном этапе ваш профиль (соответствие {score:.1f}%) не полностью соответствует требованиям."
        )
        if rec_texts:
            intro += "\n\nНиже — рекомендации по компетенциям, которые стоит подтянуть:\n"

    # 6) тело письма
    body = intro
    if rec_texts:
        body += "\n" + "\n".join(rec_texts)
    body += "\n\nС уважением,\nКоманда CV-Analyzer"

    # 7) отправка
    ok = send_email_custom(email, subject, body)
    # логируем в файл
    if ok:
        logging.info(f"BulkMail: {email} [{prof}] sent to {'A' if above else 'B'} group.")
    else:
        logging.error(f"BulkMail: {email} [{prof}] failed.")
    return ok

# ─── Настройка отправки почты ──────────────────────────────────────────────────
def send_confirmation_email(
    to_email: str,
    code: int,
    candidate_name: str,
    professions: list[str]
) -> bool:
    """
    Возвращает True, если письмо успешно отправлено,
    False — если невалидный email или при других ошибках.
    """
    # Очищаем и валидируем to_email
    to_email = str(to_email).strip()
    if "@" not in to_email or "." not in to_email.split("@")[-1]:
        st.error(f"⚠️ Некорректный email: **{to_email}**, поэтому авторассылка не придёт. Проверьте правильность почты, но мы уже внесли Ваше резюме в базу данных.")
        return False

    # Формируем тело
    profs_str = ", ".join(professions)
    subject = "Ваша заявка принята"
    body_text = f"""Здравствуйте, {candidate_name}!

Вы отправили заявку на профессии: {profs_str}
Ваша заявка (код #{code}) принята и будет рассмотрена в ближайшее время.
Спасибо за интерес к нашим вакансиям!

С уважением,
Команда CV-Analyzer
"""

    msg = MIMEText(body_text)
    msg["To"]      = to_email
    msg["From"]    = "hacaton.gpn@gmail.com"
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    service = get_gmail_service()
    try:
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True
    except HttpError as e:
        # ловим specifically Invalid To header
        msg_text = str(e)
        if e.resp.status == 400 and "Invalid To header" in msg_text:
            st.error(f"⚠️ Некорректный адрес: **{to_email}**, поэтому авторассылка не придёт. Проверьте правильность почты, но мы уже внесли Ваше резюме в базу данных.")
            return False
        # для любых других ошибок
        logging.error("Ошибка при отправке письма через Gmail API", exc_info=True)
        st.error("⚠️ Не удалось отправить письмо. Попробуйте позже.")
        return False


# ─── Общие настройки ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/errors.log",
    level=logging.ERROR,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
plt.style.use('cyberpunk')
st.set_page_config(
    page_title="Анализ резюме по матрице Альянса ИИ",
    page_icon="others/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
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

# ─── Выбор роли ────────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.title("Добро пожаловать в CV-Analyzer")
    st.write("Пожалуйста, выберите, кто вы:")
    choice = st.radio("", ["Кандидат", "HR-специалист"])
    if st.button("Продолжить"):
        st.session_state.role = "candidate" if choice == "Кандидат" else "hr"
    st.stop()

# ─── Поток кандидата ───────────────────────────────────────────────────────────
if st.session_state.role == "candidate":
    st.title("Анализ резюме по матрице Альянса ИИ")

    if "form_filled" not in st.session_state:
        st.session_state.form_filled = False

    # Шаг 1: форма
    if not st.session_state.form_filled:
        st.markdown("## Шаг 1. Заполните форму кандидата")
        with st.form("candidate_form"):
            surname          = st.text_input("Фамилия")
            name             = st.text_input("Имя")
            patronymic       = st.text_input("Отчество")
            email            = st.text_input("Email")
            professions      = st.multiselect(
                "Выберите до двух вакансий, на которые хотите податься",
                options=profession_names,
                help="Максимум 2",
                max_selections=2
            )
            st.markdown("**Описание профессий:**")
            prof_desc = {
                "Аналитик данных":           "Работает с данными, строит ML-решения, ведёт документацию.",
                "Менеджер в ИИ":             "Руководит проектом, контролирует сроки и ресурсы.",
                "Технический аналитик в ИИ": "Собирает требования, готовит ТЗ, тестирует решения.",
                "Инженер данных":            "Собирает, очищает и передаёт данные, поддерживает пайплайны."
            }
            for prof in profession_names:
                st.markdown(f"- **{prof}**: {prof_desc[prof]}")
            telegram_handle = st.text_input("Telegram-ник (например, @username)")
            phone           = st.text_input("Телефон в формате +7XXXXXXXXXX")
            consent         = st.checkbox(
                "Я даю согласие на обработку моих персональных данных в рамках отбора кандидатов. "
                "Мои данные будут использованы только для связи и принятия решения по моему резюме."
            )
            submit = st.form_submit_button("Продолжить к загрузке резюме")

        if submit:
            error = validate_candidate_form(
                surname, name, email, professions, telegram_handle, phone, consent
            )
            if error:
                st.error(error)
                logging.warning(f"Candidate form validation failed: {error}")
            else:
                st.session_state.update({
                    "surname": surname,
                    "name": name,
                    "patronymic": patronymic,
                    "email": email,
                    "selected_professions": professions,
                    "telegram_handle": telegram_handle,
                    "phone": phone,
                    "consent": consent,
                    "form_filled": True,
                    "form_submitted_at": datetime.datetime.now(datetime.timezone.utc)
                })
                st.info("✅ Форма принята, переходим к загрузке резюме…")

        if not st.session_state.form_filled:
            st.stop()

    # Шаг 2: загрузка резюме и анализ
    st.markdown("## Шаг 2. Загрузите резюме")
    uploaded_file = st.file_uploader(
        "📤 Загрузите резюме (PDF, DOCX, TXT), не более 10 MB",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ Файл больше 10 MB, загрузите меньший.")
            st.stop()

        tmp_dir = "temp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            with st.spinner("⏳ Извлечение и предобработка текста..."):
                raw = read_resume_from_file(tmp_path)
                if not raw or not raw.strip():
                    st.error("❌ Не удалось извлечь текст резюме.")
                    st.stop()

                links = extract_github_links_from_text(raw)
                st.session_state.gh_links = links
                gh_text = ""
                if links:
                    st.markdown("🔗 **GitHub-ссылки:**")
                    for link in links:
                        st.markdown(f"- {link}")
                        try:
                            gh_text += " " + collect_github_text_cached(link)
                        except Exception:
                            st.warning(f"Ошибка при загрузке GitHub-текста {link}")
                combined = raw + " " + gh_text
                text = preprocess_cached(combined)

            with st.spinner("🤖 Загрузка модели и анализ резюме..."):
                tokenizer, model = load_model_safe()
                inputs = tokenizer(text, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                preds = (probs > THRESHOLD).astype(int)

                st.session_state.prob_vector = probs
                st.session_state.pred_vector = preds
                st.session_state.uploaded_file = uploaded_file

            tab = st.tabs(["Оценка грейдов"])[0]
            with tab:
                st.subheader("Оцените уровень владения компетенциями (0–3):")
                user_grades = []
                c1, c2 = st.columns(2)
                for i, comp in enumerate(competency_list):
                    default = 1 if preds[i] else 0
                    with (c1 if i % 2 == 0 else c2):
                        grade = st.radio(comp, [0, 1, 2, 3], index=default,
                                         horizontal=True, key=f"grade_{i}")
                        user_grades.append(grade)
                st.session_state.user_grades = user_grades
                st.success("✅ Грейды сохранены!")

                # Кнопка отправки заявки
                if not st.session_state.get("submitted"):
                    if st.button("Отправить заявку"):
                        try:
                            rec_id = save_application_to_db()
                        except psycopg2.errors.UniqueViolation as e:
                            cn = e.diag.constraint_name
                            if cn == "uq_resume_phone":
                                st.error("Заявка с этим номером телефона уже отправлена.")
                            elif cn == "uq_resume_sender_email":
                                st.error("Заявка с этим email уже отправлена.")
                            elif cn == "uq_resume_telegram_handle":
                                st.error("Заявка с этим Telegram-никнеймом уже отправлена.")
                            else:
                                st.error("Заявка с такими данными уже существует.")
                            logging.warning("Duplicate application prevented", exc_info=True)
                        except Exception as e:
                            st.error("Произошла ошибка при сохранении заявки. Попробуйте ещё раз.")
                            logging.error("Error saving application", exc_info=True)
                        else:
                            # пробуем отправить письмо и смотрим на результат
                            sent = send_confirmation_email(
                                st.session_state.email,
                                rec_id,
                                st.session_state.name,
                                st.session_state.selected_professions
                            )
                            if sent:
                                st.success(
                                    f"✅ Ваша заявка №{rec_id} принята! "
                                    f"Письмо подтверждения отправлено на {st.session_state.email}"
                                )
                                st.session_state.submitted = True
                            # если sent == False, то внутри функции уже вывели st.error, флаг submitted не ставим
                else:
                    st.info("Вы уже отправили заявку.")

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# ─── Поток HR-специалиста ─────────────────────────────────────────────────────
elif st.session_state.role == "hr":
    # 1. Флаг аутентификации
    if "hr_authenticated" not in st.session_state:
        st.session_state.hr_authenticated = False

    # 2. Форма входа
    if not st.session_state.hr_authenticated:
        st.title("Вход для HR-специалиста")
        pwd = st.text_input("Пароль", type="password")
        if st.button("Войти"):
            if pwd == "duduki":
                st.session_state.hr_authenticated = True
            else:
                st.error("Неверный пароль")
        st.stop()

    # 3. Главное окно HR
    st.title("Панель HR-специалиста")

    # 5 вкладок: 4×профессии + Общая сводка
    tab_labels = profession_names + ["Общая сводка"]
    tabs = st.tabs(tab_labels)

    # Маппинг профессии на столбец score
    score_mapping = {
        "Аналитик данных":           "datan_score",
        "Менеджер в ИИ":             "ai_manager_score",
        "Технический аналитик в ИИ": "techan_score",
        "Инженер данных":            "daten_score",
    }

    for idx, prof in enumerate(profession_names):
        with tabs[idx]:
            st.subheader(f"Вакансия: {prof}")

            # —— Фильтры ——————————————————————————————————————
            with st.expander("🔍 Фильтры", expanded=True):
                # Даты
                date_range = st.date_input(
                    "Период заявок",
                    value=(
                        datetime.date.today() - datetime.timedelta(days=30),
                        datetime.date.today()
                    ),
                    key=f"filter_date_{idx}"
                )
                # HR-email
                hr_emails = st.multiselect(
                    "HR Email",
                    options=[],
                    key=f"filter_hr_{idx}"
                )
                # GitHub
                git_choice = st.selectbox(
                    "GitHub",
                    ["Любой", "Да", "Нет"],
                    key=f"filter_git_{idx}"
                )
                # Обязательные компетенции по грейдам
                st.markdown("**Обязательные компетенции по грейдам**")
                req1 = st.multiselect(
                    "Грейд 1",
                    options=competency_list,
                    key=f"req1_{idx}"
                )
                req2 = st.multiselect(
                    "Грейд 2",
                    options=[c for c in competency_list if c not in req1],
                    key=f"req2_{idx}"
                )
                req3 = st.multiselect(
                    "Грейд 3",
                    options=[c for c in competency_list if c not in req1 + req2],
                    key=f"req3_{idx}"
                )
                # Вторая профессия
                other_profs = ["Не важно"] + [p for p in profession_names if p != prof]
                sec_prof = st.selectbox(
                    "Ещё одна профессия",
                    other_profs,
                    key=f"filter_secprof_{idx}"
                )
                # Диапазон % соответствия
                col1, col2 = st.columns(2)
                min_score = col1.number_input(
                    "% от", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                    key=f"min_score_{idx}"
                )
                max_score = col2.number_input(
                    "% до", min_value=0.0, max_value=100.0, value=100.0, step=0.1,
                    key=f"max_score_{idx}"
                )
                # Сортировка по score
                sort_asc = st.radio(
                    "Сортировать по % соответствия",
                    ["По убыванию", "По возрастанию"],
                    index=0,
                    key=f"filter_sort_{idx}"
                )

            # —— Запрос данных и сортировка ——————————————————————
            score_col = score_mapping[prof]
            conditions = ["%s = ANY(selected_professions)"]
            params = [prof]

            start_date, end_date = date_range
            conditions.append("uploaded_at::date BETWEEN %s AND %s")
            params += [start_date, end_date]
            if hr_emails:
                conditions.append("hr_email = ANY(%s)")
                params.append(hr_emails)
            if git_choice != "Любой":
                conditions.append("git_available = %s")
                params.append(git_choice == "Да")
            for comp in req1:
                conditions.append("%s = ANY(grade1)")
                params.append(comp)
            for comp in req2:
                conditions.append("%s = ANY(grade2)")
                params.append(comp)
            for comp in req3:
                conditions.append("%s = ANY(grade3)")
                params.append(comp)
            if sec_prof != "Не важно":
                conditions.append("%s = ANY(selected_professions)")
                params.append(sec_prof)
            conditions.append(f"{score_col} BETWEEN %s AND %s")
            params += [min_score, max_score]
            where_clause = " AND ".join(conditions)

            sql = f"""
                SELECT
                    id,
                    form_submitted_at,
                    uploaded_at,
                    sender_email,
                    name,
                    surname,
                    patronymic,
                    telegram_handle,
                    phone,
                    {score_col} AS score,
                    git_available,
                    selected_professions,
                    code,
                    hr_email,
                    grade0,
                    grade1,
                    grade2,
                    grade3,
                    original_filename
                FROM resume_records
                WHERE {where_clause};
            """
            conn = psycopg2.connect(host="localhost", dbname="resumes", user="appuser", password="duduki")
            df = pd.read_sql(sql, conn, params=params, parse_dates=["uploaded_at"]
            )
            conn.close()

            ascending = (sort_asc == "По возрастанию")
            df = df.sort_values("score", ascending=ascending).reset_index(drop=True)

            # —— Фильтрация и рассылка —————————————————————————
            filter_key = f"filter_passed_{idx}"
            if filter_key not in st.session_state:
                st.session_state[filter_key] = False
            col_thr, col_filter, col_send = st.columns([2,1,1])
            with col_thr:
                bulk_threshold = st.number_input(
                    "Порог % для массовой рассылки", min_value=0.0, max_value=100.0,
                    value=50.0, step=0.1, key=f"bulk_thr_{idx}"
                )
            with col_filter:
                if st.button("Показать прошедших", key=f"filter_btn_{idx}"):
                    st.session_state[filter_key] = True
            with col_send:
                bulk_send = st.button(f"📤 Отправить письма для «{prof}»", key=f"bulk_send_{idx}")

            if st.session_state[filter_key]:
                st.subheader(f"Кандидаты с соответствием ≥ {bulk_threshold}%")

                # 1) статистика — показываем сразу
                total   = len(df)
                passed  = len(df[df["score"] >= bulk_threshold])
                percent = (passed / total * 100) if total else 0.0
                c1, c2, c3 = st.columns(3)
                c1.metric("Всего кандидатов", total)
                c2.metric(f"Кандидатов ≥{bulk_threshold}%", passed)
                c3.metric("Доля прошедших", f"{percent:.1f}%")
                st.caption("Нажмите кнопку «📤 Отправить письма», чтобы разослать уведомления выбранным кандидатам.")

                # 2) затем сама таблица
                df_passed = df[df["score"] >= bulk_threshold].reset_index(drop=True)
                st.dataframe(df_passed, use_container_width=True)

            st.subheader(f"Все кандидаты по вакансии «{prof}»")
            st.dataframe(df, use_container_width=True)

            if bulk_send:
                sent_A = sent_B = skipped = 0
                for _, row in df.iterrows():
                    above = row["score"] >= bulk_threshold
                    ok = send_bulk_mail(row, prof, bulk_threshold, above)
                    if ok:
                        sent_A += above
                        sent_B += (not above)
                    else:
                        skipped += 1
                st.success(
                    f"✅ Письма отправлены:\n"
                    f"  Кандидаты, которые больше порога (≥{bulk_threshold}%): {sent_A}\n."
                    f"  Кандидаты, которые меньше порога (<{bulk_threshold}%): {sent_B}\n."
                    f"  Пропущено из-за ошибок/некорректных email: {skipped}.")

            st.markdown("### 📖 Описание полей таблицы")
            descriptions = {
                "id":                   "Уникальный идентификатор заявки",
                "form_submitted_at":    "Дата и время отправки формы кандидата",
                "uploaded_at":          "Дата и время загрузки резюме",
                "sender_email":         "Email кандидата",
                "name":                 "Имя кандидата",
                "surname":              "Фамилия кандидата",
                "patronymic":           "Отчество кандидата",
                "telegram_handle":      "Telegram-ник кандидата",
                "phone":                "Телефон кандидата (+7XXXXXXXXXX)",
                "score":                f"Процент соответствия профилю «{prof}»",
                "git_available":        "Наличие GitHub-ссылки",
                "selected_professions": "Профессии, выбранные кандидатом",
                "code":                 "Внутренний код заявки",
                "hr_email":             "Email HR-специалиста, принявшего заявку",
                "grade0":               "Компетенции с грейдом 0",
                "grade1":               "Грейды 1",
                "grade2":               "Грейды 2",
                "grade3":               "Грейды 3",
                "original_filename":    "Имя загруженного файла резюме"
            }
            items = list(descriptions.items())
            half = (len(items) + 1) // 2
            col1, col2 = st.columns(2)
            for key, txt in items[:half]: col1.markdown(f"**{key}** — {txt}")
            for key, txt in items[half:]: col2.markdown(f"**{key}** — {txt}")

# ─── Пятая вкладка — общий дашборд ─────────────────────────────────────────────
    with tabs[-1]:
        st.subheader("Общая сводка по всем профессиям")

        # 1. Загрузим все данные из БД
        conn = psycopg2.connect(host="localhost", dbname="resumes", user="appuser", password="duduki")
        df_all = pd.read_sql(
            "SELECT * FROM resume_records",
            conn,
            parse_dates=["uploaded_at", "form_submitted_at"]
        )
        conn.close()

        # 2. Ключевые метрики
        total_apps     = len(df_all)
        avg_all_scores = round(np.mean([df_all[col].mean() for col in score_mapping.values()]), 1)
        apps_last_7d   = int(df_all.set_index("form_submitted_at").last("7D").shape[0])
        git_share      = f"{round(df_all['git_available'].mean()*100, 1)}%"
        hr_count       = df_all['hr_email'].nunique()
        avg_delay_mins = round(((df_all['uploaded_at'] - df_all['form_submitted_at'])
                                .dt.total_seconds() / 60).mean(), 1)

        # 3. Вывод метрик сверху (шрифт можно кастомизировать через st.markdown + HTML/CSS)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Всего заявок",          f"{total_apps}",      help="Все сохранённые заявки")
        m2.metric("Средний % по профилям", f"{avg_all_scores}%", help="Среднее из всех профилей")
        m3.metric("За последние 7 дн.",    f"{apps_last_7d}",    help="Заявки за неделю")
        m4.metric("Доля с GitHub",         git_share,            help="Кандидаты с GitHub")
        m5.metric("Уникальных HR",         f"{hr_count}",        help="Число HR-специалистов")
        m6.metric("Средняя задержка",      f"{avg_delay_mins} мин", help="Минуты между формой и загрузкой")

        # 4. Подготовка данных для графиков
        prof_counts      = df_all.explode("selected_professions")["selected_professions"] \
                                .value_counts().reindex(profession_names, fill_value=0)
        avg_scores       = pd.Series({prof: df_all[col].mean() for prof, col in score_mapping.items()}) \
                            .reindex(profession_names)
        timeseries       = df_all.set_index("form_submitted_at").resample("D").size()
        avg_per_candidate= df_all[list(score_mapping.values())].mean(axis=1)
        all_grades       = pd.concat([df_all[f"grade{i}"].explode() for i in range(4)], ignore_index=True).dropna()
        top10_comps      = all_grades.value_counts().head(10)

        # 5. Цветовые палитры
        import matplotlib.cm as cm
        cmap_prof = cm.get_cmap("tab10", len(profession_names))
        colors_prof = {prof: cmap_prof(i) for i, prof in enumerate(profession_names)}
        cmap_comp = cm.get_cmap("tab20", len(top10_comps))
        colors_comp = {comp: cmap_comp(i) for i, comp in enumerate(top10_comps.index)}

        # 6. Разметка дашборда 2×2 + один снизу
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("### Количество заявок по профессиям")
            fig1, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)  # можно менять figsize
            xs, ys = prof_counts.index, prof_counts.values
            ax1.bar(xs, ys, color=[colors_prof[x] for x in xs])
            ax1.set_ylabel("Число заявок")
            ax1.tick_params(axis="x", rotation=45)
            st.pyplot(fig1)

        with r1c2:
            st.markdown("### Средний % соответствия по профилям")
            fig2, ax2 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            xs2, ys2 = avg_scores.index, avg_scores.values
            ax2.barh(xs2, ys2, color=[colors_prof[x] for x in xs2])
            ax2.set_xlabel("Средний %")
            st.pyplot(fig2)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("### Динамика поступления заявок")
            fig3, ax3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            ax3.plot(timeseries.index, timeseries.values, marker="o")
            ax3.set_xlabel("Дата")
            ax3.set_ylabel("Число заявок")
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)

        with r2c2:
            st.markdown("### Распределение среднего % соответствия")
            fig4, ax4 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            ax4.hist(avg_per_candidate, bins=20, color="#4c72b0")
            ax4.set_xlabel("Средний %")
            ax4.set_ylabel("Частота")
            st.pyplot(fig4)

        # Для каждого грейда берём первые 5 наиболее частых компетенций
        top5 = {
            i: df_all[f"grade{i}"].explode().value_counts().head(5)
            for i in range(4)
        }

        # 1) Собираем все компетенции из топ-5 каждого грейда
        all_top5 = []
        for i in range(4):
            all_top5 += top5[i].index.tolist()
        unique_comps = list(dict.fromkeys(all_top5))  # сохраняем порядок появления

        # 2) Берём одну палитру tab20 и создаём map: компетенция → цвет
        cmap = cm.get_cmap("tab20", len(unique_comps))
        color_map = {comp: cmap(idx) for idx, comp in enumerate(unique_comps)}

        # 3) Строим по две колонки на ряд
        row3 = st.columns(2)
        with row3[0]:
            st.markdown("### Топ-5 компетенций грейда 0")
            fig_g0, ax_g0 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            data0 = top5[0]
            comps0 = data0.index[::-1]
            vals0  = data0.values[::-1]
            ax_g0.barh(comps0, vals0, color=[color_map[c] for c in comps0])
            ax_g0.set_xlabel("Частота")
            ax_g0.tick_params(axis="y", rotation=45, labelsize=6)
            st.pyplot(fig_g0)

        with row3[1]:
            st.markdown("### Топ-5 компетенций грейда 1")
            fig_g1, ax_g1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            data1 = top5[1]
            comps1 = data1.index
            vals1  = data1.values
            ax_g1.bar(comps1, vals1, color=[color_map[c] for c in comps1])
            ax_g1.set_ylabel("Частота")
            ax_g1.tick_params(axis="x", rotation=45, labelsize=6)
            st.pyplot(fig_g1)

        row4 = st.columns(2)
        with row4[0]:
            st.markdown("### Топ-5 компетенций грейда 2")
            fig_g2, ax_g2 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            data2 = top5[2]
            comps2 = data2.index
            vals2  = data2.values
            ax_g2.bar(comps2, vals2, color=[color_map[c] for c in comps2])
            ax_g2.set_ylabel("Частота")
            ax_g2.tick_params(axis="x", rotation=45, labelsize=6)
            st.pyplot(fig_g2)

        with row4[1]:
            st.markdown("### Топ-5 компетенций грейда 3")
            fig_g3, ax_g3 = plt.subplots(figsize=(4, 3), constrained_layout=True)
            data3 = top5[3]
            comps3 = data3.index[::-1]
            vals3  = data3.values[::-1]
            ax_g3.barh(comps3, vals3, color=[color_map[c] for c in comps3])
            ax_g3.set_xlabel("Частота")
            ax_g3.tick_params(axis="y", rotation=45, labelsize=6)
            st.pyplot(fig_g3)