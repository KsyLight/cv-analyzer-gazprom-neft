import os
import base64
import logging
import random

import pandas as pd
import streamlit as st

from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils.constants import (
    competency_list,
    profession_matrix,
    profession_names,
    recommendations,
    THRESHOLD,
    CREDENTIALS_FILE,
    TOKEN_FILE,
    SCOPES)

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