# 🤖 CV-Analyzer: Анализ компетенций по матрице Альянса ИИ
---
- Проект по хакатону от Газпром Нефти и СПбГЭУ
- Интерактивное веб-приложение для анализа резюме и определения уровня компетенций кандидатов по матрице профессий Альянса ИИ с последующим анализом соответствия профессиям (относительного и абсолютного), получения рекомендательных курсов (и т.п.), а также получением текста, на основе которого производился анализ.
- Интерактивное веб-приложение на базе Streamlit для автоматизированного анализа резюме и определения уровня владения ИИ-компетенциями. Основано на модели, обученной по матрице компетенций Альянса ИИ, с возможностью рекомендаций, визуализаций и интеграцией GitHub-профиля пользователя.
---

## 🔗 Демо

👉 [Веб-приложение](https://cv-analyzer-gazprom-neft.streamlit.app/)
![🌙 Dark Mode Recommended](https://img.shields.io/badge/theme-dark-blue?style=flat&logo=github) \
**Рекомендуем** использовать тёмную тему для наилучшего восприятия контента.

---

## Возможности

- Загрузка и анализ резюме (PDF, DOCX, TXT)
- Определение ИИ-компетенций на основе обученной модели
- Самостоятельная корректировка грейдов (0–3)
- Интеграция с GitHub: автоматический анализ `README.md` из репозиториев
- Визуализация соответствия профессиям по матрице
- Рекомендации по развитию слабых компетенций
- Возможность расширения с помощью интерпретируемости (LIME, Attention) — сейчас в веб-приложении отсутствует, ибо streamlit не выдерживает и падает

## Структура проекта

```plaintext
CV-ANALYZER-GAZPROM-NEFT/
├── .streamlit/                # Конфигурации Streamlit с доп. настройками
│   └── config.toml
├── others/                    # Прочие ресурсы проекта
├── utils/                     # Вспомогательные модули и логика
│   ├── __init__.py            # Инициализация пакета
│   ├── constants.py           # Список компетенций, матрица профессий и рекомендации
│   ├── cv_reader.py           # Извлечение и очистка текста из резюме
│   ├── github_reader.py       # Поиск и парсинг GitHub-профиля пользователя
│   ├── explanation.py         # Интерпретируемость модели (LIME, Attention)
│   ├── model_service.py       # Загрузка модели, предсказания с учётом threshold
│   └── resume_processor.py    # Объединение резюме + GitHub текстов
├── app.py                     # Основной Streamlit-приложение
├── requirements.txt           # Список зависимостей Python
└── README.md                  # Документация проекта
```

## Requirements

```bash
streamlit
transformers
huggingface-hub
torch
matplotlib
seaborn
scikit-learn
pdfminer.six
python-docx
requests
numpy
mplcyberpunk
lime
```
---

## Команда

## Наша команда

## Наша команда

## Наша команда

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/KsyLight.png" width="100" alt="Егор"/>
      <p><strong>Егор</strong><br/><a href="https://github.com/KsyLight">KsyLight</a></p>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/Akim-norfeg.png" width="100" alt="Аким"/>
      <p><strong>Аким</strong><br/><a href="https://github.com/Akim-norfeg">Akim-norfeg</a></p>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/Swagozavr.png" width="100" alt="Максим"/>
      <p><strong>Максим</strong><br/><a href="https://github.com/Swagozavr">Swagozavr</a></p>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/klevkina.png" width="100" alt="Катя"/>
      <p><strong>Катя</strong><br/><a href="https://github.com/klevkina">klevkina</a></p>
    </td>
    <td align="center">
      <img src="https://via.placeholder.com/100?text=%D0%90%D0%BD%D1%8F" width="100" alt="Аня"/>
      <p><strong>Аня</strong><br/><em>ссылка пока нет</em></p>
    </td>
  </tr>
</table>
