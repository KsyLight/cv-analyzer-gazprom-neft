# 🤖 CV-Analyzer: Анализ компетенций по матрице Альянса ИИ
---
- Проект по хакатону от Газпром Нефти и СПбГЭУ
- Интерактивное веб-приложение на базе Streamlit для автоматизированного анализа резюме и определения уровня владения ИИ-компетенциями. Основано на модели, обученной по матрице компетенций Альянса ИИ, с возможностью персонализированных рекомендаций, визуализации соответствия профессиям и интеграцией GitHub-профиля пользователя.
---
## 🔗 Ссылки

[Веб-приложение](https://cv-analyzer-gazprom-neft.streamlit.app/)
[![🌙 Dark Mode Recommended](https://img.shields.io/badge/theme-dark-blue?style=flat&logo=github)](https://cv-analyzer-gazprom-neft.streamlit.app/)
> **Рекомендуем** использовать **тёмную тему** для наилучшего восприятия контента.

[Документация в Google Docs](https://docs.google.com/document/d/1lgbiqXAzj9J_sWFw-ep4w4qTQOpA2A_-5ieAzwoP62M/edit?tab=t.0)
[![Google Docs](https://img.shields.io/badge/Google%20Docs-blue?style=flat)](https://docs.google.com/document/d/1lgbiqXAzj9J_sWFw-ep4w4qTQOpA2A_-5ieAzwoP62M/edit?tab=t.0)

[Презентация по хакатону]()
[![PowerPoint](https://img.shields.io/badge/PowerPoint-darkorange?style=flat&logo=microsoft-powerpoint)](https://www.office.com/launch/powerpoint?auth=2&appid=ВАШ_ИД_ПРЕЗЕНТАЦИИ)

---

## Возможности

- Загрузка и анализ резюме (PDF, DOCX, TXT)
- Определение ИИ-компетенций на основе обученной модели
- Самостоятельная корректировка грейдов (0–3)
- Интеграция с GitHub: автоматический анализ `README.md` из репозиториев
- Визуализация соответствия профессиям по матрице
- Рекомендации по развитию слабых компетенций
- Возможность расширения с помощью интерпретируемости (LIME, Attention) — сейчас в веб-приложении отсутствует, ибо streamlit не выдерживает и падает

## Проблематика и целевая аудитория

<table style="table-layout: fixed; width: 100%; border-collapse: collapse;" border="1">
  <colgroup>
    <col style="width: 50%;" />
    <col style="width: 50%;" />
  </colgroup>
  <thead>
    <tr>
      <th><strong>Проблематика</strong></th>
      <th><strong>Целевая аудитория</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <strong>Сложности с подбором IT‑специалистов</strong><br>
        Компании не успевают находить кандидатов с актуальными навыками.
      </td>
      <td>
        • Руководители отделов<br>
        • Технические интервьюеры
      </td>
    </tr>
    <tr>
      <td>
        <strong>Разрыв образования и рынка</strong><br>
        Учебные программы вузов и онлайн‑курсов отстают от трендов Big Data, ИИ, DevOps.
      </td>
      <td>
        • HR‑специалисты и рекрутеры<br>
        • Аналитики по подбору персонала
      </td>
    </tr>
    <tr>
      <td>
        <strong>Необходимость автоматизации</strong><br>
        При большом потоке резюме сложно оперативно отобрать лучших кандидатов.
      </td>
      <td></td>
    </tr>
  </tbody>
</table>

<br/>

## Функции и преимущества

<table style="table-layout: fixed; width: 100%; border-collapse: collapse;" border="1">
  <colgroup>
    <col style="width: 50%;" />
    <col style="width: 50%;" />
  </colgroup>
  <thead>
    <tr>
      <th><strong>Функции</strong></th>
      <th><strong>Преимущества</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>• Сопоставление компетенций кандидатов с требованиями вакансий</td>
      <td>• Снижение времени на отбор кандидатов</td>
    </tr>
    <tr>
      <td>• Генерация персональных карьерных рекомендаций</td>
      <td>• Повышение точности и релевантности подбора</td>
    </tr>
    <tr>
      <td>• Автоматизация обратной связи кандидатам и сотрудникам</td>
      <td>• Сокращение затрат на внешних рекрутеров</td>
    </tr>
    <tr>
      <td>• Анализ и прогнозирование карьерных траекторий сотрудников</td>
      <td>• Оптимизация обучения и повышения квалификации</td>
    </tr>
    <tr>
      <td>• Предложение курсов и сертификаций для устранения «skill‑gap»</td>
      <td>• Повышение качества обратной связи<br>• Улучшение долгосрочной карьерной стратегии сотрудников</td>
    </tr>
  </tbody>
</table>

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

## Наша команда

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/KsyLight.png?size=100" width="100" alt="Егор"/>
      <p><strong>Егор</strong><br/><a href="https://github.com/KsyLight">KsyLight</a></p>
    </td>
    <td align="center">
      <img src="https://github.com/Akim-norfeg.png?size=100" width="100" alt="Аким"/>
      <p><strong>Аким</strong><br/><a href="https://github.com/Akim-norfeg">Akim-norfeg</a></p>
    </td>
    <td align="center">
      <img src="https://github.com/Swagozavr.png?size=100" width="100" alt="Максим"/>
      <p><strong>Максим</strong><br/><a href="https://github.com/Swagozavr">Swagozavr</a></p>
    </td>
    <td align="center">
      <img src="https://github.com/klevkina.png?size=100" width="100" alt="Катя"/>
      <p><strong>Катя</strong><br/><a href="https://github.com/klevkina">klevkina</a></p>
    </td>
    <td align="center">
      <img src="https://github.com/kwarkw.png?size=100" width="100" alt="Аня"/>
      <p><strong>Аня</strong><br/><a href="https://github.com/kwarkw">kwarkw</a></p>
    </td>
  </tr>
</table>
