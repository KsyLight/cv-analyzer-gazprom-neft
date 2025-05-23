import numpy as np

# Задаем список компетенций (в том же порядке, что использовался при обучении модели)
competency_list = [
    "Определения, история развития и главные тренды ИИ",
    "Процесс, стадии и методологии разработки решений на основе ИИ (Docker, Linux/Bash, Git)",
    "Статистические методы и первичный анализ данных",
    "Промпт-инжиниринг",
    "Инструменты CitizenDS",
    "Оценка качества работы методов ИИ",
    "Языки программирования и библиотеки (Python, C++)",
    "Этика ИИ",
    "Безопасность ИИ",
    "Цифровые двойники",
    "Методы машинного обучения",
    "Методы оптимизации",
    "Информационный поиск",
    "Рекомендательные системы",
    "Анализ изображений и видео",
    "Анализ естественного языка",
    "Основы глубокого обучения",
    "Глубокое обучение для анализа и генерации изображений, видео",
    "Глубокое обучение для анализа и генерации естественного языка",
    "Обучение с подкреплением и глубокое обучение с подкреплением",
    "Гибридные модели и PIML",
    "Анализ геоданных",
    "Массово параллельные вычисления для ускорения машинного обучения (GPU)",
    "Работа с распределенной кластерной системой",
    "Машинное обучение на больших данных",
    "Потоковая обработка данных (data streaming, event processing)",
    "Графовые нейросети",
    "SQL базы данных (GreenPlum, Postgres, Oracle)",
    "NoSQL базы данных (Cassandra, MongoDB, ElasticSearch, Neo4j, Hbase)",
    "Массово параллельная обработка и анализ данных",
    "Hadoop, SPARK, Hive",
    "Шины данных (Kafka)",
    "Качество и предобработка данных, подходы и инструменты",
    "Графы знаний и онтологии"
]

profession_names = [
    "Аналитик данных",
    "Инженер данных",
    "Технический аналитик в ИИ",
    "Менеджер в ИИ"
]

# Матрица из изображения (34 строки по 4 профессии)
profession_matrix = np.array([
    [1, 1, 1, 1],
    [2, 2, 1, 1],
    [2, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 1, 1, 1],
    [2, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 2, 1, 1],
    [2, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [2, 1, 1, 1],
    [2, 1, 1, 1],
    [2, 2, 1, 1],
    [2, 2, 1, 1],
    [2, 2, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 2, 0, 0],
    [1, 2, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    [1, 3, 1, 1],
    [1, 3, 1, 1],
    [1, 2, 0, 0],
    [1, 2, 0, 0],
    [0, 0, 0, 0],
    [2, 3, 1, 1],
    [0, 0, 0, 0],
])

# Рекомендации по компетенциям – от 1 до 3 курсов/ресурсов на каждый
recommendations = {
    "Определения, история развития и главные тренды ИИ": [
       "https://artezio.ru/blog/razvitie-ii",
       "https://gb.ru/blog/razvitie-iskusstvennogo-intellekta/",
       "https://habr.com/ru/articles/861888/"
    ],
    "Процесс, стадии и методологии разработки решений на основе ИИ (Docker, Linux/Bash, Git)": [
        "https://karpov.courses/docker",
        "https://stepik.org/course/73/promo",
        "https://ru.hexlet.io/courses/bash"
    ],
    "Статистические методы и первичный анализ данных": [
        "https://openedu.ru/course/hse/STATMETODS_SPEC/",
        "https://www.specialist.ru/dictionary/definition/statistic-analysis",
        "https://www.hse.ru/edu/courses/835199376"
    ],
    "Промпт-инжиниринг": [
        "https://www.lektorium.tv/prompt-en",
        "https://stepik.org/course/204833/promo",
        "https://openedu.ru/course/hse/PROMPT_ENGINEERING/"
    ],
    "Инструменты CitizenDS": [
        "hhttps://www.techtarget.com/searchbusinessanalytics/definition/citizen-data-scientist",
        "https://wiki.loginom.ru/articles/citizen-datascience.html",
        "https://www.spotfire.com/glossary/what-is-a-citizen-data-scientist"
    ],
    "Оценка качества работы методов ИИ": [
        "https://stepik.org/course/204833/promo",
        "https://openedu.ru/course/hse/PROMPT_ENGINEERING/"
    ],
    "Языки программирования и библиотеки (Python, C++)": [
        "https://stepik.org/course/67/promo",
        "https://practicum.yandex.ru/cpp/",
        "https://stepik.org/course/7/promo"
    ],
    "Этика ИИ": [
        "https://stepik.org/course/61610/promo",
        "https://www.hse.ru/edu/courses/835167224",
        "https://stepik.org/course/118318/promo"
    ],
    "Безопасность ИИ": [
        "https://stepik.org/course/61610/promo",
        "https://www.hse.ru/edu/courses/835167224"
    ],
    "Цифровые двойники": [
        "https://openedu.ru/course/spbstu/DIGTWIN/"
    ],
    "Методы машинного обучения": [
        "https://practicum.yandex.ru/machine-learning/",
        "https://stepik.org/catalog/56",
        "https://karpov.courses/ml-start"
    ],
    "Методы оптимизации": [
        "https://stepik.org/course/91916/promo",
        "https://www.hse.ru/edu/courses/470890824",
        "https://www.hse.ru/edu/courses/835129036"
    ],
    "Информационный поиск": [
        "https://education.vk.company/program/informatsionnyy-poisk"
    ],
    "Рекомендательные системы": [
        "https://education.yandex.ru/handbook/ml/article/intro-recsys",
        "http://wiki.cs.hse.ru/%D0%A0%D0%B5%D0%BA%D0%BE%D0%BC%D0%B5%D0%BD%D0%B4%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5_%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D1%8B"
    ],
    "Анализ изображений и видео": [
        "https://www.lektorium.tv/course/22902",
        "https://www.hse.ru/edu/courses/570814531",
        "https://www.hse.ru/edu/courses/339561854"
    ],
    "Анализ естественного языка": [
        "https://blog.skillfactory.ru/glossary/nlp/",
        "https://aws.amazon.com/ru/what-is/nlp/"
    ],
    "Основы глубокого обучения": [
        "https://stepik.org/course/179805/promo",
        "https://karpov.courses/deep-learning",
        "https://stepik.org/course/230362/promo"
    ],
    "Глубокое обучение для анализа и генерации изображений, видео": [
        "https://karpov.courses/deep-learning",
        "https://stepik.org/course/230362/promo"
    ],
    "Глубокое обучение для анализа и генерации естественного языка": [
        "https://blog.skillfactory.ru/glossary/nlp/",
        "https://aws.amazon.com/ru/what-is/nlp/"
    ],
    "Обучение с подкреплением и глубокое обучение с подкреплением": [
        "https://stepik.org/course/189738/promo",
        "https://education.yandex.ru/handbook/ml/article/obuchenie-s-podkrepleniem"
    ],
    "Гибридные модели и PIML": [
        "https://www.specialist.ru/track/t-piml"
    ],
    "Анализ геоданных": [
        "https://ods.ai/tracks/geoanalytics-course-spring24",
        "https://softculture.cc/courses/architects/python-analytics",
        "https://netology.ru/programs/geoanalyst"
    ],
    "Массово параллельные вычисления для ускорения машинного обучения (GPU)": [
        "https://aws.amazon.com/ru/what-is/nlp/"
    ],
    "Работа с распределенной кластерной системой": [
        "https://edu.fors.ru/raspisanie-kursov/kurs/7089/",
        "https://intuit.ru/studies/courses/542/398/info"
    ],
    "Машинное обучение на больших данных": [
        "https://openedu.ru/course/hse/machine_learning_big_data/",
        "https://www.hse.ru/edu/courses/646526109",
        "https://practicum.yandex.ru/data-scientist/"
    ],
    "Потоковая обработка данных (data streaming, event processing)": [
        "https://bigdataschool.ru/courses/flink-stream-processing",
        "https://bigdataschool.ru/courses/apache-spark-structured-streaming",
        "https://stepik.org/course/183213/promo"
    ],
    "Графовые нейросети": [
        "https://education.yandex.ru/handbook/ml/article/grafovye-nejronnye-seti"
    ],
    "SQL базы данных (GreenPlum, Postgres, Oracle)": [
        "https://practicum.yandex.ru/sql-data-analyst/",
        "https://sql-academy.org/",
        "https://karpov.courses/simulator-sql"
    ],
    "NoSQL базы данных (Cassandra, MongoDB, ElasticSearch, Neo4j, Hbase)": [
        "https://otus.ru/lessons/cassandra/",
        "https://stepik.org/course/181467/promo",
        "https://coursehunter.net/course/nauchites-sozdavat-prilozheniya-s-pomoshchyu-neo4j"
    ],
    "Массово параллельная обработка и анализ данных": [
        "https://www.finam.ru/publications/item/massovaya-parallelnaya-obrabotka-20230629-0857/",
        "https://bigdataschool.ru/blog/what-is-mpp-greenplum.html"
    ],
    "Hadoop, SPARK, Hive": [
        "https://bigdataschool.ru/courses/hive-hadoop-sql-administrator",
        "https://stepik.org/course/115252/promo"
    ],
    "Шины данных (Kafka)": [
        "https://practicum.yandex.ru/kafka/"
    ],
    "Качество и предобработка данных, подходы и инструменты": [
        "https://practicum.yandex.ru/data-analyst/",
        "https://openedu.ru/course/hse/TEXT/"
    ],
    "Графы знаний и онтологии": [
        "https://ods.ai/tracks/kgcourse2021",
        "https://www.hse.ru/edu/courses/339556072",
        "https://kubsau.ru/upload/iblock/769/7698e43fcfcf8649fa4922d978f07b8f.pdf"
    ]
}

# Порог для бинаризации меток модели
THRESHOLD = 0.46269254347612143
# Файл учётных данных OAuth
CREDENTIALS_FILE = "client_secret_2_496304292584-focgmts10r0pc3cplngprpkiqshp5d2j.apps.googleusercontent.com.json"
# Файл для хранения токена доступа
TOKEN_FILE = "token.json"
# Область прав: отправка почты
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]