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

# Рекомендации по компетенциям – по три курса/ресурса на каждый
recommendations = {
    "Определения, история развития и главные тренды ИИ": [
        "https://www.coursera.org/search?query=ai+history",
        "https://stepik.org/search?query=искусственный+интеллект",
        "https://www.youtube.com/results?search_query=ai+trends+history"
    ],
    "Процесс, стадии и методологии разработки решений на основе ИИ (Docker, Linux/Bash, Git)": [
        "https://www.coursera.org/search?query=ai+development+process",
        "https://stepik.org/search?query=docker+machine+learning",
        "https://www.youtube.com/results?search_query=machine+learning+devops+tutorial"
    ],
    "Статистические методы и первичный анализ данных": [
        "https://www.coursera.org/learn/statistics",
        "https://stepik.org/course/120",
        "https://www.youtube.com/results?search_query=statistics+for+data+analysis"
    ],
    "Промпт-инжиниринг": [
        "https://www.coursera.org/learn/prompt-engineering",
        "https://stepik.org/search?query=prompt+engineering",
        "https://www.youtube.com/results?search_query=prompt+engineering+tutorial"
    ],
    "Инструменты CitizenDS": [
        "https://stepik.org/course/536",
        "https://github.com/CitizenDS/CDS-Docs",
        "https://www.youtube.com/results?search_query=CitizenDS+tutorial"
    ],
    "Оценка качества работы методов ИИ": [
        "https://www.coursera.org/learn/machine-learning-model-evaluation",
        "https://stepik.org/search?query=model+evaluation",
        "https://www.youtube.com/results?search_query=model+evaluation+metrics"
    ],
    "Языки программирования и библиотеки (Python, C++)": [
        "https://www.coursera.org/specializations/python",
        "https://stepik.org/course/67",
        "https://www.youtube.com/results?search_query=python+programming+tutorial"
    ],
    "Этика ИИ": [
        "https://www.coursera.org/learn/ai-ethics",
        "https://stepik.org/search?query=этика+искусственного+интеллекта",
        "https://www.youtube.com/results?search_query=ai+ethics+course"
    ],
    "Безопасность ИИ": [
        "https://www.coursera.org/search?query=ai+security",
        "https://stepik.org/search?query=безопасность+искусственного+интеллекта",
        "https://www.youtube.com/results?search_query=ai+security+tutorial"
    ],
    "Цифровые двойники": [
        "https://www.coursera.org/search?query=digital+twins",
        "https://stepik.org/search?query=цифровой+двойник",
        "https://www.youtube.com/results?search_query=digital+twins+tutorial"
    ],
    "Методы машинного обучения": [
        "https://www.coursera.org/learn/machine-learning",
        "https://stepik.org/course/4852",
        "https://www.youtube.com/results?search_query=machine+learning+course"
    ],
    "Методы оптимизации": [
        "https://www.coursera.org/search?query=optimization+methods",
        "https://stepik.org/search?query=методы+оптимизации",
        "https://www.youtube.com/results?search_query=optimization+algorithms+tutorial"
    ],
    "Информационный поиск": [
        "https://www.coursera.org/search?query=information+retrieval",
        "https://stepik.org/search?query=информационный+поиск",
        "https://www.youtube.com/results?search_query=information+retrieval+course"
    ],
    "Рекомендательные системы": [
        "https://www.coursera.org/learn/recommender-systems",
        "https://stepik.org/search?query=рекомендательные+системы",
        "https://www.youtube.com/results?search_query=recommender+systems+tutorial"
    ],
    "Анализ изображений и видео": [
        "https://www.coursera.org/learn/computer-vision",
        "https://stepik.org/search?query=анализ+изображений",
        "https://www.youtube.com/results?search_query=computer+vision+tutorial"
    ],
    "Анализ естественного языка": [
        "https://www.coursera.org/learn/natural-language-processing",
        "https://stepik.org/search?query=обработка+естественного+языка",
        "https://www.youtube.com/results?search_query=nlp+course"
    ],
    "Основы глубокого обучения": [
        "https://www.coursera.org/specializations/deep-learning",
        "https://stepik.org/search?query=глубокое+обучение",
        "https://www.youtube.com/results?search_query=deep+learning+tutorial"
    ],
    "Глубокое обучение для анализа и генерации изображений, видео": [
        "https://www.coursera.org/learn/convolutional-neural-networks",
        "https://stepik.org/search?query=сверточные+нейронные+сети",
        "https://www.youtube.com/results?search_query=cnn+tutorial"
    ],
    "Глубокое обучение для анализа и генерации естественного языка": [
        "https://www.coursera.org/learn/nlp-sequence-models",
        "https://stepik.org/search?query=рекуррентные+нейронные+сети",
        "https://www.youtube.com/results?search_query=rnn+tutorial"
    ],
    "Обучение с подкреплением и глубокое обучение с подкреплением": [
        "https://www.coursera.org/learn/reinforcement-learning",
        "https://stepik.org/search?query=обучение+с+подкреплением",
        "https://www.youtube.com/results?search_query=reinforcement+learning+tutorial"
    ],
    "Гибридные модели и PIML": [
        "https://www.coursera.org/search?query=physics+informed+ml",
        "https://stepik.org/search?query=гибридные+модели",
        "https://www.youtube.com/results?search_query=physics+informed+ml+tutorial"
    ],
    "Анализ геоданных": [
        "https://www.coursera.org/learn/geographic-information-systems",
        "https://stepik.org/search?query=геоданные+анализ",
        "https://www.youtube.com/results?search_query=geospatial+analysis+tutorial"
    ],
    "Массово параллельные вычисления для ускорения машинного обучения (GPU)": [
        "https://www.coursera.org/learn/gpu-computing",
        "https://stepik.org/search?query=gpu+вычисления",
        "https://www.youtube.com/results?search_query=gpu+cuda+tutorial"
    ],
    "Работа с распределенной кластерной системой": [
        "https://www.coursera.org/search?query=distributed+systems",
        "https://stepik.org/search?query=распределенные+системы",
        "https://www.youtube.com/results?search_query=distributed+systems+tutorial"
    ],
    "Машинное обучение на больших данных": [
        "https://www.coursera.org/specializations/big-data-machine-learning",
        "https://stepik.org/search?query=машинное+обучение+большие+данные",
        "https://www.youtube.com/results?search_query=big+data+ml+tutorial"
    ],
    "Потоковая обработка данных (data streaming, event processing)": [
        "https://www.coursera.org/learn/streaming-data",
        "https://stepik.org/search?query=потоковая+обработка",
        "https://www.youtube.com/results?search_query=data+streaming+tutorial"
    ],
    "Графовые нейросети": [
        "https://www.coursera.org/learn/graph-neural-networks",
        "https://stepik.org/search?query=графовые+нейронные+сети",
        "https://www.youtube.com/results?search_query=graph+neural+networks+tutorial"
    ],
    "SQL базы данных (GreenPlum, Postgres, Oracle)": [
        "https://www.coursera.org/learn/structured-query-language",
        "https://stepik.org/course/662",
        "https://www.youtube.com/results?search_query=sql+tutorial"
    ],
    "NoSQL базы данных (Cassandra, MongoDB, ElasticSearch, Neo4j, Hbase)": [
        "https://www.coursera.org/specializations/nosql-databases",
        "https://stepik.org/search?query=NoSQL",
        "https://www.youtube.com/results?search_query=nosql+tutorial"
    ],
    "Массово параллельная обработка и анализ данных": [
        "https://www.coursera.org/specializations/parallel-programming",
        "https://stepik.org/search?query=параллельные+вычисления",
        "https://www.youtube.com/results?search_query=parallel+computing+tutorial"
    ],
    "Hadoop, SPARK, Hive": [
        "https://www.coursera.org/learn/hadoop",
        "https://stepik.org/search?query=Hadoop+Spark",
        "https://www.youtube.com/results?search_query=hadoop+spark+tutorial"
    ],
    "Шины данных (Kafka)": [
        "https://www.coursera.org/learn/confluent-kafka",
        "https://stepik.org/search?query=Kafka",
        "https://www.youtube.com/results?search_query=apache+kafka+tutorial"
    ],
    "Качество и предобработка данных, подходы и инструменты": [
        "https://www.coursera.org/learn/data-cleaning",
        "https://stepik.org/search?query=предобработка+данных",
        "https://www.youtube.com/results?search_query=data+preprocessing+tutorial"
    ],
    "Графы знаний и онтологии": [
        "https://www.coursera.org/learn/knowledge-graphs",
        "https://stepik.org/search?query=граф+знаний",
        "https://www.youtube.com/results?search_query=knowledge+graph+tutorial"
    ]
}