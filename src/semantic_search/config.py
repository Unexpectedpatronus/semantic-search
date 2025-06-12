"""Конфигурация приложения"""

import multiprocessing
from pathlib import Path
from typing import Any, Dict

# Пути проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"

# Создаем необходимые директории
for dir_path in [DATA_DIR, MODELS_DIR, TEMP_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Поддерживаемые форматы файлов
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

# Языковая модель SpaCy
SPACY_MODEL = "ru_core_news_sm"

# Настройки обработки текста
TEXT_PROCESSING_CONFIG: Dict[str, Any] = {
    "min_text_length": 100,
    "max_text_length": 100000,
    "min_tokens_count": 10,
    "min_token_length": 2,
    "remove_stop_words": True,
    "lemmatize": True,
}

# Настройки модели Doc2Vec
DOC2VEC_CONFIG: Dict[str, Any] = {
    "vector_size": 150,
    "window": 10,
    "min_count": 2,
    "epochs": 40,
    "workers": max(1, multiprocessing.cpu_count() - 1),
    "seed": 42,
}

# Настройки GUI
GUI_CONFIG: Dict[str, Any] = {
    "window_title": "Semantic Document Search",
    "window_size": (1200, 800),
    "theme": "default",
}

# Настройки поиска
SEARCH_CONFIG: Dict[str, Any] = {
    "default_top_k": 10,
    "max_top_k": 50,
    "similarity_threshold": 0.1,
}

# Настройки суммаризации
SUMMARIZATION_CONFIG: Dict[str, Any] = {
    "default_sentences_count": 5,
    "max_sentences_count": 20,
    "min_sentence_length": 10,
}
