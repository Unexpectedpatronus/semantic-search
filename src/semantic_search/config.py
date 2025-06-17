"""Расширенная конфигурация приложения"""

import json
import multiprocessing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = DATA_DIR / "cache"
CONFIG_DIR = PROJECT_ROOT / "config"

# Создаем директории
for dir_path in [DATA_DIR, MODELS_DIR, TEMP_DIR, LOGS_DIR, CACHE_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


@dataclass
class AppConfig:
    """Главная конфигурация приложения"""

    # Обработка текста
    text_processing: Dict[str, Any] = None

    # Модель Doc2Vec
    doc2vec: Dict[str, Any] = None

    # Поиск
    search: Dict[str, Any] = None

    # GUI
    gui: Dict[str, Any] = None

    # Суммаризация
    summarization: Dict[str, Any] = None

    # Производительность
    performance: Dict[str, Any] = None

    def __post_init__(self):
        if self.text_processing is None:
            self.text_processing = {
                "min_text_length": 100,
                "max_text_length": 500000,  # Увеличено до 500KB
                "min_tokens_count": 10,
                "min_token_length": 2,
                "remove_stop_words": True,
                "lemmatize": True,
                "max_file_size_mb": 50,
                "chunk_size": 10000,  # Для больших файлов
            }

        if self.doc2vec is None:
            self.doc2vec = {
                "vector_size": 150,
                "window": 10,
                "min_count": 2,
                "epochs": 40,
                "workers": max(1, multiprocessing.cpu_count() - 1),
                "seed": 42,
                "dm": 1,  # Distributed Memory
                "negative": 5,
                "hs": 0,  # Hierarchical Softmax
                "sample": 1e-4,
            }

        if self.search is None:
            self.search = {
                "default_top_k": 10,
                "max_top_k": 100,
                "similarity_threshold": 0.1,
                "enable_caching": True,
                "cache_size": 1000,
                "enable_filtering": True,
            }

        if self.gui is None:
            self.gui = {
                "window_title": "Semantic Document Search",
                "window_size": (1400, 900),
                "theme": "default",
                "font_size": 10,
                "enable_dark_theme": False,
                "auto_save_settings": True,
            }

        if self.summarization is None:
            self.summarization = {
                "default_sentences_count": 5,
                "max_sentences_count": 20,
                "min_sentence_length": 15,
                "use_textrank": True,
                "damping_factor": 0.85,
                "max_iterations": 100,
            }

        if self.performance is None:
            self.performance = {
                "enable_monitoring": True,
                "log_slow_operations": True,
                "slow_operation_threshold": 5.0,  # секунды
                "memory_warning_threshold": 80,  # процент
                "enable_profiling": False,
            }


class ConfigManager:
    """Менеджер конфигурации"""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or (CONFIG_DIR / "app_config.json")
        self._config = None

    @property
    def config(self) -> AppConfig:
        """Получение конфигурации с ленивой загрузкой"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> AppConfig:
        """Загрузка конфигурации из файла"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                logger.info(f"Конфигурация загружена из {self.config_file}")
                return AppConfig(**config_data)

            except Exception as e:
                logger.warning(
                    f"Ошибка загрузки конфигурации: {e}. Используется конфигурация по умолчанию"
                )

        # Создаем конфигурацию по умолчанию
        default_config = AppConfig()
        self.save_config(default_config)
        return default_config

    def save_config(self, config: AppConfig) -> bool:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(asdict(config), f, indent=2, ensure_ascii=False)

            logger.info(f"Конфигурация сохранена в {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
            return False

    def update_config(self, **kwargs) -> bool:
        """Обновление конфигурации"""
        try:
            config_dict = asdict(self.config)

            for key, value in kwargs.items():
                if key in config_dict:
                    if isinstance(config_dict[key], dict) and isinstance(value, dict):
                        config_dict[key].update(value)
                    else:
                        config_dict[key] = value

            self._config = AppConfig(**config_dict)
            return self.save_config(self._config)

        except Exception as e:
            logger.error(f"Ошибка обновления конфигурации: {e}")
            return False


# Глобальный менеджер конфигурации
config_manager = ConfigManager()

# Экспортируем для обратной совместимости
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
SPACY_MODEL = "ru_core_news_sm"
TEXT_PROCESSING_CONFIG = config_manager.config.text_processing
DOC2VEC_CONFIG = config_manager.config.doc2vec
SEARCH_CONFIG = config_manager.config.search
GUI_CONFIG = config_manager.config.gui
SUMMARIZATION_CONFIG = config_manager.config.summarization
