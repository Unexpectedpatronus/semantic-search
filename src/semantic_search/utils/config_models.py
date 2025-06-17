"""Модели конфигурации"""

from dataclasses import dataclass
from typing import Set


@dataclass
class ProcessingConfig:
    """Конфигурация обработки текста"""

    min_text_length: int = 100
    max_text_length: int = 100000
    min_tokens_count: int = 10
    min_token_length: int = 2
    remove_stop_words: bool = True
    lemmatize: bool = True
    supported_extensions: Set[str] = None

    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = {".pdf", ".docx", ".doc"}


@dataclass
class ModelConfig:
    """Конфигурация модели Doc2Vec"""

    vector_size: int = 150
    window: int = 10
    min_count: int = 2
    epochs: int = 40
    workers: int = 4
    seed: int = 42


@dataclass
class SearchConfig:
    """Конфигурация поиска"""

    default_top_k: int = 10
    max_top_k: int = 50
    similarity_threshold: float = 0.1
    enable_caching: bool = True
    cache_size: int = 1000
