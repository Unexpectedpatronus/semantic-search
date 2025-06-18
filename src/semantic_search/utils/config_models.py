"""Модели конфигурации для обработки текста, обучения и поиска"""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class ProcessingConfig:
    """Конфигурация предобработки текста"""

    min_text_length: int = 100
    max_text_length: int = 100000
    min_tokens_count: int = 10
    min_token_length: int = 2
    remove_stop_words: bool = True
    lemmatize: bool = True
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".pdf", ".docx", ".doc"}
    )


@dataclass
class ModelConfig:
    """Конфигурация для обучения модели Doc2Vec"""

    vector_size: int = 150
    window: int = 10
    min_count: int = 2
    epochs: int = 40
    workers: int = 4
    seed: int = 42


@dataclass
class SearchConfig:
    """Конфигурация параметров семантического поиска"""

    default_top_k: int = 10
    max_top_k: int = 50
    similarity_threshold: float = 0.1
    enable_caching: bool = True
    cache_size: int = 1000
