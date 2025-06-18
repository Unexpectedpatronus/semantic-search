"""Основные модули для работы с документами и поиском"""

from .doc2vec_trainer import Doc2VecTrainer
from .document_processor import DocumentProcessor, ProcessedDocument
from .search_engine import SearchResult, SemanticSearchEngine
from .text_summarizer import TextSummarizer

__all__ = [
    "Doc2VecTrainer",
    "DocumentProcessor",
    "ProcessedDocument",
    "SemanticSearchEngine",
    "SearchResult",
    "TextSummarizer",
]
