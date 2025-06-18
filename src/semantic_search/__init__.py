"""Semantic Search - Интеллектуальный поиск по документам"""

__version__ = "1.0.0"
__author__ = "Evgeny Odintsov"
__email__ = "ev1genial@gmail.com"

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SearchResult, SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer

__all__ = [
    "Doc2VecTrainer",
    "DocumentProcessor",
    "SemanticSearchEngine",
    "SearchResult",
    "TextSummarizer",
]
