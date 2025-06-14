"""Semantic Document Search - приложение для семантического поиска по документам"""

__version__ = "0.1.0"
__author__ = "Evgeny Odintsov"
__email__ = "ev1genial@gmail.com"

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer

__all__ = [
    "DocumentProcessor",
    "Doc2VecTrainer",
    "SemanticSearchEngine",
    "TextSummarizer",
]
