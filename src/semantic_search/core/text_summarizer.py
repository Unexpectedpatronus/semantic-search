"""Модуль для экстрактивной суммаризации текста с использованием Doc2Vec"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from loguru import logger

from semantic_search.config import SUMMARIZATION_CONFIG
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.text_utils import TextProcessor

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec

try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn не установлен. Суммаризация будет недоступна")


def requires_sklearn(func):
    def wrapper(*args, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Для суммаризации требуется scikit-learn. Установите: pip install scikit-learn"
            )
        return func(*args, **kwargs)

    return wrapper


class TextSummarizer:
    """Класс для экстрактивной суммаризации текста"""

    def __init__(
        self, doc2vec_model: Optional[Doc2Vec] = None, config: Optional[dict] = None
    ):
        self.model = doc2vec_model
        self.config = config or SUMMARIZATION_CONFIG
        self.text_processor = TextProcessor()

    def set_model(self, model: Doc2Vec) -> None:
        self.model = model
        logger.info("Модель Doc2Vec установлена")

    @requires_sklearn
    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        """Возвращает список наиболее репрезентативных предложений"""
        if not self.model:
            raise ValueError("Модель Doc2Vec не установлена")

        sentences = self.text_processor.split_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences

        sentence_vectors = [
            self.model.infer_vector(self.text_processor.preprocess_text(s))
            for s in sentences
        ]
        doc_vector = np.mean(sentence_vectors, axis=0).reshape(1, -1)
        sent_matrix = np.stack(sentence_vectors)
        similarities = cosine_similarity(sent_matrix, doc_vector).flatten()

        top_indices = similarities.argsort()[-num_sentences:][::-1]
        summary = [sentences[i] for i in sorted(top_indices)]

        return summary

    def summarize_file(self, file_path: str) -> List[str]:
        extractor = FileExtractor()
        text = extractor.extract_text(Path(file_path))
        return self.summarize(text)

    def summarize_folder(self, folder_path: str) -> List[str]:
        extractor = FileExtractor()
        results = []
        for path in extractor.find_documents(Path(folder_path)):
            text = extractor.extract_text(path)
            summary = self.summarize(text)
            results.append(f"{path.name}:\n" + "\n".join(summary))
        return results
