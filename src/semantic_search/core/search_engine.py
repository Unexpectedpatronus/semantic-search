"""Модуль семантического поискового движка"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from semantic_search.config import CACHE_DIR, SEARCH_CONFIG
from semantic_search.utils.cache_manager import CacheManager
from semantic_search.utils.text_utils import TextProcessor

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec


@dataclass
class SearchResult:
    doc_id: str
    similarity: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }


class SemanticSearchEngine:
    """Класс для выполнения семантического поиска по векторной модели"""

    def __init__(
        self,
        model: Doc2Vec,
        cache_path: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        if model is None:
            raise ValueError("Модель Doc2Vec не может быть None")

        self.model = model
        self.config = config or SEARCH_CONFIG
        self.text_processor = TextProcessor()
        self.cache = CacheManager(cache_path or CACHE_DIR)

        logger.info("SemanticSearchEngine инициализирован")

    def _infer_vector(self, query: str) -> Optional[List[float]]:
        if not query.strip():
            logger.warning("Пустой запрос. Вектор не будет рассчитан.")
            return None

        tokens = self.text_processor.preprocess_text(query)
        if not tokens:
            logger.warning(
                "Не удалось токенизировать запрос. Результаты могут быть пустыми."
            )
            return None

        return self.model.infer_vector(tokens)

    def search(self, query: str, topn: int = 10) -> List[SearchResult]:
        """Поиск наиболее похожих документов по запросу"""
        vector = self._infer_vector(query)
        if vector is None:
            return []

        try:
            similar_docs = self.model.dv.most_similar([vector], topn=topn)
        except Exception as e:
            logger.error(f"Ошибка при поиске похожих документов: {e}")
            return []

        results: List[SearchResult] = []
        for doc_id, similarity in similar_docs:
            metadata = self._load_metadata(doc_id)
            results.append(
                SearchResult(doc_id=doc_id, similarity=similarity, metadata=metadata)
            )

        return results

    def _load_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Загрузка метаданных для найденного документа из кэша"""
        try:
            data = self.cache.get(doc_id)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Не удалось загрузить метаданные для {doc_id}: {e}")
            return {}

    def to_serializable_results(
        self, results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in results]
