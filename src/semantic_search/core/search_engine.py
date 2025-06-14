"""Модуль поискового движка"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from semantic_search.config import SEARCH_CONFIG
from semantic_search.core.doc2vec_trainer import Doc2Vec
from semantic_search.utils.text_utils import TextProcessor


class SearchResult:
    """Класс для представления результата поиска"""

    def __init__(self, doc_id: str, similarity: float, metadata: Optional[Dict] = None):
        self.doc_id = doc_id
        self.similarity = similarity
        self.metadata = metadata or {}
        self.file_path = Path(doc_id)  # doc_id это относительный путь

    def __repr__(self):
        return f"SearchResult(doc_id='{self.doc_id}', similarity={self.similarity:.3f})"


class SemanticSearchEngine:
    """Класс для семантического поиска по документам"""

    def __init__(
        self, model: Optional[Doc2Vec] = None, corpus_info: Optional[List] = None
    ):
        self.model = model
        self.corpus_info = corpus_info or []
        self.text_processor = TextProcessor()
        self.config = SEARCH_CONFIG

        # Создаем индекс метаданных для быстрого доступа
        self._metadata_index = dict()
        if self.corpus_info:
            for tokens, doc_id, metadata in self.corpus_info:
                self._metadata_index[doc_id] = metadata

    def set_model(self, model: Doc2Vec, corpus_info: Optional[List] = None):
        """
        Установка модели для поиска

        Args:
            model: Обученная модель Doc2Vec
            corpus_info: Информация о корпусе
        """
        self.model = model
        if corpus_info:
            self.corpus_info = corpus_info
            self._metadata_index = dict()
            for tokens, doc_id, metadata in corpus_info:
                self._metadata_index[doc_id] = metadata

        logger.info("Поисковая модель установлена")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Основная функция поиска

        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию из config)
            similarity_threshold: Минимальный порог схожести

        Returns:
            Список результатов поиска
        """
        if self.model is None:
            logger.error("Модель не загружена")
            return []

        if not query.strip():
            logger.warning("Пустой запрос")
            return []

        top_k = top_k or self.config["default_top_k"]
        similarity_threshold = (
            similarity_threshold or self.config["similarity_threshold"]
        )

        try:
            logger.info(f"Поиск по запросу: '{query}'")

            # Препроцессинг запроса
            query_tokens = self.text_processor.preprocess_text(query)

            if not query_tokens:
                logger.warning("Запрос не содержит значимых токенов")
                return []

            logger.info(
                f"Токены запроса: {query_tokens[:10]}..."
            )  # Показываем первые 10

            # Получаем вектор для запроса
            query_vector = self.model.infer_vector(query_tokens)

            # Ищем похожие документы
            similar_docs = self.model.dv.most_similar([query_vector], topn=top_k)

            # Фильтруем по порогу схожести и создаем результаты
            results = []
            for doc_id, similarity in similar_docs:
                if similarity >= similarity_threshold:
                    metadata = self._metadata_index.get(doc_id, {})
                    results.append(SearchResult(doc_id, similarity, metadata))

            logger.info(f"Найдено результатов: {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def search_similar_to_document(
        self, doc_id: str, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Поиск документов, похожих на указанный документ

        Args:
            doc_id: ID документа
            top_k: Количество результатов

        Returns:
            Список похожих документов
        """
        if self.model is None:
            logger.error("Модель не загружена")
            return []

        top_k = top_k or self.config["default_top_k"]

        try:
            if doc_id not in self.model.dv:
                logger.error(f"Документ не найден в модели: {doc_id}")
                return []

            # Получаем похожие документы
            similar_docs = self.model.dv.most_similar(
                doc_id, topn=top_k + 1
            )  # +1 чтобы исключить сам документ

            results = []
            for similar_doc_id, similarity in similar_docs:
                if similar_doc_id != doc_id:  # Исключаем сам документ
                    metadata = self._metadata_index.get(similar_doc_id, {})
                    results.append(SearchResult(similar_doc_id, similarity, metadata))

            return results[:top_k]  # Возвращаем только top_k результатов

        except Exception as e:
            logger.error(f"Ошибка при поиске похожих документов: {e}")
            return []

    def get_document_vector(self, doc_id: str) -> Optional[list]:
        """
        Получение вектора документа

        Args:
            doc_id: ID документа

        Returns:
            Вектор документа или None
        """
        if self.model is None or doc_id not in self.model.dv:
            return None

        return self.model.dv[doc_id].tolist()

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики поисковой системы

        Returns:
            Словарь со статистикой
        """
        if self.model is None:
            return {"status": "no_model"}

        return {
            "status": "ready",
            "documents_count": len(self.model.dv),
            "vocabulary_size": len(self.model.wv.key_to_index),
            "vector_size": self.model.vector_size,
            "indexed_documents": list(self.model.dv.key_to_index.keys())[
                :10
            ],  # Первые 10
        }
