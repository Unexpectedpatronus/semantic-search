"""Модуль поискового движка (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec
from loguru import logger

from semantic_search.config import CACHE_DIR, SEARCH_CONFIG
from semantic_search.utils.cache_manager import CacheManager
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
        self,
        model: Optional[Doc2Vec] = None,
        corpus_info: Optional[List] = None,
        documents_base_path: Optional[Path] = None,
    ):
        self.model = model
        self.corpus_info = corpus_info or []
        self.documents_base_path = documents_base_path  # Сохраняем базовый путь
        self.text_processor = TextProcessor()
        self.config = SEARCH_CONFIG
        self.cache_manager = CacheManager(CACHE_DIR)

        # Создаем индекс метаданных для быстрого доступа
        self._metadata_index = dict()
        if self.corpus_info:
            for tokens, doc_id, metadata in self.corpus_info:
                self._metadata_index[doc_id] = metadata

        if self.documents_base_path:
            logger.info(
                f"SearchEngine инициализирован с базовым путем: {self.documents_base_path}"
            )

    def set_model(
        self,
        model: Doc2Vec,
        corpus_info: Optional[List] = None,
        documents_base_path: Optional[Path] = None,
    ):
        """
        Установка модели для поиска

        Args:
            model: Обученная модель Doc2Vec
            corpus_info: Информация о корпусе
            documents_base_path: Базовый путь документов
        """
        self.model = model
        if corpus_info:
            self.corpus_info = corpus_info
            self._metadata_index = dict()
            for tokens, doc_id, metadata in corpus_info:
                self._metadata_index[doc_id] = metadata

        if documents_base_path:
            self.documents_base_path = documents_base_path
            logger.info(f"Установлен базовый путь: {self.documents_base_path}")

        logger.info("Поисковая модель установлена")

    def _search_base(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Базовая функция поиска

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

            # Препроцессор запроса
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

    def search_with_filters(
        self,
        query: str,
        top_k: Optional[int] = None,
        file_extensions: Optional[set] = None,
        date_range: Optional[tuple] = None,
        min_file_size: Optional[int] = None,
        max_file_size: Optional[int] = None,
    ) -> List[SearchResult]:
        """Поиск с фильтрами"""

        # Базовый поиск
        results = self._search_base(query, top_k=top_k or 100)

        # Применяем фильтры
        filtered_results = []
        for result in results:
            metadata = result.metadata

            # Фильтр по расширению
            if file_extensions and metadata.get("extension") not in file_extensions:
                continue

            # Фильтр по размеру файла
            file_size = metadata.get("file_size", 0)
            if min_file_size and file_size < min_file_size:
                continue
            if max_file_size and file_size > max_file_size:
                continue

            filtered_results.append(result)

            if len(filtered_results) >= (top_k or self.config["default_top_k"]):
                break

        return filtered_results

    def make_cache_key(
        self,
        query: str,
        top_k: Optional[int],
        file_extensions: Optional[set],
        date_range: Optional[tuple],
        min_file_size: Optional[int],
        max_file_size: Optional[int],
    ) -> str:
        """Генерация стабильного кэш-ключа для поискового запроса"""
        key_data = {
            "query": query.strip().lower(),
            "top_k": top_k,
            "file_extensions": sorted(file_extensions) if file_extensions else None,
            "date_range": date_range,
            "min_file_size": min_file_size,
            "max_file_size": max_file_size,
        }
        return json.dumps(key_data, sort_keys=True, ensure_ascii=False)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        file_extensions: Optional[set] = None,
        date_range: Optional[tuple] = None,
        min_file_size: Optional[int] = None,
        max_file_size: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Поиск с кэшированием и поддержкой фильтров

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            file_extensions: Фильтр по расширениям файлов
            date_range: Фильтр по дате (не используется, но оставлено для совместимости)
            min_file_size: Минимальный размер файла
            max_file_size: Максимальный размер файла

        Returns:
            Список результатов поиска
        """
        if not self.config.get("enable_caching", True):
            return self.search_with_filters(
                query,
                top_k=top_k,
                file_extensions=file_extensions,
                date_range=date_range,
                min_file_size=min_file_size,
                max_file_size=max_file_size,
            )

        # Генерируем стабильный ключ
        raw_key = self.make_cache_key(
            query, top_k, file_extensions, date_range, min_file_size, max_file_size
        )
        cache_key = f"search:{raw_key}"

        # Проверяем кэш
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Результат получен из кэша для запроса: {query}")
            return cached_result

        # Выполняем поиск
        results = self.search_with_filters(
            query,
            top_k=top_k,
            file_extensions=file_extensions,
            date_range=date_range,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
        )

        # Сохраняем в кэш
        self.cache_manager.set(cache_key, results)

        return results

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
