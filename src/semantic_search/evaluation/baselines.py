"""Базовые классы и альтернативные методы поиска"""

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from semantic_search.core.search_engine import SearchResult
from semantic_search.utils.text_utils import TextProcessor


class BaseSearchMethod(ABC):
    """Абстрактный базовый класс для методов поиска"""

    def __init__(self, name: str):
        self.name = name
        self.index_time = 0.0
        self.indexed_documents = []

    @abstractmethod
    def index(self, documents: List[Tuple[str, str, str]]) -> None:
        """
        Индексация документов

        Args:
            documents: Список кортежей (doc_id, text, metadata)
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск документов

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        pass

    def get_method_name(self) -> str:
        """Получить название метода"""
        return self.name

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику метода"""
        return {
            "method_name": self.name,
            "indexed_documents": len(self.indexed_documents),
            "index_time": self.index_time,
        }


class Doc2VecSearchAdapter(BaseSearchMethod):
    """Адаптер для существующего Doc2Vec поиска"""

    def __init__(self, search_engine, corpus_info):
        super().__init__("Doc2Vec")
        self.search_engine = search_engine
        self.corpus_info = corpus_info

    def index(self, documents: List[Tuple[str, str, str]]) -> None:
        """Doc2Vec уже проиндексирован при обучении"""
        self.indexed_documents = documents

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Использует существующий поисковый движок"""
        return self.search_engine.search(query, top_k=top_k)


class OpenAISearchBaseline(BaseSearchMethod):
    """Поиск с использованием OpenAI embeddings"""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"
    ):
        super().__init__(f"OpenAI ({model})")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embeddings = {}
        self.documents = {}
        self.text_processor = TextProcessor()

        if not self.api_key:
            raise ValueError(
                "OpenAI API key не найден. Установите переменную окружения OPENAI_API_KEY"
            )

        # Lazy import для опциональной зависимости
        try:
            import openai

            self.openai = openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Установите openai: pip install openai")

    def _get_embedding(self, text: str) -> List[float]:
        """Получить embedding для текста"""
        try:
            # Ограничиваем длину текста
            if len(text) > 8000:
                text = text[:8000]

            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Ошибка при получении embedding: {e}")
            raise

    def index(self, documents: List[Tuple[str, str, str]]) -> None:
        """
        Индексация документов через OpenAI API

        Args:
            documents: Список кортежей (doc_id, text, metadata)
        """
        start_time = time.time()
        logger.info(f"Начинаем индексацию {len(documents)} документов через OpenAI")

        for i, (doc_id, text, metadata) in enumerate(documents):
            try:
                # Создаем краткое представление документа для embedding
                # Берем первые 2000 символов + последние 1000
                if len(text) > 3000:
                    text_sample = text[:2000] + " ... " + text[-1000:]
                else:
                    text_sample = text

                # Получаем embedding
                embedding = self._get_embedding(text_sample)

                self.embeddings[doc_id] = np.array(embedding)
                self.documents[doc_id] = {"text": text, "metadata": metadata}

                if (i + 1) % 10 == 0:
                    logger.info(f"Проиндексировано {i + 1}/{len(documents)} документов")

                # Задержка для соблюдения rate limits
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Ошибка при индексации документа {doc_id}: {e}")
                continue

        self.indexed_documents = documents
        self.index_time = time.time() - start_time
        logger.info(f"Индексация завершена за {self.index_time:.2f} секунд")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск документов по запросу

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.embeddings:
            logger.error("Индекс пуст. Сначала проиндексируйте документы")
            return []

        try:
            # Получаем embedding запроса
            query_embedding = np.array(self._get_embedding(query))

            # Вычисляем косинусное сходство
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                # Косинусное сходство
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc_id, similarity))

            # Сортируем по убыванию сходства
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Создаем результаты
            results = []
            for doc_id, similarity in similarities[:top_k]:
                metadata = self.documents[doc_id].get("metadata", {})
                results.append(SearchResult(doc_id, float(similarity), metadata))

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def save_index(self, path: Path) -> None:
        """Сохранить индекс для повторного использования"""
        data = {
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "documents": self.documents,
            "model": self.model,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        logger.info(f"Индекс сохранен в {path}")

    def load_index(self, path: Path) -> None:
        """Загрузить индекс"""
        if not path.exists():
            raise FileNotFoundError(f"Файл индекса не найден: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.embeddings = {k: np.array(v) for k, v in data["embeddings"].items()}
        self.documents = data["documents"]
        self.indexed_documents = list(self.documents.keys())

        logger.info(f"Индекс загружен из {path}")
