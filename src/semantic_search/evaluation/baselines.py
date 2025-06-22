"""Базовые классы и альтернативные методы поиска (обновленная версия с TF-IDF и BM25)"""

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        # Для Doc2Vec время индексации = время обучения, которое мы не измеряем здесь
        self.index_time = 0

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Использует существующий поисковый движок"""
        return self.search_engine.search(query, top_k=top_k)


class TFIDFSearchBaseline(BaseSearchMethod):
    """
    Поиск с использованием TF-IDF
    Классический метод информационного поиска для сравнения
    """

    def __init__(self):
        super().__init__("TF-IDF")
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Униграммы и биграммы
            min_df=2,  # Минимальная частота документа
            max_df=0.95,  # Максимальная частота документа
            sublinear_tf=True,  # Логарифмическое масштабирование TF
            use_idf=True,
            smooth_idf=True,
            lowercase=True,
            tokenizer=self._custom_tokenizer,  # Используем наш токенизатор
        )
        self.tfidf_matrix = None
        self.documents = {}
        self.doc_ids = []

    def _custom_tokenizer(self, text: str) -> List[str]:
        """Использование того же токенизатора, что и в Doc2Vec для честного сравнения"""
        # Используем базовую токенизацию без SpaCy для скорости
        return self.text_processor.preprocess_basic(text)

    def index(self, documents: List[Tuple[str, str, Any]]) -> None:
        """
        Индексация документов с TF-IDF

        Args:
            documents: Список кортежей (doc_id, text, metadata)
        """
        start_time = time.time()
        logger.info(f"Начинаем индексацию {len(documents)} документов через TF-IDF")

        # Извлекаем тексты и сохраняем метаданные
        texts = []
        self.doc_ids = []

        for doc_id, text, metadata in documents:
            texts.append(text)
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = {"text": text, "metadata": metadata}

        # Создаем TF-IDF матрицу
        logger.info("Построение TF-IDF матрицы...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        self.indexed_documents = documents
        self.index_time = time.time() - start_time

        logger.info(f"Индексация TF-IDF завершена за {self.index_time:.2f} секунд")
        logger.info(f"Размер словаря: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Размер матрицы: {self.tfidf_matrix.shape}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск документов по запросу с TF-IDF

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        if self.tfidf_matrix is None:
            logger.error("Индекс пуст. Сначала проиндексируйте документы")
            return []

        try:
            # Векторизуем запрос
            query_vector = self.vectorizer.transform([query])

            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Получаем топ-k результатов
            top_indices = similarities.argsort()[-top_k:][::-1]

            # Создаем результаты
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Фильтруем нулевые схожести
                    doc_id = self.doc_ids[idx]
                    metadata = self.documents[doc_id].get("metadata", {})
                    results.append(
                        SearchResult(doc_id, float(similarities[idx]), metadata)
                    )

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске TF-IDF: {e}")
            return []


class BM25SearchBaseline(BaseSearchMethod):
    """
    Поиск с использованием BM25 (Best Matching 25)
    Улучшенная версия TF-IDF, используется в Elasticsearch и других поисковых системах
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__("BM25")
        self.k1 = k1  # Параметр насыщения TF (обычно 1.2-2.0)
        self.b = b  # Параметр нормализации длины документа (обычно 0.75)
        self.text_processor = TextProcessor()
        self.documents = {}
        self.doc_ids = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}
        self.idf = {}
        self.doc_vectors = {}
        self.total_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        """Токенизация с использованием нашего процессора"""
        return self.text_processor.preprocess_basic(text)

    def _calculate_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        """Расчет IDF (Inverse Document Frequency) для всех термов"""
        from math import log

        N = len(documents)
        self.total_docs = N
        idf = {}

        # Подсчет документной частоты
        for doc_tokens in documents:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Расчет IDF по формуле BM25
        for token, df in self.doc_freqs.items():
            # Формула IDF для BM25: log((N - df + 0.5) / (df + 0.5))
            idf[token] = log((N - df + 0.5) / (df + 0.5))

        return idf

    def index(self, documents: List[Tuple[str, str, Any]]) -> None:
        """
        Индексация документов с BM25

        Args:
            documents: Список кортежей (doc_id, text, metadata)
        """
        start_time = time.time()
        logger.info(f"Начинаем индексацию {len(documents)} документов через BM25")

        # Токенизация и сохранение
        tokenized_docs = []
        self.doc_ids = []
        self.doc_lengths = []

        logger.info("Токенизация документов...")
        for i, (doc_id, text, metadata) in enumerate(documents):
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(tokens))

            self.documents[doc_id] = {
                "text": text,
                "metadata": metadata,
                "tokens": tokens,
            }

            # Создаем вектор документа (частоты термов)
            doc_vector = {}
            for token in tokens:
                doc_vector[token] = doc_vector.get(token, 0) + 1
            self.doc_vectors[doc_id] = doc_vector

            if (i + 1) % 10 == 0:
                logger.info(f"Обработано {i + 1}/{len(documents)} документов")

        # Средняя длина документа
        self.avgdl = np.mean(self.doc_lengths)
        logger.info(f"Средняя длина документа: {self.avgdl:.1f} токенов")

        # Расчет IDF
        logger.info("Расчет IDF...")
        self.idf = self._calculate_idf(tokenized_docs)

        self.indexed_documents = documents
        self.index_time = time.time() - start_time

        logger.info(f"Индексация BM25 завершена за {self.index_time:.2f} секунд")
        logger.info(f"Размер словаря: {len(self.idf)}")

    def _score_bm25(self, query_tokens: List[str], doc_id: str) -> float:
        """Расчет BM25 score для документа"""
        score = 0.0
        doc_vector = self.doc_vectors[doc_id]
        doc_length = len(self.documents[doc_id]["tokens"])

        for token in query_tokens:
            if token not in self.idf:
                continue

            # Частота терма в документе
            tf = doc_vector.get(token, 0)

            # IDF терма
            idf = self.idf[token]

            # BM25 формула
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Поиск документов по запросу с BM25

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов поиска
        """
        if not self.doc_ids:
            logger.error("Индекс пуст. Сначала проиндексируйте документы")
            return []

        try:
            # Токенизация запроса
            query_tokens = self._tokenize(query)

            if not query_tokens:
                logger.warning("Запрос не содержит значимых токенов")
                return []

            # Расчет scores для всех документов
            scores = []
            for doc_id in self.doc_ids:
                score = self._score_bm25(query_tokens, doc_id)
                if score > 0:  # Оптимизация: пропускаем нулевые scores
                    scores.append((doc_id, score))

            # Сортировка по убыванию score
            scores.sort(key=lambda x: x[1], reverse=True)

            # Нормализация scores в диапазон [0, 1]
            if scores:
                max_score = scores[0][1]
                if max_score > 0:
                    # Нормализуем через сигмоидную функцию для лучшей интерпретации
                    scores = [
                        (doc_id, 1 / (1 + np.exp(-score / (max_score * 0.5))))
                        for doc_id, score in scores
                    ]

            # Создаем результаты
            results = []
            for doc_id, score in scores[:top_k]:
                metadata = self.documents[doc_id].get("metadata", {})
                results.append(SearchResult(doc_id, float(score), metadata))

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске BM25: {e}")
            return []


# Для обратной совместимости с OpenAI baseline (опциональный)
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
