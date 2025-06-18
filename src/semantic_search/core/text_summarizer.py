"""Модуль для суммаризации текстов"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec

import numpy as np
from loguru import logger

from semantic_search.config import SUMMARIZATION_CONFIG
from semantic_search.utils.text_utils import TextProcessor

try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn не установлен. Суммаризация будет недоступна")
    SKLEARN_AVAILABLE = False


class TextSummarizer:
    """Класс для экстрактивной суммаризации текстов"""

    def __init__(self, doc2vec_model: Optional[Doc2Vec] = None):
        self.model = doc2vec_model
        self.text_processor = TextProcessor()
        self.config = SUMMARIZATION_CONFIG

    def set_model(self, model: Doc2Vec):
        """Установка модели Doc2Vec"""
        self.model = model
        logger.info("Модель для суммаризации установлена")

    def _sentence_to_vector(self, sentence_tokens: List[str]) -> Optional[np.ndarray]:
        """
        Преобразование предложения в вектор

        Args:
            sentence_tokens: Токены предложения

        Returns:
            Векторное представление предложения
        """
        if self.model is None or not sentence_tokens:
            return None

        try:
            # Используем infer_vector для получения вектора предложения
            vector = self.model.infer_vector(sentence_tokens)
            return vector
        except Exception as e:
            logger.error(f"Ошибка при векторизации предложения: {e}")
            return None

    def _calculate_sentence_scores(
        self, sentences: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Вычисление оценок важности предложений методом TextRank

        Args:
            sentences: Список предложений

        Returns:
            Список кортежей (предложение, оценка)
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            # Fallback: простая оценка по длине предложения
            scored_sentences = []
            for sent in sentences:
                score = len(sent.split())  # Простая оценка по количеству слов
                scored_sentences.append((sent, float(score)))
            return scored_sentences

        # Получаем векторы для всех предложений
        sentence_vectors = []
        valid_sentences = []

        for sentence in sentences:
            tokens = self.text_processor.preprocess_text(sentence)
            if tokens:  # Проверяем, что есть значимые токены
                vector = self._sentence_to_vector(tokens)
                if vector is not None:
                    sentence_vectors.append(vector)
                    valid_sentences.append(sentence)

        if len(sentence_vectors) < 2:
            # Недостаточно предложений для анализа
            return [(sent, 1.0) for sent in valid_sentences]

        try:
            # Вычисляем матрицу схожести
            similarity_matrix = cosine_similarity(sentence_vectors)

            # Применяем простой алгоритм PageRank
            scores = self._pagerank_algorithm(similarity_matrix)

            # Сопоставляем оценки с предложениями
            scored_sentences = list(zip(valid_sentences, scores))

            return scored_sentences

        except Exception as e:
            logger.error(f"Ошибка при вычислении оценок предложений: {e}")
            # Fallback к простой оценке
            return [(sent, 1.0) for sent in valid_sentences]

    def _pagerank_algorithm(
        self, similarity_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100
    ) -> List[float]:
        """
        Упрощенный алгоритм PageRank для ранжирования предложений

        Args:
            similarity_matrix: Матрица схожести предложений
            damping: Коэффициент затухания
            max_iter: Максимальное количество итераций

        Returns:
            Список оценок для каждого предложения
        """
        n = similarity_matrix.shape[0]

        # Инициализация: равные веса для всех предложений
        scores = np.ones(n) / n

        # Создаем переходную матрицу
        # Заменяем нули на маленькое значение, чтобы избежать деления на ноль
        similarity_matrix = np.where(similarity_matrix == 0, 1e-8, similarity_matrix)

        # Нормализуем строки матрицы
        row_sums = similarity_matrix.sum(axis=1)
        transition_matrix = similarity_matrix / row_sums[:, np.newaxis]

        # Итеративный алгоритм PageRank
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * np.dot(
                transition_matrix.T, scores
            )

            # Проверяем сходимость
            if np.allclose(scores, new_scores, atol=1e-6):
                break

            scores = new_scores

        return scores.tolist()

    def summarize_text(
        self,
        text: str,
        sentences_count: Optional[int] = None,
        min_sentence_length: Optional[int] = None,
    ) -> List[str]:
        """
        Создание экстрактивной выжимки текста

        Args:
            text: Исходный текст
            sentences_count: Количество предложений в выжимке
            min_sentence_length: Минимальная длина предложения

        Returns:
            Список предложений выжимки
        """
        if not text.strip():
            logger.warning("Пустой текст для суммаризации")
            return []

        sentences_count = sentences_count or self.config["default_sentences_count"]
        min_sentence_length = min_sentence_length or self.config["min_sentence_length"]

        logger.info(
            f"Начинаем суммаризацию текста длиной {len(text)} символов (цель: {sentences_count} предложений)"
        )

        # Для очень длинных текстов используем упрощенный подход
        if len(text) > 1_000_000:
            logger.warning(
                f"Текст очень длинный ({len(text)} символов), используем упрощенный метод"
            )
            return self._summarize_long_text(text, sentences_count, min_sentence_length)

        sentences = self.text_processor.split_into_sentences(text)

        if not sentences:
            logger.warning("Не удалось разбить текст на предложения")
            return []

        if len(sentences) <= sentences_count:
            logger.info("Количество предложений меньше или равно требуемому")
            return sentences

        # Вычисляем оценки важности предложений
        scored_sentences = self._calculate_sentence_scores(sentences)

        # Сортируем по оценке (убывание)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Берем топ-N предложений
        top_sentences = scored_sentences[:sentences_count]

        # Восстанавливаем исходный порядок предложений
        original_sentences = sentences
        summary_sentences = []

        for original_sent in original_sentences:
            for summary_sent, score in top_sentences:
                if original_sent == summary_sent:
                    summary_sentences.append(original_sent)
                    break

        logger.info(f"Создана выжимка из {len(summary_sentences)} предложений")
        return summary_sentences

    def _summarize_long_text(
        self, text: str, sentences_count: int, min_sentence_length: int
    ) -> List[str]:
        """
        Упрощенная суммаризация для очень длинных текстов

        Args:
            text: Исходный текст
            sentences_count: Количество предложений
            min_sentence_length: Минимальная длина предложения

        Returns:
            Список предложений выжимки
        """
        # Разбиваем текст на части по 500K символов
        chunk_size = 500_000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        all_important_sentences = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Обработка части {i + 1}/{len(chunks)}")

            chunk_sentences = self.text_processor.split_into_sentences(chunk)

            if not chunk_sentences:
                continue

            # Для каждого чанка выбираем пропорциональное количество предложений
            chunk_sentence_count = max(1, sentences_count // len(chunks))
            if i == 0:  # Первый чанк может содержать больше важной информации
                chunk_sentence_count = max(2, chunk_sentence_count)

            # Простая эвристика: берем первые и последние предложения + самые длинные
            important_sentences = []

            # Первое предложение чанка
            if chunk_sentences:
                important_sentences.append(chunk_sentences[0])

            # Последнее предложение чанка
            if len(chunk_sentences) > 1:
                important_sentences.append(chunk_sentences[-1])

            # Самые длинные предложения (обычно более информативные)
            if len(chunk_sentences) > 2:
                sorted_by_length = sorted(chunk_sentences[1:-1], key=len, reverse=True)
                remaining_count = chunk_sentence_count - len(important_sentences)
                important_sentences.extend(sorted_by_length[:remaining_count])

            all_important_sentences.extend(important_sentences[:chunk_sentence_count])

        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_sentences = []
        for sent in all_important_sentences:
            if sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)

        # Возвращаем требуемое количество предложений
        return unique_sentences[:sentences_count]

    def summarize_file(self, file_path: str, **kwargs) -> List[str]:
        """
        Суммаризация файла

        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры для summarize_text

        Returns:
            Список предложений выжимки
        """
        from semantic_search.utils.file_utils import FileExtractor

        try:
            extractor = FileExtractor()
            text = extractor.extract_text(Path(file_path))

            if not text:
                logger.error(f"Не удалось извлечь текст из файла: {file_path}")
                return []

            return self.summarize_text(text, **kwargs)

        except Exception as e:
            logger.error(f"Ошибка при суммаризации файла {file_path}: {e}")
            return []

    def get_summary_statistics(self, original_text: str, summary: List[str]) -> dict:
        """
        Получение статистики суммаризации

        Args:
            original_text: Исходный текст
            summary: Выжимка

        Returns:
            Словарь со статистикой
        """
        original_sentences = self.text_processor.split_into_sentences(original_text)

        stats = {
            "original_sentences_count": len(original_sentences),
            "summary_sentences_count": len(summary),
            "compression_ratio": len(summary) / len(original_sentences)
            if original_sentences
            else 0,
            "original_chars_count": len(original_text),
            "summary_chars_count": sum(len(sent) for sent in summary),
            "chars_compression_ratio": sum(len(sent) for sent in summary)
            / len(original_text)
            if original_text
            else 0,
        }

        return stats
