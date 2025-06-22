"""Модуль для суммаризации текстов"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec

import numpy as np
from loguru import logger

from semantic_search.config import SUMMARIZATION_CONFIG, TEXT_PROCESSING_CONFIG
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
        self.chunk_size = TEXT_PROCESSING_CONFIG.get("chunk_size", 500_000)

        # Минимальная длина предложения для включения в выжимку
        self.min_summary_sentence_length = self.config.get("min_sentence_length", 15)
        # Минимальное количество слов в предложении
        self.min_words_in_sentence = self.config.get("min_words_in_sentence", 5)

    def set_model(self, model: Doc2Vec):
        """Установка модели Doc2Vec"""
        self.model = model
        logger.info("Модель для суммаризации установлена")

    def _filter_sentence(self, sentence: str) -> bool:
        """
        Проверка, подходит ли предложение для включения в выжимку

        Args:
            sentence: Предложение для проверки

        Returns:
            True если предложение подходит, False если нужно отфильтровать
        """
        # Убираем лишние пробелы
        cleaned_sentence = sentence.strip()

        # Проверка минимальной длины в символах
        if len(cleaned_sentence) < self.min_summary_sentence_length:
            return False

        # Проверка минимального количества слов
        words = cleaned_sentence.split()
        if len(words) < self.min_words_in_sentence:
            return False

        # Проверка на наличие хотя бы одного значимого слова (не только предлоги/союзы)
        meaningful_words = [w for w in words if len(w) > 3]
        if len(meaningful_words) < 2:
            return False

        # Проверка на слишком много цифр (возможно, это таблица или список)
        digit_ratio = sum(c.isdigit() for c in cleaned_sentence) / len(cleaned_sentence)
        if digit_ratio > 0.5:
            return False

        # Проверка на повторяющиеся символы (например, "............")
        for char in cleaned_sentence:
            if cleaned_sentence.count(char * 5) > 0:  # 5 одинаковых символов подряд
                return False

        return True

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
        с учетом фильтрации коротких предложений

        Args:
            sentences: Список предложений

        Returns:
            Список кортежей (предложение, оценка)
        """
        # Фильтруем предложения перед оценкой
        filtered_sentences = []
        sentence_indices = []

        for i, sentence in enumerate(sentences):
            if self._filter_sentence(sentence):
                filtered_sentences.append(sentence)
                sentence_indices.append(i)

        if not filtered_sentences:
            logger.warning("Все предложения отфильтрованы как слишком короткие")
            # Возвращаем самые длинные предложения если все отфильтрованы
            sorted_by_length = sorted(
                enumerate(sentences), key=lambda x: len(x[1]), reverse=True
            )
            return [(sent, 1.0) for _, sent in sorted_by_length[:5]]

        if not SKLEARN_AVAILABLE or self.model is None:
            # Fallback: оценка по длине и позиции
            scored_sentences = []
            for i, sent in enumerate(filtered_sentences):
                # Учитываем длину и позицию (начало текста важнее)
                position_score = 1.0 - (sentence_indices[i] / len(sentences))
                length_score = min(
                    len(sent.split()) / 20, 1.0
                )  # Нормализуем по 20 словам
                score = position_score * 0.3 + length_score * 0.7
                scored_sentences.append((sent, score))
            return scored_sentences

        # Получаем векторы для отфильтрованных предложений
        sentence_vectors = []
        valid_sentences = []

        for sentence in filtered_sentences:
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
            min_sentence_length: Минимальная длина предложения (переопределяет настройки)

        Returns:
            Список предложений выжимки
        """
        if not text.strip():
            logger.warning("Пустой текст для суммаризации")
            return []

        sentences_count = sentences_count or self.config["default_sentences_count"]

        # Временно переопределяем минимальную длину если указана
        if min_sentence_length is not None:
            original_min_length = self.min_summary_sentence_length
            self.min_summary_sentence_length = min_sentence_length

        logger.info(
            f"Начинаем суммаризацию текста длиной {len(text)} символов (цель: {sentences_count} предложений)"
        )

        # Для очень длинных текстов используем упрощенный подход
        if len(text) > 1_000_000:
            logger.warning(
                f"Текст очень длинный ({len(text)} символов), используем упрощенный метод"
            )
            result = self._summarize_long_text(
                text, sentences_count, self.min_summary_sentence_length
            )

            # Восстанавливаем оригинальную настройку
            if min_sentence_length is not None:
                self.min_summary_sentence_length = original_min_length

            return result

        sentences = self.text_processor.split_into_sentences(text)

        if not sentences:
            logger.warning("Не удалось разбить текст на предложения")

            # Восстанавливаем оригинальную настройку
            if min_sentence_length is not None:
                self.min_summary_sentence_length = original_min_length

            return []

        # Фильтруем слишком короткие предложения перед проверкой
        valid_sentences = [s for s in sentences if self._filter_sentence(s)]

        if len(valid_sentences) <= sentences_count:
            logger.info(
                f"Количество подходящих предложений ({len(valid_sentences)}) меньше или равно требуемому"
            )

            # Восстанавливаем оригинальную настройку
            if min_sentence_length is not None:
                self.min_summary_sentence_length = original_min_length

            return valid_sentences

        # Вычисляем оценки важности предложений (уже отфильтрованных)
        scored_sentences = self._calculate_sentence_scores(sentences)

        # Сортируем по оценке (убывание)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Берем топ-N предложений
        top_sentences = scored_sentences[:sentences_count]

        # Восстанавливаем исходный порядок предложений
        summary_sentences = []

        # Создаем множество для быстрого поиска
        top_sentences_set = {sent for sent, _ in top_sentences}

        for original_sent in sentences:
            if original_sent in top_sentences_set:
                summary_sentences.append(original_sent)
                if len(summary_sentences) >= sentences_count:
                    break

        logger.info(f"Создана выжимка из {len(summary_sentences)} предложений")

        # Восстанавливаем оригинальную настройку
        if min_sentence_length is not None:
            self.min_summary_sentence_length = original_min_length

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

        chunks = [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]

        all_important_sentences = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Обработка части {i + 1}/{len(chunks)}")

            chunk_sentences = self.text_processor.split_into_sentences(chunk)

            if not chunk_sentences:
                continue

            # Фильтруем короткие предложения
            valid_chunk_sentences = [
                s for s in chunk_sentences if self._filter_sentence(s)
            ]

            if not valid_chunk_sentences:
                continue

            # Для каждого чанка выбираем пропорциональное количество предложений
            chunk_sentence_count = max(1, sentences_count // len(chunks))
            if i == 0:  # Первый чанк может содержать больше важной информации
                chunk_sentence_count = max(2, chunk_sentence_count)

            # Простая эвристика: берем первые и последние предложения + самые длинные
            important_sentences = []

            # Первое предложение чанка (если оно достаточно длинное)
            if valid_chunk_sentences:
                important_sentences.append(valid_chunk_sentences[0])

            # Последнее предложение чанка
            if len(valid_chunk_sentences) > 1:
                important_sentences.append(valid_chunk_sentences[-1])

            # Самые информативные предложения (длинные, но не слишком)
            if len(valid_chunk_sentences) > 2:
                # Сортируем по "информативности" - не слишком короткие и не слишком длинные
                middle_sentences = valid_chunk_sentences[1:-1]
                sorted_by_info = sorted(
                    middle_sentences,
                    key=lambda s: min(len(s.split()), 50),  # Оптимальная длина ~50 слов
                    reverse=True,
                )
                remaining_count = chunk_sentence_count - len(important_sentences)
                important_sentences.extend(sorted_by_info[:remaining_count])

            all_important_sentences.extend(important_sentences[:chunk_sentence_count])

        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_sentences = []
        for sent in all_important_sentences:
            if sent not in seen and self._filter_sentence(sent):
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

        # Считаем только валидные предложения в оригинале
        valid_original_sentences = [
            s for s in original_sentences if self._filter_sentence(s)
        ]

        stats = {
            "original_sentences_count": len(original_sentences),
            "valid_original_sentences_count": len(valid_original_sentences),
            "summary_sentences_count": len(summary),
            "compression_ratio": len(summary) / len(original_sentences)
            if original_sentences
            else 0,
            "valid_compression_ratio": len(summary) / len(valid_original_sentences)
            if valid_original_sentences
            else 0,
            "original_chars_count": len(original_text),
            "summary_chars_count": sum(len(sent) for sent in summary),
            "chars_compression_ratio": sum(len(sent) for sent in summary)
            / len(original_text)
            if original_text
            else 0,
            "avg_sentence_length": sum(len(sent.split()) for sent in summary)
            / len(summary)
            if summary
            else 0,
        }

        return stats
