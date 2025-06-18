"""Утилиты для обработки текста"""

import re
from typing import List

from loguru import logger

from semantic_search.config import SPACY_MODEL, TEXT_PROCESSING_CONFIG

# Глобальные переменные для ленивой загрузки
_nlp = None
_spacy_available = None
_initialization_attempted = False


def _initialize_spacy():
    """Ленивая инициализация SpaCy модели"""
    global _nlp, _spacy_available, _initialization_attempted

    if _initialization_attempted:
        return _nlp, _spacy_available

    _initialization_attempted = True

    try:
        import spacy
        from spacy.lang.ru import Russian

        # Попытка загрузки русской модели
        try:
            _nlp = spacy.load(SPACY_MODEL)
            # Увеличиваем лимит для длинных текстов
            _nlp.max_length = 3_000_000  # 3 миллиона символов
            _spacy_available = True
            logger.info(f"SpaCy модель {SPACY_MODEL} загружена")
        except OSError:
            logger.warning(
                f"SpaCy модель {SPACY_MODEL} не найдена. Используется базовая обработка"
            )
            _nlp = Russian()
            _nlp.max_length = 3_000_000
            _spacy_available = False

    except ImportError:
        logger.warning("SpaCy не установлен. Используется базовая обработка текста")
        _nlp = None
        _spacy_available = False

    return _nlp, _spacy_available


def check_spacy_model_availability() -> dict:
    """
    Проверка доступности SpaCy модели

    Returns:
        Словарь с информацией о состоянии модели
    """
    info = {
        "spacy_installed": False,
        "model_found": False,
        "model_loadable": False,
        "model_path": None,
        "error": None,
    }
    try:
        import spacy

        info["spacy_installed"] = True

        try:
            nlp = spacy.load(SPACY_MODEL)
            info["model_found"] = True
            info["model_loadable"] = True
            info["model_path"] = f"SpaCy model: {nlp.meta.get('lang', 'unknown')}"

        except OSError as e:
            info["error"] = f"Модель не найдена: {e}"

    except ImportError as e:
        info["error"] = f"SpaCy не установлен: {e}"

    return info


class TextProcessor:
    """Класс для препроцессинга текста"""

    def __init__(self):
        self.config = TEXT_PROCESSING_CONFIG
        self._nlp = None
        self._spacy_available = None
        self.max_chunk_size = 800_000

    def _get_nlp(self):
        """Получить SpaCy модель (с ленивой загрузкой)"""
        if self._nlp is None:
            self._nlp, self._spacy_available = _initialize_spacy()
        return self._nlp, self._spacy_available

    def get_spacy_status(self) -> str:
        """Получить статус SpaCy для отображения пользователю"""
        info = check_spacy_model_availability()

        if info["model_loadable"]:
            return f"✅ SpaCy: модель {SPACY_MODEL} работает"
        elif info["model_found"]:
            return "⚠️ SpaCy: модель найдена, но не загружается"
        elif info["spacy_installed"]:
            return f"❌ SpaCy: модель {SPACY_MODEL} не установлена"
        else:
            return "❌ SpaCy не установлен"

    def clean_text(self, text: str) -> str:
        """
        Базовая очистка текста

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        if not text:
            return ""

        text = re.sub(r'[^\w\s\-.,!?;:()\[\]""«»]+', " ", text, flags=re.UNICODE)

        text = re.sub(r"\s+", " ", text)

        words = text.split()
        words = [word for word in words if len(word) > 1]

        return " ".join(words).strip()

    def preprocess_basic(self, text: str) -> List[str]:
        """
        Базовая обработка текста без SpaCy

        Args:
            text: Исходный текст

        Returns:
            Список токенов
        """
        text = text.lower()

        tokens = text.split()

        tokens = [
            token
            for token in tokens
            if (len(token) >= self.config["min_token_length"] and token.isalpha())
        ]

        return tokens

    def preprocess_with_spacy(self, text: str) -> List[str]:
        """
        Продвинутая обработка текста с использованием SpaCy

        Args:
            text: Исходный текст

        Returns:
            Список обработанных токенов
        """
        nlp, spacy_available = self._get_nlp()

        if not nlp:
            return self.preprocess_basic(text)

        # Для очень длинных текстов обрабатываем по частям
        if len(text) > self.max_chunk_size:
            logger.info(
                f"Текст слишком длинный ({len(text)} символов), обрабатываем по частям"
            )
            tokens = []

            # Разбиваем текст на чанки
            for i in range(0, len(text), self.max_chunk_size):
                chunk = text[i : i + self.max_chunk_size]
                chunk_tokens = self._process_spacy_chunk(chunk, nlp, spacy_available)
                tokens.extend(chunk_tokens)

            return tokens
        else:
            return self._process_spacy_chunk(text, nlp, spacy_available)

    def _process_spacy_chunk(self, text: str, nlp, spacy_available: bool) -> List[str]:
        """Обработка одного чанка текста через SpaCy"""
        doc = nlp(text)
        tokens = []

        for token in doc:
            if (
                not token.is_punct
                and not token.is_space
                and not token.is_stop
                and len(token.text) >= self.config["min_token_length"]
                and token.text.isalpha()
            ):
                if self.config["lemmatize"] and spacy_available:
                    tokens.append(token.lemma_.lower())
                else:
                    tokens.append(token.text.lower())

        return tokens

    def preprocess_text(self, text: str) -> List[str]:
        """
        Главная функция препроцессинга текста

        Args:
            text: Исходный текст

        Returns:
            Список обработанных токенов
        """
        if not text:
            return []

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        _, spacy_available = self._get_nlp()

        if spacy_available:
            logger.debug("Используется SpaCy для препроцессинга")
            tokens = self.preprocess_with_spacy(cleaned_text)
        else:
            logger.debug("Используется базовый препроцессинг")
            tokens = self.preprocess_basic(cleaned_text)

        return tokens

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        if not text:
            return []

        nlp, spacy_available = self._get_nlp()
        min_sentence_length = self.config.get("min_sentence_length", 10)

        if spacy_available and nlp:
            # Для длинных текстов используем упрощенный метод
            if len(text) > self.max_chunk_size:
                logger.info("Используем упрощенное разбиение для длинного текста")
                return self._split_sentences_basic(text, min_sentence_length)

            try:
                # Используем SpaCy для точного разбиения
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                # Фильтруем слишком короткие предложения
                sentences = [
                    sent for sent in sentences if len(sent) >= min_sentence_length
                ]
                return sentences
            except Exception as e:
                logger.warning(f"Ошибка при разбиении через SpaCy: {e}")
                return self._split_sentences_basic(text, min_sentence_length)
        else:
            return self._split_sentences_basic(text, min_sentence_length)

    def _split_sentences_basic(self, text: str, min_length: int) -> List[str]:
        """Базовое разбиение текста на предложения"""

        sentences = re.split(r"[.!?]+(?:\s+|$)", text)

        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) >= min_length:
                if sent[-1] not in ".!?":
                    sent += "."
                result.append(sent)

        return result
