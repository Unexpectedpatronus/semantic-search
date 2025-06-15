"""Утилиты для обработки текста"""

import re

from loguru import logger

from semantic_search.config import SPACY_MODEL, TEXT_PROCESSING_CONFIG

try:
    import spacy
    from spacy.lang.ru import Russian

    # Попытка загрузки модели
    try:
        nlp = spacy.load(SPACY_MODEL)
        SPACY_AVAILABLE = True
        logger.info(f"SpaCy модель {SPACY_MODEL} загружена")
    except OSError:
        logger.warning(
            f"SpaCy модель {SPACY_MODEL} не найдена. Будет использована базовая обработка"
        )
        nlp = Russian()
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("SpaCy не установлен. Будет использована базовая обработка текста")
    nlp = None
    SPACY_AVAILABLE = False


class TextProcessor:
    """Класс для препроцессинга текста"""

    def __init__(self):
        self.config = TEXT_PROCESSING_CONFIG
        self.nlp = nlp

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

        # Удаляем специальные символы (оставляем буквы, цифры, знаки препинания)
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]""«»]+', " ", text, flags=re.UNICODE)

        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r"\s+", " ", text)

        # Удаляем слишком короткие "слова" (вероятно, мусор)
        words = text.split()
        words = [word for word in words if len(word) > 1]

        return " ".join(words).strip()

    def preprocess_with_spacy(self, text: str) -> list[str]:
        """
        Продвинутая обработка текста с использованием SpaCy

        Args:
            text: Исходный текст

        Returns:
            Список обработанных токенов
        """
        if not self.nlp:
            return self.preprocess_basic(text)

        doc = self.nlp(text)
        tokens = []

        for token in doc:
            # Фильтруем токены
            if (
                not token.is_punct
                and not token.is_space
                and not token.is_stop
                and len(token.text) >= self.config["min_token_length"]
                and token.text.isalpha()
            ):
                # Используем лемму (базовую форму слова)
                if self.config["lemmatize"] and SPACY_AVAILABLE:
                    tokens.append(token.lemma_.lower())
                else:
                    tokens.append(token.text.lower())

        return tokens

    def preprocess_basic(self, text: str) -> list[str]:
        """
        Базовая обработка текста без SpaCy

        Args:
            text: Исходный текст

        Returns:
            Список токенов
        """
        text = text.lower()

        # Простая токенизация по пробелам
        tokens = text.split()

        # Базовая фильтрация
        tokens = [
            token
            for token in tokens
            if (len(token) >= self.config["min_token_length"] and token.isalpha())
        ]

        return tokens

    def preprocess_text(self, text: str) -> list[str]:
        """
        Главная функция препроцессинга текста

        Args:
            text: Исходный текст

        Returns:
            Список обработанных токенов
        """
        if not text:
            return []

        # Базовая очистка
        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        # Выбираем метод обработки в зависимости от доступности SpaCy
        if SPACY_AVAILABLE:
            logger.debug("Используется SpaCy для препроцессинга")
            tokens = self.preprocess_with_spacy(cleaned_text)
        else:
            logger.debug("Используется базовый препроцессинг")
            tokens = self.preprocess_basic(cleaned_text)

        return tokens

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Разбиение текста на предложения

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        if not text:
            return []

        if SPACY_AVAILABLE and self.nlp:
            # Используем SpaCy для точного разбиения
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            # Фильтруем слишком короткие предложения
            sentences = [
                sent
                for sent in sentences
                if len(sent) >= self.config.get("min_sentence_length", 10)
            ]
            return sentences
        else:
            # Базовое разбиение по точкам
            sentences = re.split(r"[.!?]+", text)
            sentences = [
                sent.strip()
                for sent in sentences
                if sent.strip() and len(sent.strip()) >= 10
            ]
            return sentences
