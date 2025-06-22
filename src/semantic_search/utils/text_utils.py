# В файле src/semantic_search/utils/text_utils.py
# Обновленный TextProcessor с поддержкой двух языков

import re
from typing import List, Optional, Tuple

from loguru import logger

from semantic_search.config import SPACY_MODELS, TEXT_PROCESSING_CONFIG

# Глобальные переменные для ленивой загрузки
_nlp_ru = None
_nlp_en = None
_spacy_available = None
_initialization_attempted = False


def check_spacy_model_availability() -> dict:
    """
    Проверка доступности SpaCy моделей

    Returns:
        Словарь с информацией о состоянии моделей
    """
    info = {
        "spacy_installed": False,
        "ru_model_found": False,
        "en_model_found": False,
        "ru_model_loadable": False,
        "en_model_loadable": False,
        "models_info": {},
        "error": None,
    }

    try:
        import spacy

        info["spacy_installed"] = True

        # Проверка русской модели
        ru_model = SPACY_MODELS.get("ru", "ru_core_news_sm")
        try:
            nlp_ru = spacy.load(ru_model)
            info["ru_model_found"] = True
            info["ru_model_loadable"] = True
            info["models_info"]["ru"] = ru_model
        except OSError:
            info["error"] = f"Русская модель '{ru_model}' не найдена"

        # Проверка английской модели
        en_model = SPACY_MODELS.get("en", "en_core_web_sm")
        try:
            nlp_en = spacy.load(en_model)
            info["en_model_found"] = True
            info["en_model_loadable"] = True
            info["models_info"]["en"] = en_model
        except OSError:
            if not info["error"]:
                info["error"] = f"Английская модель '{en_model}' не найдена"
            else:
                info["error"] += f", английская модель '{en_model}' не найдена"

    except ImportError as e:
        info["error"] = f"SpaCy не установлен: {e}"

    return info


def _initialize_spacy() -> Tuple[Optional[object], Optional[object], bool]:
    """Ленивая инициализация SpaCy моделей для русского и английского"""
    global _nlp_ru, _nlp_en, _spacy_available, _initialization_attempted

    if _initialization_attempted:
        return _nlp_ru, _nlp_en, _spacy_available

    _initialization_attempted = True

    try:
        import spacy

        # Загружаем русскую модель
        ru_model = SPACY_MODELS.get("ru", "ru_core_news_sm")
        try:
            _nlp_ru = spacy.load(ru_model)
            _nlp_ru.max_length = TEXT_PROCESSING_CONFIG.get(
                "spacy_max_length", 3_000_000
            )
            logger.info(f"SpaCy русская модель '{ru_model}' загружена")
        except OSError:
            logger.warning(f"SpaCy русская модель '{ru_model}' не найдена")
            _nlp_ru = None

        # Загружаем английскую модель
        en_model = SPACY_MODELS.get("en", "en_core_web_sm")
        try:
            _nlp_en = spacy.load(en_model)
            _nlp_en.max_length = TEXT_PROCESSING_CONFIG.get(
                "spacy_max_length", 3_000_000
            )
            logger.info(f"SpaCy английская модель '{en_model}' загружена")
        except OSError:
            logger.warning(f"SpaCy английская модель '{en_model}' не найдена")
            _nlp_en = None

        _spacy_available = (_nlp_ru is not None) or (_nlp_en is not None)

    except ImportError:
        logger.warning("SpaCy не установлен")
        _nlp_ru = None
        _nlp_en = None
        _spacy_available = False

    return _nlp_ru, _nlp_en, _spacy_available


class TextProcessor:
    """Класс для препроцессинга текста с поддержкой русского и английского языков"""

    def __init__(self):
        self.config = TEXT_PROCESSING_CONFIG
        self._nlp_ru = None
        self._nlp_en = None
        self._spacy_available = None
        self.max_chunk_size = self.config.get("chunk_size", 800_000)

    def _get_nlp(self):
        """Получить SpaCy модели (с ленивой загрузкой)"""
        if self._nlp_ru is None and self._nlp_en is None:
            self._nlp_ru, self._nlp_en, self._spacy_available = _initialize_spacy()
        return self._nlp_ru, self._nlp_en, self._spacy_available

    def detect_language(self, text: str) -> str:
        """
        Определение преобладающего языка текста

        Returns:
            'ru' - русский, 'en' - английский, 'mixed' - смешанный
        """
        # Анализируем первые 1000 символов
        sample = text[:1000]

        # Подсчет алфавитных символов
        cyrillic = sum(1 for c in sample if "\u0400" <= c <= "\u04ff")
        latin = sum(1 for c in sample if ("a" <= c <= "z") or ("A" <= c <= "Z"))

        total = cyrillic + latin
        if total == 0:
            return "unknown"

        cyrillic_ratio = cyrillic / total

        if cyrillic_ratio > 0.8:
            return "ru"
        elif cyrillic_ratio < 0.2:
            return "en"
        else:
            return "mixed"

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

        # Сохраняем больше символов для многоязычных текстов
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]""«»\']+', " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text)

        # Фильтруем слишком короткие слова
        words = text.split()
        words = [word for word in words if len(word) > 1 or word.lower() in ["i", "a"]]

        return " ".join(words).strip()

    def preprocess_with_spacy(self, text: str, language: str = "auto") -> List[str]:
        """
        Обработка текста с использованием SpaCy

        Args:
            text: Исходный текст
            language: 'ru', 'en' или 'auto' для автоопределения

        Returns:
            Список обработанных токенов
        """
        nlp_ru, nlp_en, spacy_available = self._get_nlp()

        if not spacy_available:
            return self.preprocess_basic(text)

        # Определяем язык если не указан
        if language == "auto":
            language = self.detect_language(text)
            logger.debug(f"Определен язык: {language}")

        # Выбираем подходящую модель
        if language == "ru" and nlp_ru:
            nlp = nlp_ru
        elif language == "en" and nlp_en:
            nlp = nlp_en
        elif language == "mixed":
            # Для смешанного текста обрабатываем по частям
            return self._process_mixed_text(text)
        else:
            # Fallback на доступную модель
            nlp = nlp_ru or nlp_en
            if not nlp:
                return self.preprocess_basic(text)

        # Обработка через SpaCy
        tokens = []

        # Для длинных текстов - по частям
        if len(text) > self.max_chunk_size:
            for i in range(0, len(text), self.max_chunk_size):
                chunk = text[i : i + self.max_chunk_size]
                chunk_tokens = self._process_spacy_chunk(chunk, nlp)
                tokens.extend(chunk_tokens)
        else:
            tokens = self._process_spacy_chunk(text, nlp)

        return tokens

    def _process_mixed_text(self, text: str) -> List[str]:
        """Обработка текста со смешанными языками"""
        nlp_ru, nlp_en, _ = self._get_nlp()

        # Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        all_tokens = []

        for sentence in sentences:
            lang = self.detect_language(sentence)

            if lang == "ru" and nlp_ru:
                tokens = self._process_spacy_chunk(sentence, nlp_ru)
            elif lang == "en" and nlp_en:
                tokens = self._process_spacy_chunk(sentence, nlp_en)
            else:
                # Если нет подходящей модели, используем базовую обработку
                tokens = self.preprocess_basic(sentence)

            all_tokens.extend(tokens)

        return all_tokens

    def _process_spacy_chunk(self, text: str, nlp) -> List[str]:
        """Обработка одного чанка текста через SpaCy"""
        doc = nlp(text)
        tokens = []

        for token in doc:
            # Фильтруем токены
            if (
                not token.is_punct
                and not token.is_space
                and not token.is_stop
                and len(token.text) >= self.config["min_token_length"]
                and (token.is_alpha or token.like_num)  # Буквы или числа
            ):
                # Лемматизация если включена
                if self.config["lemmatize"]:
                    tokens.append(token.lemma_.lower())
                else:
                    tokens.append(token.text.lower())

        return tokens

    def preprocess_basic(self, text: str) -> List[str]:
        """
        Базовая обработка текста без SpaCy
        """
        text = text.lower()

        # Простая токенизация
        tokens = re.findall(r"\b\w+\b", text, re.UNICODE)

        # Фильтрация
        tokens = [
            token for token in tokens if len(token) >= self.config["min_token_length"]
        ]

        return tokens

    def preprocess_text(self, text: str) -> List[str]:
        """
        Главная функция препроцессинга текста
        """
        if not text:
            return []

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        # Определяем язык и обрабатываем
        tokens = self.preprocess_with_spacy(cleaned_text)

        return tokens

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения с учетом многоязычности
        """
        if not text:
            return []

        nlp_ru, nlp_en, spacy_available = self._get_nlp()
        min_sentence_length = self.config.get("min_sentence_length", 10)

        if spacy_available:
            # Определяем язык
            lang = self.detect_language(text)

            # Выбираем модель
            if lang == "ru" and nlp_ru:
                nlp = nlp_ru
            elif lang == "en" and nlp_en:
                nlp = nlp_en
            else:
                # Для смешанного или неопределенного - базовый метод
                return self._split_sentences_basic(text, min_sentence_length)

            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                sentences = [
                    sent for sent in sentences if len(sent) >= min_sentence_length
                ]
                return sentences
            except Exception as e:
                logger.warning(f"Ошибка при разбиении через SpaCy: {e}")
                return self._split_sentences_basic(text, min_sentence_length)
        else:
            return self._split_sentences_basic(text, min_sentence_length)

    def _split_sentences_basic(self, text: str, min_sentence_length: int) -> List[str]:
        """
        Базовое разбиение текста на предложения без SpaCy

        Args:
            text: Исходный текст
            min_sentence_length: Минимальная длина предложения

        Returns:
            Список предложений
        """
        # Простое разбиение по знакам препинания
        # Поддержка русских и английских сокращений
        abbreviations = {
            "г.",
            "гг.",
            "т.д.",
            "т.п.",
            "др.",
            "пр.",
            "см.",
            "стр.",
            "Mr.",
            "Mrs.",
            "Dr.",
            "Prof.",
            "Inc.",
            "Ltd.",
            "Co.",
            "vs.",
            "etc.",
            "i.e.",
            "e.g.",
        }

        # Замена сокращений временными маркерами
        temp_text = text
        replacements = {}
        for i, abbr in enumerate(abbreviations):
            marker = f"__ABBR{i}__"
            replacements[marker] = abbr
            temp_text = temp_text.replace(abbr, marker)

        # Разбиение по основным знакам препинания
        import re

        sentences = re.split(r"[.!?]+", temp_text)

        # Восстановление сокращений
        result_sentences = []
        for sent in sentences:
            # Восстанавливаем сокращения
            for marker, abbr in replacements.items():
                sent = sent.replace(marker, abbr)

            sent = sent.strip()
            if len(sent) >= min_sentence_length:
                result_sentences.append(sent)

        return result_sentences

    def get_spacy_status(self) -> str:
        """Получить статус SpaCy моделей"""
        nlp_ru, nlp_en, _ = self._get_nlp()

        status_parts = []

        if nlp_ru:
            status_parts.append("✅ Русская модель")
        else:
            status_parts.append("❌ Русская модель не установлена")

        if nlp_en:
            status_parts.append("✅ Английская модель")
        else:
            status_parts.append("❌ Английская модель не установлена")

        return " | ".join(status_parts)
