"""Модуль утилит для предобработки текста с поддержкой SpaCy"""

import re
from typing import List

from loguru import logger

from semantic_search.config import SPACY_MODEL


class TextProcessor:
    """Класс для предобработки текста и предложений с поддержкой SpaCy"""

    def __init__(self):
        self._nlp = None
        self._spacy_available = False
        self._initialize_spacy()

    def _initialize_spacy(self):
        try:
            import spacy
            from spacy.lang.ru import Russian

            try:
                self._nlp = spacy.load(
                    SPACY_MODEL, exclude=["ner", "parser", "textcat"]
                )
                self._nlp.max_length = 2_000_000
                self._spacy_available = True
                logger.info(f"SpaCy модель {SPACY_MODEL} загружена")
            except OSError:
                logger.warning(
                    f"SpaCy модель {SPACY_MODEL} не найдена. Используется базовая обработка"
                )
                self._nlp = Russian()
        except ImportError:
            logger.warning("Библиотека SpaCy не установлена")

    def is_spacy_available(self) -> bool:
        return self._spacy_available

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9 .,!?\-]", "", text)
        return text.strip()

    def preprocess_basic(self, text: str) -> List[str]:
        text = self.clean_text(text)
        return text.lower().split()

    def preprocess_with_spacy(self, text: str) -> List[str]:
        if not self._nlp:
            return self.preprocess_basic(text)

        doc = self._nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha
        ]

    def preprocess_text(self, text: str) -> List[str]:
        if self._spacy_available:
            return self.preprocess_with_spacy(text)
        return self.preprocess_basic(text)

    def split_sentences(self, text: str) -> List[str]:
        if self._nlp and self._spacy_available:
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            return re.split(r"(?<=[.!?]) +", text.strip())
