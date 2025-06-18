"""Модуль для обучения и управления моделью Doc2Vec"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from semantic_search.config import DOC2VEC_CONFIG, MODELS_DIR

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument  # type: ignore

try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.error("Gensim не установлен. Установите: pip install gensim")


def requires_gensim(func):
    def wrapper(*args, **kwargs):
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "Модуль gensim не найден. Установите его: pip install gensim"
            )
        return func(*args, **kwargs)

    return wrapper


class Doc2VecTrainer:
    """Класс для обучения и управления моделью Doc2Vec"""

    def __init__(self, config: Optional[dict] = None):
        self.model: Optional[Doc2Vec] = None
        self.config = config if config is not None else DOC2VEC_CONFIG
        self.corpus_info: Optional[List[Tuple[List[str], str, dict]]] = None

    @requires_gensim
    def create_tagged_documents(
        self, corpus: List[Tuple[List[str], str, dict]]
    ) -> List[TaggedDocument]:
        """Создание TaggedDocument объектов для обучения"""
        return [TaggedDocument(words, [tag]) for words, tag, _ in corpus]

    @requires_gensim
    def train_model(
        self,
        corpus: List[Tuple[List[str], str, dict]],
        vector_size: Optional[int] = None,
        window: Optional[int] = None,
        min_count: Optional[int] = None,
        epochs: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> Doc2Vec:
        """Обучение модели Doc2Vec на основе переданного корпуса"""
        self.corpus_info = corpus
        tagged_documents = self.create_tagged_documents(corpus)

        self.model = Doc2Vec(
            documents=tagged_documents,
            vector_size=vector_size or self.config["vector_size"],
            window=window or self.config["window"],
            min_count=min_count or self.config["min_count"],
            workers=workers or self.config["workers"],
            epochs=epochs or self.config["epochs"],
            dm=self.config.get("dm", 1),
            seed=self.config.get("seed", 42),
            negative=self.config.get("negative", 5),
            hs=self.config.get("hs", 0),
            sample=self.config.get("sample", 1e-4),
        )

        logger.info("Doc2Vec модель успешно обучена.")
        return self.model

    @requires_gensim
    def save_model(self, model_name: str) -> None:
        if self.model is None:
            logger.error("Нельзя сохранить: модель не обучена.")
            return

        model_path = MODELS_DIR / f"{model_name}.model"
        self.model.save(str(model_path))
        logger.info(f"Модель сохранена в: {model_path}")

    @requires_gensim
    def load_model(self, model_name: str) -> Doc2Vec:
        model_path = MODELS_DIR / f"{model_name}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self.model = Doc2Vec.load(str(model_path))
        logger.info(f"Модель загружена из: {model_path}")
        return self.model

    def get_model_info(self) -> dict:
        if self.model is None:
            return {"status": "Модель не загружена"}

        return {
            "vector_size": self.model.vector_size,
            "corpus_count": self.model.corpus_count,
            "epochs": self.model.epochs,
        }
