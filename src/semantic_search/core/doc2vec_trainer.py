"""Модуль для обучения модели Doc2Vec"""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    GENSIM_AVAILABLE = True
except ImportError:
    logger.error("Gensim не установлен. Установите: pip install gensim")
    GENSIM_AVAILABLE = False

from semantic_search.config import DOC2VEC_CONFIG, MODELS_DIR


class Doc2VecTrainer:
    """Класс для обучения и управления моделью Doc2Vec"""

    def __init__(self):
        self.model: Optional[Doc2Vec] = None
        self.config = DOC2VEC_CONFIG
        self.corpus_info: Optional[List[Tuple[List[str], str, dict]]] = None
        self.training_metadata: Dict[str, Any] = {}

    def create_tagged_documents(
        self, corpus: List[Tuple[List[str], str, dict]]
    ) -> List[TaggedDocument]:
        """
        Создание TaggedDocument объектов для gensim

        Args:
            corpus: Список кортежей (tokens, doc_id, metadata)

        Returns:
            Список TaggedDocument объектов
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim не доступен")

        tagged_docs = [
            TaggedDocument(words=tokens, tags=[doc_id]) for tokens, doc_id, _ in corpus
        ]
        logger.info(f"Создано {len(tagged_docs)} TaggedDocument объектов")
        return tagged_docs

    def _get_training_params(
        self,
        vector_size: Optional[int],
        window: Optional[int],
        min_count: Optional[int],
        epochs: Optional[int],
        workers: Optional[int],
        dm: Optional[int],
        negative: Optional[int],
        hs: Optional[int],
        sample: Optional[float],
    ) -> Dict[str, Any]:
        """Получить параметры для обучения"""
        return {
            "vector_size": vector_size or self.config["vector_size"],
            "window": window or self.config["window"],
            "min_count": min_count or self.config["min_count"],
            "epochs": epochs or self.config["epochs"],
            "workers": workers or self.config["workers"],
            "seed": self.config["seed"],
            "dm": dm if dm is not None else self.config.get("dm", 1),
            "negative": negative
            if negative is not None
            else self.config.get("negative", 5),
            "hs": hs if hs is not None else self.config.get("hs", 0),
            "sample": sample if sample is not None else self.config.get("sample", 1e-4),
        }

    def _train_standard(
        self,
        corpus: List[Tuple[List[str], str, dict]],
        vector_size: Optional[int] = None,
        window: Optional[int] = None,
        min_count: Optional[int] = None,
        epochs: Optional[int] = None,
        workers: Optional[int] = None,
        dm: Optional[int] = None,
        negative: Optional[int] = None,
        hs: Optional[int] = None,
        sample: Optional[float] = None,
    ) -> Optional[Doc2Vec]:
        """
        Обучение модели Doc2Vec

        Args:
            corpus: Подготовленный корпус
            vector_size: Размерность векторов (по умолчанию из config)
            window: Размер окна контекста
            min_count: Минимальная частота слова
            epochs: Количество эпох обучения
            workers: Количество потоков
            dm: Distributed Memory (1) или Distributed Bag of Words (0)
            negative: Размер negative sampling (если hs=0)
            hs: Использовать Hierarchical Softmax (1) или negative sampling (0)
            sample: Порог для downsampling высокочастотных слов

        Returns:
            Обученная модель Doc2Vec или None при ошибке
        """
        if not GENSIM_AVAILABLE:
            logger.error("Gensim не доступен для обучения модели")
            return None
        if not corpus:
            logger.error("Корпус пуст, обучение невозможно")
            return None

        params = self._get_training_params(
            vector_size, window, min_count, epochs, workers, dm, negative, hs, sample
        )

        logger.info("Подготовка данных для обучения...")
        tagged_docs = self.create_tagged_documents(corpus)

        logger.info("Начинаем обучение модели Doc2Vec с параметрами:")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")

        try:
            model = Doc2Vec(tagged_docs, **params)

            self.model = model
            self.corpus_info = corpus

            logger.info("Обучение модели завершено успешно!")
            logger.info(
                f"Словарь содержит {len(model.wv.key_to_index)} уникальных слов"
            )
            logger.info(f"Обучено векторов документов: {len(model.dv)}")

            return model

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            return None

    def train_model(
        self,
        corpus: List[Tuple[List[str], str, dict]],
        vector_size: Optional[int] = None,
        window: Optional[int] = None,
        min_count: Optional[int] = None,
        epochs: Optional[int] = None,
        workers: Optional[int] = None,
        dm: Optional[int] = None,
        negative: Optional[int] = None,
        hs: Optional[int] = None,
        sample: Optional[float] = None,
    ) -> Optional[Doc2Vec]:
        """
        Обучение модели Doc2Vec с оптимизацией под объём корпуса.

        Для небольших корпусов используется стандартное обучение.
        Для больших корпусов (> 10 000 документов) применяется поэпоховое обучение.

        Args:
            corpus: Подготовленный корпус, где каждый элемент — кортеж (токены, тег, метаданные)
            vector_size: Размерность векторов (по умолчанию из self.config)
            window: Размер окна контекста (по умолчанию из self.config)
            min_count: Минимальная частота слова (по умолчанию из self.config)
            epochs: Количество эпох обучения (по умолчанию из self.config)
            workers: Количество потоков (по умолчанию из self.config)
            dm: Distributed Memory (1) или Distributed Bag of Words (0)
            negative: Размер negative sampling
            hs: Использовать Hierarchical Softmax
            sample: Порог для downsampling

        Returns:
            Обученная модель Doc2Vec или None при ошибке
        """
        if not GENSIM_AVAILABLE:
            logger.error("Gensim не доступен для обучения модели")
            return None
        if not corpus:
            logger.error("Корпус пуст, обучение невозможно")
            return None

        if len(corpus) > 10000:
            # Для больших корпусов - поэпоховое обучение
            logger.info("Большой корпус обнаружен. Используем поэпоховое обучение...")

            params = self._get_training_params(
                vector_size,
                window,
                min_count,
                epochs,
                workers,
                dm,
                negative,
                hs,
                sample,
            )

            logger.info("Подготовка данных для обучения...")
            tagged_docs = self.create_tagged_documents(corpus)

            logger.info("Создание модели с параметрами:")
            for k, v in params.items():
                logger.info(f"  {k}: {v}")

            try:
                model = Doc2Vec(**params)
                model.build_vocab(tagged_docs)
                logger.info(f"Словарь построен: {len(model.wv.key_to_index)} слов")

                for epoch in range(params["epochs"]):
                    logger.info(f"Эпоха {epoch + 1}/{params['epochs']}...")
                    model.train(
                        tagged_docs, total_examples=model.corpus_count, epochs=1
                    )

                self.model = model
                self.corpus_info = corpus

                logger.info("Обучение завершено успешно!")
                logger.info(
                    f"Словарь содержит {len(model.wv.key_to_index)} уникальных слов"
                )
                logger.info(f"Обучено векторов документов: {len(model.dv)}")

                return model

            except Exception as e:
                logger.error(f"Ошибка при обучении модели: {e}")
                return None
        else:
            # Для небольших корпусов используем стандартное обучение
            return self._train_standard(
                corpus,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                epochs=epochs,
                workers=workers,
                dm=dm,
                negative=negative,
                hs=hs,
                sample=sample,
            )

    def save_model(
        self, model: Optional[Doc2Vec] = None, model_name: str = "doc2vec_model"
    ) -> bool:
        """
        Сохранение модели на диск

        Args:
            model: Модель для сохранения (по умолчанию self.model)
            model_name: Имя файла модели

        Returns:
            True если сохранение успешно
        """
        model_to_save = model or self.model

        if model_to_save is None:
            logger.error("Нет модели для сохранения")
            return False

        # Проверяем, что имя модели не пустое
        if not model_name or model_name.strip() == "":
            logger.error("Имя модели не может быть пустым")
            return False

        try:
            if model_name.endswith(".model"):
                model_name = model_name[:-6]
            model_path = MODELS_DIR / f"{model_name}.model"
            model_to_save.save(str(model_path))

            # Сохраняем также информацию о корпусе
            if self.corpus_info:
                corpus_path = MODELS_DIR / f"{model_name}_corpus_info.pkl"
                with open(corpus_path, "wb") as f:
                    pickle.dump(self.corpus_info, f)
                logger.info(f"Информация о корпусе сохранена: {corpus_path}")

            # Сохраняем метаданные обучения если они есть
            if self.training_metadata:
                metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.training_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Метаданные обучения сохранены: {metadata_path}")

            logger.info(f"Модель сохранена: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            return False

    def load_model(self, model_name: str = "doc2vec_model") -> Optional[Doc2Vec]:
        """
        Загрузка модели с диска

        Args:
            model_name: Имя файла модели

        Returns:
            Загруженная модель Doc2Vec или None при ошибке
        """
        if not GENSIM_AVAILABLE:
            logger.error("Gensim не доступен для загрузки модели")
            return None

        if not model_name or model_name.strip() == "":
            logger.error("Имя модели не может быть пустым")
            return None

        try:
            if model_name.endswith(".model"):
                model_name = model_name[:-6]

            model_path = MODELS_DIR / f"{model_name}.model"

            if not model_path.exists():
                logger.error(f"Файл модели не найден: {model_path}")
                return None

            model = Doc2Vec.load(str(model_path))
            self.model = model

            # Загружаем информацию о корпусе если есть
            corpus_path = MODELS_DIR / f"{model_name}_corpus_info.pkl"
            if corpus_path.exists():
                try:
                    with open(corpus_path, "rb") as f:
                        self.corpus_info = pickle.load(f)
                    logger.info("Информация о корпусе загружена")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить информацию о корпусе: {e}")
                    self.corpus_info = []

            # Загружаем метаданные обучения если есть
            metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        self.training_metadata = json.load(f)
                    logger.info("Метаданные обучения загружены")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить метаданные обучения: {e}")
                    self.training_metadata = {}

            logger.info(f"Модель загружена: {model_path}")
            logger.info(f"Векторов документов: {len(model.dv)}")
            logger.info(f"Размерность векторов: {model.vector_size}")

            return model

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            return None

    def get_model_info(self) -> dict:
        """
        Получение информации о текущей модели

        Returns:
            Словарь с информацией о модели
        """
        if self.model is None:
            return {"status": "no_model"}

        info = {
            "status": "loaded",
            "vector_size": self.model.vector_size,
            "vocabulary_size": len(self.model.wv.key_to_index),
            "documents_count": len(self.model.dv),
            "window": self.model.window,
            "min_count": self.model.min_count,
            "epochs": self.model.epochs,
            "dm": self.model.dm,
            "dm_mean": getattr(self.model, "dm_mean", None),
            "dm_concat": getattr(self.model, "dm_concat", None),
            "negative": self.model.negative,
            "hs": self.model.hs,
            "sample": self.model.sample,
            "workers": self.model.workers,
        }

        info["training_time_formatted"] = self.training_metadata.get(
            "training_time_formatted", "Неизвестно"
        )

        info["training_date"] = self.training_metadata.get(
            "training_date", "Неизвестно"
        )

        return info
