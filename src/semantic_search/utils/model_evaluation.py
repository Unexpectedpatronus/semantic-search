"""Оценка когерентности модели Doc2Vec"""

import random
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


class ModelEvaluator:
    """Класс для оценки качества семантической модели"""

    def __init__(self, model, corpus_info: List[Tuple[List[str], str, dict]]):
        self.model = model
        self.corpus_info = corpus_info

    def evaluate_coherence(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Рассчитывает среднюю когерентность модели по выборке документов.

        Returns:
            Словарь с метриками когерентности
        """
        if len(self.corpus_info) == 0:
            logger.warning("Корпус пуст. Оценка невозможна.")
            return {"coherence": 0.0, "samples_used": 0}

        sample_size = min(sample_size, len(self.corpus_info))
        sample_docs = random.sample(self.corpus_info, sample_size)

        scores = []

        for tokens, doc_id, _ in sample_docs:
            if not tokens:
                continue
            try:
                doc_vector = self.model.dv[doc_id].reshape(1, -1)
                token_vectors = np.array(
                    [self.model.wv[token] for token in tokens if token in self.model.wv]
                )

                if token_vectors.size == 0:
                    continue

                mean_vector = token_vectors.mean(axis=0).reshape(1, -1)
                similarity = cosine_similarity(doc_vector, mean_vector)[0][0]
                scores.append(similarity)

            except Exception as e:
                logger.warning(f"Ошибка при вычислении когерентности для {doc_id}: {e}")

        if not scores:
            return {"coherence": 0.0, "samples_used": 0}

        return {"coherence": float(np.mean(scores)), "samples_used": len(scores)}
