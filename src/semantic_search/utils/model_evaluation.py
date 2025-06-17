"""Оценка качества модели"""

from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class ModelEvaluator:
    """Оценка качества обученной модели"""

    def __init__(self, model, corpus_info):
        self.model = model
        self.corpus_info = corpus_info

    def evaluate_coherence(self, sample_size: int = 100) -> Dict[str, float]:
        """Оценка когерентности модели"""
        if len(self.corpus_info) < sample_size:
            sample_size = len(self.corpus_info)

        # Случайная выборка документов
        import random

        sample_docs = random.sample(self.corpus_info, sample_size)

        coherence_scores = []
        similarity_scores = []

        for tokens, doc_id, metadata in sample_docs:
            try:
                # Получаем вектор документа
                doc_vector = self.model.dv[doc_id]

                # Вычисляем средний вектор слов документа
                word_vectors = []
                for token in tokens[:50]:  # Берем первые 50 токенов
                    if token in self.model.wv:
                        word_vectors.append(self.model.wv[token])

                if len(word_vectors) > 0:
                    avg_word_vector = np.mean(word_vectors, axis=0)

                    # Косинусное сходство между вектором документа и средним вектором слов
                    similarity = cosine_similarity([doc_vector], [avg_word_vector])[0][
                        0
                    ]
                    coherence_scores.append(similarity)

            except Exception as e:
                logger.debug(f"Ошибка при оценке документа {doc_id}: {e}")
                continue

        return {
            "mean_coherence": np.mean(coherence_scores) if coherence_scores else 0.0,
            "std_coherence": np.std(coherence_scores) if coherence_scores else 0.0,
            "evaluated_docs": len(coherence_scores),
        }

    def evaluate_vocabulary_coverage(self) -> Dict[str, float]:
        """Оценка покрытия словаря"""
        total_tokens = 0
        covered_tokens = 0

        for tokens, doc_id, metadata in self.corpus_info:
            for token in tokens:
                total_tokens += 1
                if token in self.model.wv:
                    covered_tokens += 1

        coverage = covered_tokens / total_tokens if total_tokens > 0 else 0.0

        return {
            "vocabulary_coverage": coverage,
            "total_tokens": total_tokens,
            "covered_tokens": covered_tokens,
            "vocabulary_size": len(self.model.wv.key_to_index),
        }

    def find_outlier_documents(self, threshold: float = 0.1) -> List[str]:
        """Поиск документов-выбросов"""
        outliers = []

        for tokens, doc_id, metadata in self.corpus_info:
            try:
                # Ищем похожие документы
                similar = self.model.dv.most_similar(doc_id, topn=5)

                # Если максимальная схожесть очень низкая - возможный выброс
                if similar and similar[0][1] < threshold:
                    outliers.append(doc_id)

            except Exception:
                continue

        return outliers
