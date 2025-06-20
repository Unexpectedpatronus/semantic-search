"""Метрики для оценки качества поиска"""

from typing import Dict, List, Set, Tuple

import numpy as np


class SearchMetrics:
    """Класс для вычисления метрик качества поиска"""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Точность на позиции k

        Args:
            retrieved: Список найденных документов (в порядке ранжирования)
            relevant: Множество релевантных документов
            k: Позиция для вычисления точности

        Returns:
            Precision@k
        """
        if k <= 0 or not retrieved:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant)

        return relevant_in_k / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Полнота на позиции k

        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            k: Позиция для вычисления полноты

        Returns:
            Recall@k
        """
        if not relevant or k <= 0:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_in_k = sum(1 for doc in retrieved_k if doc in relevant)

        return relevant_in_k / len(relevant)

    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Средняя точность (AP)

        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов

        Returns:
            Average Precision
        """
        if not relevant or not retrieved:
            return 0.0

        precisions = []
        relevant_found = 0

        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_found += 1
                precision = relevant_found / (i + 1)
                precisions.append(precision)

        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant)

    @staticmethod
    def mean_average_precision(results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Средняя точность по всем запросам (MAP)

        Args:
            results: Список кортежей (retrieved, relevant) для каждого запроса

        Returns:
            Mean Average Precision
        """
        if not results:
            return 0.0

        ap_scores = [
            SearchMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in results
        ]

        return sum(ap_scores) / len(ap_scores)

    @staticmethod
    def dcg_at_k(
        retrieved: List[str], relevance_scores: Dict[str, float], k: int
    ) -> float:
        """
        Discounted Cumulative Gain на позиции k

        Args:
            retrieved: Список найденных документов
            relevance_scores: Словарь с оценками релевантности (0-3)
            k: Позиция для вычисления DCG

        Returns:
            DCG@k
        """
        if k <= 0 or not retrieved:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0)
            # Используем log2(i+2) так как индексация начинается с 0
            dcg += (2**rel - 1) / np.log2(i + 2)

        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str], relevance_scores: Dict[str, float], k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain на позиции k

        Args:
            retrieved: Список найденных документов
            relevance_scores: Словарь с оценками релевантности
            k: Позиция для вычисления NDCG

        Returns:
            NDCG@k
        """
        dcg = SearchMetrics.dcg_at_k(retrieved, relevance_scores, k)

        # Идеальный порядок - сортировка по убыванию релевантности
        ideal_order = sorted(
            relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True
        )

        idcg = SearchMetrics.dcg_at_k(ideal_order, relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mean_reciprocal_rank(results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Mean Reciprocal Rank (MRR)

        Args:
            results: Список кортежей (retrieved, relevant) для каждого запроса

        Returns:
            MRR
        """
        if not results:
            return 0.0

        reciprocal_ranks = []

        for retrieved, relevant in results:
            # Находим позицию первого релевантного документа
            for i, doc in enumerate(retrieved):
                if doc in relevant:
                    reciprocal_ranks.append(1 / (i + 1))
                    break
            else:
                # Если релевантных документов не найдено
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        F1-мера на позиции k

        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            k: Позиция для вычисления F1

        Returns:
            F1@k
        """
        precision = SearchMetrics.precision_at_k(retrieved, relevant, k)
        recall = SearchMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_all_metrics(
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Dict[str, float] = None,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Вычислить все метрики для одного запроса

        Args:
            retrieved: Список найденных документов
            relevant: Множество релевантных документов
            relevance_scores: Оценки релевантности (для NDCG)
            k_values: Значения k для метрик

        Returns:
            Словарь с метриками
        """
        metrics = {}

        # Метрики для разных k
        for k in k_values:
            metrics[f"precision@{k}"] = SearchMetrics.precision_at_k(
                retrieved, relevant, k
            )
            metrics[f"recall@{k}"] = SearchMetrics.recall_at_k(retrieved, relevant, k)
            metrics[f"f1@{k}"] = SearchMetrics.f1_at_k(retrieved, relevant, k)

            if relevance_scores:
                metrics[f"ndcg@{k}"] = SearchMetrics.ndcg_at_k(
                    retrieved, relevance_scores, k
                )

        # Общие метрики
        metrics["average_precision"] = SearchMetrics.average_precision(
            retrieved, relevant
        )

        # MRR для одного запроса
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                metrics["reciprocal_rank"] = 1 / (i + 1)
                break
        else:
            metrics["reciprocal_rank"] = 0.0

        return metrics
