"""Модуль для сравнения различных методов поиска"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from semantic_search.config import DATA_DIR

from .baselines import BaseSearchMethod
from .metrics import SearchMetrics


class QueryTestCase:
    """Тестовый случай для оценки поиска"""

    def __init__(
        self,
        query: str,
        relevant_docs: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        description: str = "",
    ):
        self.query = query
        self.relevant_docs = relevant_docs
        self.relevance_scores = relevance_scores or {}
        self.description = description


class SearchComparison:
    """Класс для сравнения методов поиска"""

    def __init__(self, test_cases: Optional[List[QueryTestCase]] = None):
        self.test_cases = test_cases or []
        self.results = {}
        self.metrics = SearchMetrics()

    def add_test_case(self, test_case: QueryTestCase) -> None:
        """Добавить тестовый случай"""
        self.test_cases.append(test_case)

    def create_default_test_cases(self) -> List[QueryTestCase]:
        """Создать стандартный набор тестовых случаев для демонстрации"""
        test_cases = [
            QueryTestCase(
                query="машинное обучение и нейронные сети",
                relevant_docs={
                    "ml_basics.pdf",
                    "neural_networks.pdf",
                    "deep_learning.pdf",
                },
                relevance_scores={
                    "ml_basics.pdf": 3,
                    "neural_networks.pdf": 3,
                    "deep_learning.pdf": 2,
                    "ai_overview.pdf": 1,
                },
                description="Базовый запрос по ML",
            ),
            QueryTestCase(
                query="глубокое обучение для обработки изображений",
                relevant_docs={
                    "cnn_tutorial.pdf",
                    "image_processing.pdf",
                    "deep_learning.pdf",
                },
                relevance_scores={
                    "cnn_tutorial.pdf": 3,
                    "image_processing.pdf": 3,
                    "deep_learning.pdf": 2,
                    "computer_vision.pdf": 2,
                },
                description="Специализированный запрос по CV",
            ),
            QueryTestCase(
                query="обработка естественного языка трансформеры",
                relevant_docs={
                    "nlp_transformers.pdf",
                    "bert_paper.pdf",
                    "attention_mechanism.pdf",
                },
                relevance_scores={
                    "nlp_transformers.pdf": 3,
                    "bert_paper.pdf": 3,
                    "attention_mechanism.pdf": 2,
                    "nlp_basics.pdf": 1,
                },
                description="Запрос по NLP",
            ),
            QueryTestCase(
                query="градиентный спуск оптимизация",
                relevant_docs={"optimization_methods.pdf", "gradient_descent.pdf"},
                relevance_scores={
                    "optimization_methods.pdf": 3,
                    "gradient_descent.pdf": 3,
                    "ml_basics.pdf": 1,
                },
                description="Запрос по методам оптимизации",
            ),
            QueryTestCase(
                query="рекуррентные нейронные сети LSTM",
                relevant_docs={
                    "rnn_tutorial.pdf",
                    "lstm_explained.pdf",
                    "sequence_models.pdf",
                },
                relevance_scores={
                    "rnn_tutorial.pdf": 3,
                    "lstm_explained.pdf": 3,
                    "sequence_models.pdf": 2,
                },
                description="Запрос по RNN",
            ),
        ]

        return test_cases

    def evaluate_method(
        self, method: BaseSearchMethod, top_k: int = 10, verbose: bool = True
    ) -> Dict[str, any]:
        """
        Оценить метод поиска на всех тестовых случаях

        Args:
            method: Метод поиска для оценки
            top_k: Количество результатов для извлечения
            verbose: Выводить прогресс

        Returns:
            Словарь с результатами оценки
        """
        method_name = method.get_method_name()

        if verbose:
            logger.info(f"Оценка метода: {method_name}")

        all_metrics = []
        query_times = []
        all_results = []

        for i, test_case in enumerate(self.test_cases):
            if verbose and (i + 1) % 5 == 0:
                logger.info(f"Обработано запросов: {i + 1}/{len(self.test_cases)}")

            # Измеряем время выполнения запроса
            start_time = time.time()
            search_results = method.search(test_case.query, top_k=top_k)
            query_time = time.time() - start_time
            query_times.append(query_time)

            # Извлекаем ID документов из результатов
            retrieved_docs = [result.doc_id for result in search_results]

            # Вычисляем метрики
            metrics = self.metrics.calculate_all_metrics(
                retrieved=retrieved_docs,
                relevant=test_case.relevant_docs,
                relevance_scores=test_case.relevance_scores,
                k_values=[1, 5, 10],
            )

            # Добавляем информацию о запросе
            metrics["query"] = test_case.query
            metrics["query_time"] = query_time

            all_metrics.append(metrics)
            all_results.append((retrieved_docs, test_case.relevant_docs))

        # Вычисляем агрегированные метрики
        aggregated = self._aggregate_metrics(all_metrics)

        # MAP и MRR
        aggregated["MAP"] = self.metrics.mean_average_precision(all_results)
        aggregated["MRR"] = self.metrics.mean_reciprocal_rank(all_results)

        # Статистика по времени
        aggregated["avg_query_time"] = np.mean(query_times)
        aggregated["std_query_time"] = np.std(query_times)
        aggregated["median_query_time"] = np.median(query_times)

        # Сохраняем результаты
        self.results[method_name] = {
            "aggregated": aggregated,
            "detailed": all_metrics,
            "method_stats": method.get_stats(),
        }

        if verbose:
            logger.info(f"Оценка {method_name} завершена")
            logger.info(f"Среднее время запроса: {aggregated['avg_query_time']:.3f}с")
            logger.info(f"MAP: {aggregated['MAP']:.3f}")
            logger.info(f"MRR: {aggregated['MRR']:.3f}")

        return self.results[method_name]

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Агрегировать метрики по всем запросам"""
        aggregated = {}

        # Получаем все ключи метрик (исключая служебные)
        metric_keys = [
            k for k in metrics_list[0].keys() if k not in ["query", "query_time"]
        ]

        # Вычисляем среднее для каждой метрики
        for key in metric_keys:
            values = [m[key] for m in metrics_list]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)

        return aggregated

    def compare_methods(
        self,
        methods: List[BaseSearchMethod],
        top_k: int = 10,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Сравнить несколько методов

        Args:
            methods: Список методов для сравнения
            top_k: Количество результатов
            save_results: Сохранить результаты в файл

        Returns:
            DataFrame с результатами сравнения
        """
        logger.info(f"Начинаем сравнение {len(methods)} методов")

        # Оцениваем каждый метод
        for method in methods:
            self.evaluate_method(method, top_k=top_k)

        # Создаем сравнительную таблицу
        comparison_data = []

        for method_name, results in self.results.items():
            row = {
                "Method": method_name,
                "MAP": results["aggregated"]["MAP"],
                "MRR": results["aggregated"]["MRR"],
                "Avg Query Time (s)": results["aggregated"]["avg_query_time"],
                "Index Time (s)": results["method_stats"]["index_time"],
            }

            # Добавляем метрики для разных k
            for k in [1, 5, 10]:
                row[f"P@{k}"] = results["aggregated"][f"avg_precision@{k}"]
                row[f"R@{k}"] = results["aggregated"][f"avg_recall@{k}"]
                if f"avg_ndcg@{k}" in results["aggregated"]:
                    row[f"NDCG@{k}"] = results["aggregated"][f"avg_ndcg@{k}"]

            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)

        # Сохраняем результаты
        if save_results:
            results_dir = DATA_DIR / "evaluation_results"
            results_dir.mkdir(exist_ok=True)

            # Сохраняем таблицу
            df_comparison.to_csv(results_dir / "comparison_results.csv", index=False)

            # Сохраняем детальные результаты
            with open(
                results_dir / "detailed_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Результаты сохранены в {results_dir}")

        return df_comparison

    def plot_comparison(self, save_plots: bool = True) -> None:
        """Визуализация результатов сравнения"""
        if not self.results:
            logger.error("Нет результатов для визуализации")
            return

        # Подготовка данных для визуализации
        methods = list(self.results.keys())

        # Создаем фигуру с подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Сравнение методов поиска", fontsize=16)

        # 1. Сравнение основных метрик
        ax1 = axes[0, 0]
        metrics_data = {
            "MAP": [self.results[m]["aggregated"]["MAP"] for m in methods],
            "MRR": [self.results[m]["aggregated"]["MRR"] for m in methods],
            "P@10": [
                self.results[m]["aggregated"]["avg_precision@10"] for m in methods
            ],
            "R@10": [self.results[m]["aggregated"]["avg_recall@10"] for m in methods],
        }

        x = np.arange(len(methods))
        width = 0.2

        for i, (metric, values) in enumerate(metrics_data.items()):
            ax1.bar(x + i * width, values, width, label=metric)

        ax1.set_xlabel("Методы")
        ax1.set_ylabel("Значение метрики")
        ax1.set_title("Основные метрики качества поиска")
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall для разных k
        ax2 = axes[0, 1]
        k_values = [1, 5, 10]

        for method in methods:
            precisions = [
                self.results[method]["aggregated"][f"avg_precision@{k}"]
                for k in k_values
            ]
            recalls = [
                self.results[method]["aggregated"][f"avg_recall@{k}"] for k in k_values
            ]
            ax2.plot(recalls, precisions, marker="o", label=method)

        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall кривые")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Время выполнения
        ax3 = axes[1, 0]
        query_times = [self.results[m]["aggregated"]["avg_query_time"] for m in methods]
        index_times = [self.results[m]["method_stats"]["index_time"] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        ax3.bar(x - width / 2, query_times, width, label="Среднее время запроса")
        ax3.bar(x + width / 2, index_times, width, label="Время индексации")

        ax3.set_xlabel("Методы")
        ax3.set_ylabel("Время (секунды)")
        ax3.set_title("Производительность методов")
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. NDCG для разных k
        ax4 = axes[1, 1]

        for method in methods:
            if "avg_ndcg@1" in self.results[method]["aggregated"]:
                ndcg_values = [
                    self.results[method]["aggregated"][f"avg_ndcg@{k}"]
                    for k in k_values
                ]
                ax4.plot(k_values, ndcg_values, marker="s", label=method)

        ax4.set_xlabel("k")
        ax4.set_ylabel("NDCG@k")
        ax4.set_title("NDCG для различных k")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(k_values)

        plt.tight_layout()

        if save_plots:
            plots_dir = DATA_DIR / "evaluation_results" / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                plots_dir / "comparison_plots.png", dpi=300, bbox_inches="tight"
            )
            logger.info(f"Графики сохранены в {plots_dir}")

        plt.show()

    def plot_detailed_metrics(self, method_name: str, save_plot: bool = True) -> None:
        """Детальная визуализация метрик для конкретного метода"""
        if method_name not in self.results:
            logger.error(f"Результаты для метода {method_name} не найдены")
            return

        detailed = self.results[method_name]["detailed"]

        # Создаем DataFrame для удобства
        df = pd.DataFrame(detailed)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Детальный анализ метода: {method_name}", fontsize=16)

        # 1. Распределение Average Precision
        ax1 = axes[0, 0]
        ax1.hist(
            df["average_precision"], bins=20, alpha=0.7, color="blue", edgecolor="black"
        )
        ax1.axvline(
            df["average_precision"].mean(),
            color="red",
            linestyle="--",
            label=f"Среднее: {df['average_precision'].mean():.3f}",
        )
        ax1.set_xlabel("Average Precision")
        ax1.set_ylabel("Количество запросов")
        ax1.set_title("Распределение Average Precision")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Время выполнения запросов
        ax2 = axes[0, 1]
        ax2.scatter(range(len(df)), df["query_time"], alpha=0.6)
        ax2.axhline(
            df["query_time"].mean(),
            color="red",
            linestyle="--",
            label=f"Среднее: {df['query_time'].mean():.3f}s",
        )
        ax2.set_xlabel("Номер запроса")
        ax2.set_ylabel("Время (секунды)")
        ax2.set_title("Время выполнения запросов")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Метрики по k
        ax3 = axes[1, 0]
        k_values = [1, 5, 10]
        metrics = ["precision", "recall", "f1"]

        for metric in metrics:
            values = [df[f"{metric}@{k}"].mean() for k in k_values]
            ax3.plot(k_values, values, marker="o", label=metric.capitalize())

        ax3.set_xlabel("k")
        ax3.set_ylabel("Значение метрики")
        ax3.set_title("Метрики для различных k")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(k_values)

        # 4. Топ-10 лучших и худших запросов по AP
        ax4 = axes[1, 1]

        sorted_df = df.sort_values("average_precision")
        worst_5 = sorted_df.head(5)
        best_5 = sorted_df.tail(5)

        combined = pd.concat([worst_5, best_5])
        colors = ["red"] * 5 + ["green"] * 5

        y_pos = np.arange(len(combined))
        ax4.barh(y_pos, combined["average_precision"], color=colors, alpha=0.7)

        # Обрезаем длинные запросы для отображения
        labels = [q[:50] + "..." if len(q) > 50 else q for q in combined["query"]]
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel("Average Precision")
        ax4.set_title("Лучшие и худшие запросы по AP")
        ax4.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_plot:
            plots_dir = DATA_DIR / "evaluation_results" / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                plots_dir / f"{method_name.replace(' ', '_')}_detailed.png",
                dpi=300,
                bbox_inches="tight",
            )
            logger.info(f"Детальные графики сохранены для {method_name}")

        plt.show()

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Генерация текстового отчета о сравнении

        Args:
            output_path: Путь для сохранения отчета

        Returns:
            Текст отчета
        """
        if not self.results:
            return "Нет результатов для генерации отчета"

        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ О СРАВНЕНИИ МЕТОДОВ ПОИСКА")
        report.append("=" * 80)
        report.append("")

        # Общая информация
        report.append(f"Количество тестовых запросов: {len(self.test_cases)}")
        report.append(f"Оцененные методы: {', '.join(self.results.keys())}")
        report.append("")

        # Сводная таблица
        report.append("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        report.append("-" * 80)

        # Создаем DataFrame для красивого вывода
        comparison_data = []
        for method_name, results in self.results.items():
            row = {
                "Метод": method_name,
                "MAP": f"{results['aggregated']['MAP']:.3f}",
                "MRR": f"{results['aggregated']['MRR']:.3f}",
                "P@10": f"{results['aggregated']['avg_precision@10']:.3f}",
                "R@10": f"{results['aggregated']['avg_recall@10']:.3f}",
                "Время запроса (с)": f"{results['aggregated']['avg_query_time']:.3f}",
                "Время индексации (с)": f"{results['method_stats']['index_time']:.1f}",
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        report.append(df.to_string(index=False))
        report.append("")

        # Выводы
        report.append("ОСНОВНЫЕ ВЫВОДЫ")
        report.append("-" * 80)

        # Определяем лучший метод по MAP
        best_method = max(self.results.items(), key=lambda x: x[1]["aggregated"]["MAP"])
        report.append(
            f"✓ Лучший метод по MAP: {best_method[0]} ({best_method[1]['aggregated']['MAP']:.3f})"
        )

        # Определяем самый быстрый метод
        fastest_method = min(
            self.results.items(), key=lambda x: x[1]["aggregated"]["avg_query_time"]
        )
        report.append(
            f"✓ Самый быстрый метод: {fastest_method[0]} ({fastest_method[1]['aggregated']['avg_query_time']:.3f}с)"
        )

        # Сравнение Doc2Vec и OpenAI
        if "Doc2Vec" in self.results and any(
            "OpenAI" in m for m in self.results.keys()
        ):
            doc2vec_map = self.results["Doc2Vec"]["aggregated"]["MAP"]
            openai_method = next(m for m in self.results.keys() if "OpenAI" in m)
            openai_map = self.results[openai_method]["aggregated"]["MAP"]

            improvement = ((doc2vec_map - openai_map) / openai_map) * 100

            report.append("")
            report.append("СРАВНЕНИЕ DOC2VEC И OPENAI")
            report.append("-" * 80)

            if doc2vec_map > openai_map:
                report.append(
                    f"✓ Doc2Vec превосходит {openai_method} по MAP на {improvement:.1f}%"
                )
            else:
                report.append(
                    f"✗ {openai_method} превосходит Doc2Vec по MAP на {-improvement:.1f}%"
                )

            # Сравнение по скорости
            doc2vec_time = self.results["Doc2Vec"]["aggregated"]["avg_query_time"]
            openai_time = self.results[openai_method]["aggregated"]["avg_query_time"]
            time_ratio = openai_time / doc2vec_time

            report.append(f"✓ Doc2Vec быстрее {openai_method} в {time_ratio:.1f} раз")

            # Экономическая выгода
            report.append("")
            report.append("ЭКОНОМИЧЕСКАЯ ЭФФЕКТИВНОСТЬ")
            report.append("-" * 80)
            report.append("При 1000 запросов в день:")

            # Примерная стоимость OpenAI embeddings
            openai_cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens
            avg_tokens_per_query = 50  # примерно
            daily_cost = (
                1000 * avg_tokens_per_query / 1000
            ) * openai_cost_per_1k_tokens
            monthly_cost = daily_cost * 30

            report.append(f"- Стоимость OpenAI: ~${monthly_cost:.2f}/месяц")
            report.append("- Стоимость Doc2Vec: $0 (после обучения)")
            report.append(f"- Экономия: ${monthly_cost:.2f}/месяц")

        report.append("")
        report.append("=" * 80)
        report.append(f"Отчет сгенерирован: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        report_text = "\n".join(report)

        # Сохраняем отчет
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Отчет сохранен в {output_path}")

        return report_text
