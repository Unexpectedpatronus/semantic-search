"""Виджет для оценки и сравнения методов поиска в GUI"""

import os
from typing import List

from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from semantic_search.config import EVALUATION_RESULTS_DIR
from semantic_search.evaluation.baselines import (
    Doc2VecSearchAdapter,
    OpenAISearchBaseline,
)
from semantic_search.evaluation.comparison import QueryTestCase, SearchComparison


class EvaluationThread(QThread):
    """Поток для выполнения оценки"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    result_ready = pyqtSignal(dict)

    def __init__(
        self,
        search_engine,
        corpus_info,
        openai_key: str,
        test_cases: List[QueryTestCase],
    ):
        super().__init__()
        self.search_engine = search_engine
        self.corpus_info = corpus_info
        self.openai_key = openai_key
        self.test_cases = test_cases
        self.comparison = SearchComparison(test_cases)

    def run(self):
        """Выполнение сравнения"""
        try:
            # Шаг 1: Подготовка Doc2Vec адаптера
            self.progress.emit(10, "Подготовка Doc2Vec метода...")
            doc2vec_adapter = Doc2VecSearchAdapter(self.search_engine, self.corpus_info)

            # Шаг 2: Подготовка OpenAI baseline
            self.progress.emit(20, "Инициализация OpenAI...")
            try:
                openai_baseline = OpenAISearchBaseline(api_key=self.openai_key)
            except Exception as e:
                self.finished.emit(False, f"Ошибка инициализации OpenAI: {str(e)}")
                return

            # Шаг 3: Индексация документов для OpenAI
            self.progress.emit(30, "Индексация документов через OpenAI API...")

            # Подготовка документов для индексации
            documents = []
            for tokens, doc_id, metadata in self.corpus_info[
                :50
            ]:  # Ограничиваем 50 документами для демо
                # Восстанавливаем текст из токенов (упрощенно)
                text = " ".join(tokens[:500])  # Берем первые 500 токенов
                documents.append((doc_id, text, metadata))

            try:
                openai_baseline.index(documents)
            except Exception as e:
                self.finished.emit(False, f"Ошибка индексации OpenAI: {str(e)}")
                return

            # Шаг 4: Оценка методов
            self.progress.emit(50, "Оценка Doc2Vec...")
            doc2vec_results = self.comparison.evaluate_method(
                doc2vec_adapter, top_k=10, verbose=False
            )

            self.progress.emit(70, "Оценка OpenAI...")
            openai_results = self.comparison.evaluate_method(
                openai_baseline, top_k=10, verbose=False
            )

            # Шаг 5: Генерация отчетов и графиков
            self.progress.emit(85, "Генерация отчетов...")

            # Сравнительная таблица
            df_comparison = self.comparison.compare_methods(
                [doc2vec_adapter, openai_baseline], save_results=True
            )

            # Графики
            self.comparison.plot_comparison(save_plots=True)

            # Текстовый отчет
            report_path = EVALUATION_RESULTS_DIR / "comparison_report.txt"
            report_text = self.comparison.generate_report(report_path)

            # Результаты для GUI
            results = {
                "comparison_df": df_comparison,
                "report_text": report_text,
                "doc2vec_map": doc2vec_results["aggregated"]["MAP"],
                "openai_map": openai_results["aggregated"]["MAP"],
                "doc2vec_time": doc2vec_results["aggregated"]["avg_query_time"],
                "openai_time": openai_results["aggregated"]["avg_query_time"],
            }

            self.progress.emit(100, "Оценка завершена!")
            self.result_ready.emit(results)
            self.finished.emit(True, "Сравнение методов успешно завершено")

        except Exception as e:
            logger.error(f"Ошибка в потоке оценки: {e}")
            self.finished.emit(False, f"Ошибка: {str(e)}")


class EvaluationWidget(QWidget):
    """Виджет для оценки и сравнения методов поиска"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_engine = None
        self.corpus_info = None
        self.evaluation_thread = None
        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()

        # Группа настроек OpenAI
        openai_group = QGroupBox("Настройки OpenAI")
        openai_layout = QVBoxLayout()

        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("sk-...")

        # Пытаемся загрузить ключ из переменной окружения
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.api_key_edit.setText(env_key)

        key_layout.addWidget(self.api_key_edit)
        openai_layout.addLayout(key_layout)

        openai_group.setLayout(openai_layout)
        layout.addWidget(openai_group)

        # Группа тестовых запросов
        test_group = QGroupBox("Тестовые запросы")
        test_layout = QVBoxLayout()

        self.test_cases_combo = QComboBox()
        self.test_cases_combo.addItem("Стандартный набор тестов (5 запросов)")
        self.test_cases_combo.addItem("Расширенный набор (10 запросов)")
        self.test_cases_combo.addItem("Быстрый тест (3 запроса)")

        test_layout.addWidget(self.test_cases_combo)
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        # Кнопка запуска
        self.run_button = QPushButton("Запустить сравнение")
        self.run_button.clicked.connect(self.run_evaluation)
        layout.addWidget(self.run_button)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Результаты
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.setLayout(layout)

    def set_search_engine(self, search_engine, corpus_info):
        """Установка поискового движка для оценки"""
        self.search_engine = search_engine
        self.corpus_info = corpus_info

    def get_test_cases(self) -> List[QueryTestCase]:
        """Получение тестовых случаев в зависимости от выбора"""
        comparison = SearchComparison()
        default_cases = comparison.create_default_test_cases()

        selected_index = self.test_cases_combo.currentIndex()

        if selected_index == 0:  # Стандартный набор
            return default_cases[:5]
        elif selected_index == 1:  # Расширенный набор
            # Добавляем дополнительные тесты
            extra_cases = [
                QueryTestCase(
                    query="классификация изображений CNN",
                    relevant_docs={"cnn_tutorial.pdf", "image_classification.pdf"},
                    relevance_scores={
                        "cnn_tutorial.pdf": 3,
                        "image_classification.pdf": 3,
                    },
                ),
                QueryTestCase(
                    query="регуляризация в машинном обучении",
                    relevant_docs={"regularization.pdf", "overfitting.pdf"},
                    relevance_scores={"regularization.pdf": 3, "overfitting.pdf": 2},
                ),
                QueryTestCase(
                    query="word2vec и doc2vec модели",
                    relevant_docs={"word2vec_paper.pdf", "doc2vec_tutorial.pdf"},
                    relevance_scores={
                        "word2vec_paper.pdf": 3,
                        "doc2vec_tutorial.pdf": 3,
                    },
                ),
                QueryTestCase(
                    query="метрики оценки качества классификации",
                    relevant_docs={"ml_metrics.pdf", "evaluation_methods.pdf"},
                    relevance_scores={"ml_metrics.pdf": 3, "evaluation_methods.pdf": 3},
                ),
                QueryTestCase(
                    query="обратное распространение ошибки",
                    relevant_docs={"backpropagation.pdf", "neural_networks.pdf"},
                    relevance_scores={
                        "backpropagation.pdf": 3,
                        "neural_networks.pdf": 2,
                    },
                ),
            ]
            return default_cases + extra_cases
        else:  # Быстрый тест
            return default_cases[:3]

    def run_evaluation(self):
        """Запуск оценки"""
        if not self.search_engine:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель Doc2Vec")
            return

        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Введите API ключ OpenAI или установите переменную окружения OPENAI_API_KEY",
            )
            return

        # Получаем тестовые случаи
        test_cases = self.get_test_cases()

        # Отключаем кнопку и показываем прогресс
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()

        # Создаем и запускаем поток
        self.evaluation_thread = EvaluationThread(
            self.search_engine, self.corpus_info, api_key, test_cases
        )

        self.evaluation_thread.progress.connect(self.on_progress)
        self.evaluation_thread.finished.connect(self.on_finished)
        self.evaluation_thread.result_ready.connect(self.on_results_ready)

        self.evaluation_thread.start()

    def on_progress(self, value: int, message: str):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.results_text.append(f"[{value}%] {message}")

    def on_finished(self, success: bool, message: str):
        """Завершение оценки"""
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.results_text.append(f"\n✅ {message}")
            QMessageBox.information(self, "Успех", message)
        else:
            self.results_text.append(f"\n❌ {message}")
            QMessageBox.critical(self, "Ошибка", message)

    def on_results_ready(self, results: dict):
        """Обработка результатов"""
        # Добавляем основные результаты в текстовое поле
        self.results_text.append("\n" + "=" * 80)
        self.results_text.append("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        self.results_text.append("=" * 80)

        # Основные метрики
        doc2vec_map = results["doc2vec_map"]
        openai_map = results["openai_map"]
        improvement = (
            ((doc2vec_map - openai_map) / openai_map) * 100 if openai_map > 0 else 0
        )

        self.results_text.append("\n📊 MAP (Mean Average Precision):")
        self.results_text.append(f"   Doc2Vec: {doc2vec_map:.3f}")
        self.results_text.append(f"   OpenAI:  {openai_map:.3f}")

        if improvement > 0:
            self.results_text.append(f"   ✅ Doc2Vec лучше на {improvement:.1f}%")
        else:
            self.results_text.append(f"   ❌ OpenAI лучше на {-improvement:.1f}%")

        # Скорость
        doc2vec_time = results["doc2vec_time"]
        openai_time = results["openai_time"]
        speed_ratio = openai_time / doc2vec_time if doc2vec_time > 0 else 0

        self.results_text.append("\n⚡ Скорость поиска:")
        self.results_text.append(f"   Doc2Vec: {doc2vec_time:.3f}с")
        self.results_text.append(f"   OpenAI:  {openai_time:.3f}с")
        self.results_text.append(f"   ✅ Doc2Vec быстрее в {speed_ratio:.1f} раз")

        # Полный отчет
        self.results_text.append("\n" + results["report_text"])

        # Информация о сохраненных файлах
        self.results_text.append("\n📁 Файлы сохранены в:")
        self.results_text.append(f"   {EVALUATION_RESULTS_DIR}")
        self.results_text.append("   - comparison_results.csv")
        self.results_text.append("   - comparison_report.txt")
        self.results_text.append("   - plots/comparison_plots.png")
