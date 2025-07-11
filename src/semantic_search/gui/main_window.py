"""Главное окно приложения"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from semantic_search.config import GUI_CONFIG, MODELS_DIR
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer
from semantic_search.gui.evaluation_widget import EvaluationWidget
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.statistics import (
    calculate_statistics_from_processed_docs,
    format_statistics_for_display,
)

"""Исправленный класс TrainingThread для правильного подсчета времени"""


class TrainingThread(QThread):
    """Поток для обучения модели"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    statistics = pyqtSignal(dict)

    def __init__(
        self,
        documents_path: Path,
        model_name: str,
        vector_size: int,
        epochs: int,
        window: int = 15,
        min_count: int = 3,
        dm: int = 1,
        negative: int = 10,
        preset: Optional[str] = None,
    ):
        super().__init__()
        self.documents_path = documents_path
        self.model_name = model_name
        self.vector_size = vector_size
        self.epochs = epochs
        self.window = window
        self.min_count = min_count
        self.dm = dm
        self.negative = negative
        self.preset = preset
        self.is_cancelled = False

    def run(self):
        """Выполнение обучения"""
        try:
            # Начинаем отсчет общего времени
            start_time = time.time()

            # Обработка документов
            processor = DocumentProcessor()
            processed_docs = []

            self.progress.emit(10, "Поиск документов...")

            file_extractor = FileExtractor()
            file_paths = file_extractor.find_documents(self.documents_path)

            if not file_paths:
                self.finished.emit(False, "Документы не найдены")
                return

            # Обработка каждого документа
            step_size = 40 / len(file_paths)
            current_progress = 10

            for i, doc in enumerate(processor.process_documents(self.documents_path)):
                if self.is_cancelled:
                    self.finished.emit(False, "Обучение отменено")
                    return

                processed_docs.append(doc)
                current_progress += step_size
                self.progress.emit(
                    int(current_progress),
                    f"Обработан документ {i + 1}/{len(file_paths)}: {doc.relative_path}",
                )

            if not processed_docs:
                self.finished.emit(False, "Не удалось обработать документы")
                return

            # Подготовка корпуса
            corpus = [
                (doc.tokens, doc.relative_path, doc.metadata) for doc in processed_docs
            ]

            # Статистика корпуса
            stats = calculate_statistics_from_processed_docs(processed_docs)
            self.statistics.emit(stats)

            # Анализ языкового состава (новое)
            self.progress.emit(45, "Анализ языкового состава документов...")
            language_info = self._analyze_language_distribution(corpus)

            logger.info(f"Языковой состав: {language_info}")

            # Адаптация параметров на основе языкового состава
            adapted_params = self._adapt_params_for_language(language_info)

            # Применяем адаптированные параметры
            final_vector_size = adapted_params.get("vector_size", self.vector_size)
            final_window = adapted_params.get("window", self.window)
            final_min_count = adapted_params.get("min_count", self.min_count)

            if adapted_params:
                self.progress.emit(
                    48, "Параметры адаптированы для многоязычного корпуса"
                )
                logger.info(f"Адаптированные параметры: {adapted_params}")

            trainer = Doc2VecTrainer()

            # Обучение модели с адаптированными параметрами
            self.progress.emit(
                50,
                f"Обучение модели (векторы: {final_vector_size}, окно: {final_window})...",
            )

            model = trainer.train_model(
                corpus,
                vector_size=final_vector_size,  # Используем адаптированное значение
                epochs=self.epochs,
                window=final_window,  # Используем адаптированное значение
                min_count=final_min_count,  # Используем адаптированное значение
                dm=self.dm,
                negative=self.negative,
                sample=1e-5,
                preset=self.preset,
            )

            if model:
                # Вычисляем общее время включая обработку документов
                training_time = time.time() - start_time

                # Сохраняем метаданные обучения
                trainer.training_metadata = {
                    "training_time_formatted": f"{training_time:.1f}с ({training_time / 60:.1f}м)",
                    "training_date": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
                    ),
                    "corpus_size": len(processed_docs),
                    "documents_base_path": str(self.documents_path.absolute()),
                    "vector_size": self.vector_size,
                    "epochs": self.epochs,
                    "window": self.window,
                    "min_count": self.min_count,
                    "dm": self.dm,
                    "negative": self.negative,
                    "preset_used": self.preset,
                    "language_distribution": language_info,
                    "python_version": sys.version,
                    "platform": sys.platform,
                }

                self.progress.emit(90, "Сохранение модели...")
                success = trainer.save_model(model, self.model_name)

                if success:
                    self.progress.emit(100, "Обучение завершено!")
                    self.finished.emit(
                        True,
                        f"Модель '{self.model_name}' успешно обучена за {training_time / 60:.1f} минут",
                    )
                else:
                    self.finished.emit(False, "Ошибка при сохранении модели")
            else:
                self.finished.emit(False, "Ошибка при обучении модели")

        except Exception as e:
            logger.error(f"Ошибка в потоке обучения: {e}", exc_info=True)
            self.finished.emit(False, f"Ошибка: {str(e)}")

    def _analyze_language_distribution(self, corpus):
        """Анализ языкового состава корпуса"""
        language_stats = {"russian": 0, "english": 0, "mixed": 0}

        for tokens, doc_id, metadata in corpus[
            :100
        ]:  # Анализируем первые 100 документов
            # Подсчет кириллических и латинских токенов
            cyrillic_tokens = sum(
                1 for t in tokens[:200] if any("\u0400" <= c <= "\u04ff" for c in t)
            )
            latin_tokens = sum(1 for t in tokens[:200] if t.isalpha() and t.isascii())

            total = cyrillic_tokens + latin_tokens
            if total > 0:
                cyrillic_ratio = cyrillic_tokens / total

                if cyrillic_ratio > 0.8:
                    language_stats["russian"] += 1
                elif cyrillic_ratio < 0.2:
                    language_stats["english"] += 1
                else:
                    language_stats["mixed"] += 1

        # Экстраполируем на весь корпус
        sample_size = min(100, len(corpus))
        scale_factor = len(corpus) / sample_size

        return {
            "russian": int(language_stats["russian"] * scale_factor),
            "english": int(language_stats["english"] * scale_factor),
            "mixed": int(language_stats["mixed"] * scale_factor),
            "total": len(corpus),
        }

    def _adapt_params_for_language(self, language_info: dict) -> dict:
        """
        Адаптация параметров обучения на основе языкового состава

        Args:
            language_info: Статистика языков в корпусе

        Returns:
            Адаптированные параметры
        """
        total = language_info["total"]
        if total == 0:
            return {}

        # Вычисляем процентное соотношение
        russian_pct = language_info["russian"] / total
        english_pct = language_info["english"] / total
        mixed_pct = language_info["mixed"] / total

        adapted_params = {}

        # Адаптация размерности векторов
        if mixed_pct > 0.3 or (russian_pct > 0.2 and english_pct > 0.2):
            # Много смешанных документов или оба языка представлены значительно
            adapted_params["vector_size"] = min(400, self.vector_size + 50)
            logger.info(
                f"Увеличена размерность векторов до {adapted_params['vector_size']} для многоязычного корпуса"
            )

        # Адаптация размера окна
        if english_pct > 0.5:
            # Английские тексты часто имеют более короткие предложения
            adapted_params["window"] = max(10, self.window - 2)
        elif mixed_pct > 0.3:
            # Смешанные тексты требуют большего контекста
            adapted_params["window"] = min(20, self.window + 3)

        # Адаптация минимальной частоты
        if total < 100:
            # Маленький корпус - снижаем порог
            adapted_params["min_count"] = max(1, self.min_count - 1)
        elif mixed_pct > 0.3:
            # Смешанный корпус - повышаем порог для фильтрации шума
            adapted_params["min_count"] = self.min_count + 1

        return adapted_params

    def cancel(self):
        """Отмена обучения"""
        self.is_cancelled = True


class SearchThread(QThread):
    """Поток для поиска"""

    results = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, search_engine: SemanticSearchEngine, query: str, top_k: int):
        super().__init__()
        self.search_engine = search_engine
        self.query = query
        self.top_k = top_k

    def run(self):
        """Выполнение поиска"""
        try:
            results = self.search_engine.search(self.query, top_k=self.top_k)
            self.results.emit(results)
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.current_model = None
        self.search_engine = None
        self.training_thread = None
        self.search_thread = None
        self.summarizer = None

        self.init_ui()
        self.load_models()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle(GUI_CONFIG["window_title"])
        self.setGeometry(100, 100, *GUI_CONFIG["window_size"])

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout(central_widget)

        # Создаем меню
        self.create_menu_bar()

        # Создаем панель инструментов
        self.create_toolbar()

        # Создаем вкладки
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Вкладка обучения
        self.create_training_tab()

        # Вкладка поиска
        self.create_search_tab()

        # Вкладка суммаризации
        self.create_summarization_tab()

        # Вкладка статистики
        self.create_statistics_tab()

        # Вкладка оценки и сравнения
        self.create_evaluation_tab()

        # Статус бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")

        # Применяем стили
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0066cc;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #0066cc;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
            QLineEdit, QTextEdit, QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 3px;
            }
        """)

    def create_menu_bar(self):
        """Создание меню"""
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu("Файл")

        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню Модель
        model_menu = menubar.addMenu("Модель")

        load_model_action = QAction("Загрузить модель", self)
        load_model_action.triggered.connect(self.load_model_dialog)
        model_menu.addAction(load_model_action)

        # Меню Помощь
        help_menu = menubar.addMenu("Помощь")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Комбобокс для выбора модели
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        toolbar.addWidget(QLabel("Модель: "))
        toolbar.addWidget(self.model_combo)
        toolbar.addSeparator()

        # Индикатор статуса модели
        self.model_status_label = QLabel("Модель не загружена")
        self.model_status_label.setStyleSheet("color: red;")
        toolbar.addWidget(self.model_status_label)

    def create_search_tab(self):
        """Создание вкладки поиска"""
        search_widget = QWidget()
        layout = QVBoxLayout(search_widget)

        # Панель поиска
        search_panel = QWidget()
        search_layout = QHBoxLayout(search_panel)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Введите поисковый запрос...")
        self.search_input.returnPressed.connect(self.perform_search)

        self.search_button = QPushButton("Поиск")
        self.search_button.clicked.connect(self.perform_search)

        self.results_count_spin = QSpinBox()
        self.results_count_spin.setMinimum(1)
        self.results_count_spin.setMaximum(100)
        self.results_count_spin.setValue(10)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(QLabel("Результатов:"))
        search_layout.addWidget(self.results_count_spin)
        search_layout.addWidget(self.search_button)

        layout.addWidget(search_panel)

        # Разделитель для результатов
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Список результатов
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_selected)
        splitter.addWidget(self.results_list)

        # Просмотр документа
        self.document_viewer = QTextEdit()
        self.document_viewer.setReadOnly(True)
        splitter.addWidget(self.document_viewer)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        self.tab_widget.addTab(search_widget, "🔍 Поиск")

    def create_training_tab(self):
        """Создание вкладки обучения"""
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)

        # Группа выбора документов
        docs_group = QGroupBox("Документы для обучения")
        docs_layout = QHBoxLayout()

        self.docs_path_edit = QLineEdit()
        self.docs_path_edit.setPlaceholderText("Путь к папке с документами...")

        browse_button = QPushButton("Обзор...")
        browse_button.clicked.connect(self.browse_documents)

        docs_layout.addWidget(self.docs_path_edit)
        docs_layout.addWidget(browse_button)
        docs_group.setLayout(docs_layout)

        layout.addWidget(docs_group)

        # Параметры модели
        params_group = QGroupBox("Параметры модели")
        params_layout = QVBoxLayout()

        # Имя модели
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Имя модели:"))
        self.model_name_edit = QLineEdit("doc2vec_model")
        name_layout.addWidget(self.model_name_edit)
        params_layout.addLayout(name_layout)

        # Выбор пресета
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Пресет настроек:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "Сбалансированный (рекомендуется)",
                "Быстрый (для тестирования)",
                "Качественный (медленный)",
                "Пользовательский",
            ]
        )
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        params_layout.addLayout(preset_layout)

        # Размерность векторов
        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Размерность векторов:"))
        self.vector_size_spin = QSpinBox()
        self.vector_size_spin.setMinimum(50)
        self.vector_size_spin.setMaximum(500)
        self.vector_size_spin.setValue(300)
        vector_layout.addWidget(self.vector_size_spin)
        params_layout.addLayout(vector_layout)

        # Количество эпох
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Количество эпох:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(100)
        self.epochs_spin.setValue(30)
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)

        # Дополнительные параметры (расширенные)
        advanced_group = QGroupBox("Расширенные параметры (необязательно)")
        advanced_layout = QVBoxLayout()

        # Размер окна
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Размер окна контекста:"))
        self.window_spin = QSpinBox()
        self.window_spin.setMinimum(5)
        self.window_spin.setMaximum(50)
        self.window_spin.setValue(15)
        window_layout.addWidget(self.window_spin)
        advanced_layout.addLayout(window_layout)

        # Минимальная частота
        min_count_layout = QHBoxLayout()
        min_count_layout.addWidget(QLabel("Минимальная частота слова:"))
        self.min_count_spin = QSpinBox()
        self.min_count_spin.setMinimum(1)
        self.min_count_spin.setMaximum(10)
        self.min_count_spin.setValue(3)
        min_count_layout.addWidget(self.min_count_spin)
        advanced_layout.addLayout(min_count_layout)

        # DM режим
        dm_layout = QHBoxLayout()
        dm_layout.addWidget(QLabel("Режим обучения:"))
        self.dm_combo = QComboBox()
        self.dm_combo.addItems(
            ["Distributed Memory (DM)", "Distributed Bag of Words (DBOW)"]
        )
        self.dm_combo.setCurrentIndex(0)
        dm_layout.addWidget(self.dm_combo)
        advanced_layout.addLayout(dm_layout)

        # Negative sampling
        negative_layout = QHBoxLayout()
        negative_layout.addWidget(QLabel("Negative sampling:"))
        self.negative_spin = QSpinBox()
        self.negative_spin.setMinimum(0)
        self.negative_spin.setMaximum(20)
        self.negative_spin.setValue(10)
        negative_layout.addWidget(self.negative_spin)
        advanced_layout.addLayout(negative_layout)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Кнопка обучения
        self.train_button = QPushButton("Начать обучение")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)

        # Прогресс бар
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)

        # Лог обучения
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)

        self.tab_widget.addTab(training_widget, "🧠 Обучение")

    def on_preset_changed(self, index):
        """Обработка изменения пресета настроек"""
        if index == 0:  # Сбалансированный
            self.vector_size_spin.setValue(300)
            self.epochs_spin.setValue(30)
            self.window_spin.setValue(15)
            self.min_count_spin.setValue(3)
            self.negative_spin.setValue(10)
            self.training_log.append(
                "📋 Выбран сбалансированный пресет (рекомендуется для большинства случаев)"
            )

        elif index == 1:  # Быстрый
            self.vector_size_spin.setValue(200)
            self.epochs_spin.setValue(15)
            self.window_spin.setValue(10)
            self.min_count_spin.setValue(5)
            self.negative_spin.setValue(5)
            self.training_log.append("⚡ Выбран быстрый пресет (для тестирования)")

        elif index == 2:  # Качественный
            self.vector_size_spin.setValue(400)
            self.epochs_spin.setValue(50)
            self.window_spin.setValue(20)
            self.min_count_spin.setValue(2)
            self.negative_spin.setValue(15)
            self.training_log.append(
                "🏆 Выбран качественный пресет (максимальное качество, медленное обучение)"
            )

        elif index == 3:  # Пользовательский
            self.training_log.append(
                "🔧 Пользовательский режим - настройте параметры вручную"
            )

    def create_summarization_tab(self):
        """Создание вкладки суммаризации"""
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)

        # Выбор файла
        file_group = QGroupBox("Выбор документа")
        file_layout = QHBoxLayout()

        self.summary_file_edit = QLineEdit()
        self.summary_file_edit.setPlaceholderText("Путь к файлу...")

        browse_file_button = QPushButton("Обзор...")
        browse_file_button.clicked.connect(self.browse_summary_file)

        file_layout.addWidget(self.summary_file_edit)
        file_layout.addWidget(browse_file_button)
        file_group.setLayout(file_layout)

        layout.addWidget(file_group)

        # Параметры суммаризации
        params_group = QGroupBox("Параметры выжимки")
        params_layout = QVBoxLayout()

        # Количество предложений
        sentences_layout = QHBoxLayout()
        sentences_layout.addWidget(QLabel("Количество предложений:"))

        self.sentences_spin = QSpinBox()
        self.sentences_spin.setMinimum(1)
        self.sentences_spin.setMaximum(20)
        self.sentences_spin.setValue(5)
        self.sentences_spin.setToolTip("Количество предложений в выжимке")
        sentences_layout.addWidget(self.sentences_spin)

        sentences_layout.addStretch()
        params_layout.addLayout(sentences_layout)

        # Минимальная длина предложения
        min_length_layout = QHBoxLayout()
        min_length_layout.addWidget(QLabel("Минимальная длина предложения:"))

        self.min_sentence_length_spin = QSpinBox()
        self.min_sentence_length_spin.setMinimum(10)
        self.min_sentence_length_spin.setMaximum(100)
        self.min_sentence_length_spin.setValue(15)
        self.min_sentence_length_spin.setSuffix(" символов")
        self.min_sentence_length_spin.setToolTip(
            "Предложения короче этого значения не будут включены в выжимку"
        )
        min_length_layout.addWidget(self.min_sentence_length_spin)

        min_length_layout.addStretch()
        params_layout.addLayout(min_length_layout)

        # Минимальное количество слов
        min_words_layout = QHBoxLayout()
        min_words_layout.addWidget(QLabel("Минимум слов в предложении:"))

        self.min_words_spin = QSpinBox()
        self.min_words_spin.setMinimum(3)
        self.min_words_spin.setMaximum(20)
        self.min_words_spin.setValue(5)
        self.min_words_spin.setToolTip(
            "Предложения с меньшим количеством слов будут отфильтрованы"
        )
        min_words_layout.addWidget(self.min_words_spin)

        min_words_layout.addStretch()
        params_layout.addLayout(min_words_layout)

        # Флажок для фильтрации
        self.filter_short_checkbox = QCheckBox(
            "Фильтровать короткие и малоинформативные предложения"
        )
        self.filter_short_checkbox.setChecked(True)
        self.filter_short_checkbox.toggled.connect(self.on_filter_toggled)
        params_layout.addWidget(self.filter_short_checkbox)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Кнопка создания выжимки
        button_layout = QHBoxLayout()
        self.summarize_button = QPushButton("Создать выжимку")
        self.summarize_button.clicked.connect(self.create_summary)
        button_layout.addWidget(self.summarize_button)

        # Кнопка сохранения выжимки
        self.save_summary_button = QPushButton("Сохранить выжимку")
        self.save_summary_button.clicked.connect(self.save_summary)
        self.save_summary_button.setEnabled(False)
        button_layout.addWidget(self.save_summary_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Результат
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Оригинальный текст
        original_group = QGroupBox("Оригинальный текст")
        original_layout = QVBoxLayout()
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        original_layout.addWidget(self.original_text)
        original_group.setLayout(original_layout)
        splitter.addWidget(original_group)

        # Выжимка
        summary_group = QGroupBox("Выжимка")
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        splitter.addWidget(summary_group)

        layout.addWidget(splitter)

        self.tab_widget.addTab(summary_widget, "📝 Суммаризация")

        # Сохраняем текущую выжимку для возможности сохранения
        self.current_summary = []

    def on_filter_toggled(self, checked):
        """Обработчик переключения фильтрации"""
        self.min_sentence_length_spin.setEnabled(checked)
        self.min_words_spin.setEnabled(checked)

        if checked:
            self.status_bar.showMessage("Фильтрация коротких предложений включена")
        else:
            self.status_bar.showMessage(
                "Фильтрация отключена - все предложения будут учитываться"
            )

    def save_summary(self):
        """Сохранение выжимки в файл"""
        if not self.current_summary:
            QMessageBox.warning(self, "Ошибка", "Нет выжимки для сохранения")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить выжимку", "", "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # Заголовок
                    f.write(f"Выжимка документа: {self.summary_file_edit.text()}\n")
                    f.write(f"Создано: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")

                    # Предложения выжимки
                    for i, sentence in enumerate(self.current_summary, 1):
                        f.write(f"{i}. {sentence.strip()}\n\n")

                    # Статистика если есть
                    if hasattr(self, "last_summary_stats"):
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("СТАТИСТИКА СУММАРИЗАЦИИ\n")
                        f.write("=" * 60 + "\n")
                        stats = self.last_summary_stats
                        f.write(
                            f"Исходных предложений: {stats['original_sentences_count']}\n"
                        )
                        f.write(
                            f"Валидных предложений: {stats.get('valid_original_sentences_count', 'н/д')}\n"
                        )
                        f.write(
                            f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                        )
                        f.write(
                            f"Коэффициент сжатия: {stats['compression_ratio']:.1%}\n"
                        )
                        f.write(
                            f"Средняя длина предложения: {stats.get('avg_sentence_length', 0):.1f} слов\n"
                        )

                QMessageBox.information(
                    self, "Успех", f"Выжимка сохранена в:\n{file_path}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Ошибка при сохранении:\n{str(e)}"
                )

    def create_statistics_tab(self):
        """Создание вкладки статистики"""
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)

        # Кнопка обновления
        refresh_button = QPushButton("Обновить статистику")
        refresh_button.clicked.connect(self.update_statistics)
        layout.addWidget(refresh_button)

        # Текстовое поле для статистики
        self.statistics_text = QTextEdit()
        self.statistics_text.setReadOnly(True)
        self.statistics_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.statistics_text)

        self.tab_widget.addTab(stats_widget, "📊 Статистика")

    def create_evaluation_tab(self):
        """Создание вкладки оценки и сравнения"""
        self.evaluation_widget = EvaluationWidget()
        self.tab_widget.addTab(self.evaluation_widget, "📚 Оценка методов")

    def load_models(self):
        """Загрузка списка доступных моделей"""
        self.model_combo.clear()
        self.model_combo.addItem("Не выбрано")

        try:
            # Создаем директорию если её нет
            MODELS_DIR.mkdir(exist_ok=True, parents=True)

            # Ищем модели в директории
            model_files = list(MODELS_DIR.glob("*.model"))

            for model_file in model_files:
                model_name = model_file.stem
                if model_name:  # Проверяем что имя не пустое
                    self.model_combo.addItem(model_name)

            if len(model_files) > 0:
                self.model_combo.setCurrentIndex(1)
        except Exception as e:
            logger.error(f"Ошибка при загрузке списка моделей: {e}")
            QMessageBox.warning(
                self, "Ошибка", f"Не удалось загрузить список моделей: {e}"
            )

    def on_model_changed(self, model_name: str):
        """Обработчик изменения модели"""
        if not model_name or model_name == "Не выбрано":
            self.current_model = None
            self.search_engine = None
            self.summarizer = None
            self.model_status_label.setText("Модель не загружена")
            self.model_status_label.setStyleSheet("color: red;")

            # Отключаем evaluation widget
            if hasattr(self, "evaluation_widget"):
                self.evaluation_widget.set_search_engine(None, None)
            return

        # Загружаем модель
        try:
            logger.info(f"Загрузка модели: {model_name}")
            trainer = Doc2VecTrainer()
            model = trainer.load_model(model_name)

            if model:
                self.current_model = model

                # Создаем SearchEngine с базовым путем
                self.search_engine = SemanticSearchEngine(
                    model,
                    trainer.corpus_info,
                    trainer.documents_base_path,  # Передаем базовый путь
                )

                self.summarizer = TextSummarizer(model)

                # Передаем данные в evaluation widget
                if hasattr(self, "evaluation_widget"):
                    self.evaluation_widget.set_search_engine(
                        self.search_engine, trainer.corpus_info
                    )

                # Обновляем статус
                status_text = f"Модель '{model_name}' загружена"
                if trainer.documents_base_path:
                    status_text += f" (база: {trainer.documents_base_path.name})"

                self.model_status_label.setText(status_text)
                self.model_status_label.setStyleSheet("color: green;")

                self.status_bar.showMessage(f"Модель '{model_name}' успешно загружена")

                # Логирование для отладки
                if trainer.documents_base_path:
                    logger.info(
                        f"Базовый путь документов: {trainer.documents_base_path}"
                    )
                    logger.info(
                        f"Путь существует: {trainer.documents_base_path.exists()}"
                    )
                else:
                    logger.warning("Базовый путь документов не загружен из модели")

            else:
                logger.error(f"Модель {model_name} не может быть загружена")
                self.current_model = None
                self.search_engine = None
                self.summarizer = None
                self.model_status_label.setText("Ошибка загрузки модели")
                self.model_status_label.setStyleSheet("color: red;")

                if hasattr(self, "evaluation_widget"):
                    self.evaluation_widget.set_search_engine(None, None)

                QMessageBox.warning(
                    self, "Ошибка", f"Не удалось загрузить модель '{model_name}'"
                )
        except Exception as e:
            logger.error(f"Исключение при загрузке модели: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при загрузке модели: {str(e)}"
            )

    def browse_documents(self):
        """Выбор папки с документами"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с документами")
        if folder:
            self.docs_path_edit.setText(folder)

    def browse_summary_file(self):
        """Выбор файла для суммаризации"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "Документы (*.pdf *.docx *.doc);;Все файлы (*.*)"
        )
        if file_path:
            self.summary_file_edit.setText(file_path)

    def start_training(self):
        """Начало обучения модели с сохранением конфигурации"""
        documents_path = self.docs_path_edit.text()
        if not documents_path:
            QMessageBox.warning(self, "Ошибка", "Выберите папку с документами")
            return

        documents_path = Path(documents_path)
        if not documents_path.exists():
            QMessageBox.warning(self, "Ошибка", "Указанная папка не существует")
            return

        model_name = self.model_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Ошибка", "Введите имя модели")
            return

        # Проверяем валидность имени модели
        if "/" in model_name or "\\" in model_name or ":" in model_name:
            QMessageBox.warning(
                self, "Ошибка", "Имя модели содержит недопустимые символы"
            )
            return

        # Проверяем, не существует ли уже такая модель
        existing_model = MODELS_DIR / f"{model_name}.model"
        if existing_model.exists():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                f"Модель '{model_name}' уже существует. Перезаписать?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Собираем текущие параметры из GUI
        current_params = {
            "vector_size": self.vector_size_spin.value(),
            "window": self.window_spin.value(),
            "min_count": self.min_count_spin.value(),
            "epochs": self.epochs_spin.value(),
            "dm": 1 if self.dm_combo.currentIndex() == 0 else 0,
            "negative": self.negative_spin.value(),
        }

        # Обновляем конфигурацию если параметры изменились
        from semantic_search.config import config_manager

        config_changed = False
        current_config = config_manager.config.doc2vec

        for param, value in current_params.items():
            if current_config.get(param) != value:
                config_changed = True
                break

        if config_changed:
            # Обновляем конфигурацию
            config_manager.update_config(doc2vec=current_params)
            logger.info("Конфигурация обновлена с новыми параметрами обучения")

            # Информируем пользователя
            self.status_bar.showMessage("Параметры обучения сохранены в конфигурацию")

        # Отключаем кнопку
        self.train_button.setEnabled(False)
        self.training_progress.setVisible(True)
        self.training_progress.setValue(0)

        # Очищаем лог
        self.training_log.clear()
        self.training_log.append("Начинаем обучение модели...\n")

        # Показываем используемые параметры
        self.training_log.append("📋 Параметры обучения:")
        self.training_log.append(
            f"   Размерность векторов: {current_params['vector_size']}"
        )
        self.training_log.append(f"   Размер окна: {current_params['window']}")
        self.training_log.append(
            f"   Минимальная частота: {current_params['min_count']}"
        )
        self.training_log.append(f"   Количество эпох: {current_params['epochs']}")
        self.training_log.append(
            f"   Режим: {'DM' if current_params['dm'] == 1 else 'DBOW'}"
        )
        self.training_log.append(
            f"   Negative sampling: {current_params['negative']}\n"
        )

        # Определяем пресет
        preset_index = self.preset_combo.currentIndex()
        preset_map = {0: "balanced", 1: "fast", 2: "quality", 3: None}
        preset = preset_map.get(preset_index)

        # Создаем и запускаем поток
        self.training_thread = TrainingThread(
            documents_path,
            model_name,
            current_params["vector_size"],
            current_params["epochs"],
            window=current_params["window"],
            min_count=current_params["min_count"],
            dm=current_params["dm"],
            negative=current_params["negative"],
            preset=preset,
        )

        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.statistics.connect(self.on_training_statistics)

        self.training_thread.start()

    def on_training_progress(self, value: int, message: str):
        """Обновление прогресса обучения"""
        self.training_progress.setValue(value)
        self.training_log.append(message)
        self.status_bar.showMessage(message)

    def on_training_statistics(self, stats: dict):
        """Отображение статистики корпуса"""
        stats_text = format_statistics_for_display(stats)
        self.training_log.append("\n" + stats_text + "\n")

    def on_training_finished(self, success: bool, message: str):
        """Завершение обучения"""
        self.train_button.setEnabled(True)
        self.training_progress.setVisible(False)

        if success:
            self.training_log.append(f"\n✅ {message}")
            QMessageBox.information(self, "Успех", message)

            # Обновляем список моделей
            self.load_models()

            # Пытаемся выбрать только что обученную модель
            model_name = self.model_name_edit.text().strip()
            if model_name:
                index = self.model_combo.findText(model_name)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                else:
                    logger.warning(
                        f"Не удалось найти модель '{model_name}' в списке после обучения"
                    )
        else:
            self.training_log.append(f"\n❌ {message}")
            QMessageBox.critical(self, "Ошибка", message)

        self.status_bar.showMessage("Готов к работе")

    def perform_search(self):
        """Выполнение поиска"""
        if not self.search_engine:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return

        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Ошибка", "Введите поисковый запрос")
            return

        # Очищаем предыдущие результаты
        self.results_list.clear()
        self.document_viewer.clear()

        # Отключаем кнопку
        self.search_button.setEnabled(False)
        self.status_bar.showMessage("Выполняется поиск...")

        # Создаем поток поиска
        self.search_thread = SearchThread(
            self.search_engine, query, self.results_count_spin.value()
        )

        self.search_thread.results.connect(self.on_search_results)
        self.search_thread.error.connect(self.on_search_error)

        self.search_thread.start()

    def on_search_results(self, results: List):
        """Обработка результатов поиска"""
        self.search_button.setEnabled(True)

        if not results:
            self.status_bar.showMessage("Результатов не найдено")
            QMessageBox.information(
                self, "Поиск", "По вашему запросу ничего не найдено"
            )
            return

        self.status_bar.showMessage(f"Найдено результатов: {len(results)}")

        # Отображаем результаты
        for i, result in enumerate(results, 1):
            item = QListWidgetItem(
                f"{i}. {result.doc_id} (схожесть: {result.similarity:.3f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, result)
            self.results_list.addItem(item)

    def on_search_error(self, error: str):
        """Обработка ошибки поиска"""
        self.search_button.setEnabled(True)
        self.status_bar.showMessage("Ошибка поиска")
        QMessageBox.critical(self, "Ошибка", f"Ошибка при поиске: {error}")

    def on_result_selected(self, item: QListWidgetItem):
        """Обработка выбора результата"""
        result = item.data(Qt.ItemDataRole.UserRole)

        try:
            # Получаем относительный путь из результата
            relative_path = result.doc_id

            logger.info(f"Выбран документ: {relative_path}")

            file_path = None

            # Основной способ: используем базовый путь из SearchEngine
            if (
                self.search_engine
                and hasattr(self.search_engine, "documents_base_path")
                and self.search_engine.documents_base_path
            ):
                # Строим полный путь: базовый путь + относительный путь
                file_path = self.search_engine.documents_base_path / relative_path

                logger.info(f"Базовый путь: {self.search_engine.documents_base_path}")
                logger.info(f"Полный путь: {file_path}")
                logger.info(f"Файл существует: {file_path.exists()}")

                # Если файл не найден, пробуем с нормализацией слешей
                if not file_path.exists():
                    # Нормализуем слеши для текущей ОС
                    normalized_relative = relative_path.replace("/", os.sep).replace(
                        "\\", os.sep
                    )
                    file_path = (
                        self.search_engine.documents_base_path / normalized_relative
                    )

                    if file_path.exists():
                        logger.info("Файл найден после нормализации путей")
            else:
                logger.warning("Базовый путь не доступен в SearchEngine")

            # Запасной вариант: проверяем полный путь из метаданных
            if (
                (not file_path or not file_path.exists())
                and result.metadata
                and "full_path" in result.metadata
            ):
                test_path = Path(result.metadata["full_path"])
                if test_path.exists():
                    file_path = test_path
                    logger.info(f"Использован full_path из метаданных: {file_path}")

            # Отображение результата
            if file_path and file_path.exists():
                extractor = FileExtractor()
                text = extractor.extract_text(file_path)

                if text:
                    # Показываем первые 5000 символов
                    preview = text[:5000]
                    if len(text) > 5000:
                        preview += "\n\n... (текст обрезан) ..."

                    self.document_viewer.setPlainText(preview)

                    # Добавляем метаданные
                    metadata_text = "\n\n" + "=" * 60 + "\n"
                    metadata_text += "ИНФОРМАЦИЯ О ДОКУМЕНТЕ\n"
                    metadata_text += "=" * 60 + "\n"
                    metadata_text += f"📄 Документ: {relative_path}\n"
                    metadata_text += f"📁 Полный путь: {file_path}\n"
                    metadata_text += f"📊 Схожесть: {result.similarity:.3f}\n"

                    if self.search_engine and self.search_engine.documents_base_path:
                        metadata_text += f"📂 Базовая папка модели: {self.search_engine.documents_base_path}\n"

                    if result.metadata:
                        metadata_text += (
                            f"💾 Размер: {result.metadata.get('file_size', 0):,} байт\n"
                        )
                        metadata_text += (
                            f"📝 Токенов: {result.metadata.get('tokens_count', 0):,}\n"
                        )
                        metadata_text += f"📑 Расширение: {result.metadata.get('extension', 'н/д')}\n"

                    self.document_viewer.append(metadata_text)
                else:
                    self.document_viewer.setPlainText(
                        f"❌ Не удалось извлечь текст из документа:\n{file_path}\n\n"
                        f"Возможно, файл поврежден или имеет неподдерживаемый формат."
                    )
            else:
                # Подробное сообщение об ошибке
                error_msg = "❌ ФАЙЛ НЕ НАЙДЕН\n\n"
                error_msg += f"Искомый документ: {relative_path}\n\n"

                if self.search_engine and hasattr(
                    self.search_engine, "documents_base_path"
                ):
                    if self.search_engine.documents_base_path:
                        error_msg += f"Базовая папка модели: {self.search_engine.documents_base_path}\n"
                        error_msg += f"Ожидаемый путь: {self.search_engine.documents_base_path / relative_path}\n"
                        error_msg += f"Базовая папка существует: {'✅ Да' if self.search_engine.documents_base_path.exists() else '❌ Нет'}\n"
                    else:
                        error_msg += "⚠️ Базовый путь не сохранен в модели\n"

                error_msg += "\n📋 Возможные причины:\n"
                error_msg += "1. Файлы были перемещены после обучения модели\n"
                error_msg += "2. Модель была обучена на другом компьютере\n"
                error_msg += "3. Изменилась структура папок\n"

                if not (self.search_engine and self.search_engine.documents_base_path):
                    error_msg += "4. Модель была обучена старой версией программы без сохранения базового пути\n"

                error_msg += "\n💡 Рекомендации:\n"
                error_msg += "• Переобучите модель с текущим расположением документов\n"
                error_msg += "• Или переместите документы в исходное расположение\n"

                self.document_viewer.setPlainText(error_msg)

        except Exception as e:
            logger.error(f"Ошибка при отображении документа: {e}", exc_info=True)
            self.document_viewer.setPlainText(
                f"❌ ОШИБКА ПРИ ЗАГРУЗКЕ ДОКУМЕНТА\n\n"
                f"Документ: {result.doc_id}\n"
                f"Ошибка: {str(e)}\n\n"
                f"Проверьте логи для подробной информации."
            )

    def create_summary(self):
        """Создание выжимки документа с учетом параметров фильтрации"""
        if not self.summarizer:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return

        file_path = self.summary_file_edit.text()
        if not file_path:
            QMessageBox.warning(self, "Ошибка", "Выберите файл для суммаризации")
            return

        file_path = Path(file_path)
        if not file_path.exists():
            QMessageBox.warning(self, "Ошибка", "Файл не существует")
            return

        try:
            # Загружаем текст
            extractor = FileExtractor()
            text = extractor.extract_text(file_path)

            if not text:
                QMessageBox.warning(self, "Ошибка", "Не удалось извлечь текст из файла")
                return

            # Проверяем размер текста
            text_length = len(text)
            logger.info(f"Загружен текст длиной {text_length} символов")

            # Предупреждение для очень больших файлов
            if text_length > 2_000_000:
                reply = QMessageBox.question(
                    self,
                    "Большой файл",
                    f"Файл содержит {text_length:,} символов.\n"
                    "Обработка может занять несколько минут.\n"
                    "Продолжить?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Показываем оригинальный текст (ограничиваем превью)
            preview_length = min(5000, text_length)
            preview = text[:preview_length]
            if text_length > preview_length:
                preview += f"\n\n... (показаны первые {preview_length:,} из {text_length:,} символов) ..."
            self.original_text.setPlainText(preview)

            # Обновляем параметры суммаризатора если включена фильтрация
            if self.filter_short_checkbox.isChecked():
                self.summarizer.min_summary_sentence_length = (
                    self.min_sentence_length_spin.value()
                )
                self.summarizer.min_words_in_sentence = self.min_words_spin.value()
            else:
                # Если фильтрация отключена, устанавливаем минимальные значения
                self.summarizer.min_summary_sentence_length = 1
                self.summarizer.min_words_in_sentence = 1

            # Создаем выжимку
            self.status_bar.showMessage(
                "Создание выжимки... Это может занять время для больших файлов"
            )
            QApplication.processEvents()  # Обновляем UI

            sentences_count = self.sentences_spin.value()

            # Отключаем кнопки на время обработки
            self.summarize_button.setEnabled(False)
            self.save_summary_button.setEnabled(False)

            try:
                # Создаем выжимку с учетом фильтрации
                summary = self.summarizer.summarize_text(
                    text,
                    sentences_count=sentences_count,
                    min_sentence_length=self.min_sentence_length_spin.value()
                    if self.filter_short_checkbox.isChecked()
                    else None,
                )

                if summary:
                    # Сохраняем текущую выжимку
                    self.current_summary = summary

                    # Показываем выжимку
                    summary_text = "\n\n".join(
                        f"{i}. {sent.strip()}" for i, sent in enumerate(summary, 1)
                    )
                    self.summary_text.setPlainText(summary_text)

                    # Показываем статистику
                    stats = self.summarizer.get_summary_statistics(text, summary)
                    self.last_summary_stats = (
                        stats  # Сохраняем для последующего сохранения
                    )

                    stats_text = "\n\n--- Статистика ---\n"
                    stats_text += (
                        f"Исходных предложений: {stats['original_sentences_count']}\n"
                    )

                    # Показываем информацию о фильтрации
                    if (
                        self.filter_short_checkbox.isChecked()
                        and "valid_original_sentences_count" in stats
                    ):
                        filtered_count = (
                            stats["original_sentences_count"]
                            - stats["valid_original_sentences_count"]
                        )
                        stats_text += f"Отфильтровано коротких: {filtered_count}\n"
                        stats_text += f"Валидных предложений: {stats['valid_original_sentences_count']}\n"

                    stats_text += (
                        f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                    )
                    stats_text += f"Сжатие: {stats['compression_ratio']:.1%}\n"
                    stats_text += (
                        f"Исходных символов: {stats['original_chars_count']:,}\n"
                    )
                    stats_text += (
                        f"Символов в выжимке: {stats['summary_chars_count']:,}\n"
                    )

                    if "avg_sentence_length" in stats:
                        stats_text += f"Средняя длина предложения: {stats['avg_sentence_length']:.1f} слов\n"

                    self.summary_text.append(stats_text)

                    # Включаем кнопку сохранения
                    self.save_summary_button.setEnabled(True)
                    self.status_bar.showMessage("Выжимка создана успешно")
                else:
                    QMessageBox.warning(
                        self,
                        "Ошибка",
                        "Не удалось создать выжимку.\n"
                        "Возможно, все предложения слишком короткие.\n"
                        "Попробуйте уменьшить минимальную длину предложения.",
                    )
                    self.status_bar.showMessage("Ошибка создания выжимки")

            finally:
                self.summarize_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Ошибка при создании выжимки: {e}")
            self.summarize_button.setEnabled(True)

            # Специальная обработка ошибки SpaCy
            if "exceeds maximum" in str(e):
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    "Файл слишком большой для обработки.\n"
                    "Попробуйте файл меньшего размера или разделите его на части.",
                )
            else:
                QMessageBox.critical(
                    self, "Ошибка", f"Ошибка при создании выжимки: {str(e)}"
                )

            self.status_bar.showMessage("Ошибка")

    def update_statistics(self):
        """Обновление статистики"""
        stats_text = "📊 СТАТИСТИКА СИСТЕМЫ\n" + "=" * 50 + "\n\n"

        # Статистика моделей
        stats_text += "🧠 МОДЕЛИ:\n"
        model_files = list(MODELS_DIR.glob("*.model"))
        stats_text += f"Всего моделей: {len(model_files)}\n\n"

        # Получаем имя текущей модели
        current_model_name = (
            self.model_combo.currentText() if hasattr(self, "model_combo") else None
        )
        is_current_model_shown = False

        for model_file in model_files:
            model_name = model_file.stem
            file_size_mb = model_file.stat().st_size / 1024 / 1024

            # Проверяем, является ли это текущей моделью
            is_current = (
                current_model_name == model_name and self.current_model is not None
            )

            if is_current:
                is_current_model_shown = True
                stats_text += f"📍 ТЕКУЩАЯ МОДЕЛЬ: {model_name}\n"
            else:
                stats_text += f"📁 {model_name}:\n"

            stats_text += f"   Размер файла: {file_size_mb:.1f} МБ\n"

            # Пытаемся загрузить метаданные модели
            metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Показываем только базовую информацию для не-текущих моделей
                    if not is_current:
                        training_time = metadata.get(
                            "training_time_formatted", "Неизвестно"
                        )
                        training_date = metadata.get("training_date", "Неизвестно")
                        corpus_size = metadata.get("corpus_size", 0)
                        stats_text += f"   Дата обучения: {training_date}\n"
                        stats_text += f"   Время обучения: {training_time}\n"
                        stats_text += f"   Размер корпуса при обучении: {corpus_size}\n"

                    else:
                        # Для текущей модели показываем подробную информацию
                        trainer = Doc2VecTrainer()
                        trainer.model = self.current_model
                        trainer.training_metadata = metadata

                        model_info = trainer.get_model_info()

                        stats_text += (
                            f"   Размерность векторов: {model_info['vector_size']}\n"
                        )
                        stats_text += f"   Размер словаря: {model_info['vocabulary_size']:,} слов\n"
                        stats_text += (
                            f"   Документов в модели: {model_info['documents_count']}\n"
                        )
                        stats_text += f"   Размер окна: {model_info['window']}\n"
                        stats_text += (
                            f"   Минимальная частота: {model_info['min_count']}\n"
                        )
                        stats_text += f"   Эпох обучения: {model_info['epochs']}\n"

                        stats_text += f"   Время обучения: {model_info['training_time_formatted']}\n"
                        stats_text += (
                            f"   Дата обучения: {model_info['training_date']}\n"
                        )

                        # Режим обучения
                        if model_info["dm"] == 1:
                            stats_text += "   Режим: Distributed Memory (DM)\n"
                        else:
                            stats_text += "   Режим: Distributed Bag of Words (DBOW)\n"

                except Exception as e:
                    logger.debug(
                        f"Не удалось загрузить метаданные для {model_name}: {e}"
                    )

            stats_text += "\n"

        # Если текущая модель не была показана в списке (не загружена правильно)
        if self.current_model and not is_current_model_shown:
            stats_text += "⚠️ Текущая модель загружена, но метаданные недоступны\n\n"

        # Системная информация
        try:
            import psutil

            stats_text += "💻 СИСТЕМА:\n"
            stats_text += f"CPU: {psutil.cpu_percent()}% загрузка\n"
            stats_text += f"Память: {psutil.virtual_memory().percent}% использовано\n"
            stats_text += f"Свободно памяти: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} ГБ\n"
        except ImportError:
            stats_text += "\n💻 СИСТЕМА:\n"
            stats_text += "Установите psutil для отображения системной информации\n"

        self.statistics_text.setPlainText(stats_text)

    def load_model_dialog(self):
        """Диалог загрузки модели"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            str(MODELS_DIR),
            "Модели (*.model);;Все файлы (*.*)",
        )

        if file_path:
            model_name = Path(file_path).stem

            # Проверяем, есть ли модель в списке
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                # Копируем модель в нашу директорию
                import shutil

                try:
                    shutil.copy2(file_path, MODELS_DIR / Path(file_path).name)
                    self.load_models()
                    self.model_combo.setCurrentText(model_name)
                except Exception as e:
                    QMessageBox.critical(
                        self, "Ошибка", f"Не удалось загрузить модель: {str(e)}"
                    )

    def show_about(self):
        """Показать информацию о программе"""
        about_text = """
        <h2>Semantic Document Search</h2>
        <p>Версия 1.0.0</p>
        <p>Приложение для семантического поиска по документам с использованием технологии Doc2Vec.</p>
        <p><b>Возможности:</b></p>
        <ul>
        <li>Семантический поиск по содержимому документов</li>
        <li>Поддержка форматов PDF, DOCX, DOC</li>
        <li>Создание выжимок из документов</li>
        <li>Обучение собственных моделей</li>
        </ul>
        <p><b>Автор:</b> Evgeny Odintsov</p>
        <p><b>Email:</b> ev1genial@gmail.com</p>
        """

        QMessageBox.about(self, "О программе", about_text)

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        # Останавливаем потоки если они запущены
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.cancel()
            self.training_thread.wait()

        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.wait()

        event.accept()
