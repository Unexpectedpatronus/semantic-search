"""Главное окно приложения"""

from pathlib import Path
from typing import List

from loguru import logger
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
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

from semantic_search.config import DATA_DIR, GUI_CONFIG, MODELS_DIR
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.statistics import (
    calculate_statistics_from_processed_docs,
    format_statistics_for_display,
)


class TrainingThread(QThread):
    """Поток для обучения модели"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    statistics = pyqtSignal(dict)

    def __init__(
        self, documents_path: Path, model_name: str, vector_size: int, epochs: int
    ):
        super().__init__()
        self.documents_path = documents_path
        self.model_name = model_name
        self.vector_size = vector_size
        self.epochs = epochs
        self.is_cancelled = False

    def run(self):
        """Выполнение обучения"""
        try:
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

            # Обучение модели
            self.progress.emit(50, "Начинаем обучение модели...")

            trainer = Doc2VecTrainer()

            # Обучение с отслеживанием прогресса
            model = trainer.train_model(
                corpus, vector_size=self.vector_size, epochs=self.epochs
            )

            if model:
                self.progress.emit(90, "Сохранение модели...")
                trainer.save_model(model, self.model_name)
                self.progress.emit(100, "Обучение завершено!")
                self.finished.emit(True, f"Модель '{self.model_name}' успешно обучена")
            else:
                self.finished.emit(False, "Ошибка при обучении модели")

        except Exception as e:
            logger.error(f"Ошибка в потоке обучения: {e}")
            self.finished.emit(False, f"Ошибка: {str(e)}")

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

        # Вкладка поиска
        self.create_search_tab()

        # Вкладка обучения
        self.create_training_tab()

        # Вкладка суммаризации
        self.create_summarization_tab()

        # Вкладка статистики
        self.create_statistics_tab()

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

        # Размерность векторов
        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Размерность векторов:"))
        self.vector_size_spin = QSpinBox()
        self.vector_size_spin.setMinimum(50)
        self.vector_size_spin.setMaximum(500)
        self.vector_size_spin.setValue(150)
        vector_layout.addWidget(self.vector_size_spin)
        params_layout.addLayout(vector_layout)

        # Количество эпох
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Количество эпох:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(100)
        self.epochs_spin.setValue(40)
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)

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
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Количество предложений:"))

        self.sentences_spin = QSpinBox()
        self.sentences_spin.setMinimum(1)
        self.sentences_spin.setMaximum(20)
        self.sentences_spin.setValue(5)
        params_layout.addWidget(self.sentences_spin)

        self.summarize_button = QPushButton("Создать выжимку")
        self.summarize_button.clicked.connect(self.create_summary)
        params_layout.addWidget(self.summarize_button)

        params_layout.addStretch()
        layout.addLayout(params_layout)

        # Результат
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Оригинальный текст
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        splitter.addWidget(QLabel("Оригинальный текст:"))
        splitter.addWidget(self.original_text)

        # Выжимка
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        splitter.addWidget(QLabel("Выжимка:"))
        splitter.addWidget(self.summary_text)

        layout.addWidget(splitter)

        self.tab_widget.addTab(summary_widget, "📝 Суммаризация")

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
            return

        # Загружаем модель
        try:
            logger.info(f"Загрузка модели: {model_name}")
            trainer = Doc2VecTrainer()
            model = trainer.load_model(model_name)

            if model:
                self.current_model = model
                self.search_engine = SemanticSearchEngine(model, trainer.corpus_info)
                self.summarizer = TextSummarizer(model)

                self.model_status_label.setText(f"Модель '{model_name}' загружена")
                self.model_status_label.setStyleSheet("color: green;")

                self.status_bar.showMessage(f"Модель '{model_name}' успешно загружена")
            else:
                logger.error(f"Модель {model_name} не может быть загружена")
                self.current_model = None
                self.search_engine = None
                self.summarizer = None
                self.model_status_label.setText("Ошибка загрузки модели")
                self.model_status_label.setStyleSheet("color: red;")
                QMessageBox.warning(
                    self, "Ошибка", f"Не удалось загрузить модель '{model_name}'"
                )
        except Exception as e:
            logger.error(f"Исключение при загрузке модели: {e}")
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
        """Начало обучения модели"""
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

        # Отключаем кнопку
        self.train_button.setEnabled(False)
        self.training_progress.setVisible(True)
        self.training_progress.setValue(0)

        # Очищаем лог
        self.training_log.clear()
        self.training_log.append("Начинаем обучение модели...\n")

        # Создаем и запускаем поток
        self.training_thread = TrainingThread(
            documents_path,
            model_name,
            self.vector_size_spin.value(),
            self.epochs_spin.value(),
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

        # Пытаемся загрузить и показать документ
        try:
            file_path = Path(result.doc_id)
            if not file_path.is_absolute():
                # Если путь относительный, пытаемся найти файл
                for parent in Path.cwd().parents:
                    full_path = parent / file_path
                    if full_path.exists():
                        file_path = full_path
                        break

            if file_path.exists():
                extractor = FileExtractor()
                text = extractor.extract_text(file_path)

                if text:
                    # Показываем первые 5000 символов
                    preview = text[:5000]
                    if len(text) > 5000:
                        preview += "\n\n... (текст обрезан) ..."

                    self.document_viewer.setPlainText(preview)

                    # Добавляем метаданные
                    metadata_text = "\n\n--- Метаданные ---\n"
                    metadata_text += f"Файл: {result.doc_id}\n"
                    metadata_text += f"Схожесть: {result.similarity:.3f}\n"

                    if result.metadata:
                        metadata_text += (
                            f"Размер: {result.metadata.get('file_size', 0)} байт\n"
                        )
                        metadata_text += (
                            f"Токенов: {result.metadata.get('tokens_count', 0)}\n"
                        )

                    self.document_viewer.append(metadata_text)
                else:
                    self.document_viewer.setPlainText(
                        "Не удалось извлечь текст из документа"
                    )
            else:
                # Попробуем найти файл относительно DATA_DIR
                alternative_path = DATA_DIR.parent / file_path
                if alternative_path.exists():
                    file_path = alternative_path
                    extractor = FileExtractor()
                    text = extractor.extract_text(file_path)

                    if text:
                        preview = text[:5000]
                        if len(text) > 5000:
                            preview += "\n\n... (текст обрезан) ..."
                        self.document_viewer.setPlainText(preview)
                    else:
                        self.document_viewer.setPlainText(
                            "Не удалось извлечь текст из документа"
                        )
                else:
                    self.document_viewer.setPlainText(f"Файл не найден: {file_path}")

        except Exception as e:
            logger.error(f"Ошибка при отображении документа: {e}")
            self.document_viewer.setPlainText(
                f"Ошибка при загрузке документа: {str(e)}"
            )

    def create_summary(self):
        """Создание выжимки документа"""
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

            # Показываем оригинальный текст
            self.original_text.setPlainText(
                text[:5000]
            )  # Показываем первые 5000 символов

            # Создаем выжимку
            self.status_bar.showMessage("Создание выжимки...")
            sentences_count = self.sentences_spin.value()

            summary = self.summarizer.summarize_text(
                text, sentences_count=sentences_count
            )

            if summary:
                # Показываем выжимку
                summary_text = "\n\n".join(
                    f"{i}. {sent}" for i, sent in enumerate(summary, 1)
                )
                self.summary_text.setPlainText(summary_text)

                # Показываем статистику
                stats = self.summarizer.get_summary_statistics(text, summary)
                stats_text = "\n\n--- Статистика ---\n"
                stats_text += (
                    f"Исходных предложений: {stats['original_sentences_count']}\n"
                )
                stats_text += (
                    f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                )
                stats_text += f"Сжатие: {stats['compression_ratio']:.1%}\n"
                stats_text += f"Исходных символов: {stats['original_chars_count']:,}\n"
                stats_text += f"Символов в выжимке: {stats['summary_chars_count']:,}\n"

                self.summary_text.append(stats_text)
                self.status_bar.showMessage("Выжимка создана")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось создать выжимку")

        except Exception as e:
            logger.error(f"Ошибка при создании выжимки: {e}")
            QMessageBox.critical(
                self, "Ошибка", f"Ошибка при создании выжимки: {str(e)}"
            )

    def update_statistics(self):
        """Обновление статистики"""
        stats_text = "📊 СТАТИСТИКА СИСТЕМЫ\n" + "=" * 50 + "\n\n"

        # Статистика моделей
        stats_text += "🧠 МОДЕЛИ:\n"
        model_files = list(MODELS_DIR.glob("*.model"))
        stats_text += f"Всего моделей: {len(model_files)}\n"

        for model_file in model_files:
            stats_text += f"  - {model_file.stem} ({model_file.stat().st_size / 1024 / 1024:.1f} МБ)\n"

        stats_text += "\n"

        # Статистика текущей модели
        if self.current_model:
            stats_text += "📍 ТЕКУЩАЯ МОДЕЛЬ:\n"
            trainer = Doc2VecTrainer()
            trainer.model = self.current_model
            model_info = trainer.get_model_info()

            stats_text += f"Размерность векторов: {model_info['vector_size']}\n"
            stats_text += f"Размер словаря: {model_info['vocabulary_size']:,} слов\n"
            stats_text += f"Документов в модели: {model_info['documents_count']}\n"
            stats_text += f"Размер окна: {model_info['window']}\n"
            stats_text += f"Минимальная частота: {model_info['min_count']}\n"
            stats_text += f"Эпох обучения: {model_info['epochs']}\n"
        else:
            stats_text += "❌ Модель не загружена\n"

        stats_text += "\n"

        # Системная информация
        try:
            import psutil

            stats_text += "💻 СИСТЕМА:\n"
            stats_text += f"CPU: {psutil.cpu_percent()}% загрузка\n"
            stats_text += f"Память: {psutil.virtual_memory().percent}% использовано\n"
            stats_text += f"Свободно памяти: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} ГБ\n"
        except ImportError:
            pass

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
