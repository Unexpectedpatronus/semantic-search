"""Главное окно приложения (базовая заглушка)"""

from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from semantic_search.config import GUI_CONFIG


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(GUI_CONFIG["window_title"])
        self.setGeometry(100, 100, *GUI_CONFIG["window_size"])

        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Создаем основной layout
        layout = QVBoxLayout(central_widget)

        # Добавляем заглушку
        welcome_label = QLabel("🔍 Semantic Document Search")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")

        status_label = QLabel(
            "Backend загружен успешно! GUI будет реализован на следующем этапе."
        )
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet("font-size: 14px; color: green; margin: 10px;")

        # Кнопка для тестирования backend
        test_button = QPushButton("Тест Backend")
        test_button.clicked.connect(self.test_backend)

        layout.addWidget(welcome_label)
        layout.addWidget(status_label)
        layout.addWidget(test_button)

        logger.info("Главное окно инициализировано")

    def test_backend(self):
        """Тестирование backend компонентов"""
        try:
            from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
            from semantic_search.core.document_processor import DocumentProcessor
            from semantic_search.core.search_engine import SemanticSearchEngine
            from semantic_search.core.text_summarizer import TextSummarizer

            # Создаем экземпляры классов
            processor = DocumentProcessor()
            trainer = Doc2VecTrainer()
            search_engine = SemanticSearchEngine()
            summarizer = TextSummarizer()

            QMessageBox.information(
                self,
                "Тест Backend",
                "✅ Все backend компоненты загружены успешно!\n\n"
                "Компоненты:\n"
                "• DocumentProcessor\n"
                "• Doc2VecTrainer\n"
                "• SemanticSearchEngine\n"
                "• TextSummarizer\n\n"
                "Готов к реализации GUI!",
            )

            logger.info("Backend тест прошел успешно")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка Backend",
                f"❌ Ошибка при тестировании backend:\n\n{str(e)}",
            )
            logger.error(f"Backend тест не прошел: {e}")
