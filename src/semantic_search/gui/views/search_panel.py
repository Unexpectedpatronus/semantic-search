"""Виджет панели поиска по загруженным документам"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


class SearchPanel(QWidget):
    search_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout()

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Введите поисковый запрос")
        self.query_input.returnPressed.connect(self._emit_search)

        self.search_button = QPushButton("🔍 Найти")
        self.search_button.clicked.connect(self._emit_search)

        layout.addWidget(self.query_input)
        layout.addWidget(self.search_button)

        self.setLayout(layout)

    def _emit_search(self):
        query = self.query_input.text().strip()
        if query:
            self.search_requested.emit(query)
