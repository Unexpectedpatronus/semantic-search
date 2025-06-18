"""Виджет для выбора директории и запуска обработки документов"""

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DocumentLoader(QWidget):
    directory_selected = pyqtSignal(Path)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        self.info_label = QLabel("Выберите папку с документами")
        self.select_button = QPushButton("📁 Выбрать папку")

        self.select_button.clicked.connect(self._choose_directory)

        layout.addWidget(self.info_label)
        layout.addWidget(self.select_button)

        self.setLayout(layout)

    def _choose_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите директорию")
        if folder:
            self.directory_selected.emit(Path(folder))
        else:
            QMessageBox.information(
                self, "Выбор отменён", "Выбор директории был отменён."
            )
