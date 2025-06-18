"""Кастомный виджет индикатора прогресса для загрузки документов"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QProgressBar


class LoadingProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("Прогресс: %p%")
        self.setFixedHeight(20)
        self.hide()

    def start(self):
        self.setValue(0)
        self.show()

    def finish(self):
        self.setValue(100)
        self.hide()
