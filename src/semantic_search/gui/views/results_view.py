"""Виджет для отображения результатов поиска"""

from typing import Dict, List

from PyQt6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultsView(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        self.title_label = QLabel("Результаты поиска")
        self.title_label.setStyleSheet(
            "font-weight: bold; font-size: 16px; margin: 10px 0"
        )

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Документ", "Сходство", "Путь"])
        self.table.setSortingEnabled(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_results(self, results: List[Dict[str, str]]) -> None:
        self.table.setRowCount(len(results))

        for row, res in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(res.get("title", "")))
            self.table.setItem(row, 1, QTableWidgetItem(str(res.get("score", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(res.get("path", "")))

        self.table.resizeColumnsToContents()
