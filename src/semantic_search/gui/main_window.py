from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from semantic_search.gui.controller import AppController
from semantic_search.gui.views.document_loader import DocumentLoader
from semantic_search.gui.views.results_view import ResultsView
from semantic_search.gui.views.search_panel import SearchPanel
from semantic_search.gui.widgets import LoadingProgressBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantic Document Search")
        self.setGeometry(100, 100, 960, 720)

        self.controller = AppController()
        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.loader = DocumentLoader()
        self.search_panel = SearchPanel()
        self.results_view = ResultsView()
        self.progress_bar = LoadingProgressBar()

        self.loader.directory_selected.connect(self._on_folder_selected)
        self.search_panel.search_requested.connect(self._on_search)

        layout.addWidget(self.loader)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.search_panel)
        layout.addWidget(self.results_view)

        self.setCentralWidget(central_widget)

    def _on_folder_selected(self, folder):
        self.progress_bar.start()
        result = self.controller.load_documents(folder)
        self.loader.info_label.setText(result["status"])
        self.progress_bar.finish()

    def _on_search(self, query):
        results = self.controller.search(query)
        self.results_view.update_results(results)
