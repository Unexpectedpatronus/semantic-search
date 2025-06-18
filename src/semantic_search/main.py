# main.py — главная точка входа: GUI или CLI переключение
import sys

from PyQt6.QtWidgets import QApplication

from semantic_search.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--gui":
        from semantic_search.cli import cli

        cli()
    else:
        main()
