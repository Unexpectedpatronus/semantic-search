"""Точка входа в приложение"""

import sys
from pathlib import Path

from loguru import logger

from semantic_search.config import GUI_CONFIG
from semantic_search.utils.logging_config import setup_logging


def main():
    """Главная функция приложения"""
    # Настройка логирования
    setup_logging()
    logger.info("Запуск приложения Semantic Document Search")

    try:
        # Проверяем доступность PyQt6
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        # Создание приложения Qt
        app = QApplication(sys.argv)
        app.setApplicationName(GUI_CONFIG["window_title"])
        app.setOrganizationName("Semantic Search")

        # Устанавливаем стиль
        app.setStyle("Fusion")  # Современный стиль

        # Создание и отображение главного окна
        from semantic_search.gui.main_window import MainWindow

        main_window = MainWindow()
        main_window.show()

        logger.info("Главное окно создано и отображено")

        # Запуск цикла событий
        exit_code = app.exec()
        logger.info(f"Приложение завершено с кодом: {exit_code}")
        sys.exit(exit_code)

    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены:")
        print("poetry install")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def cli_mode():
    """Консольный режим для тестирования"""
    import argparse

    from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
    from semantic_search.core.document_processor import DocumentProcessor
    from semantic_search.core.search_engine import SemanticSearchEngine

    parser = argparse.ArgumentParser(description="Semantic Document Search CLI")
    parser.add_argument(
        "--documents", "-d", required=True, help="Путь к папке с документами"
    )
    parser.add_argument(
        "--train", "-t", action="store_true", help="Обучить новую модель"
    )
    parser.add_argument("--search", "-s", help="Поисковый запрос")
    parser.add_argument("--model", "-m", default="doc2vec_model", help="Имя модели")

    args = parser.parse_args()

    setup_logging()

    documents_path = Path(args.documents)
    if not documents_path.exists():
        logger.error(f"Папка не найдена: {documents_path}")
        return

    if args.train:
        logger.info("Режим обучения модели")

        # Обработка документов
        processor = DocumentProcessor()
        corpus = processor.prepare_corpus_for_doc2vec(documents_path)

        if not corpus:
            logger.error("Не удалось подготовить корпус")
            return

        # Обучение модели
        trainer = Doc2VecTrainer()
        model = trainer.train_model(corpus)

        if model:
            trainer.save_model(model, args.model)
            logger.info("Модель обучена и сохранена")
        else:
            logger.error("Ошибка обучения модели")

    elif args.search:
        logger.info(f"Режим поиска: {args.search}")

        # Загрузка модели
        trainer = Doc2VecTrainer()
        model = trainer.load_model(args.model)

        if model is None:
            logger.error("Не удалось загрузить модель")
            return

        # Поиск
        search_engine = SemanticSearchEngine(model, trainer.corpus_info)
        results = search_engine.search(args.search)

        print(f"\nРезультаты поиска для '{args.search}':")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.doc_id} (сходство: {result.similarity:.3f})")

    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        main()
