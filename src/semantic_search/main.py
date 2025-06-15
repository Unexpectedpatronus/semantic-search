"""Точка входа в приложение"""

import sys
from pathlib import Path

import click
from loguru import logger

from semantic_search.config import GUI_CONFIG
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.utils.logging_config import setup_logging


def main():
    """Главная функция приложения"""

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


@click.group()
def cli():
    """Semantic Document Search CLI"""
    setup_logging()


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документами")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
@click.option("--vector-size", default=150, help="Размерность векторов")
@click.option("--epochs", default=40, help="Количество эпох обучения")
def train(documents, model, vector_size, epochs):
    """Обучить модель"""

    documents_path = Path(documents)
    if not documents_path.exists():
        logger.error(f"Папка не найдена: {documents_path}")
        return

    logger.info("Режим обучения модели")

    # Обработка документов
    processor = DocumentProcessor()
    corpus = processor.prepare_corpus_for_doc2vec(documents_path)

    if not corpus:
        logger.error("Не удалось подготовить корпус")
        return

    # Обучение модели
    trainer = Doc2VecTrainer()
    trained_model = trainer.train_model(corpus, vector_size=vector_size, epochs=epochs)

    if trained_model:
        trainer.save_model(trained_model, model)
        logger.info("✅ Модель обучена и сохранена")

        # Показываем статистику
        stats = processor.get_processing_statistics(documents_path)
        click.echo("\n📊 Статистика обучения:")
        click.echo(f"📁 Обработано документов: {stats['processed_files']}")
        click.echo(f"🔤 Общее количество токенов: {stats['total_tokens']}")
        click.echo(f"📄 Среднее токенов на документ: {stats['avg_tokens_per_doc']:.1f}")
        click.echo(f"📋 Форматы файлов: {stats['extensions_count']}")
    else:
        logger.error("❌ Ошибка обучения модели")


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документами")
@click.option("--query", "-q", required=True, help="Поисковый запрос")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
@click.option("--top-k", default=10, help="Количество результатов")
def search(documents, query, model, top_k):
    """Поиск по документам"""
    logger.info(f"Режим поиска: {query}")

    # Загрузка модели
    trainer = Doc2VecTrainer()
    loaded_model = trainer.load_model(model)

    if loaded_model is None:
        logger.error("❌ Не удалось загрузить модель")
        click.echo("Сначала обучите модель командой: train")
        return

    # Поиск
    search_engine = SemanticSearchEngine(loaded_model, trainer.corpus_info)
    results = search_engine.search(query, top_k=top_k)

    if results:
        click.echo(f"\n🔍 Результаты поиска для '{query}':")
        click.echo("=" * 50)
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.doc_id}")
            click.echo(f"   📊 Сходство: {result.similarity:.3f}")
            if result.metadata:
                tokens_count = result.metadata.get("tokens_count", "N/A")
                file_size = result.metadata.get("file_size", 0)
                click.echo(f"   📝 Токенов: {tokens_count}, Размер: {file_size} байт")
            click.echo()
    else:
        click.echo(f"❌ Результатов не найдено для запроса '{query}'")


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документов")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
def stats(documents, model):
    """Показать статистику модели и корпуса"""
    documents_path = Path(documents)

    # Статистика корпуса
    if documents_path.exists():
        processor = DocumentProcessor()
        corpus_stats = processor.get_processing_statistics(documents_path)

        click.echo("📁 Статистика корпуса:")
        click.echo(f"  Документов: {corpus_stats['processed_files']}")
        click.echo(f"  Токенов: {corpus_stats['total_tokens']}")
        click.echo(
            f"  Среднее токенов/документ: {corpus_stats['avg_tokens_per_doc']:.1f}"
        )
        click.echo(f"  Форматы: {corpus_stats['extensions_count']}")

        if corpus_stats["largest_doc"]:
            click.echo(
                f"  Самый большой: {corpus_stats['largest_doc']['path']} ({corpus_stats['largest_doc']['tokens']} токенов)"
            )
        if corpus_stats["smallest_doc"]:
            click.echo(
                f"  Самый маленький: {corpus_stats['smallest_doc']['path']} ({corpus_stats['smallest_doc']['tokens']} токенов)"
            )

    # Статистика модели
    trainer = Doc2VecTrainer()
    if trainer.load_model(model):
        model_info = trainer.get_model_info()

        click.echo(f"\n🧠 Статистика модели '{model}':")
        click.echo(f"  Статус: {model_info['status']}")
        click.echo(f"  Размерность векторов: {model_info['vector_size']}")
        click.echo(f"  Размер словаря: {model_info['vocabulary_size']}")
        click.echo(f"  Документов в модели: {model_info['documents_count']}")
        click.echo(f"  Окно контекста: {model_info['window']}")
        click.echo(f"  Минимальная частота: {model_info['min_count']}")
        click.echo(f"  Эпох обучения: {model_info['epochs']}")
    else:
        click.echo(f"\n❌ Модель '{model}' не найдена")


def cli_mode():
    """Запуск CLI режима"""
    cli()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        main()
