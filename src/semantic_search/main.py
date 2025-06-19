"""Точка входа в приложение"""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from semantic_search.config import GUI_CONFIG, SPACY_MODEL
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.logging_config import setup_logging
from semantic_search.utils.notification_system import notification_manager
from semantic_search.utils.performance_monitor import PerformanceMonitor
from semantic_search.utils.statistics import (
    calculate_model_statistics,
    calculate_statistics_from_processed_docs,
    format_statistics_for_display,
)
from semantic_search.utils.task_manager import task_manager
from semantic_search.utils.text_utils import check_spacy_model_availability
from semantic_search.utils.validators import DataValidator, FileValidator

performance_monitor = PerformanceMonitor()


def main():
    """Главная функция приложения"""

    notification_manager.start()

    try:
        setup_logging()

        # Проверка SpaCy с уведомлением
        spacy_info = check_spacy_model_availability()
        if not spacy_info["model_loadable"]:
            notification_manager.warning(
                "SpaCy недоступен",
                f"Модель {SPACY_MODEL} не найдена",
                "Используйте: poetry run python scripts/setup_spacy.py",
            )
        else:
            notification_manager.success(
                "SpaCy готов", "Языковая модель загружена успешно"
            )

        # Проверяем доступность PyQt6
        try:
            from PyQt6.QtWidgets import QApplication

            from semantic_search.gui.main_window import MainWindow
        except ImportError as e:
            logger.error(f"PyQt6 не установлен: {e}")
            notification_manager.error(
                "Ошибка импорта",
                "PyQt6 не установлен",
                "Установите зависимости: poetry install",
            )
            print("\n❌ PyQt6 не установлен!")
            print("Установите зависимости командой: poetry install")
            print("\nВы можете использовать CLI режим:")
            print("poetry run semantic-search-cli --help")
            sys.exit(1)

        # Создание приложения Qt
        app = QApplication(sys.argv)
        app.setApplicationName(GUI_CONFIG["window_title"])
        app.setOrganizationName("Semantic Search")

        # Устанавливаем стиль
        app.setStyle("Fusion")  # Современный стиль

        # Создание и отображение главного окна
        main_window = MainWindow()
        main_window.show()

        logger.info("Главное окно создано и отображено")

        # Запуск цикла событий
        exit_code = app.exec()
        logger.info(f"Приложение завершено с кодом: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        notification_manager.error(
            "Критическая ошибка", "Ошибка при запуске приложения", str(e)
        )
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        print("\nПроверьте логи для деталей")
        sys.exit(1)
    finally:
        notification_manager.stop()
        task_manager.shutdown()


@click.group()
def cli():
    """Semantic Document Search CLI"""
    setup_logging()


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документами")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
@click.option("--vector-size", default=150, help="Размерность векторов")
@click.option("--epochs", default=40, help="Количество эпох обучения")
@click.option("--async-mode", "-a", is_flag=True, help="Асинхронное выполнение")
def train(documents: str, model: str, vector_size: int, epochs: int, async_mode: bool):
    """
    Обучить модель Doc2Vec на корпусе документов

    Args:
        documents: Путь к папке с документами
        model: Имя модели для сохранения
        vector_size: Размерность векторов
        epochs: Количество эпох обучения
    """
    try:
        # Валидация параметров
        documents_path = DataValidator.validate_directory_path(Path(documents))
        model_params = DataValidator.validate_model_params(
            vector_size=vector_size, epochs=epochs
        )

        logger.info("Валидация прошла успешно")

    except Exception as e:
        click.echo(f"❌ Ошибка валидации: {e}")
        return

    def train_task(progress_tracker=None):
        """Задача обучения модели"""

        with performance_monitor.measure_operation("document_processing"):
            # Обработка документов
            processor = DocumentProcessor()
            processed_docs = []

            file_extractor = FileExtractor()
            file_paths = file_extractor.find_documents(documents_path)

            if progress_tracker:
                progress_tracker.total_steps = len(file_paths) + epochs + 2
                progress_tracker.update(0, "Начинаем обработку документов...")

            for i, doc in enumerate(processor.process_documents(documents_path)):
                processed_docs.append(doc)
                if progress_tracker:
                    progress_tracker.update(
                        i + 1, f"Обработан документ: {doc.relative_path}"
                    )

            if not processed_docs:
                raise ValueError("Не удалось обработать документы")

            corpus = [
                (doc.tokens, doc.relative_path, doc.metadata) for doc in processed_docs
            ]

            if progress_tracker:
                progress_tracker.update(message="Подготовка к обучению модели...")

        with performance_monitor.measure_operation("model_training"):
            # Обучение модели
            trainer = Doc2VecTrainer()

            # Создаем модель с прогресс трекингом
            class ProgressDoc2Vec:
                def __init__(self, trainer, progress_tracker):
                    self.trainer = trainer
                    self.progress_tracker = progress_tracker

                def train_with_progress(self, corpus, **kwargs):
                    """Обучение с отслеживанием прогресса"""

                    # Создаем tagged documents
                    tagged_docs = self.trainer.create_tagged_documents(corpus)

                    from gensim.models.doc2vec import Doc2Vec

                    model = Doc2Vec(
                        vector_size=kwargs.get("vector_size", 150),
                        window=kwargs.get("window", 10),
                        min_count=kwargs.get("min_count", 2),
                        workers=kwargs.get("workers", 4),
                        seed=42,
                    )

                    # Построение словаря
                    model.build_vocab(tagged_docs)

                    if self.progress_tracker:
                        self.progress_tracker.update(
                            message="Словарь построен, начинаем обучение..."
                        )

                    # Обучение по эпохам
                    for epoch in range(kwargs.get("epochs", 40)):
                        model.train(
                            tagged_docs, total_examples=model.corpus_count, epochs=1
                        )

                        if self.progress_tracker:
                            progress_step = len(processed_docs) + epoch + 1
                            self.progress_tracker.update(
                                progress_step,
                                f"Эпоха {epoch + 1}/{kwargs.get('epochs', 40)}",
                            )

                    return model

            progress_trainer = ProgressDoc2Vec(trainer, progress_tracker)
            trained_model = progress_trainer.train_with_progress(
                corpus, vector_size=vector_size, epochs=epochs
            )

            if trained_model:
                trainer.model = trained_model
                trainer.corpus_info = corpus
                trainer.save_model(trained_model, model)

                if progress_tracker:
                    progress_tracker.finish("Модель обучена и сохранена")

                # Возвращаем статистику
                stats = calculate_statistics_from_processed_docs(processed_docs)
                return {
                    "model_saved": True,
                    "documents_processed": len(processed_docs),
                    "vocabulary_size": len(trained_model.wv.key_to_index),
                    "statistics": stats,
                }
            else:
                raise ValueError("Не удалось обучить модель")

    if async_mode:
        # Асинхронное выполнение
        notification_manager.start()

        task_id = task_manager.submit_task(
            train_task,
            name=f"Обучение модели {model}",
            description=f"Обучение на документах из {documents_path}",
            track_progress=True,
            total_steps=100,  # Примерное количество шагов
        )

        click.echo(f"🔄 Задача обучения запущена (ID: {task_id})")
        click.echo("Используйте команду 'status' для проверки прогресса")

        # Подписка на уведомления для консоли
        def console_notification_handler(notification):
            if notification.type.value == "success":
                click.echo(f"✅ {notification.title}: {notification.message}")
            elif notification.type.value == "error":
                click.echo(f"❌ {notification.title}: {notification.message}")
            elif notification.type.value == "warning":
                click.echo(f"⚠️ {notification.title}: {notification.message}")

        notification_manager.subscribe(console_notification_handler)

    else:
        # Синхронное выполнение
        try:
            with performance_monitor.measure_operation("full_training"):
                result = train_task()

            click.echo("✅ Обучение завершено успешно!")
            click.echo(f"📊 Обработано документов: {result['documents_processed']}")
            click.echo(f"📚 Размер словаря: {result['vocabulary_size']:,}")

            # Выводим детальную статистику
            stats_display = format_statistics_for_display(result["statistics"])
            click.echo(f"\n{stats_display}")

        except Exception as e:
            click.echo(f"❌ Ошибка при обучении: {e}")
            logger.error(f"Ошибка обучения модели: {e}")


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документами")
@click.option("--query", "-q", required=True, help="Поисковый запрос")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
@click.option("--top-k", "-k", default=10, help="Количество результатов")
def search(documents: str, query: str, model: str, top_k: int):
    """
    Поиск по документам

    Args:
        documents: Путь к папке с документами
        query: Поисковый запрос
        model: Имя модели
        top_k: Количество результатов для вывода
    """
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
@click.option("--documents", "-d", help="Путь к папке с документами")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
def stats(documents: Optional[str], model: str):
    """
    Показать статистику модели и корпуса

    Args:
        documents: Путь к папке с документами (опционально)
        model: Имя модели
    """
    # Статистика корпуса
    if documents:
        documents_path = Path(documents)
        if documents_path.exists():
            processor = DocumentProcessor()
            processed_docs = list(processor.process_documents(documents_path))

            if processed_docs:
                stats_data = calculate_statistics_from_processed_docs(processed_docs)
                click.echo(format_statistics_for_display(stats_data))
            else:
                click.echo("❌ Не удалось обработать документы")
        else:
            click.echo(f"❌ Папка не найдена: {documents_path}")

    # Статистика модели
    trainer = Doc2VecTrainer()
    if trainer.load_model(model):
        model_info = trainer.get_model_info()
        click.echo(f"\n{calculate_model_statistics(model_info)}")
    else:
        click.echo(f"\n❌ Модель '{model}' не найдена")


# КОМАНДЫ СУММАРИЗАЦИИ


@cli.command()
@click.option("--file", "-f", required=True, help="Путь к файлу для суммаризации")
@click.option("--model", "-m", default="doc2vec_model", help="Имя Doc2Vec модели")
@click.option("--sentences", "-s", default=5, help="Количество предложений в выжимке")
@click.option("--output", "-o", help="Файл для сохранения выжимки")
def summarize_file(
    file: str, model: str, sentences: int, output: Optional[str]
) -> None:
    """
    Создать выжимку из файла

    Args:
        file: Путь к файлу для суммаризации
        model: Имя Doc2Vec модели для улучшенной суммаризации
        sentences: Количество предложений в выжимке
        output: Путь для сохранения выжимки (опционально)
    """
    file_path = Path(file)
    if not file_path.exists():
        click.echo(f"❌ Файл не найден: {file_path}")
        return

    # Загрузка модели Doc2Vec
    trainer = Doc2VecTrainer()
    loaded_model = trainer.load_model(model)

    if loaded_model is None:
        click.echo("⚠️ Модель Doc2Vec не найдена. Используется базовая суммаризация")
        summarizer = TextSummarizer()
    else:
        click.echo("✅ Используется продвинутая суммаризация с Doc2Vec")
        summarizer = TextSummarizer(loaded_model)

    logger.info(f"Создание выжимки файла: {file_path}")

    # Создание выжимки
    try:
        summary = summarizer.summarize_file(str(file_path), sentences_count=sentences)

        if not summary:
            click.echo(
                "❌ Не удалось создать выжимку. Проверьте файл и попробуйте снова"
            )
            return

        # Вывод результата в консоль
        click.echo(f"\n📄 Выжимка файла: {file_path.name}")
        click.echo("=" * 60)

        for i, sentence in enumerate(summary, 1):
            click.echo(f"{i}. {sentence.strip()}")
            click.echo()  # Пустая строка между предложениями

        # Статистика суммаризации
        try:
            from semantic_search.utils.file_utils import FileExtractor

            extractor = FileExtractor()
            original_text = extractor.extract_text(file_path)

            if original_text:
                stats = summarizer.get_summary_statistics(original_text, summary)

                click.echo("📊 Статистика суммаризации:")
                click.echo("-" * 30)
                click.echo(
                    f"  📑 Исходных предложений: {stats['original_sentences_count']}"
                )
                click.echo(
                    f"  📄 Предложений в выжимке: {stats['summary_sentences_count']}"
                )
                click.echo(
                    f"  📉 Коэффициент сжатия предложений: {stats['compression_ratio']:.1%}"
                )
                click.echo(f"  🔤 Исходных символов: {stats['original_chars_count']:,}")
                click.echo(f"  ✂️ Символов в выжимке: {stats['summary_chars_count']:,}")
                click.echo(
                    f"  📊 Сокращение текста: {stats['chars_compression_ratio']:.1%}"
                )

        except Exception as e:
            logger.error(f"Ошибка при расчете статистики: {e}")

        # Сохранение в файл
        if output:
            output_path = Path(output)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"Выжимка файла: {file_path.name}\n")
                    f.write(
                        f"Создано: {click.get_current_context().meta.get('timestamp', 'неизвестно')}\n"
                    )
                    f.write("=" * 60 + "\n\n")

                    for i, sentence in enumerate(summary, 1):
                        f.write(f"{i}. {sentence.strip()}\n\n")

                    # Добавляем статистику в файл
                    if "stats" in locals():
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("СТАТИСТИКА СУММАРИЗАЦИИ\n")
                        f.write("=" * 60 + "\n")
                        f.write(
                            f"Исходных предложений: {stats['original_sentences_count']}\n"
                        )
                        f.write(
                            f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                        )
                        f.write(
                            f"Коэффициент сжатия предложений: {stats['compression_ratio']:.1%}\n"
                        )
                        f.write(
                            f"Исходных символов: {stats['original_chars_count']:,}\n"
                        )
                        f.write(
                            f"Символов в выжимке: {stats['summary_chars_count']:,}\n"
                        )
                        f.write(
                            f"Сокращение текста: {stats['chars_compression_ratio']:.1%}\n"
                        )

                click.echo(f"💾 Выжимка сохранена в: {output_path}")

            except Exception as e:
                click.echo(f"❌ Ошибка при сохранении: {e}")
                logger.error(f"Ошибка при сохранении выжимки: {e}")

    except Exception as e:
        click.echo(f"❌ Ошибка при создании выжимки: {e}")
        logger.error(f"Ошибка суммаризации файла {file_path}: {e}")


@cli.command()
@click.option("--text", "-t", required=True, help="Текст для суммаризации")
@click.option("--model", "-m", default="doc2vec_model", help="Имя Doc2Vec модели")
@click.option("--sentences", "-s", default=5, help="Количество предложений в выжимке")
@click.option("--output", "-o", help="Файл для сохранения выжимки")
def summarize_text(
    text: str, model: str, sentences: int, output: Optional[str]
) -> None:
    """
    Создать выжимку из текста

    Args:
        text: Текст для суммаризации (строка)
        model: Имя Doc2Vec модели для улучшенной суммаризации
        sentences: Количество предложений в выжимке
        output: Путь для сохранения выжимки (опционально)
    """
    # Базовая валидация входного текста
    if not text or len(text.strip()) < 100:
        click.echo("❌ Текст слишком короткий для суммаризации (минимум 100 символов)")
        return

    # Проверяем количество предложений в исходном тексте
    temp_processor = TextSummarizer()
    original_sentences = temp_processor.text_processor.split_into_sentences(text)

    if len(original_sentences) <= sentences:
        click.echo(
            f"⚠️ В тексте всего {len(original_sentences)} предложений, что меньше или равно запрошенному количеству ({sentences})"
        )
        click.echo("Выводим весь текст:")
        for i, sentence in enumerate(original_sentences, 1):
            click.echo(f"{i}. {sentence.strip()}")
        return

    # Загрузка модели Doc2Vec
    trainer = Doc2VecTrainer()
    loaded_model = trainer.load_model(model)

    if loaded_model is None:
        click.echo("⚠️ Модель Doc2Vec не найдена. Используется базовая суммаризация")
        summarizer = TextSummarizer()
    else:
        click.echo("✅ Используется продвинутая суммаризация с Doc2Vec")
        summarizer = TextSummarizer(loaded_model)

    logger.info("Создание выжимки текста")

    try:
        # Создание выжимки
        summary = summarizer.summarize_text(text, sentences_count=sentences)

        if not summary:
            click.echo("❌ Не удалось создать выжимку")
            return

        # Вывод результата
        click.echo("\n📄 Выжимка текста:")
        click.echo("=" * 60)

        for i, sentence in enumerate(summary, 1):
            click.echo(f"{i}. {sentence.strip()}")
            click.echo()  # Пустая строка между предложениями

        # Статистика суммаризации
        stats = summarizer.get_summary_statistics(text, summary)

        click.echo("📊 Статистика суммаризации:")
        click.echo("-" * 30)
        click.echo(f"  📑 Исходных предложений: {stats['original_sentences_count']}")
        click.echo(f"  📄 Предложений в выжимке: {stats['summary_sentences_count']}")
        click.echo(
            f"  📉 Коэффициент сжатия предложений: {stats['compression_ratio']:.1%}"
        )
        click.echo(f"  🔤 Исходных символов: {stats['original_chars_count']:,}")
        click.echo(f"  ✂️ Символов в выжимке: {stats['summary_chars_count']:,}")
        click.echo(f"  📊 Сокращение текста: {stats['chars_compression_ratio']:.1%}")

        # Сохранение в файл
        if output:
            output_path = Path(output)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("Выжимка текста\n")
                    f.write("=" * 60 + "\n\n")

                    for i, sentence in enumerate(summary, 1):
                        f.write(f"{i}. {sentence.strip()}\n\n")

                    # Добавляем статистику
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("СТАТИСТИКА СУММАРИЗАЦИИ\n")
                    f.write("=" * 60 + "\n")
                    f.write(
                        f"Исходных предложений: {stats['original_sentences_count']}\n"
                    )
                    f.write(
                        f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                    )
                    f.write(
                        f"Коэффициент сжатия предложений: {stats['compression_ratio']:.1%}\n"
                    )
                    f.write(f"Исходных символов: {stats['original_chars_count']:,}\n")
                    f.write(f"Символов в выжимке: {stats['summary_chars_count']:,}\n")
                    f.write(
                        f"Сокращение текста: {stats['chars_compression_ratio']:.1%}\n"
                    )

                click.echo(f"💾 Выжимка сохранена в: {output_path}")

            except Exception as e:
                click.echo(f"❌ Ошибка при сохранении: {e}")
                logger.error(f"Ошибка при сохранении выжимки: {e}")

    except Exception as e:
        click.echo(f"❌ Ошибка при создании выжимки: {e}")
        logger.error(f"Ошибка суммаризации текста: {e}")


@cli.command()
@click.option("--documents", "-d", required=True, help="Путь к папке с документами")
@click.option("--model", "-m", default="doc2vec_model", help="Имя Doc2Vec модели")
@click.option(
    "--sentences",
    "-s",
    default=3,
    help="Количество предложений в выжимке каждого файла",
)
@click.option("--output-dir", "-o", help="Папка для сохранения выжимок")
@click.option(
    "--extensions", default="pdf,docx,doc", help="Расширения файлов (через запятую)"
)
@click.option(
    "--max-files",
    default=0,
    help="Максимальное количество файлов для обработки (0 = все)",
)
def summarize_batch(
    documents: str,
    model: str,
    sentences: int,
    output_dir: Optional[str],
    extensions: str,
    max_files: int,
) -> None:
    """
    Создать выжимки для всех документов в папке

    Args:
        documents: Путь к папке с документами
        model: Имя Doc2Vec модели
        sentences: Количество предложений в каждой выжимке
        output_dir: Папка для сохранения выжимок
        extensions: Обрабатываемые расширения файлов (через запятую)
        max_files: Максимальное количество файлов (0 = без ограничений)
    """
    documents_path = Path(documents)
    if not documents_path.exists():
        click.echo(f"❌ Папка не найдена: {documents_path}")
        return

    # Загрузка модели Doc2Vec
    trainer = Doc2VecTrainer()
    loaded_model = trainer.load_model(model)

    if loaded_model is None:
        click.echo("⚠️ Модель Doc2Vec не найдена. Используется базовая суммаризация")
        summarizer = TextSummarizer()
    else:
        click.echo("✅ Используется продвинутая суммаризация с Doc2Vec")
        summarizer = TextSummarizer(loaded_model)

    # Подготовка расширений
    allowed_extensions = {f".{ext.strip().lower()}" for ext in extensions.split(",")}
    click.echo(f"🔍 Поиск файлов с расширениями: {allowed_extensions}")

    # Поиск файлов
    all_files = []
    for file_path in documents_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            all_files.append(file_path)

    if not all_files:
        click.echo(f"❌ Файлы с расширениями {allowed_extensions} не найдены")
        return

    # Ограничение количества файлов
    if max_files > 0 and len(all_files) > max_files:
        all_files = all_files[:max_files]
        click.echo(f"📁 Ограничено до {max_files} файлов из найденных")

    click.echo(f"📁 Найдено файлов для обработки: {len(all_files)}")

    # Подготовка папки для выходных файлов
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        click.echo(f"💾 Выжимки будут сохранены в: {output_path}")
    else:
        click.echo("📺 Выжимки будут выведены только в консоль")

    successful = 0
    failed = 0

    # Обработка файлов
    for i, file_path in enumerate(all_files, 1):
        click.echo(f"\n🔄 Обработка {i}/{len(all_files)}: {file_path.name}")

        try:
            # Создание выжимки
            summary = summarizer.summarize_file(
                str(file_path), sentences_count=sentences
            )

            if not summary:
                click.echo(f"   ⚠️ Не удалось создать выжимку для {file_path.name}")
                failed += 1
                continue

            # Краткий вывод в консоль
            click.echo(f"   ✅ Создана выжимка: {len(summary)} предложений")

            # Показываем первое предложение как превью
            if summary:
                preview = (
                    summary[0][:100] + "..." if len(summary[0]) > 100 else summary[0]
                )
                click.echo(f"   👁️ Превью: {preview}")

            # Сохранение в файл
            if output_dir:
                summary_filename = f"{file_path.stem}_summary.txt"
                summary_path = output_path / summary_filename

                try:
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(f"Выжимка файла: {file_path.name}\n")
                        f.write(f"Исходный путь: {file_path}\n")
                        f.write(f"Количество предложений: {len(summary)}\n")
                        f.write("=" * 60 + "\n\n")

                        for j, sentence in enumerate(summary, 1):
                            f.write(f"{j}. {sentence.strip()}\n\n")

                    click.echo(f"   💾 Сохранено: {summary_filename}")
                except Exception as save_error:
                    click.echo(f"   ❌ Ошибка сохранения: {save_error}")
                    failed += 1
                    continue

            successful += 1

        except Exception as e:
            click.echo(f"   ❌ Ошибка при обработке: {e}")
            logger.error(f"Ошибка при обработке {file_path}: {e}")
            failed += 1

    # Итоговая статистика
    click.echo("\n📊 Итоговая статистика пакетной суммаризации:")
    click.echo("=" * 50)
    click.echo(f"  ✅ Успешно обработано: {successful}")
    click.echo(f"  ❌ Ошибок: {failed}")
    click.echo(f"  📁 Всего файлов: {len(all_files)}")
    click.echo(f"  📈 Процент успеха: {(successful / len(all_files) * 100):.1f}%")

    if output_dir and successful > 0:
        click.echo(f"  💾 Выжимки сохранены в: {output_path}")


@cli.command()
def status():
    """Проверка статуса выполняющихся задач"""

    tasks = task_manager.get_all_tasks()

    if not tasks:
        click.echo("📭 Активных задач нет")
        return

    click.echo("📋 Статус задач:")
    click.echo("=" * 60)

    for task in tasks:
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "cancelled": "⏹️",
        }.get(task.status.value, "❓")

        click.echo(f"{status_icon} {task.name}")
        click.echo(f"   ID: {task.id}")
        click.echo(f"   Статус: {task.status.value}")

        if task.progress > 0:
            progress_bar = "█" * int(task.progress * 20) + "░" * (
                20 - int(task.progress * 20)
            )
            click.echo(f"   Прогресс: [{progress_bar}] {task.progress:.1%}")

        if task.duration:
            click.echo(f"   Время: {task.duration:.1f}с")

        if task.error:
            click.echo(f"   Ошибка: {task.error}")

        click.echo()


@cli.command()
@click.argument("task_id")
def cancel(task_id: str):
    """Отмена задачи"""

    if task_manager.cancel_task(task_id):
        click.echo(f"✅ Задача {task_id} отменена")
    else:
        click.echo(f"❌ Не удалось отменить задачу {task_id}")


@cli.command()
@click.option(
    "--max-keep", default=50, help="Максимальное количество задач для хранения"
)
def cleanup(max_keep: int):
    """Очистка завершенных задач"""

    before_count = len(task_manager.get_all_tasks())
    task_manager.cleanup_finished_tasks(max_keep)
    after_count = len(task_manager.get_all_tasks())

    removed = before_count - after_count
    click.echo(f"🧹 Удалено {removed} завершенных задач")


@cli.command()
@click.option("--documents", "-d", help="Путь к папке с документами")
@click.option("--output", "-o", help="Файл для сохранения отчета")
def system_info(documents: Optional[str], output: Optional[str]):
    """Системная информация и диагностика"""

    info_lines = []

    # Системная информация
    system_info = performance_monitor.get_system_info()
    info_lines.extend(
        [
            "🖥️ Системная информация:",
            f"   CPU: {system_info['cpu_count']} ядер, загрузка {system_info['cpu_percent']}%",
            f"   ОЗУ: {system_info['memory_available']:.1f}/{system_info['memory_total']:.1f} ГБ свободно",
            f"   Диск: {100 - system_info['disk_usage']:.1f}% свободно",
            "",
        ]
    )

    # Статус SpaCy
    spacy_info = check_spacy_model_availability()
    spacy_status = "✅ Готов" if spacy_info["model_loadable"] else "❌ Не готов"
    info_lines.extend(
        [
            "🧠 Языковая модель:",
            f"   SpaCy: {spacy_status}",
            f"   Модель: {SPACY_MODEL}",
            "",
        ]
    )

    # Информация о документах
    if documents:
        try:
            docs_path = Path(documents)
            if docs_path.exists():
                file_extractor = FileExtractor()
                found_files = file_extractor.find_documents(docs_path)

                # Валидация файлов
                valid_files = 0
                invalid_files = 0
                total_size = 0

                for file_path in found_files:
                    validation = FileValidator.validate_document_file(file_path)
                    if validation["valid"]:
                        valid_files += 1
                        total_size += validation["file_info"]["size"]
                    else:
                        invalid_files += 1

                info_lines.extend(
                    [
                        "📁 Анализ документов:",
                        f"   Всего найдено: {len(found_files)}",
                        f"   Валидных: {valid_files}",
                        f"   С ошибками: {invalid_files}",
                        f"   Общий размер: {total_size / 1024 / 1024:.1f} МБ",
                        "",
                    ]
                )

        except Exception as e:
            info_lines.extend(["📁 Анализ документов:", f"   ❌ Ошибка: {e}", ""])

    # Производительность
    if performance_monitor.metrics:
        info_lines.extend(
            [
                "⚡ Последние операции:",
            ]
        )

        for op_name, metrics in list(performance_monitor.metrics.items())[-5:]:
            info_lines.append(f"   {op_name}: {metrics['duration']:.2f}с")

        info_lines.append("")

    # Активные задачи
    running_tasks = task_manager.get_running_tasks()
    if running_tasks:
        info_lines.extend(
            [
                "🔄 Активные задачи:",
            ]
        )

        for task in running_tasks:
            info_lines.append(f"   {task.name}: {task.progress:.1%}")

        info_lines.append("")

    report = "\n".join(info_lines)

    # Вывод в консоль
    click.echo(report)

    # Сохранение в файл
    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                f.write(f"Системный отчет - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)

            click.echo(f"💾 Отчет сохранен в: {output}")

        except Exception as e:
            click.echo(f"❌ Ошибка сохранения отчета: {e}")


@cli.command()
@click.option("--show", is_flag=True, help="Показать текущую конфигурацию")
@click.option("--reset", is_flag=True, help="Сбросить к значениям по умолчанию")
@click.option("--reload", is_flag=True, help="Перезагрузить конфигурацию из файла")
@click.option(
    "--set", nargs=2, multiple=True, help="Установить параметр: --set key value"
)
def config(show: bool, reset: bool, reload: bool, set: tuple):
    """Управление конфигурацией приложения"""

    from semantic_search.config import config_manager

    if reset:
        if click.confirm(
            "Вы уверены, что хотите сбросить конфигурацию к значениям по умолчанию?"
        ):
            config_manager.reset_to_defaults()
            click.echo("✅ Конфигурация сброшена к значениям по умолчанию")
        else:
            click.echo("❌ Сброс отменен")
        return

    if reload:
        config_manager.reload_config()
        click.echo("✅ Конфигурация перезагружена из файла")

    if set:
        for key, value in set:
            # Пытаемся преобразовать значение в правильный тип
            try:
                # Числа
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                # Булевы значения
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                # Числа с подчеркиваниями
                elif "_" in value and value.replace("_", "").isdigit():
                    value = int(value.replace("_", ""))

            except Exception:
                pass  # Оставляем как строку

            # Определяем секцию и параметр
            if "." in key:
                section, param = key.split(".", 1)
                config_manager.update_config(**{section: {param: value}})
                click.echo(f"✅ Установлено: {section}.{param} = {value}")
            else:
                click.echo("❌ Неверный формат ключа. Используйте: section.parameter")

    if show or (not reset and not reload and not set):
        # Показываем текущую конфигурацию
        current_config = config_manager.config

        click.echo("\n📋 Текущая конфигурация:")
        click.echo("=" * 60)

        # Обработка текста
        click.echo("\n📝 Обработка текста (text_processing):")
        for key, value in current_config.text_processing.items():
            if isinstance(value, int) and value > 1000:
                click.echo(f"  {key}: {value:,}")
            else:
                click.echo(f"  {key}: {value}")

        # Doc2Vec
        click.echo("\n🧠 Параметры Doc2Vec (doc2vec):")
        for key, value in current_config.doc2vec.items():
            click.echo(f"  {key}: {value}")

        # Поиск
        click.echo("\n🔍 Параметры поиска (search):")
        for key, value in current_config.search.items():
            click.echo(f"  {key}: {value}")

        # GUI
        click.echo("\n💻 Параметры интерфейса (gui):")
        for key, value in current_config.gui.items():
            click.echo(f"  {key}: {value}")

        # Суммаризация
        click.echo("\n📄 Параметры суммаризации (summarization):")
        for key, value in current_config.summarization.items():
            click.echo(f"  {key}: {value}")

        click.echo("\n💡 Примеры изменения параметров:")
        click.echo(
            "  semantic-search-cli config --set text_processing.max_text_length 10000000"
        )
        click.echo("  semantic-search-cli config --set doc2vec.vector_size 200")
        click.echo("  semantic-search-cli config --set search.default_top_k 20")


def cli_mode():
    """Запуск CLI режима"""
    cli()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        main()
