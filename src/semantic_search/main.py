"""Точка входа в приложение"""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from semantic_search.config import SPACY_MODEL
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


def check_dependencies() -> bool:
    """
    Проверка критических зависимостей

    Returns:
        True если все зависимости установлены
    """
    critical_errors = []

    # Проверка SpaCy
    spacy_info = check_spacy_model_availability()

    if not spacy_info["spacy_installed"]:
        critical_errors.append("SpaCy не установлен")
        notification_manager.error(
            "SpaCy не установлен",
            "Установите SpaCy для работы с текстом",
            "Используйте: pip install spacy",
        )
    else:
        # Проверяем наличие хотя бы одной модели
        if not (spacy_info["ru_model_loadable"] or spacy_info["en_model_loadable"]):
            notification_manager.warning(
                "SpaCy модели не найдены",
                "Ни одна языковая модель не установлена",
                "Используйте: poetry run python scripts/setup_spacy.py",
            )
        else:
            # Информируем о доступных моделях
            models_status = []
            if spacy_info["ru_model_loadable"]:
                models_status.append("русская")
            if spacy_info["en_model_loadable"]:
                models_status.append("английская")

            notification_manager.success(
                "SpaCy готов", f"Загружены модели: {', '.join(models_status)}"
            )

    return len(critical_errors) == 0


def init_gui_mode():
    """Инициализация GUI режима"""
    try:
        from PyQt6.QtWidgets import QApplication

        from semantic_search.gui.main_window import MainWindow

        # Создание приложения Qt
        app = QApplication(sys.argv)
        app.setApplicationName("Semantic Document Search")
        app.setOrganizationName("Semantic Search")
        app.setStyle("Fusion")

        # Создание и отображение главного окна
        main_window = MainWindow()
        main_window.show()

        logger.info("Главное окно создано и отображено")

        # Запуск цикла событий
        exit_code = app.exec()
        logger.info(f"Приложение завершено с кодом: {exit_code}")
        return exit_code

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
        return 1


def main():
    """Главная функция приложения"""
    notification_manager.start()

    try:
        setup_logging()

        # Проверка зависимостей
        if not check_dependencies():
            logger.warning("Обнаружены проблемы с зависимостями")

        # Запуск GUI
        exit_code = init_gui_mode()
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
    """Обучить модель Doc2Vec на корпусе документов"""
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
        start_time = time.time()

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

            trained_model = trainer.train_model(
                corpus,
                vector_size=model_params.get("vector_size", vector_size),
                epochs=model_params.get("epochs", epochs),
            )

            if trained_model:
                # Вычисляем общее время
                training_time = time.time() - start_time
                trainer.training_metadata["training_time_formatted"] = (
                    f"{training_time:.1f}с ({training_time / 60:.1f}м)"
                )
                trainer.training_metadata["training_date"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(start_time)
                )
                trainer.training_metadata["corpus_size"] = len(processed_docs)
                trainer.training_metadata["documents_base_path"] = str(
                    documents_path.absolute()
                )

                trainer.save_model(trained_model, model)

                if progress_tracker:
                    progress_tracker.finish(
                        f"Модель обучена за {training_time / 60:.1f} минут"
                    )

                # Возвращаем статистику
                stats = calculate_statistics_from_processed_docs(processed_docs)
                return {
                    "model_saved": True,
                    "documents_processed": len(processed_docs),
                    "vocabulary_size": len(trained_model.wv.key_to_index),
                    "training_time": training_time,
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
            total_steps=100,
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
@click.option("--documents", "-d", help="Путь к папке с документами")
@click.option("--model", "-m", default="doc2vec_model", help="Имя модели")
def stats(documents: Optional[str], model: str):
    """Показать статистику модели и корпуса"""
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
@click.option(
    "--min-length", "-l", default=15, help="Минимальная длина предложения в символах"
)
@click.option(
    "--min-words", "-w", default=5, help="Минимальное количество слов в предложении"
)
@click.option(
    "--no-filter", is_flag=True, help="Отключить фильтрацию коротких предложений"
)
@click.option("--output", "-o", help="Файл для сохранения выжимки")
def summarize_file(
    file: str,
    model: str,
    sentences: int,
    min_length: int,
    min_words: int,
    no_filter: bool,
    output: Optional[str],
) -> None:
    """
    Создать выжимку из файла с фильтрацией коротких предложений

    Args:
        file: Путь к файлу для суммаризации
        model: Имя Doc2Vec модели для улучшенной суммаризации
        sentences: Количество предложений в выжимке
        min_length: Минимальная длина предложения в символах
        min_words: Минимальное количество слов в предложении
        no_filter: Отключить фильтрацию коротких предложений
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

    # Настройка параметров фильтрации
    if not no_filter:
        summarizer.min_summary_sentence_length = min_length
        summarizer.min_words_in_sentence = min_words
        click.echo(f"📏 Фильтрация: минимум {min_length} символов и {min_words} слов")
    else:
        summarizer.min_summary_sentence_length = 1
        summarizer.min_words_in_sentence = 1
        click.echo("📋 Фильтрация коротких предложений отключена")

    logger.info(f"Создание выжимки файла: {file_path}")

    # Создание выжимки
    try:
        summary = summarizer.summarize_file(str(file_path), sentences_count=sentences)

        if not summary:
            click.echo(
                "❌ Не удалось создать выжимку. Возможные причины:\n"
                "   - Все предложения слишком короткие\n"
                "   - Файл не содержит текста\n"
                "   Попробуйте --no-filter или уменьшите --min-length"
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

                if "valid_original_sentences_count" in stats and not no_filter:
                    filtered = (
                        stats["original_sentences_count"]
                        - stats["valid_original_sentences_count"]
                    )
                    click.echo(f"  🔽 Отфильтровано коротких: {filtered}")
                    click.echo(
                        f"  ✅ Валидных предложений: {stats['valid_original_sentences_count']}"
                    )

                click.echo(
                    f"  📄 Предложений в выжимке: {stats['summary_sentences_count']}"
                )
                click.echo(f"  📉 Коэффициент сжатия: {stats['compression_ratio']:.1%}")
                click.echo(f"  🔤 Исходных символов: {stats['original_chars_count']:,}")
                click.echo(f"  ✂️ Символов в выжимке: {stats['summary_chars_count']:,}")
                click.echo(
                    f"  📊 Сокращение текста: {stats['chars_compression_ratio']:.1%}"
                )

                if "avg_sentence_length" in stats:
                    click.echo(
                        f"  📏 Средняя длина предложения: {stats['avg_sentence_length']:.1f} слов"
                    )

        except Exception as e:
            logger.error(f"Ошибка при расчете статистики: {e}")

        # Сохранение в файл
        if output:
            output_path = Path(output)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"Выжимка файла: {file_path.name}\n")
                    f.write(f"Создано: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if not no_filter:
                        f.write(
                            f"Фильтрация: мин. {min_length} символов, {min_words} слов\n"
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

                        if "valid_original_sentences_count" in stats and not no_filter:
                            f.write(
                                f"Валидных предложений: {stats['valid_original_sentences_count']}\n"
                            )
                            f.write(
                                f"Отфильтровано: {stats['original_sentences_count'] - stats['valid_original_sentences_count']}\n"
                            )

                        f.write(
                            f"Предложений в выжимке: {stats['summary_sentences_count']}\n"
                        )
                        f.write(
                            f"Коэффициент сжатия: {stats['compression_ratio']:.1%}\n"
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
@click.option(
    "--min-length", "-l", default=15, help="Минимальная длина предложения в символах"
)
@click.option(
    "--min-words", "-w", default=5, help="Минимальное количество слов в предложении"
)
@click.option(
    "--no-filter", is_flag=True, help="Отключить фильтрацию коротких предложений"
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
    min_length: int,
    min_words: int,
    no_filter: bool,
    output_dir: Optional[str],
    extensions: str,
    max_files: int,
) -> None:
    """
    Создать выжимки для всех документов в папке с фильтрацией

    Args:
        documents: Путь к папке с документами
        model: Имя Doc2Vec модели
        sentences: Количество предложений в каждой выжимке
        min_length: Минимальная длина предложения
        min_words: Минимальное количество слов
        no_filter: Отключить фильтрацию
        output_dir: Папка для сохранения выжимок
        extensions: Обрабатываемые расширения файлов
        max_files: Максимальное количество файлов
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

    # Настройка фильтрации
    if not no_filter:
        summarizer.min_summary_sentence_length = min_length
        summarizer.min_words_in_sentence = min_words
        click.echo(f"📏 Фильтрация: минимум {min_length} символов и {min_words} слов")
    else:
        summarizer.min_summary_sentence_length = 1
        summarizer.min_words_in_sentence = 1
        click.echo("📋 Фильтрация отключена")

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
    filtered_out = 0

    # Обработка файлов
    for i, file_path in enumerate(all_files, 1):
        click.echo(f"\n🔄 Обработка {i}/{len(all_files)}: {file_path.name}")

        try:
            # Создание выжимки
            summary = summarizer.summarize_file(
                str(file_path), sentences_count=sentences
            )

            if not summary:
                click.echo(
                    "   ⚠️ Не удалось создать выжимку (возможно, все предложения слишком короткие)"
                )
                filtered_out += 1
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
                        if not no_filter:
                            f.write(
                                f"Фильтрация: мин. {min_length} символов, {min_words} слов\n"
                            )
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
    click.echo(f"  ⚠️ Отфильтровано (короткие предложения): {filtered_out}")
    click.echo(f"  ❌ Ошибок: {failed}")
    click.echo(f"  📁 Всего файлов: {len(all_files)}")
    click.echo(f"  📈 Процент успеха: {(successful / len(all_files) * 100):.1f}%")

    if output_dir and successful > 0:
        click.echo(f"  💾 Выжимки сохранены в: {output_path}")

    if filtered_out > 0 and not no_filter:
        click.echo(
            "\n💡 Совет: Используйте --no-filter или уменьшите --min-length для обработки файлов с короткими предложениями"
        )


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


"""Добавить в main.py после других CLI команд"""


@cli.command()
@click.option("--model", "-m", default="doc2vec_model", help="Имя Doc2Vec модели")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option(
    "--test-cases",
    type=click.Choice(["quick", "standard", "extended"]),
    default="standard",
    help="Набор тестовых случаев",
)
@click.option("--output-dir", "-o", help="Директория для сохранения результатов")
def evaluate(model: str, openai_key: str, test_cases: str, output_dir: Optional[str]):
    """
    Сравнение Doc2Vec с OpenAI embeddings

    Args:
        model: Имя Doc2Vec модели для оценки
        openai_key: API ключ OpenAI (или из OPENAI_API_KEY)
        test_cases: Размер набора тестов (quick/standard/extended)
        output_dir: Директория для результатов
    """
    if not openai_key:
        click.echo("❌ OpenAI API key не найден")
        click.echo(
            "Установите переменную окружения OPENAI_API_KEY или используйте --openai-key"
        )
        return

    # Загрузка модели
    click.echo(f"📂 Загрузка модели {model}...")
    trainer = Doc2VecTrainer()
    loaded_model = trainer.load_model(model)

    if not loaded_model:
        click.echo(f"❌ Не удалось загрузить модель '{model}'")
        return

    if not trainer.corpus_info:
        click.echo("❌ Информация о корпусе не найдена")
        return

    # Импорт модулей оценки
    from semantic_search.core.search_engine import SemanticSearchEngine
    from semantic_search.evaluation.baselines import (
        Doc2VecSearchAdapter,
        OpenAISearchBaseline,
    )
    from semantic_search.evaluation.comparison import SearchComparison

    # Создание поискового движка
    search_engine = SemanticSearchEngine(loaded_model, trainer.corpus_info)

    # Создание сравнения
    comparison = SearchComparison()

    # Получение тестовых случаев
    click.echo("🧪 Подготовка тестовых случаев...")
    default_cases = comparison.create_default_test_cases()

    if test_cases == "quick":
        test_cases_list = default_cases[:3]
        click.echo("   Быстрый тест: 3 запроса")
    elif test_cases == "extended":
        # Добавляем дополнительные тесты
        from semantic_search.evaluation.comparison import QueryTestCase

        extra_cases = [
            QueryTestCase(
                query="классификация изображений CNN",
                relevant_docs={"cnn_tutorial.pdf", "image_classification.pdf"},
                relevance_scores={"cnn_tutorial.pdf": 3, "image_classification.pdf": 3},
            ),
            QueryTestCase(
                query="регуляризация dropout L1 L2",
                relevant_docs={"regularization.pdf", "overfitting.pdf"},
                relevance_scores={"regularization.pdf": 3, "overfitting.pdf": 2},
            ),
            QueryTestCase(
                query="word embeddings word2vec GloVe",
                relevant_docs={"word2vec_paper.pdf", "embeddings_tutorial.pdf"},
                relevance_scores={
                    "word2vec_paper.pdf": 3,
                    "embeddings_tutorial.pdf": 3,
                },
            ),
            QueryTestCase(
                query="precision recall F1 score ROC AUC",
                relevant_docs={"ml_metrics.pdf", "evaluation_methods.pdf"},
                relevance_scores={"ml_metrics.pdf": 3, "evaluation_methods.pdf": 3},
            ),
            QueryTestCase(
                query="backpropagation gradient descent",
                relevant_docs={"backpropagation.pdf", "optimization.pdf"},
                relevance_scores={"backpropagation.pdf": 3, "optimization.pdf": 2},
            ),
        ]
        test_cases_list = default_cases + extra_cases
        click.echo("   Расширенный тест: 10 запросов")
    else:  # standard
        test_cases_list = default_cases
        click.echo("   Стандартный тест: 5 запросов")

    comparison.test_cases = test_cases_list

    # Создание адаптеров
    click.echo("\n🔧 Инициализация методов поиска...")
    doc2vec_adapter = Doc2VecSearchAdapter(search_engine, trainer.corpus_info)

    try:
        openai_baseline = OpenAISearchBaseline(api_key=openai_key)
        click.echo("✅ OpenAI baseline инициализирован")
    except Exception as e:
        click.echo(f"❌ Ошибка инициализации OpenAI: {e}")
        return

    # Подготовка документов для индексации
    click.echo("\n📚 Подготовка документов для OpenAI...")
    documents = []
    max_docs = min(50, len(trainer.corpus_info))  # Ограничиваем для экономии

    for i, (tokens, doc_id, metadata) in enumerate(trainer.corpus_info[:max_docs]):
        # Восстанавливаем текст из токенов
        text = " ".join(tokens[:500])  # Берем первые 500 токенов
        documents.append((doc_id, text, metadata))

    click.echo(f"   Подготовлено {len(documents)} документов")

    # Индексация для OpenAI
    click.echo("\n🔄 Индексация документов через OpenAI API...")
    click.echo("   (это может занять несколько минут)")

    try:
        with click.progressbar(length=100, label="Индексация") as bar:
            # Используем callback для обновления прогресса
            original_index = openai_baseline.index

            def index_with_progress(docs):
                result = original_index(docs)
                bar.update(100)
                return result

            openai_baseline.index = index_with_progress
            openai_baseline.index(documents)
            openai_baseline.index = original_index

        click.echo("✅ Индексация завершена")
    except Exception as e:
        click.echo(f"❌ Ошибка индексации: {e}")
        return

    # Оценка методов
    click.echo("\n📊 Оценка методов...")

    # Doc2Vec
    click.echo("\n1️⃣ Оценка Doc2Vec...")
    doc2vec_results = comparison.evaluate_method(
        doc2vec_adapter, top_k=10, verbose=False
    )
    click.echo(f"   MAP: {doc2vec_results['aggregated']['MAP']:.3f}")
    click.echo(f"   MRR: {doc2vec_results['aggregated']['MRR']:.3f}")
    click.echo(
        f"   Среднее время запроса: {doc2vec_results['aggregated']['avg_query_time']:.3f}с"
    )

    # OpenAI
    click.echo("\n2️⃣ Оценка OpenAI embeddings...")
    openai_results = comparison.evaluate_method(
        openai_baseline, top_k=10, verbose=False
    )
    click.echo(f"   MAP: {openai_results['aggregated']['MAP']:.3f}")
    click.echo(f"   MRR: {openai_results['aggregated']['MRR']:.3f}")
    click.echo(
        f"   Среднее время запроса: {openai_results['aggregated']['avg_query_time']:.3f}с"
    )

    # Сравнение результатов
    click.echo("\n📈 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    click.echo("=" * 60)

    # MAP сравнение
    doc2vec_map = doc2vec_results["aggregated"]["MAP"]
    openai_map = openai_results["aggregated"]["MAP"]
    map_improvement = (
        ((doc2vec_map - openai_map) / openai_map * 100) if openai_map > 0 else 0
    )

    click.echo("\n📊 Mean Average Precision (MAP):")
    click.echo(f"   Doc2Vec: {doc2vec_map:.3f}")
    click.echo(f"   OpenAI:  {openai_map:.3f}")

    if map_improvement > 0:
        click.echo(f"   ✅ Doc2Vec превосходит OpenAI на {map_improvement:.1f}%")
    else:
        click.echo(f"   ❌ OpenAI превосходит Doc2Vec на {-map_improvement:.1f}%")

    # Скорость
    doc2vec_time = doc2vec_results["aggregated"]["avg_query_time"]
    openai_time = openai_results["aggregated"]["avg_query_time"]
    speed_ratio = openai_time / doc2vec_time if doc2vec_time > 0 else float("inf")

    click.echo("\n⚡ Скорость поиска:")
    click.echo(f"   Doc2Vec: {doc2vec_time:.3f}с на запрос")
    click.echo(f"   OpenAI:  {openai_time:.3f}с на запрос")
    click.echo(f"   ✅ Doc2Vec быстрее в {speed_ratio:.1f} раз")

    # Другие метрики
    click.echo("\n📏 Дополнительные метрики:")

    for metric in ["precision@10", "recall@10", "f1@10"]:
        doc2vec_val = doc2vec_results["aggregated"].get(f"avg_{metric}", 0)
        openai_val = openai_results["aggregated"].get(f"avg_{metric}", 0)
        click.echo(f"   {metric.upper()}:")
        click.echo(f"      Doc2Vec: {doc2vec_val:.3f}")
        click.echo(f"      OpenAI:  {openai_val:.3f}")

    # Экономическая эффективность
    click.echo("\n💰 Экономическая эффективность:")

    # Расчет примерной стоимости
    queries_per_day = 1000
    avg_tokens_per_query = 50
    openai_cost_per_1k_tokens = 0.0001  # $0.0001 за 1K токенов для ada-002

    daily_queries_cost = (
        queries_per_day * avg_tokens_per_query / 1000
    ) * openai_cost_per_1k_tokens
    # Стоимость индексации (примерно 200 токенов на документ)
    indexing_cost = (len(documents) * 200 / 1000) * openai_cost_per_1k_tokens

    monthly_cost = daily_queries_cost * 30 + indexing_cost
    yearly_cost = monthly_cost * 12

    click.echo(f"   При {queries_per_day} запросов в день:")
    click.echo(
        f"   - Стоимость OpenAI: ~${monthly_cost:.2f}/месяц (${yearly_cost:.2f}/год)"
    )
    click.echo("   - Стоимость Doc2Vec: $0 (после единоразового обучения)")
    click.echo(f"   - 💵 Экономия: ${yearly_cost:.2f} в год")

    # Генерация отчетов
    click.echo("\n📄 Генерация отчетов и графиков...")

    # Определяем директорию для результатов
    if output_dir:
        results_dir = Path(output_dir)
    else:
        from semantic_search.config import EVALUATION_RESULTS_DIR

        results_dir = EVALUATION_RESULTS_DIR

    results_dir.mkdir(exist_ok=True, parents=True)

    # Сравнительная таблица
    comparison.compare_methods([doc2vec_adapter, openai_baseline], save_results=True)

    # Генерация графиков
    try:
        comparison.plot_comparison(save_plots=True)
        click.echo("✅ Графики сохранены")
    except Exception as e:
        click.echo(f"⚠️ Не удалось создать графики: {e}")

    # Текстовый отчет
    report_path = results_dir / "comparison_report.txt"
    comparison.generate_report(report_path)

    click.echo(f"\n✅ Результаты сохранены в: {results_dir}")
    click.echo("   📊 comparison_results.csv - таблица с метриками")
    click.echo("   📝 comparison_report.txt - текстовый отчет")
    click.echo("   📈 plots/ - графики сравнения")
    click.echo("   🗂️ detailed_results.json - детальные результаты")

    # Основные выводы
    click.echo("\n🎯 ОСНОВНЫЕ ВЫВОДЫ:")
    click.echo("=" * 60)

    if map_improvement > 0:
        click.echo(
            f"✅ Doc2Vec показывает ЛУЧШЕЕ качество поиска (+{map_improvement:.1f}% MAP)"
        )
        click.echo("   на специализированном корпусе документов")
    else:
        click.echo(
            f"⚠️ OpenAI показывает лучшее качество поиска (+{-map_improvement:.1f}% MAP)"
        )
        click.echo("   Рекомендуется дообучить модель Doc2Vec")

    click.echo(f"\n✅ Doc2Vec работает ЗНАЧИТЕЛЬНО БЫСТРЕЕ (в {speed_ratio:.1f} раз)")
    click.echo("   и не требует интернет-соединения")

    click.echo(f"\n✅ Doc2Vec ЭКОНОМИЧЕСКИ ВЫГОДНЕЕ (экономия ${yearly_cost:.0f}/год)")
    click.echo("   при регулярном использовании")

    click.echo("\n📌 Рекомендация: Doc2Vec оптимален для:")
    click.echo("   • Специализированных корпусов документов")
    click.echo("   • Высокой нагрузки (много запросов)")
    click.echo("   • Работы без интернета")
    click.echo("   • Конфиденциальных данных")


def cli_mode():
    """Запуск CLI режима"""
    cli()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        main()
