"""Точка входа в приложение"""

import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from semantic_search.config import GUI_CONFIG
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.core.text_summarizer import TextSummarizer
from semantic_search.utils.logging_config import setup_logging
from semantic_search.utils.statistics import (
    calculate_model_statistics,
    calculate_statistics_from_processed_docs,
    format_statistics_for_display,
)


def main():
    """Главная функция приложения"""

    setup_logging()
    # Проверяем SpaCy модель при запуске
    from semantic_search.utils.text_utils import check_spacy_model_availability

    spacy_info = check_spacy_model_availability()

    if not spacy_info["model_loadable"]:
        logger.warning(f"SpaCy модель недоступна: {spacy_info['error']}")
        logger.info("Для установки: poetry run python scripts/setup_spacy.py")
    else:
        logger.info("✅ SpaCy модель готова к работе")

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
def train(documents: str, model: str, vector_size: int, epochs: int):
    """
    Обучить модель Doc2Vec на корпусе документов

    Args:
        documents: Путь к папке с документами
        model: Имя модели для сохранения
        vector_size: Размерность векторов
        epochs: Количество эпох обучения
    """

    documents_path = Path(documents)
    if not documents_path.exists():
        logger.error(f"Папка не найдена: {documents_path}")
        return

    logger.info("Режим обучения модели")

    # Создаем обработчик документов
    processor = DocumentProcessor()

    # Обработка документов
    processed_docs = list(processor.process_documents(documents_path))
    if not processed_docs:
        logger.error("Не удалось подготовить корпус")
        return
    corpus = [(doc.tokens, doc.relative_path, doc.metadata) for doc in processed_docs]
    logger.info(f"Подготовлен корпус из {len(corpus)} документов")

    # Обучение модели
    trainer = Doc2VecTrainer()
    trained_model = trainer.train_model(corpus, vector_size=vector_size, epochs=epochs)

    if trained_model:
        trainer.save_model(trained_model, model)
        logger.info("✅ Модель обучена и сохранена")

        # Показываем статистику
        stats = calculate_statistics_from_processed_docs(processed_docs)
        click.echo(f"\n{format_statistics_for_display(stats)}")
    else:
        logger.error("❌ Ошибка обучения модели")


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


def cli_mode():
    """Запуск CLI режима"""
    cli()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        main()
