"""Основной модуль для обработки документов"""

from pathlib import Path
from typing import Generator, List, NamedTuple

from loguru import logger

from semantic_search.config import TEXT_PROCESSING_CONFIG
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.text_utils import TextProcessor


class ProcessedDocument(NamedTuple):
    """Структура для хранения обрабатываемого документа"""

    file_path: Path
    relative_path: str
    raw_text: str
    tokens: List[str]
    metadata: dict


class DocumentProcessor:
    """Главный класс для обработки коллекции документов"""

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.text_processor = TextProcessor()
        self.config = TEXT_PROCESSING_CONFIG

    def process_documents(
        self, root_path: Path
    ) -> Generator[ProcessedDocument, None, None]:
        """
        Основная функция обработки документов

        Args:
            root_path: Путь к корневой директории с документами

        Yields:
            ProcessedDocument объекты
        """

        if not root_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {root_path}")

        if not root_path.is_dir():
            raise NotADirectoryError(f"Путь не является директорией: {root_path}")

        logger.info(f"Начинаем обработку документов в: {root_path}")

        file_paths = self.file_extractor.find_documents(root_path)

        if not file_paths:
            logger.warning("Документы не найдены")
            return

        processed_count = 0
        skipped_count = 0

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Обработка {i}/{len(file_paths)}: {file_path.name}")

            try:
                file_size = file_path.stat().st_size
                max_file_size_bytes = (
                    self.config.get("max_file_size_mb", 100) * 1024 * 1024
                )

                if file_size > max_file_size_bytes:
                    logger.warning(
                        f"Файл слишком большой ({file_size / 1024 / 1024:.1f}MB): {file_path}"
                    )
                    skipped_count += 1
                    continue

                raw_text = self.file_extractor.extract_text(file_path)

                if len(raw_text) < self.config["min_text_length"]:
                    logger.warning(
                        f"Текст слишком короткий ({len(raw_text)} символов): {file_path}"
                    )
                    skipped_count += 1
                    continue

                max_text_length = self.config.get("max_text_length", 5_000_000)

                if len(raw_text) > max_text_length:
                    logger.info(
                        f"Текст обрезан с {len(raw_text):,} до {max_text_length:,} символов"
                    )
                    raw_text = raw_text[:max_text_length]

                tokens = self.text_processor.preprocess_text(raw_text)

                if len(tokens) < self.config["min_tokens_count"]:
                    logger.warning(f"Слишком мало токенов ({len(tokens)}): {file_path}")
                    skipped_count += 1
                    continue

                relative_path = str(file_path.relative_to(root_path))
                relative_path = relative_path.replace("\\", "/")

                metadata = {
                    "file_size": file_path.stat().st_size,
                    "extension": file_path.suffix,
                    "tokens_count": len(tokens),
                    "text_length": len(raw_text),
                }

                processed_count += 1
                yield ProcessedDocument(
                    file_path=file_path,
                    relative_path=relative_path,
                    raw_text=raw_text,
                    tokens=tokens,
                    metadata=metadata,
                )

            except PermissionError:
                logger.error(f"Нет доступа к файлу: {file_path}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")
                skipped_count += 1
                continue

        logger.info(
            f"Обработка завершена. Успешно: {processed_count}, Пропущено: {skipped_count}"
        )
