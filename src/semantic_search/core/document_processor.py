"""Основной модуль для обработки документов"""

from collections import Counter
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
        logger.info(f"Начинаем обработку документов в: {root_path}")

        # Находим все документы
        file_paths = self.file_extractor.find_documents(root_path)

        if not file_paths:
            logger.warning("Документы не найдены")
            return

        processed_count = 0
        skipped_count = 0

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Обработка {i}/{len(file_paths)}: {file_path.name}")

            try:
                # Извлекаем текст
                raw_text = self.file_extractor.extract_text(file_path)

                # Обрезаем слишком длинный текст
                if len(raw_text) > self.config["max_text_length"]:
                    raw_text = raw_text[: self.config["max_text_length"]]
                    logger.info(
                        f"Текст обрезан до {self.config['max_text_length']} символов"
                    )

                if len(raw_text) < self.config["min_text_length"]:
                    logger.warning(
                        f"Текст слишком короткий ({len(raw_text)} символов): {file_path}"
                    )
                    skipped_count += 1
                    continue

                # Препроцессинг
                tokens = self.text_processor.preprocess_text(raw_text)

                if len(tokens) < self.config["min_tokens_count"]:
                    logger.warning(f"Слишком мало токенов ({len(tokens)}): {file_path}")
                    skipped_count += 1
                    continue

                # Создаем относительный путь для ID документа
                relative_path = str(file_path.relative_to(root_path))

                # Собираем метаданные
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

            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")
                skipped_count += 1
                continue

        logger.info(
            f"Обработка завершена. Успешно: {processed_count}, Пропущено: {skipped_count}"
        )

    def prepare_corpus_for_doc2vec(self, root_path: Path) -> List[tuple]:
        """
        Подготовка корпуса для обучения Doc2Vec

        Args:
            root_path: Путь к корневой директории

        Returns:
            Список кортежей (tokens, doc_id, metadata)
        """
        corpus = []

        for doc in self.process_documents(root_path):
            corpus.append(
                (
                    doc.tokens,
                    doc.relative_path,
                    doc.metadata,
                )
            )

        logger.info(f"Подготовлен корпус из {len(corpus)} документов")
        return corpus

    def get_processing_statistics(self, root_path: Path) -> dict:
        """
        Получение статистики по обработке документов

        Args:
            root_path: Путь к корневой директории

        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "total_tokens": 0,
            "avg_tokens_per_doc": 0,
            "extensions_count": dict,
            "largest_doc": None,
            "smallest_doc": None,
        }

        docs_data = list(self.process_documents(root_path))

        if not docs_data:
            return stats

        stats["processed_files"] = len(docs_data)
        stats["total_tokens"] = sum(doc.metadata["tokens_count"] for doc in docs_data)
        stats["avg_tokens_per_doc"] = stats["total_tokens"] / stats["processed_files"]

        # Статистика по расширениям
        extensions_count = Counter(doc.metadata["extension"] for doc in docs_data)
        stats["extensions_count"] = dict(extensions_count)

        # Самый большой и маленький документы
        docs_by_tokens = sorted(docs_data, key=lambda x: x.metadata["tokens_count"])
        stats["smallest_doc"] = {
            "path": docs_by_tokens[0].relative_path,
            "tokens": docs_by_tokens[0].metadata["tokens_count"],
        }
        stats["largest_doc"] = {
            "path": docs_by_tokens[-1].relative_path,
            "tokens": docs_by_tokens[-1].metadata["tokens_count"],
        }

        return stats
