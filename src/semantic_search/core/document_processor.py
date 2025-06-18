"""Модуль для обработки документов"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, List, Optional

from loguru import logger

from semantic_search.config import TEXT_PROCESSING_CONFIG
from semantic_search.utils.file_utils import FileExtractor
from semantic_search.utils.text_utils import TextProcessor


@dataclass
class ProcessedDocument:
    """Класс для представления обработанного документа"""

    file_path: Path
    relative_path: str
    raw_text: str
    tokens: List[str]
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["file_path"] = str(data["file_path"])
        return data


class DocumentProcessor:
    """Класс для обработки коллекции документов"""

    def __init__(
        self,
        file_extractor: Optional[FileExtractor] = None,
        text_processor: Optional[TextProcessor] = None,
        config: Optional[dict] = None,
    ):
        self.file_extractor = file_extractor or FileExtractor()
        self.text_processor = text_processor or TextProcessor()
        self.config = config or TEXT_PROCESSING_CONFIG

    def process_documents(
        self, root_path: Path
    ) -> Generator[ProcessedDocument, None, None]:
        """Обрабатывает документы в директории"""
        file_paths = self.file_extractor.find_documents(root_path)
        logger.info(f"Найдено {len(file_paths)} файлов для обработки.")

        for file_path in file_paths:
            try:
                raw_text = self.file_extractor.extract_text(file_path)
                tokens = self.text_processor.preprocess_text(raw_text)
                rel_path = str(file_path.relative_to(root_path))

                yield ProcessedDocument(
                    file_path=file_path,
                    relative_path=rel_path,
                    raw_text=raw_text,
                    tokens=tokens,
                    metadata={"file_size": file_path.stat().st_size},
                )
            except Exception as e:
                logger.warning(f"Ошибка при обработке файла {file_path}: {e}")
