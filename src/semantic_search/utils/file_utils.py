"""Утилиты для работы с файлами: чтение PDF, DOCX, DOC"""

from pathlib import Path
from typing import List, Optional

import fitz  # type: ignore
from docx import Document as DocxDocument
from loguru import logger

from semantic_search.config import SUPPORTED_EXTENSIONS

try:
    import win32com.client

    DOC_SUPPORT = True
except ImportError:
    DOC_SUPPORT = False
    logger.warning("pywin32 не установлен. Поддержка .doc недоступна")


class FileExtractor:
    """Класс для извлечения текста из различных типов документов"""

    def __init__(self):
        self.word_app: Optional[object] = None
        if DOC_SUPPORT:
            try:
                self.word_app = win32com.client.Dispatch("Word.Application")
                self.word_app.Visible = False
            except Exception as e:
                logger.warning(f"Не удалось запустить Word: {e}")

    def extract_text(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._extract_from_pdf(path)
        elif ext == ".docx":
            return self._extract_from_docx(path)
        elif ext == ".doc" and DOC_SUPPORT:
            return self._extract_from_doc(path)
        raise ValueError(f"Неподдерживаемый тип файла: {ext}")

    def find_documents(self, root_path: Path) -> List[Path]:
        return [
            p
            for p in root_path.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        ]

    def _extract_from_pdf(self, path: Path) -> str:
        try:
            doc = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF {path.name}: {e}")
            return ""

    def _extract_from_docx(self, path: Path) -> str:
        try:
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except Exception as e:
            logger.error(f"Ошибка при чтении DOCX {path.name}: {e}")
            return ""

    def _extract_from_doc(self, path: Path) -> str:
        try:
            doc = self.word_app.Documents.Open(str(path))
            text = doc.Content.Text
            doc.Close(False)
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка при чтении DOC {path.name}: {e}")
            return ""

    def __del__(self):
        if self.word_app:
            try:
                self.word_app.Quit()
                logger.debug("Word Application завершена")
            except Exception as e:
                logger.warning(f"Ошибка при завершении Word: {e}")
