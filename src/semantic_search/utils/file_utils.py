"""Утилиты для работы с файлами"""

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from loguru import logger

from semantic_search.config import SUPPORTED_EXTENSIONS

try:
    import win32com.client

    DOC_SUPPORT = True
except ImportError:
    DOC_SUPPORT = False
    logger.warning("pywin32 не установлен. Поддержка .doc файлов недоступна")


class FileExtractor:
    """Класс для извлечения текста из различных форматов файлов"""

    def __init__(self):
        self.word_app: Optional[object] = None
        if DOC_SUPPORT:
            try:
                self.word_app = win32com.client.Dispatch("Word.Application")
                self.word_app.Visible = False
                logger.info(
                    "Word Application инициализирован для работы с .doc файлами"
                )
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Word Application: {e}")
                global DOC_SUPPORT
                DOC_SUPPORT = False

    def __del__(self):
        """Освобождение ресурсов"""
        if self.word_app is not None:
            try:
                # Проверяем, что объект поддерживает метод Quit
                if hasattr(self.word_app, "Quit"):
                    self.word_app.Quit()
            except Exception:
                pass

    def find_documents(self, root_path: Path) -> List[Path]:
        """
        Рекурсивный поиск документов в директории

        Args:
            root_path: Путь к корневой директории

        Returns:
            Список путей к найденным файлам
        """
        if not root_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {root_path}")

        found_files = []
        logger.info(f"Поиск документов в: {root_path}")

        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found_files.append(file_path)

        # Логирование статистики
        logger.info(f"Найдено файлов: {len(found_files)}")
        for ext in SUPPORTED_EXTENSIONS:
            count = sum(1 for f in found_files if f.suffix.lower() == ext)
            if count > 0:
                logger.info(f"  {ext}: {count}")

        return found_files

    def extract_from_pdf(self, file_path: Path) -> str:
        """Извлечение текста из PDF файла"""
        try:
            doc = fitz.open(file_path)
            text = ""

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()

            doc.close()
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из PDF {file_path}: {e}")
            return ""

    def extract_from_docx(self, file_path: Path) -> str:
        """Извлечение текста из DOCX файла"""
        try:
            doc = DocxDocument(file_path)
            text_parts = []

            # Извлекаем текст из параграфов
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())

            # Извлекаем текст из таблиц
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text.strip())

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из DOCX {file_path}: {e}")
            return ""

    def extract_from_doc(self, file_path: Path) -> str:
        """Извлечение текста из DOC файла (только Windows)"""
        if not DOC_SUPPORT or self.word_app is None:
            logger.warning(f"Поддержка .doc файлов недоступна: {file_path}")
            return ""

        try:
            doc = self.word_app.Documents.Open(str(file_path.absolute()))
            text = doc.Content.Text
            doc.Close()
            return text.strip()

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из DOC {file_path}: {e}")
            return ""

    def extract_text(self, file_path: Path) -> str:
        """
        Универсальная функция извлечения текста

        Args:
            file_path: Путь к файлу

        Returns:
            Извлеченный текст
        """
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self.extract_from_pdf(file_path)
        elif extension == ".docx":
            return self.extract_from_docx(file_path)
        elif extension == ".doc":
            return self.extract_from_doc(file_path)
        else:
            logger.warning(f"Неподдерживаемый формат файла: {file_path}")
            return ""
