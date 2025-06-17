"""Утилиты для работы с файлами"""

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf
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
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Word Application: {e}")

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
        ext_counter = Counter(f.suffix.lower() for f in found_files)
        for ext in SUPPORTED_EXTENSIONS:
            count = ext_counter.get(ext, 0)
            if count > 0:
                logger.info(f"  {ext}: {count}")

        return found_files

    def extract_from_pdf(self, file_path: Path) -> str:
        """Улучшенная извлечение текста из PDF с обработкой больших файлов"""
        try:
            doc = pymupdf.open(file_path)
            text_parts = []

            # Проверяем количество страниц
            total_pages = len(doc)
            if total_pages > 1000:
                logger.warning(
                    f"PDF содержит {total_pages} страниц. Обработка может занять время"
                )

            # Обрабатываем с прогресс-баром для больших файлов
            page_iterator = range(len(doc))
            if total_pages > 50:
                try:
                    from tqdm import tqdm

                    page_iterator = tqdm(
                        page_iterator, desc=f"Извлечение из {file_path.name}"
                    )
                except ImportError:
                    pass

            for page_num in page_iterator:
                try:
                    page = doc[page_num]
                    page_text = page.get_text()

                    # Фильтруем явно мусорные страницы
                    if len(page_text.strip()) > 50:  # Минимум 50 символов
                        text_parts.append(page_text)

                except Exception as e:
                    logger.warning(f"Ошибка при обработке страницы {page_num}: {e}")
                    continue

            doc.close()

            full_text = "\n".join(text_parts)
            logger.info(f"Извлечено {len(full_text)} символов из {total_pages} страниц")

            return full_text.strip()

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из PDF {file_path}: {e}")
            return ""

    def extract_text_with_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Извлечение текста с метаданными"""
        result = {
            "text": "",
            "metadata": {
                "file_path": str(file_path),
                "file_size": 0,
                "creation_time": None,
                "modification_time": None,
                "pages_count": 0,
                "extraction_time": 0,
            },
        }

        start_time = time.time()

        try:
            # Базовые метаданные файла
            stat = file_path.stat()
            result["metadata"].update(
                {
                    "file_size": stat.st_size,
                    "creation_time": stat.st_ctime,
                    "modification_time": stat.st_mtime,
                }
            )

            # Извлечение текста
            text = self.extract_text(file_path)
            result["text"] = text

            # Дополнительные метаданные для PDF
            if file_path.suffix.lower() == ".pdf":
                try:
                    doc = pymupdf.open(file_path)
                    result["metadata"]["pages_count"] = len(doc)

                    # Метаданные документа PDF
                    pdf_metadata = doc.metadata
                    if pdf_metadata:
                        result["metadata"].update(
                            {
                                "title": pdf_metadata.get("title", ""),
                                "author": pdf_metadata.get("author", ""),
                                "subject": pdf_metadata.get("subject", ""),
                                "creator": pdf_metadata.get("creator", ""),
                            }
                        )
                    doc.close()

                except Exception as e:
                    logger.debug(f"Не удалось получить PDF метаданные: {e}")

            result["metadata"]["extraction_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста с метаданными {file_path}: {e}")
            result["metadata"]["extraction_time"] = time.time() - start_time

        return result

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
