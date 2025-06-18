"""Конфигурация логирования через loguru"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


class LoggingManager:
    """Менеджер логирования для проекта"""

    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir
        self.is_configured = False

    def setup_logging(
        self,
        level: str = "INFO",
        enable_file_logging: bool = True,
        enable_rotation: bool = True,
        custom_format: Optional[str] = None,
    ) -> None:
        if self.is_configured:
            return

        logger.remove()
        log_format = custom_format or DEFAULT_FORMAT

        # Консольный логгер
        logger.add(sys.stdout, level=level, format=log_format, enqueue=True)

        # Файловый логгер (если задан каталог)
        if enable_file_logging and self.logs_dir:
            log_file = self.logs_dir / "app.log"
            logger.add(
                str(log_file),
                level=level,
                format=log_format,
                rotation="1 week" if enable_rotation else None,
                retention="1 month",
                encoding="utf-8",
                enqueue=True,
            )

        logger.debug("Система логирования настроена")
        self.is_configured = True
