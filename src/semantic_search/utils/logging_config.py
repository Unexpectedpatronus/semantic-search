"""Конфигурация логирования"""

import sys

from loguru import logger

from semantic_search.config import LOGS_DIR


def setup_logging(level: str = "INFO") -> None:
    """
    Настройка системы логирования

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    # Удаляем стандартный handler
    logger.remove()

    # Добавляем вывод в консоль
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
    )

    # Добавляем запись в файл
    log_file = LOGS_DIR / "semantic_search.log"
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    logger.info("Система логирования настроена")
