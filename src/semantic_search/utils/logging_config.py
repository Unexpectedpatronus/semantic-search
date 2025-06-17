"""Улучшенная конфигурация логирования"""

import sys
from typing import Optional

from loguru import logger

from semantic_search.config import LOGS_DIR


class LoggingManager:
    """Менеджер системы логирования"""

    def __init__(self):
        self.handlers = {}
        self.is_configured = False

    def setup_logging(
        self,
        level: str = "INFO",
        enable_file_logging: bool = True,
        enable_rotation: bool = True,
        custom_format: Optional[str] = None,
    ) -> None:
        """
        Расширенная настройка системы логирования

        Args:
            level: Уровень логирования
            enable_file_logging: Включить запись в файл
            enable_rotation: Включить ротацию логов
            custom_format: Пользовательский формат логов
        """
        if self.is_configured:
            return

        # Удаляем стандартный handler
        logger.remove()

        # Формат для консоли
        console_format = custom_format or (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Консольный вывод
        console_handler = logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            format=console_format,
            diagnose=True,
            backtrace=True,
        )
        self.handlers["console"] = console_handler

        if enable_file_logging:
            # Основной лог файл
            main_log = LOGS_DIR / "semantic_search.log"
            file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"

            if enable_rotation:
                main_handler = logger.add(
                    main_log,
                    level=level,
                    format=file_format,
                    rotation="10 MB",
                    retention="2 weeks",
                    compression="zip",
                    serialize=False,
                )
            else:
                main_handler = logger.add(main_log, level=level, format=file_format)

            self.handlers["main"] = main_handler

            # Отдельный файл для ошибок
            error_log = LOGS_DIR / "errors.log"
            error_handler = logger.add(
                error_log,
                level="ERROR",
                format=file_format,
                rotation="5 MB",
                retention="1 month",
                compression="zip",
            )
            self.handlers["error"] = error_handler

            # Лог производительности
            perf_log = LOGS_DIR / "performance.log"
            perf_handler = logger.add(
                perf_log,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
                filter=lambda record: "PERF" in record["message"],
                rotation="50 MB",
                retention="1 week",
            )
            self.handlers["performance"] = perf_handler

        self.is_configured = True
        logger.info("Система логирования настроена")

        # Логируем системную информацию
        self._log_system_info()

    def _log_system_info(self):
        """Логирование информации о системе"""
        try:
            import platform

            import psutil

            logger.info(f"Система: {platform.system()} {platform.release()}")
            logger.info(f"Python: {platform.python_version()}")
            logger.info(f"Процессор: {platform.processor()}")
            logger.info(f"ОЗУ: {psutil.virtual_memory().total / 1024**3:.1f} ГБ")
            logger.info(
                f"Свободное место: {psutil.disk_usage('/').free / 1024**3:.1f} ГБ"
            )

        except Exception as e:
            logger.debug(f"Не удалось получить системную информацию: {e}")

    def add_performance_log(self, operation: str, duration: float, **kwargs):
        """Добавление записи о производительности"""
        perf_data = {"operation": operation, "duration": duration, **kwargs}

        logger.info(f"PERF: {perf_data}")

    def cleanup(self):
        """Очистка ресурсов логирования"""
        for handler_id in self.handlers.values():
            try:
                logger.remove(handler_id)
            except ValueError:
                pass

        self.handlers.clear()
        self.is_configured = False


# Глобальный экземпляр
logging_manager = LoggingManager()


def setup_logging(level: str = "INFO") -> None:
    """Обратная совместимость"""
    logging_manager.setup_logging(level)
