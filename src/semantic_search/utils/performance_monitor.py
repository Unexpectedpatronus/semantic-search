"""Мониторинг производительности"""

import time
from contextlib import contextmanager
from typing import Any, Dict

import psutil
from loguru import logger


class PerformanceMonitor:
    """Мониторинг производительности операций"""

    def __init__(self):
        self.metrics = {}

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Контекстный менеджер для измерения времени операции"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            self.metrics[operation_name] = {
                "duration": duration,
                "memory_start": start_memory,
                "memory_end": end_memory,
                "memory_delta": memory_delta,
                "timestamp": time.time(),
            }

            logger.info(
                f"{operation_name}: {duration:.2f}s, Память: {memory_delta:+.1f}MB"
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total": psutil.virtual_memory().total / 1024**3,  # GB
            "memory_available": psutil.virtual_memory().available / 1024**3,  # GB
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
            if psutil.disk_usage("/")
            else 0,
        }
