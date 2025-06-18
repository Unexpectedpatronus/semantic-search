"""Контекстный мониторинг производительности операций"""

import time
from contextlib import contextmanager
from typing import Dict, Generator

import psutil
from loguru import logger


class PerformanceMonitor:
    """Класс для измерения времени и использования памяти"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.process = psutil.Process()

    @contextmanager
    def measure_operation(self, name: str) -> Generator[None, None, None]:
        start_time = time.perf_counter()
        start_mem = self._get_memory()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_mem = self._get_memory()

            self.metrics[name] = {
                "duration_sec": round(end_time - start_time, 4),
                "memory_start_mb": round(start_mem, 3),
                "memory_end_mb": round(end_mem, 3),
                "memory_delta_mb": round(end_mem - start_mem, 3),
            }

            logger.debug(
                f"[{name}] duration: {end_time - start_time:.4f}s, memory delta: {end_mem - start_mem:.3f}MB"
            )

    def _get_memory(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.metrics
