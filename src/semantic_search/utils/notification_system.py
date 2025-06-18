"""Система уведомлений и трекинга прогресса"""

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional

from loguru import logger


class NotificationType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"


@dataclass
class Notification:
    type: NotificationType
    title: str
    message: str
    details: Optional[str] = None
    progress: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


class ProgressTracker:
    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()

    def step(self, count: int = 1) -> None:
        self.current_step = min(self.total_steps, self.current_step + count)

    def percent(self) -> float:
        return (
            round(self.current_step / self.total_steps, 4) if self.total_steps else 0.0
        )

    def elapsed(self) -> float:
        return round(time.time() - self.start_time, 2)


class NotificationManager:
    def __init__(self):
        self._queue: Queue[Notification] = Queue()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._callback: Optional[Callable[[Notification], None]] = None

    def start(self, callback: Callable[[Notification], None]) -> None:
        self._callback = callback
        self._stop_event.clear()
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.debug("Менеджер уведомлений запущен")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            logger.debug("Менеджер уведомлений остановлен")

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                notification = self._queue.get(timeout=0.5)
                if self._callback:
                    self._callback(notification)
            except Empty:
                continue

    def send(self, notification: Notification) -> None:
        self._queue.put(notification)
        logger.debug(
            f"Уведомление отправлено: {notification.title} [{notification.type.value}]"
        )
