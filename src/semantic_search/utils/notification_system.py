"""Система уведомлений и прогресса"""

import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional

from loguru import logger


class NotificationType(Enum):
    """Типы уведомлений"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"


@dataclass
class Notification:
    """Структура уведомления"""

    type: NotificationType
    title: str
    message: str
    details: Optional[str] = None
    progress: Optional[float] = None  # 0.0 - 1.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ProgressTracker:
    """Трекер прогресса операций"""

    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.callbacks = []

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Добавление callback для обновлений прогресса"""
        self.callbacks.append(callback)

    def update(self, step: Optional[int] = None, message: str = ""):
        """Обновление прогресса"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        progress = min(self.current_step / self.total_steps, 1.0)
        elapsed = time.time() - self.start_time

        if progress > 0:
            eta = elapsed / progress * (1 - progress)
        else:
            eta = 0

        progress_info = {
            "progress": progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed": elapsed,
            "eta": eta,
            "message": message,
            "description": self.description,
        }

        # Вызываем callbacks
        for callback in self.callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                logger.error(f"Ошибка в progress callback: {e}")

    def finish(self, message: str = "Завершено"):
        """Завершение операции"""
        self.current_step = self.total_steps
        self.update(message=message)


class NotificationManager:
    """Менеджер системы уведомлений"""

    def __init__(self):
        self.subscribers = []
        self.notification_queue = Queue()
        self.is_running = False
        self.worker_thread = None
        self.stop_event = Event()

    def start(self):
        """Запуск системы уведомлений"""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        logger.info("Система уведомлений запущена")

    def stop(self):
        """Остановка системы уведомлений"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

        logger.info("Система уведомлений остановлена")

    def subscribe(self, callback: Callable[[Notification], None]):
        """Подписка на уведомления"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Notification], None]):
        """Отписка от уведомлений"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def notify(self, notification: Notification):
        """Отправка уведомления"""
        if self.is_running:
            self.notification_queue.put(notification)
        else:
            # Если система не запущена, отправляем сразу
            self._send_notification(notification)

    def _worker(self):
        """Рабочий поток для обработки уведомлений"""
        while not self.stop_event.is_set():
            try:
                notification = self.notification_queue.get(timeout=0.1)
                self._send_notification(notification)
                self.notification_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в worker уведомлений: {e}")

    def _send_notification(self, notification: Notification):
        """Отправка уведомления подписчикам"""
        for callback in self.subscribers:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Ошибка в callback уведомления: {e}")

    # Удобные методы для создания уведомлений
    def info(self, title: str, message: str, details: Optional[str] = None):
        """Информационное уведомление"""
        self.notify(Notification(NotificationType.INFO, title, message, details))

    def warning(self, title: str, message: str, details: Optional[str] = None):
        """Предупреждение"""
        self.notify(Notification(NotificationType.WARNING, title, message, details))

    def error(self, title: str, message: str, details: Optional[str] = None):
        """Уведомление об ошибке"""
        self.notify(Notification(NotificationType.ERROR, title, message, details))

    def success(self, title: str, message: str, details: Optional[str] = None):
        """Уведомление об успехе"""
        self.notify(Notification(NotificationType.SUCCESS, title, message, details))


# Глобальный экземпляр
notification_manager = NotificationManager()
