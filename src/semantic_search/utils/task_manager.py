"""Менеджер долгосрочных задач"""

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .notification_system import ProgressTracker, notification_manager


class TaskStatus(Enum):
    """Статусы задач"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Структура задачи"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    future: Optional[Future] = None
    progress_tracker: Optional[ProgressTracker] = None

    @property
    def duration(self) -> Optional[float]:
        """Длительность выполнения задачи"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None


class TaskManager:
    """Менеджер задач для выполнения длительных операций"""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Task] = {}
        self.lock = Lock()

        logger.info(f"TaskManager инициализирован с {max_workers} потоками")

    def submit_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        name: str = "",
        description: str = "",
        track_progress: bool = False,
        total_steps: int = 0,
    ) -> str:
        """
        Создание и запуск задачи

        Args:
            func: Функция для выполнения
            args: Аргументы функции
            kwargs: Именованные аргументы
            name: Название задачи
            description: Описание задачи
            track_progress: Включить отслеживание прогресса
            total_steps: Общее количество шагов (для прогресса)

        Returns:
            ID задачи
        """
        kwargs = kwargs or {}

        task = Task(name=name or func.__name__, description=description)

        if track_progress and total_steps > 0:
            task.progress_tracker = ProgressTracker(total_steps, name)
            # Добавляем tracker в kwargs если функция его ожидает
            if "progress_tracker" in func.__code__.co_varnames:
                kwargs["progress_tracker"] = task.progress_tracker

        # Оборачиваем функцию для отслеживания статуса
        def wrapped_func():
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            try:
                notification_manager.info(
                    "Задача запущена", f"Начато выполнение: {task.name}"
                )

                result = func(*args, **kwargs)

                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 1.0

                notification_manager.success(
                    "Задача завершена",
                    f"Успешно завершена: {task.name}",
                    f"Время выполнения: {task.duration:.1f}с",
                )

                return result

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)

                notification_manager.error(
                    "Ошибка задачи", f"Ошибка в задаче: {task.name}", str(e)
                )

                logger.error(f"Ошибка в задаче {task.id}: {e}")
                raise

            finally:
                task.end_time = time.time()

        # Запускаем задачу
        future = self.executor.submit(wrapped_func)
        task.future = future

        with self.lock:
            self.tasks[task.id] = task

        logger.info(f"Задача {task.id} ({name}) поставлена в очередь")
        return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Получение информации о задаче"""
        with self.lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Получение всех задач"""
        with self.lock:
            return list(self.tasks.values())

    def get_running_tasks(self) -> List[Task]:
        """Получение выполняющихся задач"""
        return [
            task for task in self.get_all_tasks() if task.status == TaskStatus.RUNNING
        ]

    def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        task = self.get_task(task_id)
        if not task or not task.future:
            return False

        if task.future.cancel():
            task.status = TaskStatus.CANCELLED
            logger.info(f"Задача {task_id} отменена")
            return True

        return False

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Ожидание завершения задачи"""
        task = self.get_task(task_id)
        if not task or not task.future:
            raise ValueError(f"Задача {task_id} не найдена")

        return task.future.result(timeout=timeout)

    def cleanup_finished_tasks(self, max_keep: int = 100):
        """Очистка завершенных задач"""
        with self.lock:
            finished_tasks = [
                (task_id, task)
                for task_id, task in self.tasks.items()
                if task.status
                in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]

            if len(finished_tasks) > max_keep:
                # Сортируем по времени завершения и удаляем старые
                finished_tasks.sort(key=lambda x: x[1].end_time or 0)
                to_remove = finished_tasks[:-max_keep]

                for task_id, _ in to_remove:
                    del self.tasks[task_id]

                logger.info(f"Удалено {len(to_remove)} завершенных задач")

    def shutdown(self, wait: bool = True):
        """Завершение работы менеджера задач"""
        logger.info("Завершение работы TaskManager...")
        self.executor.shutdown(wait=wait)


# Глобальный экземпляр
task_manager = TaskManager()
