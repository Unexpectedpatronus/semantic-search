"""Асинхронный менеджер задач с прогрессом и статусами"""

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional

from loguru import logger

from semantic_search.config import CACHE_DIR
from semantic_search.utils.cache_manager import CacheManager
from semantic_search.utils.notification_system import ProgressTracker


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
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

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class TaskManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Task] = {}
        self.lock = Lock()

    def submit(
        self,
        func: Callable[..., Any],
        *args,
        name: str = "",
        description: str = "",
        **kwargs,
    ) -> Task:
        task = Task(name=name, description=description, start_time=time.time())

        def wrapper():
            task.status = TaskStatus.RUNNING
            try:
                result = func(*args, **kwargs)
                task.status = TaskStatus.COMPLETED
                task.result = result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.exception(f"Задача {task.id} завершилась с ошибкой")
            finally:
                task.end_time = time.time()

        future = self.executor.submit(wrapper)
        task.future = future

        with self.lock:
            self.tasks[task.id] = task

        logger.debug(f"Добавлена задача {task.id} [{task.name}]")
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self.lock:
            return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.future and not task.future.done():
                cancelled = task.future.cancel()
                if cancelled:
                    task.status = TaskStatus.CANCELLED
                return cancelled
        return False

    def get_status(self) -> str:
        with self.lock:
            if not self.tasks:
                return "Нет активных задач."
            return "\n".join(
                f"[{task.id}] {task.name} — {task.status.value}, прогресс: {task.progress * 100:.1f}%"
                for task in self.tasks.values()
            )

    def cancel_all(self) -> None:
        with self.lock:
            for task_id in list(self.tasks.keys()):
                self.cancel(task_id)

    def cleanup(self) -> None:
        cache = CacheManager(CACHE_DIR)
        cache.clear()
