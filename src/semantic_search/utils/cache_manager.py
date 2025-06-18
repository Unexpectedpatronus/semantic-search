"""Менеджер кэширования на основе pickle и md5"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class CacheManager:
    """Менеджер кэша для хранения промежуточных результатов"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Инициализирован CacheManager по пути {self.cache_dir}")

    def _get_cache_key(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{self._get_cache_key(key)}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._get_cache_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Ошибка при чтении кэша {path.name}: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        path = self._get_cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
            logger.debug(f"Сохранено в кэш: {path.name}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при записи в кэш {path.name}: {e}")
            return False

    def exists(self, key: str) -> bool:
        return self._get_cache_path(key).exists()

    def clear(self) -> None:
        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
                logger.debug(f"Удалён кэш-файл: {file.name}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {file.name}: {e}")
