"""Менеджер кэширования"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class CacheManager:
    """Менеджер кэширования для результатов поиска и обработки"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _get_cache_key(self, data: str) -> str:
        """Генерация ключа кэша"""
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        cache_file = self.cache_dir / f"{self._get_cache_key(key)}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Ошибка чтения кэша: {e}")

        return None

    def set(self, key: str, value: Any) -> bool:
        """Сохранение данных в кэш"""
        cache_file = self.cache_dir / f"{self._get_cache_key(key)}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            logger.error(f"Ошибка записи в кэш: {e}")
            return False

    def clear(self) -> bool:
        """Очистка кэша"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Кэш очищен")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
            return False
