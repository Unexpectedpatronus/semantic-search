"""Валидаторы для проверки различных типов данных"""

from pathlib import Path
from typing import Any, Union


class ValidationError(Exception):
    """Ошибка валидации"""

    pass


class DataValidator:
    """Класс с методами проверки различных типов входных данных"""

    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
        path = Path(path) if isinstance(path, str) else path

        if not isinstance(path, Path):
            raise ValidationError(f"Ожидался Path или str, получен: {type(path)}")

        if must_exist and not path.exists():
            raise ValidationError(f"Файл не найден: {path}")

        if must_exist and not path.is_file():
            raise ValidationError(f"Путь не является файлом: {path}")

        return path

    @staticmethod
    def validate_directory(path: Union[str, Path], must_exist: bool = True) -> Path:
        path = Path(path) if isinstance(path, str) else path

        if not isinstance(path, Path):
            raise ValidationError(f"Ожидался Path или str, получен: {type(path)}")

        if must_exist and not path.exists():
            raise ValidationError(f"Каталог не найден: {path}")

        if must_exist and not path.is_dir():
            raise ValidationError(f"Путь не является директорией: {path}")

        return path

    @staticmethod
    def validate_positive_int(value: Any, name: str = "значение") -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(
                f"{name} должно быть положительным целым числом, получено: {value}"
            )
        return value

    @staticmethod
    def validate_non_empty_string(value: Any, name: str = "строка") -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} должна быть непустой строкой")
        return value
