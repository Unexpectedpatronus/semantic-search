"""Валидаторы для проверки данных"""

from pathlib import Path
from typing import Any, Dict, Optional, Union


class ValidationError(Exception):
    """Ошибка валидации"""

    pass


class DataValidator:
    """Валидатор для различных типов данных"""

    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
        """
        Валидация пути к файлу

        Args:
            path: Путь к файлу
            must_exist: Файл должен существовать

        Returns:
            Валидный Path объект

        Raises:
            ValidationError: При невалидном пуге
        """
        if isinstance(path, str):
            path = Path(path)

        if not isinstance(path, Path):
            raise ValidationError(
                f"Путь должен быть строкой или Path объектом: {type(path)}"
            )

        if must_exist and not path.exists():
            raise ValidationError(f"Файл не найден: {path}")

        if must_exist and not path.is_file():
            raise ValidationError(f"Путь не является файлом: {path}")

        return path

    @staticmethod
    def validate_directory_path(
        path: Union[str, Path], must_exist: bool = True
    ) -> Path:
        """Валидация пути к директории"""
        if isinstance(path, str):
            path = Path(path)

        if not isinstance(path, Path):
            raise ValidationError(
                f"Путь должен быть строкой или Path объектом: {type(path)}"
            )

        if must_exist and not path.exists():
            raise ValidationError(f"Директория не найдена: {path}")

        if must_exist and not path.is_dir():
            raise ValidationError(f"Путь не является директорией: {path}")

        return path

    @staticmethod
    def validate_text(
        text: str, min_length: int = 1, max_length: Optional[int] = None
    ) -> str:
        """Валидация текста"""
        if not isinstance(text, str):
            raise ValidationError(f"Текст должен быть строкой: {type(text)}")

        text = text.strip()

        if len(text) < min_length:
            raise ValidationError(
                f"Текст слишком короткий. Минимум: {min_length}, получено: {len(text)}"
            )

        if max_length and len(text) > max_length:
            raise ValidationError(
                f"Текст слишком длинный. Максимум: {max_length}, получено: {len(text)}"
            )

        return text

    @staticmethod
    def validate_search_params(
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Валидация параметров поиска"""

        # Валидация запроса
        query = DataValidator.validate_text(query, min_length=2, max_length=1000)

        # Валидация количества результатов
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
                raise ValidationError(
                    f"top_k должно быть целым числом от 1 до 1000: {top_k}"
                )

        # Валидация порога схожести
        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)) or not (
                0 <= similarity_threshold <= 1
            ):
                raise ValidationError(
                    f"similarity_threshold должен быть числом от 0 до 1: {similarity_threshold}"
                )

        return {
            "query": query,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

    @staticmethod
    def validate_model_params(**params) -> Dict[str, Any]:
        """Валидация параметров модели Doc2Vec"""
        validated = {}

        # Размерность векторов
        if "vector_size" in params:
            vs = params["vector_size"]
            if not isinstance(vs, int) or not (50 <= vs <= 1000):
                raise ValidationError(f"vector_size должен быть от 50 до 1000: {vs}")
            validated["vector_size"] = vs

        # Размер окна
        if "window" in params:
            w = params["window"]
            if not isinstance(w, int) or not (1 <= w <= 50):
                raise ValidationError(f"window должен быть от 1 до 50: {w}")
            validated["window"] = w

        # Минимальная частота
        if "min_count" in params:
            mc = params["min_count"]
            if not isinstance(mc, int) or not (1 <= mc <= 100):
                raise ValidationError(f"min_count должен быть от 1 до 100: {mc}")
            validated["min_count"] = mc

        # Количество эпох
        if "epochs" in params:
            e = params["epochs"]
            if not isinstance(e, int) or not (1 <= e <= 1000):
                raise ValidationError(f"epochs должен быть от 1 до 1000: {e}")
            validated["epochs"] = e

        # Количество потоков
        if "workers" in params:
            w = params["workers"]
            if not isinstance(w, int) or not (1 <= w <= 32):
                raise ValidationError(f"workers должен быть от 1 до 32: {w}")
            validated["workers"] = w

        return validated


class FileValidator:
    """Валидатор для файлов"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    @classmethod
    def validate_document_file(cls, file_path: Path) -> Dict[str, Any]:
        """
        Комплексная валидация файла документа

        Returns:
            Словарь с результатами валидации
        """
        result = {"valid": True, "errors": [], "warnings": [], "file_info": {}}

        try:
            # Проверка существования
            if not file_path.exists():
                result["errors"].append(f"Файл не найден: {file_path}")
                result["valid"] = False
                return result

            # Проверка, что это файл
            if not file_path.is_file():
                result["errors"].append(f"Путь не является файлом: {file_path}")
                result["valid"] = False
                return result

            # Информация о файле
            stat = file_path.stat()
            result["file_info"] = {
                "size": stat.st_size,
                "size_mb": stat.st_size / 1024 / 1024,
                "extension": file_path.suffix.lower(),
                "name": file_path.name,
            }

            # Проверка расширения
            if file_path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
                result["errors"].append(
                    f"Неподдерживаемое расширение: {file_path.suffix}"
                )
                result["valid"] = False

            # Проверка размера
            if stat.st_size > cls.MAX_FILE_SIZE:
                result["errors"].append(
                    f"Файл слишком большой: {stat.st_size / 1024 / 1024:.1f}MB"
                )
                result["valid"] = False
            elif stat.st_size > cls.MAX_FILE_SIZE * 0.8:
                result["warnings"].append(
                    f"Большой файл: {stat.st_size / 1024 / 1024:.1f}MB"
                )

            # Проверка пустого файла
            if stat.st_size == 0:
                result["errors"].append("Файл пустой")
                result["valid"] = False
            elif stat.st_size < 100:  # Меньше 100 байт
                result["warnings"].append("Очень маленький файл")

            # Проверка доступности для чтения
            try:
                with open(file_path, "rb") as f:
                    f.read(1)
            except PermissionError:
                result["errors"].append("Нет прав на чтение файла")
                result["valid"] = False
            except Exception as e:
                result["errors"].append(f"Ошибка при чтении файла: {e}")
                result["valid"] = False

        except Exception as e:
            result["errors"].append(f"Неожиданная ошибка валидации: {e}")
            result["valid"] = False

        return result
