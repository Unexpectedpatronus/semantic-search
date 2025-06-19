import fnmatch
import os
from pathlib import Path


def print_tree(start_path, prefix="", ignore=None, leaf_dirs=None):
    if ignore is None:
        ignore = set()
    if leaf_dirs is None:
        leaf_dirs = {"cache", "models", "logs"}

    # Собираем содержимое директории с учетом игнорирования
    contents = []
    for item in os.listdir(start_path):
        skip = False
        # Проверяем шаблоны игнорирования
        for pattern in ignore:
            if fnmatch.fnmatch(item, pattern):
                skip = True
                break
        if skip:
            continue
        contents.append(item)

    # Сортируем для последовательного отображения
    contents.sort()

    for i, item in enumerate(contents):
        path = Path(start_path) / item
        is_last = i == len(contents) - 1

        # Добавляем '/' к директориям
        display_name = f"{item}/" if os.path.isdir(path) else item
        print(f"{prefix}{'└── ' if is_last else '├── '}{display_name}")

        # Рекурсивный обход только для НЕлистовых директорий
        if os.path.isdir(path) and item not in leaf_dirs:
            ext = "    " if is_last else "│   "
            print_tree(path, prefix + ext, ignore, leaf_dirs)


if __name__ == "__main__":
    print(".")
    ignore_set = {
        "__pycache__",
        ".git",
        ".venv",
        "*.pyc",
        ".pytest_cache",
        "tree.py",
        ".benchmarks",
        ".coverage",
        "*__pycache__*",
        "poetry.lock",
        "__init__.py",
        ".gitignore",
    }
    print_tree(".", ignore=ignore_set)
