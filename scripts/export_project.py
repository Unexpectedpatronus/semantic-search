#!/usr/bin/env python
"""Скрипт для экспорта всего проекта в один текстовый файл для проверки"""

import os
from datetime import datetime
from pathlib import Path


def should_include_file(file_path: Path, output_file: str) -> bool:
    """Определяет, нужно ли включать файл"""

    # ВАЖНО: Исключаем сам файл экспорта!
    if file_path.name == output_file or file_path.name == "project_export.txt":
        return False

    # Исключаем временные файлы экспорта
    if file_path.name.startswith("project_export") and file_path.suffix == ".txt":
        return False

    # Исключаем директории
    exclude_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".idea",
        ".vscode",
        "logs",
        "data/models",
        "data/cache",
        "data/temp",
        ".ruff_cache",
        "htmlcov",
        ".coverage",
    }

    # Исключаем расширения
    exclude_extensions = {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".pkl",
        ".npy",
        ".model",
        ".log",
        ".tmp",
        ".cache",
        ".lock",
        ".db",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
    }

    # Проверяем, не находится ли файл в исключенной директории
    for parent in file_path.parents:
        if parent.name in exclude_dirs:
            return False

    # Проверяем расширение
    if file_path.suffix in exclude_extensions:
        return False

    # Исключаем большие файлы (больше 1MB)
    try:
        if file_path.is_file() and file_path.stat().st_size > 1_000_000:
            return False
    except:
        return False

    return True


def collect_project_files(root_path: Path, output_file: str) -> list[Path]:
    """Собирает все файлы проекта, которые нужно включить"""
    included_files = []
    seen_paths = set()  # Для отслеживания уже добавленных файлов

    # Определяем приоритет файлов
    priority_order = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        ".gitignore",
        "src/semantic_search/__init__.py",
        "src/semantic_search/config.py",
        "src/semantic_search/main.py",
        "src/semantic_search/core",
        "src/semantic_search/gui",
        "src/semantic_search/utils",
        "src/semantic_search/evaluation",
        "scripts",
        "tests",
    ]

    # Рекурсивно обходим директории
    for path in root_path.rglob("*"):
        # Преобразуем в абсолютный путь для сравнения
        abs_path = path.absolute()

        # Проверяем, не добавляли ли уже этот файл
        if abs_path in seen_paths:
            continue

        if path.is_file() and should_include_file(path, output_file):
            included_files.append(path)
            seen_paths.add(abs_path)

    # Сортируем файлы по приоритету
    def get_sort_key(file_path: Path):
        rel_path = str(file_path.relative_to(root_path)).replace("\\", "/")

        # Ищем приоритет
        for i, priority in enumerate(priority_order):
            if rel_path == priority or rel_path.startswith(priority + "/"):
                return (i, rel_path)

        # Остальные файлы в конец
        return (len(priority_order), rel_path)

    included_files.sort(key=get_sort_key)

    # Удаляем дубликаты (на всякий случай)
    unique_files = []
    seen_contents = set()

    for file_path in included_files:
        file_key = (file_path.name, file_path.stat().st_size)
        if file_key not in seen_contents:
            unique_files.append(file_path)
            seen_contents.add(file_key)

    return unique_files


def export_project(output_file: str = "project_export.txt", include_data: bool = False):
    """Экспортирует проект в текстовый файл"""
    project_root = Path(__file__).parent.parent

    # Удаляем старый файл экспорта если существует
    output_path = Path(output_file)
    if output_path.exists():
        print(f"🗑️  Удаляем старый файл экспорта: {output_file}")
        output_path.unlink()

    # Собираем файлы
    print("📂 Сканирование проекта...")
    all_files = collect_project_files(project_root, output_file)

    # Фильтруем data файлы если не нужны
    if not include_data:
        all_files = [
            f for f in all_files if "data" not in f.parts or f.name == ".gitkeep"
        ]

    print(f"📄 Найдено файлов для экспорта: {len(all_files)}")

    # Проверяем на дубликаты
    file_names = [f.name for f in all_files]
    duplicates = [name for name in file_names if file_names.count(name) > 1]
    if duplicates:
        print(f"⚠️  Обнаружены файлы с одинаковыми именами: {set(duplicates)}")

    # Записываем в файл
    with open(output_file, "w", encoding="utf-8") as f:
        # Заголовок
        f.write("=" * 80 + "\n")
        f.write("SEMANTIC SEARCH PROJECT EXPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root: {project_root}\n")
        f.write("=" * 80 + "\n\n")

        # Структура проекта (простой список)
        f.write("PROJECT STRUCTURE:\n")
        f.write("-" * 40 + "\n")

        current_dir = None
        for file_path in all_files:
            rel_path = file_path.relative_to(project_root)
            parent_dir = rel_path.parent

            # Показываем директорию при смене
            if parent_dir != current_dir:
                current_dir = parent_dir
                if str(parent_dir) != ".":
                    f.write(f"\n{parent_dir}/\n")
                else:
                    f.write("\n[root]\n")

            f.write(f"  - {file_path.name}\n")

        # Содержимое файлов
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILE CONTENTS:\n")
        f.write("=" * 80 + "\n")

        # Записываем содержимое каждого файла
        written_files = set()  # Отслеживаем записанные файлы

        for file_path in all_files:
            # Создаем уникальный ключ для файла
            file_key = str(file_path.absolute())

            # Проверяем, не записали ли уже
            if file_key in written_files:
                print(f"⚠️  Пропускаем дубликат: {file_path.relative_to(project_root)}")
                continue

            written_files.add(file_key)
            rel_path = file_path.relative_to(project_root)

            f.write(f"\n\n{'=' * 40}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"{'=' * 40}\n")

            try:
                # Текстовые файлы
                if file_path.suffix in [
                    ".py",
                    ".toml",
                    ".md",
                    ".txt",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".bat",
                    ".sh",
                    ".cfg",
                    ".ini",
                ]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")
                else:
                    # Бинарные файлы
                    f.write(f"[Binary file - {file_path.stat().st_size} bytes]\n")

            except Exception as e:
                f.write(f"[Error reading file: {e}]\n")

        # Итоговая статистика
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPORT SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total files included: {len(written_files)}\n")
        f.write(f"Export file: {output_file}\n")
        f.write(f"Export size: {os.path.getsize(output_file):,} bytes\n")
        f.write("=" * 80 + "\n")

    # Результат
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"✅ Проект экспортирован в: {output_file}")
    print(f"📊 Размер файла: {file_size_mb:.2f} MB")
    print(f"📁 Включено файлов: {len(written_files)}")

    if file_size_mb > 10:
        print("\n⚠️  Файл больше 10 MB, может быть слишком большим для некоторых чатов")
        print("💡 Попробуйте исключить некоторые файлы или использовать архив")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Экспорт проекта в текстовый файл для проверки"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="project_export.txt",
        help="Имя выходного файла (по умолчанию: project_export.txt)",
    )
    parser.add_argument(
        "--include-data", action="store_true", help="Включить файлы из директории data/"
    )

    args = parser.parse_args()

    export_project(args.output, args.include_data)
