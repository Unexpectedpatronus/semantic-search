#!/usr/bin/env python
"""Экспорт выбранных частей проекта: поддержка каталогов и отдельных файлов"""

from datetime import datetime
from pathlib import Path

# Можно указывать папки и конкретные файлы
INCLUDE_STRINGS = [
    "src\\semantic_search\\evaluation",
    "scripts\\diploma_demo.py",
    "src\\semantic_search\\core",
    "src\\semantic_search\\utils",
]
INCLUDE_PATHS = [
    Path(s) if "\\" in s else Path(s.replace("/", "\\")) for s in INCLUDE_STRINGS
]


def should_include_file(file_path: Path) -> bool:
    """Определяет, включать ли файл"""

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

    if file_path.suffix in exclude_extensions:
        return False

    try:
        if file_path.is_file() and file_path.stat().st_size > 1_000_000:
            return False
    except:
        return False

    return True


def collect_selected_items(root_path: Path) -> list[Path]:
    """Собирает файлы из указанных директорий и отдельных файлов"""
    included_files = []

    for rel_path in INCLUDE_PATHS:
        target = root_path / rel_path

        if target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and should_include_file(path):
                    included_files.append(path)

        elif target.is_file():
            if should_include_file(target):
                included_files.append(target)
            else:
                print(f"⚠️  Пропущен неподходящий файл: {target}")

        else:
            print(f"⚠️  Не найден путь: {target}")

    return included_files


def export_selected_parts(output_file: str = "partial_export.txt"):
    """Экспортирует выбранные элементы проекта"""
    project_root = Path(__file__).parent.parent
    output_path = Path(output_file)

    if output_path.exists():
        print(f"🗑️  Удаляем старый файл экспорта: {output_file}")
        output_path.unlink()

    print("📂 Сканирование выбранных элементов...")
    files = collect_selected_items(project_root)

    print(f"📄 Найдено файлов для экспорта: {len(files)}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SELECTIVE PROJECT EXPORT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root: {project_root}\n")
        f.write("=" * 80 + "\n\n")

        f.write("STRUCTURE:\n")
        f.write("-" * 40 + "\n")

        current_dir = None
        for file_path in files:
            rel_path = file_path.relative_to(project_root)
            parent_dir = rel_path.parent

            if parent_dir != current_dir:
                current_dir = parent_dir
                f.write(f"\n{parent_dir}/\n")

            f.write(f"  - {file_path.name}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FILE CONTENTS:\n")
        f.write("=" * 80 + "\n")

        for file_path in files:
            rel_path = file_path.relative_to(project_root)

            f.write(f"\n\n{'=' * 40}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"{'=' * 40}\n")

            try:
                if file_path.suffix in [
                    ".py",
                    ".txt",
                    ".md",
                    ".json",
                    ".toml",
                    ".yaml",
                    ".yml",
                    ".cfg",
                    ".ini",
                    ".bat",
                    ".sh",
                ]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")
                else:
                    f.write(f"[Binary file - {file_path.stat().st_size} bytes]\n")

            except Exception as e:
                f.write(f"[Error reading file: {e}]\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total files included: {len(files)}\n")
        f.write("=" * 80 + "\n")

    print(f"\n✅ Экспорт завершен: {output_file}")
    print(f"📁 Включено файлов: {len(files)}")


if __name__ == "__main__":
    export_selected_parts()
