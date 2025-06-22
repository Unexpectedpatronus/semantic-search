#!/usr/bin/env python
"""Скрипт для очистки временных файлов и логов"""

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from semantic_search.config import CACHE_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, TEMP_DIR

# Добавляем путь к src в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def format_file_size(size_bytes):
    """Форматирование размера файла"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_directory_size(directory: Path) -> int:
    """Получить размер директории"""
    total_size = 0
    if directory.exists():
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    return total_size


def clean_temp_files():
    """Очистка временных файлов"""
    print("\n🗑️  Очистка временных файлов...")

    if not TEMP_DIR.exists():
        print("   ℹ️  Директория temp не найдена")
        return 0

    temp_files = list(TEMP_DIR.rglob("*"))
    temp_size = get_directory_size(TEMP_DIR)

    if not temp_files:
        print("   ✅ Временные файлы отсутствуют")
        return 0

    print(f"   📁 Найдено файлов: {len([f for f in temp_files if f.is_file()])}")
    print(f"   💾 Размер: {format_file_size(temp_size)}")

    try:
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(exist_ok=True)
        print(f"   ✅ Очищено {format_file_size(temp_size)}")
        return temp_size
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return 0


def clean_old_logs(days: int = 7):
    """Очистка старых логов"""
    print(f"\n📜 Очистка логов старше {days} дней...")

    if not LOGS_DIR.exists():
        print("   ℹ️  Директория логов не найдена")
        return 0

    cutoff_date = datetime.now() - timedelta(days=days)
    old_logs = []
    total_size = 0

    for log_file in LOGS_DIR.glob("*.log*"):
        if log_file.is_file():
            modified_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if modified_time < cutoff_date:
                old_logs.append(log_file)
                total_size += log_file.stat().st_size

    if not old_logs:
        print("   ✅ Старые логи отсутствуют")
        return 0

    print(f"   📁 Найдено старых логов: {len(old_logs)}")
    print(f"   💾 Размер: {format_file_size(total_size)}")

    removed_size = 0
    for log_file in old_logs:
        try:
            size = log_file.stat().st_size
            log_file.unlink()
            removed_size += size
        except Exception as e:
            print(f"   ⚠️  Не удалось удалить {log_file.name}: {e}")

    print(f"   ✅ Очищено {format_file_size(removed_size)}")
    return removed_size


def clean_cache():
    """Очистка кэша"""
    print("\n💾 Очистка кэша...")

    if not CACHE_DIR.exists():
        print("   ℹ️  Директория кэша не найдена")
        return 0

    from semantic_search.utils.cache_manager import CacheManager

    cache_size = get_directory_size(CACHE_DIR)
    cache_files = list(CACHE_DIR.glob("*.pkl"))

    if not cache_files:
        print("   ✅ Кэш пуст")
        return 0

    print(f"   📁 Найдено файлов: {len(cache_files)}")
    print(f"   💾 Размер: {format_file_size(cache_size)}")

    cache_manager = CacheManager(CACHE_DIR)
    if cache_manager.clear():
        print(f"   ✅ Очищено {format_file_size(cache_size)}")
        return cache_size
    else:
        print("   ❌ Ошибка при очистке кэша")
        return 0


def clean_evaluation_results():
    """Очистка результатов оценки"""
    print("\n📊 Очистка результатов оценки...")

    eval_dir = DATA_DIR / "evaluation_results"
    if not eval_dir.exists():
        print("   ℹ️  Директория результатов не найдена")
        return 0

    eval_size = get_directory_size(eval_dir)
    eval_files = list(eval_dir.rglob("*"))

    if not eval_files:
        print("   ✅ Результаты отсутствуют")
        return 0

    print(f"   📁 Найдено файлов: {len([f for f in eval_files if f.is_file()])}")
    print(f"   💾 Размер: {format_file_size(eval_size)}")

    try:
        shutil.rmtree(eval_dir)
        eval_dir.mkdir(exist_ok=True)
        print(f"   ✅ Очищено {format_file_size(eval_size)}")
        return eval_size
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return 0


def show_disk_usage():
    """Показать использование диска"""
    print("\n📊 Использование дискового пространства:")
    print("=" * 50)

    directories = [
        (DATA_DIR, "Данные"),
        (MODELS_DIR, "Модели"),
        (CACHE_DIR, "Кэш"),
        (TEMP_DIR, "Временные файлы"),
        (LOGS_DIR, "Логи"),
        (DATA_DIR / "evaluation_results", "Результаты оценки"),
    ]

    total_size = 0
    for directory, name in directories:
        if directory.exists():
            size = get_directory_size(directory)
            total_size += size
            print(f"{name:.<30} {format_file_size(size):>15}")

    print("-" * 50)
    print(f"{'ИТОГО':.<30} {format_file_size(total_size):>15}")


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Очистка временных файлов и кэша")
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Очистить все (временные файлы, кэш, старые логи)",
    )
    parser.add_argument(
        "--temp", "-t", action="store_true", help="Очистить только временные файлы"
    )
    parser.add_argument(
        "--cache", "-c", action="store_true", help="Очистить только кэш"
    )
    parser.add_argument(
        "--logs", "-l", action="store_true", help="Очистить старые логи"
    )
    parser.add_argument(
        "--eval", "-e", action="store_true", help="Очистить результаты оценки"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Удалять логи старше указанного количества дней (по умолчанию: 7)",
    )
    parser.add_argument(
        "--usage", "-u", action="store_true", help="Показать использование диска"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Не запрашивать подтверждение"
    )

    args = parser.parse_args()

    # Если не указаны флаги, показываем использование диска
    if not any([args.all, args.temp, args.cache, args.logs, args.eval, args.usage]):
        args.usage = True

    print("🧹 Утилита очистки Semantic Search")
    print("=" * 50)

    if args.usage:
        show_disk_usage()
        return

    # Определяем что будем очищать
    actions = []
    if args.all or args.temp:
        actions.append(("временные файлы", clean_temp_files))
    if args.all or args.cache:
        actions.append(("кэш", clean_cache))
    if args.all or args.logs:
        actions.append(
            (f"логи старше {args.days} дней", lambda: clean_old_logs(args.days))
        )
    if args.eval:
        actions.append(("результаты оценки", clean_evaluation_results))

    # Показываем что будет очищено
    print("\n🎯 Будут очищены:")
    for name, _ in actions:
        print(f"   • {name}")

    # Запрашиваем подтверждение
    if not args.yes:
        response = input("\n❓ Продолжить? (y/n): ")
        if response.lower() != "y":
            print("❌ Очистка отменена.")
            return

    # Выполняем очистку
    total_cleaned = 0
    for name, action in actions:
        cleaned = action()
        total_cleaned += cleaned

    print(f"\n✅ Всего очищено: {format_file_size(total_cleaned)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Операция прервана.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
