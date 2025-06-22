#!/usr/bin/env python
"""Скрипт для очистки кэша приложения"""

import sys
from pathlib import Path

from semantic_search.config import CACHE_DIR
from semantic_search.utils.cache_manager import CacheManager

# Добавляем путь к src в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def clear_cache():
    """Очистка кэша приложения"""
    print("🧹 Очистка кэша Semantic Search")
    print(f"📁 Директория кэша: {CACHE_DIR}")

    # Проверяем существование директории
    if not CACHE_DIR.exists():
        print("ℹ️  Директория кэша не существует. Нечего очищать.")
        return

    # Подсчитываем размер кэша
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    if not cache_files:
        print("ℹ️  Кэш пуст. Нечего очищать.")
        return

    print(f"📊 Найдено файлов: {len(cache_files)}")
    print(f"💾 Общий размер: {total_size / 1024 / 1024:.2f} МБ")

    # Запрашиваем подтверждение
    response = input("\n❓ Вы уверены, что хотите очистить кэш? (y/n): ")

    if response.lower() != "y":
        print("❌ Очистка отменена.")
        return

    # Очищаем кэш
    cache_manager = CacheManager(CACHE_DIR)
    if cache_manager.clear():
        print("✅ Кэш успешно очищен.")
        print(f"🗑️  Удалено {len(cache_files)} файлов")
        print(f"💨 Освобождено {total_size / 1024 / 1024:.2f} МБ")
    else:
        print("❌ Произошла ошибка при очистке кэша.")


def main():
    """Главная функция"""
    try:
        clear_cache()
    except KeyboardInterrupt:
        print("\n\n⚠️  Операция прервана пользователем.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
