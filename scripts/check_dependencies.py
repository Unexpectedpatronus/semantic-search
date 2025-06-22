#!/usr/bin/env python
"""Скрипт для проверки установленных зависимостей"""

import importlib
import subprocess
import sys
from pathlib import Path


def check_module(module_name: str, import_name: str = None) -> tuple[bool, str]:
    """Проверка доступности модуля"""
    import_name = import_name or module_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_spacy_models():
    """Проверка установленных моделей SpaCy"""
    models = {
        "ru_core_news_sm": "Русская модель (маленькая)",
        "en_core_web_sm": "Английская модель (маленькая)",
    }

    results = {}
    try:
        import spacy

        for model_name, description in models.items():
            try:
                nlp = spacy.load(model_name)
                results[model_name] = (True, description)
            except OSError:
                results[model_name] = (False, description)
    except ImportError:
        for model_name, description in models.items():
            results[model_name] = (False, description)

    return results


def check_system_dependencies():
    """Проверка системных зависимостей"""
    results = {}

    # Проверка Python версии
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    results["Python"] = (
        sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12,
        python_version,
    )

    # Проверка Poetry
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            results["Poetry"] = (True, version)
        else:
            results["Poetry"] = (False, None)
    except FileNotFoundError:
        results["Poetry"] = (False, None)

    # Проверка Microsoft Word (только на Windows)
    if sys.platform == "win32":
        try:
            import win32com.client

            word = win32com.client.Dispatch("Word.Application")
            results["Microsoft Word"] = (True, "installed")
            word.Quit()
        except:
            results["Microsoft Word"] = (False, None)

    return results


def main():
    """Главная функция"""
    print("🔍 Проверка зависимостей Semantic Search")
    print("=" * 60)

    # Основные зависимости
    core_dependencies = [
        ("pymupdf", "fitz", "PDF обработка"),
        ("python-docx", "docx", "DOCX обработка"),
        ("spacy", None, "NLP обработка"),
        ("gensim", None, "Doc2Vec модель"),
        ("PyQt6", "PyQt6.QtCore", "GUI интерфейс"),
        ("loguru", None, "Логирование"),
        ("click", None, "CLI интерфейс"),
        ("scikit-learn", "sklearn", "ML утилиты"),
        ("psutil", None, "Системная информация"),
        ("numpy", None, "Численные вычисления"),
        ("pandas", None, "Обработка данных"),
        ("matplotlib", None, "Графики"),
        ("tqdm", None, "Прогресс-бары"),
    ]

    # Опциональные зависимости
    optional_dependencies = [
        ("openai", None, "OpenAI интеграция"),
        ("seaborn", None, "Улучшенные графики"),
    ]

    # Windows-специфичные
    if sys.platform == "win32":
        core_dependencies.append(("pywin32", "win32com", "DOC обработка"))

    print("\n📦 Основные зависимости:")
    print("-" * 60)

    all_ok = True
    for package_name, import_name, description in core_dependencies:
        installed, version = check_module(package_name, import_name)
        if installed:
            print(f"✅ {package_name:<20} {version:<15} {description}")
        else:
            print(f"❌ {package_name:<20} {'НЕ УСТАНОВЛЕН':<15} {description}")
            all_ok = False

    print("\n📦 Опциональные зависимости:")
    print("-" * 60)

    for package_name, import_name, description in optional_dependencies:
        installed, version = check_module(package_name, import_name)
        if installed:
            print(f"✅ {package_name:<20} {version:<15} {description}")
        else:
            print(f"⚠️  {package_name:<20} {'не установлен':<15} {description}")

    # Проверка SpaCy моделей
    print("\n🧠 SpaCy модели:")
    print("-" * 60)

    spacy_models = check_spacy_models()
    spacy_ok = True
    for model_name, (installed, description) in spacy_models.items():
        if installed:
            print(f"✅ {model_name:<20} {description}")
        else:
            print(f"❌ {model_name:<20} {description} - НЕ УСТАНОВЛЕНА")
            spacy_ok = False

    # Системные зависимости
    print("\n💻 Системные зависимости:")
    print("-" * 60)

    system_deps = check_system_dependencies()
    for dep_name, (installed, version) in system_deps.items():
        if installed:
            print(f"✅ {dep_name:<20} {version}")
        else:
            if dep_name == "Microsoft Word":
                print(f"⚠️  {dep_name:<20} не установлен (требуется для .doc файлов)")
            else:
                print(f"❌ {dep_name:<20} НЕ УСТАНОВЛЕН")
                all_ok = False

    # Итоговый статус
    print("\n" + "=" * 60)

    if all_ok and spacy_ok:
        print("✅ Все зависимости установлены корректно!")
        print("\n🚀 Приложение готово к запуску:")
        print("   GUI: poetry run semantic-search")
        print("   CLI: poetry run semantic-search-cli --help")
    else:
        print("⚠️  Обнаружены проблемы с зависимостями!")

        print("\n📋 Рекомендации по устранению:")

        if not all_ok:
            print("\n1. Установите недостающие зависимости:")
            print("   poetry install")

        if not spacy_ok:
            print("\n2. Установите языковые модели SpaCy:")
            print("   poetry run python scripts/setup_spacy.py")
            print("   или вручную:")
            print("   poetry run python -m spacy download ru_core_news_sm")
            print("   poetry run python -m spacy download en_core_web_sm")

        if not system_deps.get("Poetry", (False, None))[0]:
            print("\n3. Установите Poetry:")
            print("   Инструкции: https://python-poetry.org/docs/#installation")

    # Проверка путей
    print("\n📁 Проверка структуры проекта:")
    print("-" * 60)

    project_root = Path(__file__).parent.parent
    important_paths = [
        (project_root / "src" / "semantic_search", "Исходный код"),
        (project_root / "data", "Директория данных"),
        (project_root / "pyproject.toml", "Конфигурация Poetry"),
    ]

    for path, description in important_paths:
        if path.exists():
            print(f"✅ {str(path.relative_to(project_root)):<30} {description}")
        else:
            print(
                f"❌ {str(path.relative_to(project_root)):<30} {description} - НЕ НАЙДЕН"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка при проверке: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
