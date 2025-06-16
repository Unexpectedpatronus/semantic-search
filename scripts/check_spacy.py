"""Проверка установки SpaCy модели"""

import sys
from pathlib import Path

from semantic_search.config import SPACY_MODEL


def check_spacy_model(model_name=SPACY_MODEL):
    """Комплексная проверка SpaCy модели"""
    print(f"🔍 Проверка модели {model_name}")
    print("=" * 50)

    # 1. Проверка установки SpaCy
    try:
        import spacy

        print(f"✅ SpaCy установлен: версия {spacy.__version__}")
    except ImportError:
        print("❌ SpaCy не установлен")
        return False

    # 2. Проверка наличия и загрузки модели (в одном шаге)
    try:
        nlp = spacy.load(model_name)
        print(f"✅ Модель {model_name} найдена и загружается")
        print(f"📊 Словарь содержит: {len(nlp.vocab)} записей")
        print(f"📋 Компоненты: {nlp.component_names}")

        # Получаем путь к модели
        try:
            model_path = nlp.meta.get("lang", "unknown")
            print(f"🗂️ Язык модели: {model_path}")
            print(f"📖 Версия модели: {nlp.meta.get('version', 'unknown')}")
        except:  # noqa: E722
            pass

        # 3. Тест обработки текста
        test_text = "Привет мир! Это тестовое предложение для проверки."
        doc = nlp(test_text)

        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        print(f"🧪 Тест лемматизации: {tokens[:8]}...")  # Показываем первые 8

        sentences = [sent.text.strip() for sent in doc.sents]
        print(f"📝 Тест разбиения на предложения: {len(sentences)} предложений")

        return True

    except OSError:
        print(f"❌ Модель {model_name} не найдена")
        print("💡 Модель не установлена или повреждена")
        return False
    except Exception as e:
        print(f"❌ Ошибка при работе с моделью: {e}")
        return False


def show_available_models():
    """Показать доступные модели SpaCy"""
    try:
        import subprocess

        print("\n📦 Доступные модели SpaCy:")

        # Пытаемся получить список моделей через spacy validate
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "validate"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Не удалось получить список моделей")

    except Exception as e:
        print(f"Ошибка получения списка моделей: {e}")


def show_spacy_info():
    """Показать общую информацию о SpaCy"""
    try:
        import spacy

        print("\n📊 Информация о SpaCy:")
        print(f"   Версия: {spacy.__version__}")
        print(f"   Путь установки: {Path(spacy.__file__).parent}")

        # Попытка показать системную информацию
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "-m", "spacy", "info"], capture_output=True, text=True
            )

            if result.returncode == 0:
                print("\n🔧 Системная информация:")
                # Показываем только важные строки
                lines = result.stdout.split("\n")
                for line in lines:
                    if any(
                        key in line.lower() for key in ["version", "platform", "python"]
                    ):
                        print(f"   {line.strip()}")
        except:  # noqa: E722
            pass

    except Exception as e:
        print(f"Ошибка получения информации: {e}")


def get_model_download_command(model_name=SPACY_MODEL):
    """Получить команды для установки модели"""
    return [
        f"poetry run python -m spacy download {model_name}",
    ]


if __name__ == "__main__":
    model_installed = check_spacy_model()

    if not model_installed:
        print("\n💡 Для установки модели выполните:")
        for cmd in get_model_download_command():
            print(f"   {cmd}")

    show_spacy_info()
    show_available_models()
