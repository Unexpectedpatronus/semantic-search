#!/usr/bin/env python
"""Скрипт для просмотра доступных моделей"""

import json
import sys
from datetime import datetime
from pathlib import Path

from semantic_search.config import MODELS_DIR
from semantic_search.core.doc2vec_trainer import Doc2VecTrainer

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


def list_models():
    """Список всех доступных моделей"""
    print("📚 Доступные модели Doc2Vec")
    print("=" * 80)

    if not MODELS_DIR.exists():
        print("❌ Директория моделей не найдена.")
        return

    model_files = list(MODELS_DIR.glob("*.model"))

    if not model_files:
        print("ℹ️  Модели не найдены.")
        print(f"   Директория моделей: {MODELS_DIR}")
        print("\n💡 Создайте модель командой:")
        print("   poetry run semantic-search-cli train -d /path/to/documents")
        return

    print(f"📁 Директория моделей: {MODELS_DIR}")
    print(f"🔢 Найдено моделей: {len(model_files)}\n")

    # Сортируем по дате изменения (новые первые)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for i, model_file in enumerate(model_files, 1):
        model_name = model_file.stem
        file_size = model_file.stat().st_size
        modified_time = datetime.fromtimestamp(model_file.stat().st_mtime)

        print(f"{i}. 🧠 {model_name}")
        print(f"   📏 Размер: {format_file_size(file_size)}")
        print(f"   📅 Изменен: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Пытаемся загрузить метаданные
        metadata_file = MODELS_DIR / f"{model_name}_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if "corpus_size" in metadata:
                    print(f"   📄 Документов: {metadata['corpus_size']}")
                if "vector_size" in metadata:
                    print(f"   🎯 Размерность: {metadata['vector_size']}")
                if "epochs" in metadata:
                    print(f"   🔄 Эпох: {metadata['epochs']}")
                if "training_time_formatted" in metadata:
                    print(
                        f"   ⏱️  Время обучения: {metadata['training_time_formatted']}"
                    )
                if "documents_base_path" in metadata:
                    base_path = Path(metadata["documents_base_path"])
                    print(f"   📂 База документов: {base_path.name}")

            except Exception as e:
                print(f"   ⚠️  Не удалось загрузить метаданные: {e}")

        # Проверяем наличие corpus_info
        corpus_info_file = MODELS_DIR / f"{model_name}_corpus_info.pkl"
        if corpus_info_file.exists():
            corpus_size = format_file_size(corpus_info_file.stat().st_size)
            print(f"   💾 Corpus info: {corpus_size}")

        print()  # Пустая строка между моделями


def show_model_details(model_name: str):
    """Показать детальную информацию о модели"""
    print(f"\n📋 Детальная информация о модели: {model_name}")
    print("=" * 80)

    trainer = Doc2VecTrainer()
    model = trainer.load_model(model_name)

    if model is None:
        print(f"❌ Не удалось загрузить модель '{model_name}'")
        return

    info = trainer.get_model_info()

    print(f"✅ Статус: {info['status']}")
    print(f"📏 Размерность векторов: {info['vector_size']}")
    print(f"📚 Размер словаря: {info['vocabulary_size']:,} слов")
    print(f"📄 Документов в модели: {info['documents_count']}")
    print(f"🔍 Размер окна: {info['window']}")
    print(f"📊 Минимальная частота: {info['min_count']}")
    print(f"🔄 Эпох обучения: {info['epochs']}")

    if info["dm"] == 1:
        print("🔧 Режим: Distributed Memory (DM)")
    else:
        print("🔧 Режим: Distributed Bag of Words (DBOW)")

    print(f"➖ Negative sampling: {info['negative']}")
    print(f"🌳 Hierarchical Softmax: {'Да' if info['hs'] == 1 else 'Нет'}")
    print(f"📉 Sample threshold: {info['sample']}")

    if "training_time_formatted" in info:
        print(f"\n⏱️  Время обучения: {info['training_time_formatted']}")
    if "training_date" in info:
        print(f"📅 Дата обучения: {info['training_date']}")

    # Показываем примеры документов
    if trainer.corpus_info:
        print("\n📑 Примеры документов в модели:")
        for i, (tokens, doc_id, metadata) in enumerate(trainer.corpus_info[:5]):
            print(f"   {i + 1}. {doc_id}")
            if "tokens_count" in metadata:
                print(f"      Токенов: {metadata['tokens_count']}")


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Управление моделями Doc2Vec")
    parser.add_argument(
        "--details", "-d", help="Показать детальную информацию о модели"
    )

    args = parser.parse_args()

    if args.details:
        show_model_details(args.details)
    else:
        list_models()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Операция прервана.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
