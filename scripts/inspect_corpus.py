"""scripts/inspect_corpus.py - Скрипт для изучения документов в модели"""

import sys
from pathlib import Path

from loguru import logger

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer


def inspect_corpus(model_name: str = "doc2vec_model"):
    """Показать информацию о документах в модели"""

    trainer = Doc2VecTrainer()
    model = trainer.load_model(model_name)

    if not model:
        logger.error(f"Не удалось загрузить модель {model_name}")
        return

    if not trainer.corpus_info:
        logger.error("Информация о корпусе не найдена")
        return

    print(f"\n📚 ДОКУМЕНТЫ В МОДЕЛИ '{model_name}':")
    print("=" * 80)
    print(f"Всего документов: {len(trainer.corpus_info)}\n")

    # Группируем по расширениям
    by_extension = {}
    for tokens, doc_id, metadata in trainer.corpus_info:
        ext = Path(doc_id).suffix
        if ext not in by_extension:
            by_extension[ext] = []
        by_extension[ext].append(doc_id)

    # Показываем статистику по типам
    print("📊 По типам файлов:")
    for ext, docs in sorted(by_extension.items()):
        print(f"  {ext}: {len(docs)} файлов")

    # Показываем все документы
    print("\n📄 Список документов:")
    for i, (tokens, doc_id, metadata) in enumerate(trainer.corpus_info, 1):
        tokens_count = metadata.get("tokens_count", len(tokens))
        print(f"{i:3d}. {doc_id:<50} ({tokens_count} токенов)")

    # Примеры для создания тестов
    print("\n💡 Примеры для создания тестовых случаев:")
    print("relevant_docs = {")
    for i, (_, doc_id, _) in enumerate(trainer.corpus_info[:5]):
        print(f'    "{doc_id}",')
    print("}")

    print("\n✅ Используйте эти имена файлов в QueryTestCase!")


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "doc2vec_model"
    inspect_corpus(model_name)
