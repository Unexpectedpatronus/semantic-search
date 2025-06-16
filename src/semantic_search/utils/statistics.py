"""Утилиты для вычисления статистики"""

from collections import Counter
from typing import Any, Dict, List

from semantic_search.core.document_processor import ProcessedDocument


def calculate_statistics_from_processed_docs(
    docs_data: List[ProcessedDocument],
) -> Dict[str, Any]:
    """
    Вычисление статистики из уже обработанных документов

    Args:
        docs_data: Список обработанных документов

    Returns:
        Словарь со статистикой:
        - processed_files: количество обработанных файлов
        - total_tokens: общее количество токенов
        - avg_tokens_per_doc: среднее количество токенов на документ
        - extensions_count: количество файлов по расширениям
        - largest_doc: информация о самом большом документе
        - smallest_doc: информация о самом маленьком документе
        - total_chars: общее количество символов
        - avg_chars_per_doc: среднее количество символов на документ
    """
    if not docs_data:
        return {
            "processed_files": 0,
            "total_tokens": 0,
            "avg_tokens_per_doc": 0.0,
            "extensions_count": {},
            "largest_doc": None,
            "smallest_doc": None,
            "total_chars": 0,
            "avg_chars_per_doc": 0.0,
        }

    # Основная статистика
    stats = {
        "processed_files": len(docs_data),
        "total_tokens": sum(doc.metadata["tokens_count"] for doc in docs_data),
        "total_chars": sum(doc.metadata["text_length"] for doc in docs_data),
        "extensions_count": dict(
            Counter(doc.metadata["extension"] for doc in docs_data)
        ),
    }

    # Средние значения
    stats["avg_tokens_per_doc"] = stats["total_tokens"] / stats["processed_files"]
    stats["avg_chars_per_doc"] = stats["total_chars"] / stats["processed_files"]

    # Самый большой и маленький документы
    docs_by_tokens = sorted(docs_data, key=lambda x: x.metadata["tokens_count"])
    stats["smallest_doc"] = {
        "path": docs_by_tokens[0].relative_path,
        "tokens": docs_by_tokens[0].metadata["tokens_count"],
        "chars": docs_by_tokens[0].metadata["text_length"],
    }
    stats["largest_doc"] = {
        "path": docs_by_tokens[-1].relative_path,
        "tokens": docs_by_tokens[-1].metadata["tokens_count"],
        "chars": docs_by_tokens[-1].metadata["text_length"],
    }

    return stats


def format_statistics_for_display(stats: Dict[str, Any]) -> str:
    """
    Форматирование статистики для красивого вывода

    Args:
        stats: Словарь со статистикой из calculate_statistics_from_processed_docs

    Returns:
        Отформатированная строка со статистикой
    """
    if stats["processed_files"] == 0:
        return "❌ Нет обработанных документов"

    lines = [
        "📊 Статистика корпуса:",
        f"📁 Обработано документов: {stats['processed_files']}",
        f"🔤 Общее количество токенов: {stats['total_tokens']:,}",
        f"📄 Среднее токенов на документ: {stats['avg_tokens_per_doc']:.1f}",
        f"📝 Общее количество символов: {stats['total_chars']:,}",
        f"📖 Среднее символов на документ: {stats['avg_chars_per_doc']:.1f}",
        f"📋 Форматы файлов: {stats['extensions_count']}",
    ]

    if stats["largest_doc"]:
        lines.extend(
            [
                "📈 Самый большой документ:",
                f"   📄 {stats['largest_doc']['path']}",
                f"   🔤 {stats['largest_doc']['tokens']} токенов, {stats['largest_doc']['chars']} символов",
            ]
        )

    if stats["smallest_doc"]:
        lines.extend(
            [
                "📉 Самый маленький документ:",
                f"   📄 {stats['smallest_doc']['path']}",
                f"   🔤 {stats['smallest_doc']['tokens']} токенов, {stats['smallest_doc']['chars']} символов",
            ]
        )

    return "\n".join(lines)


def calculate_model_statistics(model_info: Dict[str, Any]) -> str:
    """
    Форматирование статистики модели для вывода

    Args:
        model_info: Информация о модели из Doc2VecTrainer.get_model_info()

    Returns:
        Отформатированная строка со статистикой модели
    """
    if model_info.get("status") != "loaded":
        return f"❌ Модель недоступна: {model_info.get('status', 'неизвестно')}"

    lines = [
        "🧠 Статистика модели:",
        f"✅ Статус: {model_info['status']}",
        f"📏 Размерность векторов: {model_info['vector_size']}",
        f"📚 Размер словаря: {model_info['vocabulary_size']:,} слов",
        f"📄 Документов в модели: {model_info['documents_count']}",
        f"🔍 Размер окна контекста: {model_info['window']}",
        f"📊 Минимальная частота слова: {model_info['min_count']}",
        f"🔄 Количество эпох обучения: {model_info['epochs']}",
    ]

    return "\n".join(lines)
