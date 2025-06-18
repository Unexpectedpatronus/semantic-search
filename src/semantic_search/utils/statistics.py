"""Утилита для вычисления статистики по обработанным документам"""

from collections import Counter
from typing import Any, Dict, List

from semantic_search.core.document_processor import ProcessedDocument


def calculate_statistics_from_processed_docs(
    docs_data: List[ProcessedDocument],
) -> Dict[str, Any]:
    if not docs_data:
        return {
            "processed_files": 0,
            "total_tokens": 0,
            "avg_tokens_per_doc": 0.0,
            "extensions_count": {},
            "largest_doc": {},
            "smallest_doc": {},
            "total_chars": 0,
            "avg_chars_per_doc": 0.0,
        }

    total_tokens = 0
    total_chars = 0
    extensions = Counter()
    largest = None
    smallest = None

    for doc in docs_data:
        num_tokens = len(doc.tokens)
        num_chars = len(doc.raw_text)
        total_tokens += num_tokens
        total_chars += num_chars
        extensions[doc.file_path.suffix.lower()] += 1

        if largest is None or num_tokens > len(largest.tokens):
            largest = doc
        if smallest is None or num_tokens < len(smallest.tokens):
            smallest = doc

    return {
        "processed_files": len(docs_data),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(total_tokens / len(docs_data), 2),
        "extensions_count": dict(extensions),
        "largest_doc": {
            "file": str(largest.relative_path),
            "tokens": len(largest.tokens),
            "chars": len(largest.raw_text),
        },
        "smallest_doc": {
            "file": str(smallest.relative_path),
            "tokens": len(smallest.tokens),
            "chars": len(smallest.raw_text),
        },
        "total_chars": total_chars,
        "avg_chars_per_doc": round(total_chars / len(docs_data), 2),
    }
