"""Контроллер для связи GUI и логики обработки/поиска документов"""

from pathlib import Path
from typing import Dict, List

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor, ProcessedDocument
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.utils.notification_system import NotificationManager
from semantic_search.utils.statistics import calculate_statistics_from_processed_docs


class AppController:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.documents: List[ProcessedDocument] = []
        self.engine: SemanticSearchEngine | None = None
        self.notifier = NotificationManager()

    def load_documents(self, folder: Path) -> Dict[str, str]:
        self.documents.clear()
        self.documents.extend(self.processor.process_documents(folder))

        # Тренируем модель
        trainer = Doc2VecTrainer()
        model = trainer.train_model(
            [(doc.tokens, doc.relative_path, doc.metadata) for doc in self.documents]
        )

        # Передаём модель в поисковик
        self.engine = SemanticSearchEngine(model)

        stats = calculate_statistics_from_processed_docs(self.documents)
        return {
            "status": f"Загружено документов: {stats['processed_files']}",
            "details": str(stats),
        }

    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.engine:
            return []

        results = self.engine.search(query)

        for res in results:
            doc_id = res.doc_id
            similarity = res.similarity
            metadata = res.metadata

            title = metadata.get("relative_path", doc_id)
            path = metadata.get("file_path", "")

            results.append(
                {
                    "title": title,
                    "path": str(path),
                    "score": round(similarity, 4),
                }
            )
