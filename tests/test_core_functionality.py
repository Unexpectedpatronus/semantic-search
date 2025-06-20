"""Тесты основной функциональности (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from semantic_search.core.doc2vec_trainer import Doc2VecTrainer
from semantic_search.core.document_processor import DocumentProcessor
from semantic_search.core.search_engine import SemanticSearchEngine
from semantic_search.utils.validators import DataValidator, ValidationError


class TestDocumentProcessor:
    """Тесты обработчика документов"""

    def test_processor_initialization(self):
        """Тест инициализации процессора"""
        processor = DocumentProcessor()
        assert processor is not None
        assert processor.file_extractor is not None
        assert processor.text_processor is not None

    def test_empty_directory_processing(self):
        """Тест обработки пустой директории"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = DocumentProcessor()
            docs = list(processor.process_documents(Path(temp_dir)))
            assert len(docs) == 0

    @patch("semantic_search.core.document_processor.FileExtractor")
    def test_document_processing_with_mock(self, mock_extractor):
        """Тест обработки с мокированием"""
        # Настройка мока
        mock_instance = Mock()
        mock_extractor.return_value = mock_instance

        # Используем относительный путь вместо абсолютного
        test_path = Path("test_documents")
        mock_instance.find_documents.return_value = [test_path / "test_doc.pdf"]
        mock_instance.extract_text.return_value = "Test document content " * 20

        processor = DocumentProcessor()
        processor.file_extractor = mock_instance

        docs = list(processor.process_documents(test_path))
        assert len(docs) > 0
        assert docs[0].relative_path == "test_doc.pdf"


class TestValidators:
    """Тесты валидаторов"""

    def test_text_validation_success(self):
        """Тест успешной валидации текста"""
        text = DataValidator.validate_text("Hello world", min_length=5)
        assert text == "Hello world"

    def test_text_validation_failure(self):
        """Тест неудачной валидации текста"""
        with pytest.raises(ValidationError):
            DataValidator.validate_text("Hi", min_length=10)

    def test_search_params_validation(self):
        """Тест валидации параметров поиска"""
        params = DataValidator.validate_search_params(
            query="test query", top_k=10, similarity_threshold=0.5
        )

        assert params["query"] == "test query"
        assert params["top_k"] == 10
        assert params["similarity_threshold"] == 0.5

    def test_model_params_validation(self):
        """Тест валидации параметров модели"""
        params = DataValidator.validate_model_params(vector_size=100, epochs=20)

        assert params["vector_size"] == 100
        assert params["epochs"] == 20


@pytest.fixture
def sample_corpus():
    """Фикстура с образцом корпуса"""
    return [
        (["машинное", "обучение", "алгоритм"], "doc1.pdf", {"tokens_count": 3}),
        (
            ["нейронная", "сеть", "глубокое", "обучение"],
            "doc2.pdf",
            {"tokens_count": 4},
        ),
        (["анализ", "данных", "статистика"], "doc3.pdf", {"tokens_count": 3}),
    ]


class TestDoc2VecTrainer:
    """Тесты тренера Doc2Vec"""

    def test_trainer_initialization(self):
        """Тест инициализации тренера"""
        trainer = Doc2VecTrainer()
        assert trainer is not None
        assert trainer.model is None

    def test_tagged_documents_creation(self, sample_corpus):
        """Тест создания TaggedDocument объектов"""
        trainer = Doc2VecTrainer()

        if not hasattr(trainer, "create_tagged_documents"):
            pytest.skip("Gensim не доступен")

        tagged_docs = trainer.create_tagged_documents(sample_corpus)
        assert len(tagged_docs) == 3
        assert tagged_docs[0].tags == ["doc1.pdf"]


class TestSearchEngine:
    """Тесты поискового движка"""

    def test_search_engine_initialization(self):
        """Тест инициализации поискового движка"""
        engine = SemanticSearchEngine()
        assert engine is not None
        assert engine.model is None

    def test_empty_query_handling(self):
        """Тест обработки пустого запроса"""
        engine = SemanticSearchEngine()
        results = engine.search("")
        assert len(results) == 0

    def test_search_without_model(self):
        """Тест поиска без модели"""
        engine = SemanticSearchEngine()
        results = engine.search("test query")
        assert len(results) == 0


# Бенчмарк тесты
class TestPerformance:
    """Тесты производительности"""

    def test_text_processing_speed(self, benchmark):
        """Бенчмарк скорости обработки текста"""
        from semantic_search.utils.text_utils import TextProcessor

        processor = TextProcessor()
        test_text = "Это тестовый текст для проверки скорости обработки. " * 100

        result = benchmark(processor.preprocess_text, test_text)
        assert isinstance(result, list)
        assert len(result) > 0
