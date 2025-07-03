class ProcessedDocument(NamedTuple):
    """Structure for processed document"""
    file_path: Path
    relative_path: str
    raw_text: str
    tokens: list
    metadata: dict


class DocumentProcessor:
    """Document collection processor"""

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.text_processor = TextProcessor()

    def process_documents(self, root_path):
        """Process files in folder"""
        file_paths = self.file_extractor.find_documents(root_path)
        for file_path in file_paths:
            raw_text = self.file_extractor.extract_text(file_path)
            tokens = self.text_processor.preprocess_text(raw_text)
            yield ProcessedDocument(
                file_path=file_path,
                relative_path=str(file_path.relative_to(root_path)).replace("\\", "/"),
                raw_text=raw_text,
                tokens=tokens,
                metadata={"tokens_count": len(tokens)}
            )
