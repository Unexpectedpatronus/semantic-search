class SearchResult:
    """Search result structure"""

    def __init__(self, doc_id, similarity, metadata=None):
        self.doc_id = doc_id
        self.similarity = similarity
        self.metadata = metadata or {}
        self.file_path = Path(doc_id)

    def to_dict(self):
        return {"doc_id": self.doc_id, "similarity": self.similarity, "metadata": self.metadata}


class SemanticSearchEngine:
    """Semantic search engine"""

    def __init__(self, model=None, corpus_info=None, documents_base_path=None):
        self.model = model
        self.corpus_info = corpus_info or []
        self.documents_base_path = documents_base_path
        self.text_processor = TextProcessor()
        self.cache_manager = CacheManager(CACHE_DIR)
        self.config = SEARCH_CONFIG
        self._metadata_index = self._build_metadata_index()

    def _build_metadata_index(self):
        return {doc_id: metadata for tokens, doc_id, metadata in self.corpus_info}

    def _validate_search_params(self, query, top_k=None, threshold=None):
        query = query.strip()
        top_k = top_k or self.config["default_top_k"]
        threshold = threshold or self.config["similarity_threshold"]
        return query, top_k, threshold

    def _search_base(self, query, top_k=None, threshold=None):
        if self.model is None:
            return []
        query, top_k, threshold = self._validate_search_params(query, top_k, threshold)
        tokens = self.text_processor.preprocess_text(query)
        if not tokens:
            return []
        q_vector = self.model.infer_vector(tokens)
        similar_docs = self.model.dv.most_similar([q_vector], topn=top_k)
        return [SearchResult(doc_id, sim, self._metadata_index.get(doc_id)) for doc_id, sim in similar_docs if sim >= threshold]

    def search(self, query, top_k=None):
        """Search with cache"""
        key = f"search:{query.lower()}"
        cached = self.cache_manager.get(key)
        if cached:
            return [SearchResult.from_dict(r) for r in cached]
        results = self._search_base(query, top_k)
        self.cache_manager.set(key, [r.to_dict() for r in results])
        return results
