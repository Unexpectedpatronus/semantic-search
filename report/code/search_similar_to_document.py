def search_similar_to_document(
    self, doc_id: str, top_k: Optional[int] = None
) -> List[SearchResult]:
    if self.model is None:
        logger.error("Model is not loaded")
        return []

    if not doc_id:
        logger.error("Document ID is missing")
        return []

    top_k = top_k or self.config["default_top_k"]

    try:
        if doc_id not in self.model.dv:
            logger.error(f"Document not found in model: {doc_id}")
            return []

        similar_docs = self.model.dv.most_similar(doc_id, topn=top_k + 1)

        results = []
        for sim_doc_id, similarity in similar_docs:
            if sim_doc_id != doc_id:
                metadata = self._metadata_index.get(sim_doc_id, {})
                results.append(SearchResult(sim_doc_id, similarity, metadata))

        return results[:top_k]

    except Exception as e:
        logger.error(f"Error while searching similar documents: {e}")
        return []
