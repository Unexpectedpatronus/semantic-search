class TextSummarizer:
    """Extractive text summarizer"""

    def __init__(self, doc2vec_model=None):
        self.model = doc2vec_model
        self.text_processor = TextProcessor()
        self.chunk_size = TEXT_PROCESSING_CONFIG.get("chunk_size", 500_000)
        self.min_sentence_len = SUMMARIZATION_CONFIG.get("min_sentence_length", 15)

    def set_model(self, model):
        self.model = model

    def _filter_sentence(self, sentence):
        """Filter unsuitable sentences"""
        s = sentence.strip()
        return len(s) >= self.min_sentence_len and len(s.split()) >= 5

    def _sentence_to_vector(self, tokens):
        return self.model.infer_vector(tokens) if self.model and tokens else None

    def _calculate_sentence_scores(self, sentences):
        filtered = [s for s in sentences if self._filter_sentence(s)]
        vectors = [self._sentence_to_vector(self.text_processor.preprocess_text(s)) for s in filtered]
        if len(vectors) < 2:
            return [(s, 1.0) for s in filtered]
        sim_matrix = cosine_similarity(vectors)
        scores = self._pagerank(sim_matrix)
        return list(zip(filtered, scores))

    def _pagerank(self, matrix, damping=0.85, max_iter=100):
        n = matrix.shape[0]
        scores = np.ones(n) / n
        matrix = np.where(matrix == 0, 1e-8, matrix)
        matrix /= matrix.sum(axis=1)[:, None]
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * matrix.T @ scores
            if np.allclose(scores, new_scores, atol=1e-6):
                break
            scores = new_scores
        return scores.tolist()

    def summarize_text(self, text, sentences_count=5):
        """Summarize text"""
        sentences = self.text_processor.split_into_sentences(text)
        scored = self._calculate_sentence_scores(sentences)
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, _ in scored[:sentences_count]]
        return top_sentences
