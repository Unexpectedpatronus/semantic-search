class Doc2VecTrainer:
    """Trainer for Doc2Vec model"""

    def __init__(self):
        self.model = None
        self.config = DOC2VEC_CONFIG

    def create_tagged_documents(self, corpus):
        """Prepare TaggedDocument objects"""
        return [TaggedDocument(words=tokens, tags=[doc_id]) for tokens, doc_id, _ in corpus]

    def _get_training_params(self, vector_size=None, window=None, min_count=None, epochs=None):
        """Assemble training params"""
        return {
            "vector_size": vector_size or self.config["vector_size"],
            "window": window or self.config["window"],
            "min_count": min_count or self.config["min_count"],
            "epochs": epochs or self.config["epochs"],
        }

    def train_model(self, corpus, vector_size=None, window=None, min_count=None, epochs=None):
        """Train Doc2Vec"""
        params = self._get_training_params(vector_size, window, min_count, epochs)
        tagged_docs = self.create_tagged_documents(corpus)
        self.model = Doc2Vec(tagged_docs, **params)
        return self.model

    def save_model(self, model_name="doc2vec_model"):
        """Save trained model"""
        path = MODELS_DIR / f"{model_name}.model"
        self.model.save(str(path))

    def load_model(self, model_name="doc2vec_model"):
        """Load model from disk"""
        path = MODELS_DIR / f"{model_name}.model"
        self.model = Doc2Vec.load(str(path))
