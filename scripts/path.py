import pickle
import pprint
from pathlib import Path

with open(
    r"C:\Users\evgen\Evgeny\Dev_projects\Dev_Python\diplom\semantic-search\data\models\doc2vec_model_corpus_info.pkl",
    "rb",
) as f:
    corpus_info = pickle.load(f)
print(f"Всего документов: {len(corpus_info)}")
for tokens, doc_id, metadata in corpus_info:
    pprint.pprint(str(Path(doc_id).absolute()))
    pprint.pprint(metadata)
