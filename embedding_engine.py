"""
Этот модуль превращает тексты в числовые векторы
и ищет ближайшие тексты по косинусному сходству
"""
import numpy as np
import config
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingEngine:

    def __init__(self):
        # погружаем модельку
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Словарь индексов: {"faq": {...}, "kb": {...}, ...}
        self._indexes = {}
    
    def build_index(self, name: str, texts: list, meta: list = None):
        # encode: пропускает каждый текст через нейросеть
        # normalize_embeddings=True: делает длину каждого вектора = 1
        emb = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        emb = np.array(emb).astype("float32")

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity через dot product
        index.add(emb)

        self._indexes[name] = {
            "index": index,          
            "texts": texts,
            "meta": meta or texts}

    
    def search(self, name: str, query: str, top_k: int = 5) -> list:
        if name not in self._indexes:
            return []

        idx = self._indexes[name]

        # Превращаем запрос в вектор
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.array(q).astype("float32")

        # FAISS сразу возвращает top-K
        scores, indices = idx["index"].search(q, top_k)

        scores = scores[0]
        indices = indices[0]

        return [
            (float(scores[i]), idx["texts"][indices[i]], idx["meta"][indices[i]])
            for i in range(len(indices))
            if indices[i] != -1
        ]
