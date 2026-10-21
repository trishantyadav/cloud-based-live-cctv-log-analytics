import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed_text(self, text: str) -> list[float]: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def validate_vector(vector, dimension):
    if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
        raise ValueError('embedding dimension or values invalid')
    if not any(vector):
        raise ValueError('zero embedding is not searchable')
    return vector


class MiniLM:
    dimension = 384

    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.model = SentenceTransformer(model_id)
        if self.model.get_sentence_embedding_dimension() != self.dimension:
            raise ValueError('selected model must produce 384 dimensions')

    def embed_texts(self, texts):
        return [validate_vector(v, self.dimension) for v in self.model.encode(texts, normalize_embeddings=True).tolist()]

    def embed_text(self, text):
        return self.embed_texts([text])[0]
