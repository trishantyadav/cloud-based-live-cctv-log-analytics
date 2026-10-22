from functools import lru_cache
from .config import Config
from .storage import SQLiteStore
from .ingestion import Ingestor
from .summarizer import SummaryProcessor
from .retriever import Retriever
from .service import QueryService


@lru_cache(maxsize=2)
def load_embedder(model):
    from .embeddings import MiniLM
    return MiniLM(model)


class Runtime:
    def __init__(self, config=None):
        self.config = config or Config.from_env()
        self.config.validate()
        embedder = None
        self.embedder_error = None
        if self.config.embedding_mode == 'minilm':
            try:
                embedder = load_embedder(self.config.embedding_model)
            except Exception:
                self.embedder_error = 'MiniLM embedding model unavailable; falling back to metadata-only retrieval.'
        llm = None
        if self.config.gemini_api_key:
            from .rag_generate import Gemini
            llm = Gemini(self.config.gemini_api_key, self.config.gemini_model)
        if self.config.backend == 'postgres':
            from .database import PostgresStore
            self.store = PostgresStore(self.config.database_url)
        else:
            self.store = SQLiteStore(self.config.sqlite_path)
        self.embedder = embedder
        self.llm = llm
        self.processor = SummaryProcessor(self.store, embedder)
        self.ingestor = Ingestor(self.store, self.processor, self.config.window_minutes)
        self.service = QueryService(self.store, Retriever(self.store, embedder), llm, self.config.timezone)

    def status(self):
        batches = self.store.batches()
        return dict(
            latest_success=max((b.get('succeeded_at', '') for b in batches if b['status'] == 'succeeded'), default=None),
            counts={s: sum(b['status'] == s for b in batches) for s in ['pending', 'processing', 'succeeded', 'failed', 'partially_failed']}
        )

    def close(self):
        self.store.close()
