import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(repr=False)
class Config:
    backend: str = 'sqlite'
    sqlite_path: str = 'cctv_rag.sqlite3'
    database_url: str = ''
    gemini_api_key: str = ''
    gemini_model: str = ''
    embedding_mode: str = 'disabled'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    timezone: str = 'UTC'
    window_minutes: int = 60

    @classmethod
    def from_env(cls):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        gemini_key = os.getenv('GEMINI_API_KEY', '')
        gemini_model = os.getenv('GEMINI_MODEL', '')
        if gemini_key and not gemini_model:
            gemini_model = 'gemini-2.5-flash'
        config = cls(backend=os.getenv('STORAGE_BACKEND', 'sqlite'), sqlite_path=os.getenv('SQLITE_PATH', 'cctv_rag.sqlite3'),
                     database_url=os.getenv('DATABASE_URL', ''), gemini_api_key=gemini_key,
                     gemini_model=gemini_model, embedding_mode=os.getenv('EMBEDDING_MODE', 'disabled'),
                     embedding_model=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                     timezone=os.getenv('TIME_ZONE', 'UTC'), window_minutes=int(os.getenv('SUMMARY_WINDOW_MINUTES', '60')))
        config.validate()
        return config

    def validate(self):
        if self.backend not in {'sqlite', 'postgres'}:
            raise ValueError('STORAGE_BACKEND must be sqlite or postgres')
        if self.backend == 'postgres' and not self.database_url:
            raise ValueError('Set DATABASE_URL for PostgreSQL')
        if self.embedding_mode not in {'disabled', 'minilm'}:
            raise ValueError('EMBEDDING_MODE must be disabled or minilm')
        if bool(self.gemini_api_key) != bool(self.gemini_model):
            raise ValueError('Set both GEMINI_API_KEY and GEMINI_MODEL, or leave both unset')
        if self.gemini_model and not self.gemini_model.startswith('gemini-'):
            raise ValueError(f'Unsupported Gemini model: {self.gemini_model}')
        if not 1 <= self.window_minutes <= 1440:
            raise ValueError('SUMMARY_WINDOW_MINUTES must be 1..1440')
        ZoneInfo(self.timezone)
