import hashlib


class FakeEmbedder:
    """Test-only vectors. Never represent these as semantic embeddings."""
    model_id = 'test-only-sha256-v1'
    dimension = 8

    def __init__(self):
        self.calls = 0
        self.fail = False

    def embed_text(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError('secret-provider-token-must-not-leak')
        return [x / 255 + 0.01 for x in hashlib.sha256(text.encode()).digest()[:8]]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]
