import requests
from .BaseEmbedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
        Embedder using Ollama's embedding models.
        Default model is "EmbeddingGemma".
    """
    def __init__(self, model: str = "EmbeddingGemma", api_url: str = "http://localhost:11434", batch_limit: int = None):
        self.model = model
        self.api_url = api_url
        self.batch_limit = batch_limit

    def _embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.api_url}/api/embed",
            json={"model": self.model, "input": text}
        )
        response.raise_for_status()
        return response.json()['embeddings'][0]

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.api_url}/api/embed",
            json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        return [embedding for embedding in response.json()['embeddings']]
    

