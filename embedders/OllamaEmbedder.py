import requests
from .BaseEmbedder import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
        Embedder using Ollama's embedding models.
        Default model is "EmbeddingGemma".
    """
    def __init__(self, model_name: str = "EmbeddingGemma", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.api_url}/api/embed",
            json={"model": self.model_name, "input": text}
        )
        response.raise_for_status()
        return response.json()['embeddings'][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.api_url}/api/embed",
            json={"model": self.model_name, "input": texts}
        )
        response.raise_for_status()
        return [embedding for embedding in response.json()['embeddings']]
    

