from typing import Union
from openai import OpenAI
import requests

class BaseEmbedder:
    """Base class for embedding models."""
    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        raise NotImplementedError("Embed method must be implemented by subclasses")

class OpenAIEmbedder(BaseEmbedder):
    """
        Embedder using OpenAI's embedding models.
        Default model is "text-embedding-3-small".
    """
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]

        client = OpenAI(api_key=self.api_key)

        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings



class OllamaEmbedder(BaseEmbedder):
    """
        Embedder using Ollama's embedding models.
        Default model is "EmbeddingGemma".    
    """
    def __init__(self, model_name: str = "EmbeddingGemma", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.api_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data['embedding'])
        
        return embeddings