from openai import OpenAI
from .BaseEmbedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
        Embedder using OpenAI's embedding models.
        Default model is "text-embedding-3-small".
    """
    def __init__(self, model: str, api_key: str, batch_limit: int = None):
        self.model = model
        self.api_key = api_key
        self.batch_limit = batch_limit

    def _embed(self, text: str) -> list[float]:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
