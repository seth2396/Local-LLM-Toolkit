from openai import OpenAI
from .BaseEmbedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
        Embedder using OpenAI's embedding models.
        Default model is "text-embedding-3-small".
    """
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
