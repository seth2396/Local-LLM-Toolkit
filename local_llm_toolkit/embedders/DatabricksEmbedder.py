from databricks_openai import DatabricksOpenAI
from .BaseEmbedder import BaseEmbedder


class DatabricksEmbedder(BaseEmbedder):
    """
    Embedder using a Databricks model serving endpoint via the DatabricksOpenAI client.

    Attributes:
        model (str): The name of the Databricks embedding endpoint.
        client (DatabricksOpenAI): A pre-configured DatabricksOpenAI client instance.
    """

    def __init__(self, model: str, client: DatabricksOpenAI):
        self.model = model
        self.client = client

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
