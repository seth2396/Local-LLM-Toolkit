from abc import ABC, abstractmethod

class BaseEmbedder:
    """Base class for embedding models."""
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embed method must be implemented by subclasses")

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Embed_documents method must be implemented by subclasses")
