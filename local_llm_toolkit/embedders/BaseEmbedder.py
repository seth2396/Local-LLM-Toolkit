import logging
from abc import ABC, abstractmethod
from openai import BadRequestError


class BaseEmbedder(ABC):
    """Base class for embedding models."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, returning an empty list on BadRequestError."""
        try:
            return self._embed(text)
        except BadRequestError as e:
            logging.error(f"{self.__class__.__name__}.embed failed: {e}")
            return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning an empty list on BadRequestError."""
        try:
            return self._embed_documents(texts)
        except BadRequestError as e:
            logging.error(f"{self.__class__.__name__}.embed_documents failed: {e}")
            return []

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
