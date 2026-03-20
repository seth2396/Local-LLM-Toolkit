import logging
from abc import ABC, abstractmethod
from openai import BadRequestError


class BaseEmbedder(ABC):
    """
    Base class for embedding models.

    Attributes:
        batch_limit (int | None): Maximum number of texts per API call in
            embed_documents. If None, all texts are sent in a single call.
    """

    batch_limit: int | None = None

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, returning an empty list on BadRequestError."""
        try:
            return self._embed(text)
        except BadRequestError as e:
            logging.error(f"{self.__class__.__name__}.embed failed: {e}")
            return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, batching calls according to batch_limit.

        Returns:
            Flat list of embeddings in the same order as the input texts.
            Returns an empty list if a BadRequestError is raised.
        """
        try:
            if self.batch_limit is None:
                return self._embed_documents(texts)
            results = []
            for i in range(0, len(texts), self.batch_limit):
                results.extend(self._embed_documents(texts[i:i + self.batch_limit]))
            return results
        except BadRequestError as e:
            logging.error(f"{self.__class__.__name__}.embed_documents failed: {e}")
            return []

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
