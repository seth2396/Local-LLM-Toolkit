from abc import ABC, abstractmethod
from typing import Any

from ..embedders import BaseEmbedder
from ..chunkers import Chunk


class BaseVectorStore(ABC):
    """
        Abstract base class for vector stores.
    """
    @abstractmethod
    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder

    @abstractmethod
    def add(self, content: list[str] | str, embedding_content: list[str] | str, metadata: list[dict] | dict = None) -> None:
        """
            Adds content to the vector database.

            Params:
                content: Text content to be stored
                embedding_content: Text content to be embedded if embedding is to differ from content in the entry
                metadata: Any metadata to be stored with the entry
        """

    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> Any:
        """
            Query the vector database.

            Params:
                query: String to query against
                top_k: Number of results to return
        """

    @abstractmethod
    def count(self) -> int:
        """Returns the number of entries in the vector store."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> dict:
        """
            Add or update chunks, skipping any whose hash_id matches what is already stored.
            Deletes stale chunk positions if the document has fewer chunks than before.

            Params:
                chunks: List of Chunk objects from a single document. Each chunk's metadata
                        must include 'file_name' and 'hash_id'.

            Returns:
                {"added": int, "updated": int, "skipped": int, "deleted": int}
        """
