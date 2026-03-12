from abc import ABC, abstractmethod

from .Chunk import Chunk
from ..loaders import Document

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 100


class BaseChunker(ABC):
    """
        Abstract base class for text chunking strategies.

        Subclasses must implement the `chunk` method. Configuration (e.g. chunk
        size, overlap) is the responsibility of each strategy, as not all strategies
        share the same parameters.

        Example:
            >>> strategy = MyChunkStrategy(...)
            >>> chunks = strategy.chunk(document)
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into a list of Chunk objects.

        Args:
            document: Document to chunk. Content must be a string or convertible to one.

        Returns:
            list[Chunk]: Ordered chunks with the document's metadata attached to each.
        """

    def chunk_documents(self, documents: list[Document]) -> list[list[Chunk]]:
        """Chunk each document in the list, returning one chunk list per document."""
        return [self.chunk(document) for document in documents]