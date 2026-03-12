from abc import ABC, abstractmethod

from .Chunk import Chunk
from ..loaders import Document

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 100


class BaseChunkStrategy(ABC):
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
        pass
