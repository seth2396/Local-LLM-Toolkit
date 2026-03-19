from abc import ABC, abstractmethod

from .Chunk import Chunk
from .ChunkFilter import DEFAULT_CHUNK_FILTER
from ..loaders import Document

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 100


class BaseChunker(ABC):
    """
        Abstract base class for text chunking strategies.

        Subclasses must implement `_chunk()`. Configuration (e.g. chunk size, overlap)
        is the responsibility of each strategy, as not all strategies share the same
        parameters.

        Filtering is applied automatically in `chunk()` via a ChunkFilter. Pass a
        custom ChunkFilter to override the defaults, or False to skip filtering entirely.

        Example:
            >>> strategy = MyChunkStrategy(...)
            >>> chunks = strategy.chunk(document)
    """

    def chunk(self, document: Document, chunk_filter=None) -> list[Chunk]:
        """
        Split a document into filtered Chunk objects.

        Calls _chunk() to produce raw chunks, then applies chunk_filter.

        Params:
            document: Document to chunk.
            chunk_filter: ChunkFilter to apply. None uses DEFAULT_CHUNK_FILTER. False skips filtering entirely.

        Returns:
            list[Chunk]: Filtered, ordered chunks with the document's metadata attached.
        """
        chunks = self._chunk(document)
        f = chunk_filter if chunk_filter is not None else DEFAULT_CHUNK_FILTER
        return f.filter(chunks) if f else chunks

    @abstractmethod
    def _chunk(self, document: Document) -> list[Chunk]:
        """
        Produce raw chunks from a document. Do not call directly — use chunk() instead.

        Args:
            document: Document to chunk. Content must be a string or convertible to one.

        Returns:
            list[Chunk]: Unfiltered, ordered chunks with the document's metadata attached.
        """

    def chunk_documents(self, documents: list[Document], chunk_filter=None) -> list[list[Chunk]]:
        """Chunk each document in the list, applying chunk_filter to each."""
        return [self.chunk(document, chunk_filter) for document in documents]