from abc import ABC, abstractmethod

from .Chunk import Chunk
from ..loaders import Document

_DEFAULT_CHUNK_SIZE = 200
_DEFAULT_OVERLAP = 40


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

    
    #Helper functions
    def _sanitize_texts(self, texts: list[str]) -> list[str]:
        cleaned = []
        for text in texts:
            if not text:
                continue
                
            # 1. Remove non-printable/control characters (like \x00, \x01)
            # 2. Force UTF-8 encoding and ignore any malformed byte sequences
            # 3. Strip leading/trailing whitespace
            safe_text = "".join(ch for ch in text if ch.isprintable())
            safe_text = safe_text.encode("utf-8", "ignore").decode("utf-8").strip()
            
            # Only add if the string isn't empty after cleaning
            if safe_text:
                cleaned.append(safe_text)
                
        return cleaned