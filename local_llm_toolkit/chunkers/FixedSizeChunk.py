from .Chunk import Chunk
from .BaseChunker import BaseChunker, _DEFAULT_CHUNK_SIZE, _DEFAULT_OVERLAP
from ..loaders import Document


class FixedSizeChunk(BaseChunker):
    """
    Chunking strategy that splits text into fixed-size segments with optional overlap.

    Text is split sequentially by character count. Each chunk may overlap with the
    previous by overlap_tokens characters to preserve context across boundaries.

    Attributes:
        max_tokens (int): Maximum number of characters per chunk.
        overlap_tokens (int): Number of characters to overlap between consecutive chunks.
    """
    def __init__(self, max_tokens: int = _DEFAULT_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split_text(self, text: str) -> list[str]:
        """
            Split a string into fixed-size segments with optional overlap.

            Params:
                text: The string to split.

            Returns:
                List of text segments, each at most max_tokens characters long.
        """
        segments = []
        start = 0
        end = self.max_tokens
        while start < len(text):
            segments.append(text[start:end])
            start = end - self.overlap_tokens
            end = start + self.max_tokens
        return segments

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into fixed-size chunks.

        Args:
            document: Document to chunk. Content must be a string or convertible to one.

        Returns:
            list[Chunk]: Ordered chunks with the document's metadata attached to each.

        Raises:
            ValueError: If document.content cannot be converted to a string.
        """
        if not isinstance(document.content, str):
            try:
                text = str(document.content)
            except Exception as e:
                raise ValueError(f"FixedSizeChunkStrategy can only support string content. The provided document content is of type {type(document.content)} and cannot be converted to string. \n Error: {e}")
        else:
            text = document.content

        return [Chunk(content=segment, metadata=document.metadata) for segment in self.split_text(text)]
