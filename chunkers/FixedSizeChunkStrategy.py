from .Chunk import Chunk
from .BaseChunkStrategy import BaseChunkStrategy, _DEFAULT_CHUNK_SIZE, _DEFAULT_OVERLAP
from loaders import Document


class FixedSizeChunkStrategy(BaseChunkStrategy):
    """
        Chunking strategy that splits text into fixed-size segments with optional overlap.

        This strategy produces sequential chunks of a specified maximum token size.
        Each chunk may optionally overlap with the previous one, which can improve
        embedding coherence when used in retrieval pipelines.

        Attributes:
            max_tokens (int): Maximum number of tokens (or characters) per chunk.
                document (Document): A document object containing `content` (str or overlap_tokens (int): Number of tokens (or characters) that overlap
                    convertible to str) and `metadata` (dict).

            Returns:
                list[Chunk]: A list of `Chunk` objects, each containing a segment of
                    text and inherited metadata from the original document.

            Raises:
                ValueError: If `document.content` is not a string and cannot be
                    converted to one.

            Example:
                >>> strategy = FixedSizeChunkStrategy(max_tokens=100, overlap_tokens=20)
                >>> chunks = strategy.chunk(document)
                >>> len(chunks) = 5
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
        if not isinstance(document.content, str):
            try:
                text = str(document.content)
            except Exception as e:
                raise ValueError(f"FixedSizeChunkStrategy can only support string content. The provided document content is of type {type(document.content)} and cannot be converted to string. \n Error: {e}")
        else:
            text = document.content

        return [Chunk(content=segment, metadata=document.metadata) for segment in self.split_text(text)]
