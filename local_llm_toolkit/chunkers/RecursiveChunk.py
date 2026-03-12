from .BaseChunkStrategy import BaseChunkStrategy, _DEFAULT_CHUNK_SIZE, _DEFAULT_OVERLAP
from .Chunk import Chunk
from .FixedSizeChunk import FixedSizeChunk
from ..loaders import Document


class RecursiveChunk(BaseChunkStrategy):
    """
        Chunking strategy that recursively splits text using progressively smaller
        separators until all chunks fit within max_tokens.

        Split order: paragraphs → lines → sentences → words → characters.

        Attributes:
            max_tokens (int): Maximum number of characters per chunk.
            overlap_tokens (int): Number of characters to overlap between chunks.
            separators (list[str]): Ordered list of separators to try. Defaults to
                ["\n\n", "\n", ". ", "? ", "! ", " ", ""].
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        max_tokens: int = _DEFAULT_CHUNK_SIZE,
        overlap_tokens: int = _DEFAULT_OVERLAP,
        separators: list[str] = None
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        if not isinstance(document.content, str):
            try:
                text = str(document.content)
            except Exception as e:
                raise ValueError(
                    f"RecursiveChunk can only support string content. "
                    f"The provided document content is of type {type(document.content)} "
                    f"and cannot be converted to string.\nError: {e}"
                )
        else:
            text = document.content

        pieces = self._split(text, self.separators)
        pieces = self._apply_overlap(pieces)
        return [Chunk(content=piece, metadata=document.metadata) for piece in pieces]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that produces chunks
        within max_tokens, falling back to the next separator as needed."""

        # Base case: no more separators — split by character
        if not separators or separators[0] == "":
            return self._character_split(text)

        separator = separators[0]
        remaining = separators[1:]

        splits = text.split(separator)
        chunks = []
        current = ""

        for split in splits:
            if not split:
                continue

            candidate = (current + separator + split) if current else split

            if len(candidate) <= self.max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                # This split alone is still too large — recurse with next separator
                if len(split) > self.max_tokens:
                    chunks.extend(self._split(split, remaining))
                    current = ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return chunks

    def _character_split(self, text: str) -> list[str]:
        """Last resort: split by fixed character count with no separator."""
        return FixedSizeChunk(max_tokens=self.max_tokens, overlap_tokens=0).split_text(text)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of the previous chunk to each subsequent chunk."""
        if self.overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-self.overlap_tokens:]
            overlapped.append(tail + chunks[i])

        return overlapped
