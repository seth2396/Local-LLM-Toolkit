from .BaseChunker import BaseChunker, _DEFAULT_CHUNK_SIZE, _DEFAULT_OVERLAP
from .Chunk import Chunk
from .FixedSizeChunk import FixedSizeChunk
from .separators import SEPARATORS_BY_DOCTYPE
from ..loaders import Document


class RecursiveChunk(BaseChunker):
    """
        Chunking strategy that recursively splits text using progressively smaller
        separators until all chunks fit within max_tokens.

        Separators are resolved from the document's doctype via SEPARATORS_BY_DOCTYPE
        at chunk time. Pass separators explicitly to override the doctype-based lookup.

        Attributes:
            max_tokens (int): Maximum number of characters per chunk.
            overlap_tokens (int): Number of characters to overlap between chunks.
            separators (list[str] | None): Override separator list. If None, resolved
                from the document's doctype at chunk time.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_CHUNK_SIZE,
        overlap_tokens: int = _DEFAULT_OVERLAP,
        separators: list[str] = None
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = separators

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks using recursive separator-based splitting.

        Separators are resolved from the document's doctype via SEPARATORS_BY_DOCTYPE
        unless overridden at construction time. Overlap is applied after splitting.

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
                raise ValueError(
                    f"RecursiveChunk can only support string content. "
                    f"The provided document content is of type {type(document.content)} "
                    f"and cannot be converted to string.\nError: {e}"
                )
        else:
            text = document.content

        separators = self.separators or SEPARATORS_BY_DOCTYPE.get(
            document.metadata.get("doctype", "default"),
            SEPARATORS_BY_DOCTYPE["default"]
        )

        pieces = self._split(text, separators)
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
        """
        Prepend the tail of the previous chunk to each subsequent chunk.

        Overlap length is controlled by self.overlap_tokens. The first chunk is
        returned unchanged. No-ops when overlap_tokens <= 0 or there is only one chunk.
        """
        if self.overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-self.overlap_tokens:]
            overlapped.append(tail + chunks[i])

        return overlapped
