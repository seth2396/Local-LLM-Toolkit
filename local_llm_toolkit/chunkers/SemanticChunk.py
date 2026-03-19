import re

from .BaseChunker import _DEFAULT_CHUNK_SIZE
from .Chunk import Chunk
from .RecursiveChunk import RecursiveChunk
from ..embedders import BaseEmbedder
from ..loaders import Document


class SemanticChunk(RecursiveChunk):
    """
        Chunking strategy that groups sentences by semantic similarity.

        Sentences are embedded and compared pairwise. A new chunk begins wherever
        the cosine similarity between adjacent sentences drops below the threshold,
        indicating a shift in topic or meaning. max_tokens acts as a safety cap —
        any group that exceeds it is split by character using FixedSizeChunkStrategy.

        Attributes:
            embedder (BaseEmbedder): Embedder used to generate sentence vectors.
            breakpoint_threshold (float): Cosine similarity below which a chunk
                boundary is inserted. Range [0, 1]. Lower = fewer, larger chunks.
            max_tokens (int): Maximum characters per chunk. Oversized groups are
                split by character count after semantic grouping.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        breakpoint_threshold: float = 0.7,
        max_tokens: int = _DEFAULT_CHUNK_SIZE,
    ):
        super().__init__(max_tokens=max_tokens, overlap_tokens=0)
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold

    def _chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into semantically coherent chunks.

        Sentences are extracted, embedded, and compared pairwise. A chunk boundary
        is inserted wherever cosine similarity drops below breakpoint_threshold.
        Groups that exceed max_tokens are further split by character count.

        Args:
            document: Document to chunk. Content must be a string or convertible to one.

        Returns:
            list[Chunk]: Semantically grouped chunks with the document's metadata attached.

        Raises:
            ValueError: If document.content cannot be converted to a string.
        """
        if not isinstance(document.content, str):
            try:
                text = str(document.content)
            except Exception as e:
                raise ValueError(
                    f"SemanticChunk can only support string content. "
                    f"The provided document content is of type {type(document.content)} "
                    f"and cannot be converted to string.\nError: {e}"
                )
        else:
            text = document.content

        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [Chunk(content=text, metadata=document.metadata)]

        embeddings = self.embedder.embed_documents(sentences)
        similarities = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        groups = self._group_by_breakpoints(sentences, similarities)

        chunks = []
        for group in groups:
            if len(group) > self.max_tokens:
                for segment in self._character_split(group):
                    chunks.append(Chunk(content=segment, metadata=document.metadata))
            else:
                chunks.append(Chunk(content=group, metadata=document.metadata))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences on .  !  ? boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _group_by_breakpoints(self, sentences: list[str], similarities: list[float]) -> list[str]:
        """Merge sentences into groups, splitting where similarity drops below threshold."""
        groups = []
        current = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                groups.append(" ".join(current))
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])

        if current:
            groups.append(" ".join(current))

        return groups
