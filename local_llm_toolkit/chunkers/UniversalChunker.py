from ..loaders import Document
from ..embedders import BaseEmbedder

from .BaseChunker import BaseChunker
from .Chunk import Chunk
from .FixedSizeChunk import FixedSizeChunk
from .RecursiveChunk import RecursiveChunk
from .SemanticChunk import SemanticChunk
from .LLMChunk import LLMChunk
from .TableChunk import TableChunk


STRATEGY_REGISTRY = {
    "fixed": FixedSizeChunk,
    "recursive": RecursiveChunk,
    "semantic": SemanticChunk,
    "llm": LLMChunk,
    "table": TableChunk
}

DEFAULT_STRATEGY_FOR_TYPE = {
    '.pdf': SemanticChunk,
    '.docx': SemanticChunk,
    '.txt': RecursiveChunk,
    '.md': RecursiveChunk,
    '.html': FixedSizeChunk,
    '.json': FixedSizeChunk,
    '.csv': TableChunk,
    '.xlsx': TableChunk
}


class UniversalChunker(BaseChunker):
    """
    Chunker that automatically selects a strategy based on the document's file extension.

    Uses DEFAULT_STRATEGY_FOR_TYPE to pick the appropriate chunking class, or accepts
    a caller-supplied chunk_strategy dict to override the defaults per extension.
    SemanticChunk is only available when an embedder is provided at construction time.

    Attributes:
        embedder (BaseEmbedder | None): Required when SemanticChunk may be selected.
            Pass None to disable semantic chunking.
    """

    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder = embedder

    def chunk(self, document: Document, chunk_strategy: dict = None, chunk_filter=None) -> list[Chunk]:
        """
        Chunk a document using the strategy appropriate for its file extension.

        Args:
            document: Document to chunk. Must have 'extension' in its metadata.
            chunk_strategy: Optional mapping of file extension to strategy name
                (a key from STRATEGY_REGISTRY) to override the defaults.
            chunk_filter: ChunkFilter to apply after chunking. Defaults to
                DEFAULT_CHUNK_FILTER. Pass None to skip filtering.

        Returns:
            list[Chunk]: Chunks produced by the selected strategy.

        Raises:
            NotImplementedError: If the extension has no matching strategy.
            ValueError: If SemanticChunk is selected but no embedder was provided.
        """
        extension = document.metadata['extension']
        if chunk_strategy and extension in chunk_strategy:
            strategy = STRATEGY_REGISTRY[chunk_strategy[extension]]
        elif extension in DEFAULT_STRATEGY_FOR_TYPE:
            strategy = DEFAULT_STRATEGY_FOR_TYPE[extension]
        else:
            raise NotImplementedError(f"{extension} Not found in chunk_strategy dict or default dictionary. \nPlease add {extension} to chunk_strategy dict.")

        if strategy is SemanticChunk:
            if self.embedder is None:
                raise ValueError("SemanticChunk requires an embedder. Pass one when constructing UniversalChunker(embedder=...).")
            selected_class = SemanticChunk(embedder=self.embedder)
        else:
            selected_class = strategy()

        return selected_class.chunk(document, chunk_filter)

    def _chunk(self, document: Document) -> list[Chunk]:
        return self.chunk(document)