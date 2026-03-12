from ..loaders import Document
from ..embedders import BaseEmbedder

from .BaseChunkStrategy import BaseChunkStrategy
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


class UniversalChunker(BaseChunkStrategy):

    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder = embedder

    def chunk(self, document: Document, chunk_strategy: dict = None) -> list[Chunk]:
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

        return selected_class.chunk(document)
