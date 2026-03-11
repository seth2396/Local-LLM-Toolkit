from .BaseChunkStrategy import BaseChunkStrategy
from .Chunk import Chunk
from loaders import Document


class TableChunk(BaseChunkStrategy):
    """
        Table-based chunker - Not yet implemented.
    """
    def __init__(self):
        raise NotImplementedError("Table-based chunking not yet implemented")

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("Table-based chunking not yet implemented")
