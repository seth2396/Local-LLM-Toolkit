from .BaseChunker import BaseChunker
from .Chunk import Chunk
from ..loaders import Document


class TableChunk(BaseChunker):
    """
    Chunking strategy specialized for tabular data (DataFrames, CSV, Excel).

    Not yet implemented. Intended to split structured data row-by-row or by
    logical groupings rather than treating it as plain text.
    """
    def __init__(self):
        raise NotImplementedError("Table-based chunking not yet implemented")

    def _chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("Table-based chunking not yet implemented")
