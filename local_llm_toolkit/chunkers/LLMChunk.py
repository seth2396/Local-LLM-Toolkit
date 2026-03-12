from .BaseChunkStrategy import BaseChunkStrategy
from .Chunk import Chunk
from ..loaders import Document


class LLMChunk(BaseChunkStrategy):
    """
        LLM based chunker - Not yet implemented.
    """
    def __init__(self):
        raise NotImplementedError("LLM-based chunking not yet implemented")

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("LLM-based chunking not yet implemented")
