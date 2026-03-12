from .BaseChunker import BaseChunker
from .Chunk import Chunk
from ..loaders import Document


class LLMChunk(BaseChunker):
    """
    Chunking strategy that uses an LLM to determine semantically meaningful boundaries.

    Not yet implemented. Intended to prompt a language model to identify natural
    split points in the document rather than relying on fixed rules or embeddings.
    """
    def __init__(self):
        raise NotImplementedError("LLM-based chunking not yet implemented")

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("LLM-based chunking not yet implemented")
