from pathlib import Path

_DEFAULET_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 50


class Chunk:

    def __init__(self, content: str, index: int = None, metadata: dict = None):
        self.index = index
        self.content = content
        self.metadata = metadata if metadata else {}

class BaseChunkStrategy:
    """Base class for text chunking strategies.

    This class defines the interface and common attributes for chunking
    large text data into smaller segments. Subclasses should implement
    the `chunk` method to provide specific chunking logic.

    Attributes:
        chunk_size (int): Maximum number of characters (or tokens) per chunk.
        overlap (int): Number of characters (or tokens) to overlap between chunks.
        chunks (list): Stores the generated chunks after processing.

    Example:
        >>> strategy = MyChunkStrategy(chunk_size=500, overlap=50)
        >>> chunks = strategy.chunk(document: Document)
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50, chunk_strategy: dict = None):
        self.chunk_size = chunk_size
        self.overlap = overlap  
        self.chunks = []

    def chunk(self, document) -> list[Chunk]:
        raise NotImplementedError("Chunk method must be implemented by subclasses")

class FixedSizeChunkStrategy(BaseChunkStrategy):
    """
    Fixed chunk strategy that splits text into fixed-size chunks with optional overlap.
    """
    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        super().__init__(chunk_size=max_tokens, overlap=overlap_tokens)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunks = []

    def chunk(self, document) -> list[Chunk]:
        if not isinstance(document.content, str):
            try:
                text = str(document.content)
            except Exception as e:
                raise ValueError(f"FixedSizeChunkStrategy can only support string content. The provided document content is of type {type(document.content)} and cannot be converted to string. \n Error: {e}")
        else:
            text = document.content

        chunks = []
        start = 0
        end = self.max_tokens
        while start < len(text):
            chunk_text = text[start:end]
            chunks.append(Chunk(content=chunk_text, metadata=document.metadata))

            # Move forward with overlap
            start = end - self.overlap_tokens
            end = start + self.max_tokens

        return chunks

class RecursiveChunk(BaseChunkStrategy):
    """
        if not is is_iterable()
            if size > max_size:
                split_chunk

        for item in iterable: 
            recurse(iterable)
    """
    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        raise NotImplementedError("recursive chunking not yet implemented")

class SemanticChunk(RecursiveChunk):
    """
    Semantic chunk strategy that uses recursive chunking based on content semantics. Goes only as deep as it needs to to break up chunks. Keeps track of hierarchy.
    Inherits from RecursiveChunkStrategy.
    """
    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        raise NotImplementedError("Semantic chunking not yet implemented")

class LLMChunk(BaseChunkStrategy):

    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        raise NotImplementedError("LLM-based chunking not yet implemented")

class TableChunk(BaseChunkStrategy):

    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        raise NotImplementedError("Table-based chunking not yet implemented")
    

STRATEGY_REGISTRY = {
    "fixed": FixedSizeChunkStrategy,
    "recursive": RecursiveChunk,
    "semantic": SemanticChunk,
    "llm": LLMChunk,
    "table": TableChunk}

DEFAULT_STRATEGY_FOR_TYPE = {
    '.pdf': FixedSizeChunkStrategy, 
    '.docx': FixedSizeChunkStrategy,
    '.txt': FixedSizeChunkStrategy,
    '.md': FixedSizeChunkStrategy,
    '.html': FixedSizeChunkStrategy, 
    '.json': RecursiveChunk,
    '.csv': TableChunk,
    '.xlsx': TableChunk}

class UniversalChunker(BaseChunkStrategy):

    def chunk(self, document, chunk_strategy: dict = None) -> list[Chunk]:
        #check if a chunking_strategy has been provided and return the correct strategy, if not use default
        extension = document.metadata['extension']
        if chunk_strategy and extension in chunk_strategy:
            strategy = STRATEGY_REGISTRY[chunk_strategy[extension]]
        elif extension in DEFAULT_STRATEGY_FOR_TYPE:
            strategy =  DEFAULT_STRATEGY_FOR_TYPE[extension]
        else:
            raise NotImplementedError(f"{extension} Not found in chunk_strategy dict or default dictionary. \nPlease add {extension} to chunk_strategy dict.")

        selected_class = strategy()
        return selected_class.chunk(document)

if __name__ == "__main__":
    from loaders import Document
    doc = Document()
    chunker = UniversalChunker().chunk()