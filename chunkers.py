from pathlib import Path

_DEFAULET_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 100


#TODO: add method/interface for using a tokenizer to set max character limits and overlap by token count, not just raw character count
#TODO: implement recursive chunking strategy
#TODO: implement semantic chunking strategygio
#TODO: implement LLM based chunking strategy


class Chunk:
    """
        Data class for holding chunk information

        Attributes:
            index (int): index value of the chunk in the document
            content (str): string representation of the content stored in the chunk
            metadata (dict): dictionary containing metadata for the chunk which should contain source information
    """
    def __init__(self, content: str, index: int = None, metadata: dict = None):
        self.index = index
        self.content = content
        self.metadata = metadata if metadata else {}

class BaseChunkStrategy:
    """
        Base class for text chunking strategies.

        This class defines the interface and common attributes for chunking
        large text data into smaller segments. Subclasses should implement
        the `chunk` method to provide specific chunking logic.

        Attributes:
            chunk_size (int): Maximum number of characters (or tokens) per chunk default is 500 characters.
            overlap (int): Number of characters (or tokens) to overlap between chunks default is 100 characters.
            chunks (list): Stores the generated chunks after processing.

        Example:
            >>> strategy = MyChunkStrategy(chunk_size=500, overlap=50)
            >>> chunks = strategy.chunk(document: Document) -> list[Chunk]
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50, chunk_strategy: dict = None):
        self.chunk_size = chunk_size
        self.overlap = overlap  
        self.chunks = []

    def chunk(self, document) -> list[Chunk]:
        raise NotImplementedError("Chunk method must be implemented by subclasses")

class FixedSizeChunkStrategy(BaseChunkStrategy):
    """
        Chunking strategy that splits text into fixed-size segments with optional overlap.

        This strategy produces sequential chunks of a specified maximum token size.
        Each chunk may optionally overlap with the previous one, which can improve
        embedding coherence when used in retrieval pipelines.

        Attributes:
            max_tokens (int): Maximum number of tokens (or characters) per chunk.
                document (Document): A document object containing `content` (str or overlap_tokens (int): Number of tokens (or characters) that overlap
                    convertible to str) and `metadata` (dict).

            Returns:
                list[Chunk]: A list of `Chunk` objects, each containing a segment of
                    text and inherited metadata from the original document.

            Raises:
                ValueError: If `document.content` is not a string and cannot be
                    converted to one.

            Example:
                >>> strategy = FixedSizeChunkStrategy(max_tokens=100, overlap_tokens=20)
                >>> chunks = strategy.chunk(document)
                >>> len(chunks) = 5
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
        Recursive Chunking Strategy - Not yet implemented.
    """
    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        raise NotImplementedError("recursive chunking not yet implemented")

class SemanticChunk(RecursiveChunk):
    """
        Semantic chunk strategy - Not yet implemented.
    """
    def __init__(self, max_tokens: int = _DEFAULET_CHUNK_SIZE, overlap_tokens: int = _DEFAULT_OVERLAP):
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        raise NotImplementedError("Semantic chunking not yet implemented")

class LLMChunk(BaseChunkStrategy):
    """
        LLM based chunker - Not yet implemented.
    """
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