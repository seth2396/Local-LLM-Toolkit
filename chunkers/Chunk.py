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
