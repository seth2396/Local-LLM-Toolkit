from abc import ABC, abstractmethod
from .Document import Document
from ..ingesters import BaseItem

class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    Subclasses implement load() for a specific file or item type. The
    load_documents() convenience method calls load() for each item in a list.
    """

    @abstractmethod
    def load(self, item: BaseItem) -> Document:
        """
        Load a single item and return a populated Document.

        Args:
            item: A BaseItem (FileItem, WebItem, etc.) describing the source.

        Returns:
            Document: Loaded document with content and metadata populated.
        """
        raise NotImplementedError("Load method not implemented.")

    def load_documents(self, items: list[BaseItem]) -> list[Document]:
        """Load multiple items and return a list of Documents."""
        return [self.load(item) for item in items]
