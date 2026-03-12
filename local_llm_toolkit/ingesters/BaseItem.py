from abc import ABC, abstractmethod


class BaseItem(ABC):
    """
    Abstract base for all ingester item types.

    Each ingester returns a list of items that implement this interface.
    The to_metadata() method serializes the item's fields into a flat dict
    suitable for storage as Document metadata in a vector database.
    """

    @abstractmethod
    def to_metadata(self) -> dict:
        """
        Return a flat dict of this item's fields, omitting None values
        and converting any non-primitive types (e.g. Path) to str.
        """
        pass
