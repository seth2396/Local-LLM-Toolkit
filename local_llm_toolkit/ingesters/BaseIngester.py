from abc import ABC, abstractmethod


class BaseIngester(ABC):
    """
    Abstract base class for ingesters.

    An ingester's responsibility is to discover and return a collection of items
    from a source (filesystem, S3, database, web, etc.) for downstream loading.
    """

    @abstractmethod
    def collect(self) -> list:
        """
        Discover and return all items from the source.

        Returns:
            A list of items to be passed to a loader. The concrete type
            depends on the ingester implementation (e.g., list[FileItem]).
        """
        pass
