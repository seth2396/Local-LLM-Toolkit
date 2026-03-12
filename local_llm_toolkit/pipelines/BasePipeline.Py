from abc import ABC

from ..ingesters import BaseIngester, BaseItem
from ..loaders import BaseLoader
from ..chunkers import BaseChunker
from ..embedders import BaseEmbedder
from ..vectorstores import BaseVectorStore


class BasePipeline(ABC):
    """
    Abstract base class for ingestion pipelines.

    Orchestrates the processing flow: collect → load → chunk → embed → store.
    The ingester is passed to run(), keeping discovery decoupled from processing.

    Attributes:
        loader:      Loader used to extract content from items.
        chunker:     Chunker used to split documents into chunks.
        embedder:    Embedder used to generate vectors from chunk content.
        vectorstore: Vector store used to persist chunks and their embeddings.
    """

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        ingester: BaseIngester,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.ingester = ingester

    def run(self) -> dict:
        """
        Collect items from the ingester and process each one.

        Args:
            ingester: Any BaseIngester implementation (FileIngester, WebIngester, etc.)

        Returns:
            Aggregated upsert stats: {"added": int, "updated": int, "skipped": int, "deleted": int}
        """
        items = self.ingester.collect()
        return self._process_items(items)

    def process_item(self, item: BaseItem) -> dict:
        """
        Load, chunk, and upsert a single item.

        Args:
            item: Any BaseItem returned by an ingester.

        Returns:
            Upsert stats: {"added": int, "updated": int, "skipped": int, "deleted": int}
        """
        document = self.loader.load(item)
        chunks = self.chunker.chunk(document)
        return self.vectorstore.upsert(chunks)

    def _process_items(self, items: list[BaseItem]) -> dict:
        totals = {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}
        for item in items:
            result = self.process_item(item)
            for key in totals:
                totals[key] += result.get(key, 0)
        return totals
