from pathlib import Path
from typing import Optional

from .BasePipeline import BasePipeline
from ..ingesters import FileIngester
from ..loaders import UniversalLoader, BaseLoader
from ..chunkers import UniversalChunker, BaseChunker
from ..embedders import BaseEmbedder
from ..vectorstores import ChromaVectorStore


class FilePipeline(BasePipeline):
    """
    Pipeline for ingesting local files into a vector store.

    Works out of the box with just a source and an embedder.
    Defaults to UniversalLoader and UniversalChunker.

    Example:
        # Ephemeral (in-memory):
        pipeline = FilePipeline.create("./docs", embedder)

        # Persistent (on disk):
        pipeline = FilePipeline.create("./docs", embedder,
                                       destination="./my_db", collection="my_docs")

        pipeline.run()
    """

    @classmethod
    def create(
        cls,
        source: str | Path,
        embedder: BaseEmbedder,
        destination: Optional[str | Path] = None,
        collection: str = "default",
        loader: BaseLoader = None,
        chunker: BaseChunker = None,
    ) -> "FilePipeline":
        """
        Construct a FilePipeline from a source directory.

        Args:
            source:      Root directory to ingest files from.
            embedder:    Embedder used to generate vectors.
            destination: Path for persistent storage. If None, uses ephemeral (in-memory) storage.
            collection:  Vector store collection name. Defaults to 'default'.
            loader:      Loader to use. Defaults to UniversalLoader.
            chunker:     Chunker to use. Defaults to UniversalChunker.

        Returns:
            A configured FilePipeline ready to run.
        """
        loader = loader or UniversalLoader()
        chunker = chunker or UniversalChunker()
        ingester = FileIngester(roots=[source])
        vectorstore = (
            ChromaVectorStore.persistent(embedder, str(destination), collection)
            if destination
            else ChromaVectorStore.ephemeral(embedder, collection)
        )
        return cls(loader, chunker, embedder, vectorstore, ingester)
