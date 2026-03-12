import chromadb

from ..embedders import BaseEmbedder
from ..chunkers import Chunk
from .BaseVectorStore import BaseVectorStore

from typing import Optional


class ChromaVectorStore(BaseVectorStore):
    """
        Vector store backed by ChromaDB collections.

        Use the named constructors instead of instantiating directly:
            ChromaVectorStore.ephemeral(embedder)
            ChromaVectorStore.persistent(embedder, path="./my_db")

        Attributes:
            embedder: Embedder used to generate vectors.
            collection: The ChromaDB collection used to store documents and embeddings.
    """
    def __init__(self, embedder: BaseEmbedder, collection_name: str, client: chromadb.Client):
        """
            Initialise the vector store. Prefer the named constructors over calling this directly.

            Params:
                embedder: Embedder used to generate vectors for stored and queried content.
                collection_name: Name of the ChromaDB collection to open or create.
                client: A pre-built ChromaDB client (EphemeralClient or PersistentClient).
        """
        self.embedder = embedder
        self.collection = client.get_or_create_collection(name=collection_name)

    @classmethod
    def ephemeral(cls, embedder: BaseEmbedder, collection_name: str = "default") -> "ChromaVectorStore":
        """
            Create an in-memory vector store. Data is lost when the process exits.

            Params:
                embedder: Embedder used to generate vectors.
                collection_name: Name of the collection to open or create. Defaults to 'default'.

            Returns:
                A ChromaVectorStore backed by an in-memory ChromaDB client.
        """
        return cls(embedder, collection_name, chromadb.EphemeralClient())

    @classmethod
    def persistent(cls, embedder: BaseEmbedder, path: str = "./ChromaVectorStore", collection_name: str = "default") -> "ChromaVectorStore":
        """
            Create a persistent vector store. Data is written to disk and survives process restarts.
            If a database already exists at the given path it will be opened; otherwise a new one is created.

            Params:
                embedder: Embedder used to generate vectors. Must produce the same dimension as when
                          the collection was first created, otherwise ChromaDB will raise.
                path: Directory where the ChromaDB database is stored. Defaults to './ChromaVectorStore'.
                collection_name: Name of the collection to open or create. Defaults to 'default'.

            Returns:
                A ChromaVectorStore backed by a persistent ChromaDB client.
        """
        return cls(embedder, collection_name, chromadb.PersistentClient(path))

    def add(self, content: list[str] | str, embedding_content: list[str] | str = None, metadata: list[dict] | dict = None) -> None:
        current_row_count = self.count()
        if isinstance(content, str):
            ids = [str(current_row_count + 1)]
        else:
            ids = [str(i + current_row_count) for i in range(len(content))]

        if embedding_content:
            embedded_texts = self.embedder.embed(embedding_content)
        else:
            embedded_texts = self.embedder.embed(content)

        self.collection.add(ids=ids, documents=content, embeddings=embedded_texts, metadatas=metadata)


    def query(
        self,
        query: str,
        top_k: int = 5,
        include: Optional[list[str]] = None,
        max_distance: float = 1.0,
        overfetch: float = 1.0,  # float multiplier
    ) -> list[dict]:
        """
        Query the vector database and filter results by distance threshold.

        Args:
            query: text query
            top_k: number of results to return (after filtering)
            include: fields to return ['ids','documents','metadatas','embeddings','distances']
            max_distance: keep results where distance <= max_distance
            overfetch: multiplier for initial retrieval; may be float.
                      n_results is computed by rounding to nearest int.

        Returns:
            List[dict] hits sorted by distance asc, filtered by max_distance.
        """
        if top_k <= 0:
            return []

        if include is None:
            include = ["documents"]

        # Don't mutate caller list
        include = list(include)

        # Ensure distances present for filtering
        if "distances" not in include:
            include.append("distances")

        # Compute n_results as an int from float overfetch
        # Round to nearest int, then clamp to at least top_k
        n_results = int(round(top_k * float(overfetch)))
        n_results = max(top_k, n_results, 1)

        embedded_query = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=embedded_query,
            n_results=n_results,
            include=include,
        )

        # Chroma returns list-of-lists for single query: [ [...hits...] ]
        distances = results.get("distances", [[]])[0] or []

        hits = []
        num_hits = len(distances)

        for i in range(num_hits):
            d = distances[i]
            if d is None or d > max_distance:
                continue

            hit = {"distance": d}

            # Add requested fields (where present)
            if "ids" in results and results["ids"] is not None:
                hit["id"] = results["ids"][0][i]
            if "documents" in results and results["documents"] is not None:
                hit["document"] = results["documents"][0][i]
            if "metadatas" in results and results["metadatas"] is not None:
                hit["metadata"] = results["metadatas"][0][i]
            if "embeddings" in results and results["embeddings"] is not None:
                hit["embedding"] = results["embeddings"][0][i]

            hits.append(hit)

        # Sort by distance (smaller is closer for typical distance metrics)
        hits.sort(key=lambda x: x["distance"])

        return hits[:top_k]


    def count(self) -> int:
        """Returns the total number of entries in the collection."""
        return self.collection.count()

    def upsert(self, chunks: list[Chunk]) -> dict:
        """
            Add or update chunks from a single document, skipping any whose content has not changed.
            Stale chunk positions (from a document that now produces fewer chunks) are deleted.

            Chunk identity is based on a stable ID derived from the source file name and chunk
            position. Content change is detected by comparing the hash_id stored in metadata.

            Params:
                chunks: List of Chunk objects from a single document. Each chunk's metadata
                        must include 'file_name' and 'hash_id'.

            Returns:
                {"added": int, "updated": int, "skipped": int, "deleted": int}
        """
        if not chunks:
            return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}

        source_file = chunks[0].metadata.get("file_name", "unknown")
        new_ids = [f"{source_file}_chunk_{i}" for i in range(len(chunks))]

        existing = self.collection.get(where={"file_name": source_file}, include=["metadatas"])
        existing_ids = set(existing["ids"])
        existing_hash_map = {
            id_: meta.get("hash_id")
            for id_, meta in zip(existing["ids"], existing["metadatas"])
        }

        to_upsert_ids, to_upsert_contents, to_upsert_metadatas = [], [], []
        skipped = 0

        for chunk_id, chunk in zip(new_ids, chunks):
            if existing_hash_map.get(chunk_id) == chunk.metadata.get("hash_id"):
                skipped += 1
            else:
                to_upsert_ids.append(chunk_id)
                to_upsert_contents.append(chunk.content)
                to_upsert_metadatas.append(chunk.metadata)

        added, updated = 0, 0
        if to_upsert_ids:
            embeddings = self.embedder.embed_documents(to_upsert_contents)
            self.collection.upsert(
                ids=to_upsert_ids,
                documents=to_upsert_contents,
                embeddings=embeddings,
                metadatas=to_upsert_metadatas
            )
            for id_ in to_upsert_ids:
                if id_ in existing_ids:
                    updated += 1
                else:
                    added += 1

        stale_ids = existing_ids - set(new_ids)
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))

        return {"added": added, "updated": updated, "skipped": skipped, "deleted": len(stale_ids)}
