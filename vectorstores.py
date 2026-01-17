from typing import Union
import chromadb
from embedders import BaseEmbedder


class BaseVectorStore:
    """
        Abstract base class for vector stores.

        This is a minimal placeholder. Replace or extend with your interface
        used across your project.
    """
    def __init__(self, embedder: BaseEmbedder ) -> None:
        raise NotImplementedError()

    def add(self, content: list[str] | str, embedding_content: list[str] | str, metadata: list[dict] | dict = None) -> None:
        """
            Adds content to the vector database. 

            Params: 
                content: Text content to be embedded and stored
                embed_content: Text content to be embedded if embedding is to differ from content in the entry
                metadata: Any metadata to be stored with the entry
        """
        raise NotImplementedError()

    def query(self, embeddings: Union[list[float], list[list[float]]], top_k: int = 55) -> list[dict]:
        raise NotImplementedError()
    
class ChromaVectorStore(BaseVectorStore):
    """
        Vector store backed by ChromaDB collections.

        Supports both in-memory and persistent ChromaDB clients.

        By default, this class uses a persistent client so data is stored on disk
        and survives process restarts. You can switch to in-memory for ephemeral
        use cases by setting `persist=False`.

        Attributes:
            client_type: The ChromaDB client instance (PersistentClient or EphemeralClient).
            collection: The n ChromaDB collection used to store documents and embeddings.
            collaction_location: The path string where the DB should be stored 
    """
    def __init__(self, embedder: BaseEmbedder, collection_name: str = "default",  client_type: str = "ephemeral", collection_location: str = "./ChromaVectorStore"):
        if client_type == "persistent":
            client = chromadb.PersistentClient(collection_location)
        else:
            client = chromadb.EphemeralClient()
        self.client = client
        self.embedder = embedder
        self.collection = self.client.get_or_create_collection(name=collection_name)

        
    def add(self, content: list[str] | str, embedding_content: list[str] | str = None, metadata: list[dict] | dict = None) -> None:
        #Id looks like it is just assigned by position of list. Might be good to refine it so that it is more precise. 
        current_row_count = self.count()
        if isinstance(content, str):
            ids = [str(current_row_count + 1)]
        else:
            ids = [str(i + current_row_count) for i in range(len(content))]

        if embedding_content:
            embedded_texts = self.embedder.embed(embedding_content)
        else: 
            embedded_texts = self.embedder.embed(content)

        self.collection.add(ids=ids, documents=content, embeddings = embedded_texts, metadatas=metadata )
    

    
    def query(self, query: str, top_k: int = 5, include: list[str] = ['documents'] ) -> list[dict]: 
        """
            Query the vector database:
            
            parameters: 
                query: String to query against
                top_k: number of results to return
                include: Type of results to return ['ids', 'documents', 'metadatas', 'embeddings'] default = ['documents']

            returns: 
                {ids: List[ID]
                embeddings: Optional[Embeddings],
                documents: Optional[List[Document]],
                metadatas: Optional[List[Metadata]]
                included: Include}
        """

        embedded_query = self.embedder.embed(query)
        results = self.collection.query(query_embeddings=embedded_query, n_results=top_k, include = include)
        return results


    def count(self):
        return self.collection.count()