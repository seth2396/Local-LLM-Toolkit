# Local LLM Toolkit

A Python toolkit for local LLM inference, agentic workflows, and RAG (retrieval-augmented generation) pipelines.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (for local inference and embeddings)
- A model that supports tool use installed in Ollama (e.g. `llama3.1`, `mistral-nemo`)

## Installation

```bash
pip install -e .
```

---

## Package Structure

```
local_llm_toolkit/
├── agents/          # LLM-backed agent types
├── chunkers/        # Text chunking strategies
├── embedders/       # Embedding model wrappers
├── ingesters/       # Source crawlers (file system, web, SharePoint)
├── loaders/         # Document loaders by file type
├── pipelines/       # End-to-end ingest pipelines
└── vectorstores/    # Vector database wrappers
```

---

## Agents

Agents wrap an LLM and expose different interaction patterns.

| Class | Description |
|---|---|
| `ChatAgent` | Stateful multi-turn conversation |
| `StructuredOutputAgent` | Forces JSON output matching a Pydantic schema |
| `BinaryDecisionAgent` | Returns a yes/no decision with reasoning |
| `TaskAgent` | Executes a list of tasks in sequence |
| `OrchestratorAgent` | Routes tasks to specialist sub-agents |

```python
from local_llm_toolkit.agents import ChatAgent, Tool

# Basic chat
agent = ChatAgent(model="llama3.1")
response = agent.chat("Explain RAG in one paragraph.")

# With tools
def get_weather(location: str) -> str:
    return f"Sunny in {location}"

tool = Tool(name="get_weather", func=get_weather, description="Get weather for a location")
agent = ChatAgent(model="llama3.1", tools=[tool])
```

---

## Embedders

Wrappers around embedding models that produce fixed-size vectors from text.

| Class | Backend |
|---|---|
| `OllamaEmbedder` | Local Ollama model |
| `OpenAIEmbedder` | OpenAI API |

```python
from local_llm_toolkit.embedders import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")
vector = embedder.embed("Hello world")
vectors = embedder.embed_documents(["doc one", "doc two"])
```

---

## Ingesters

Ingesters crawl a source and return a list of `BaseItem` objects describing each discovered document. Items carry metadata but not content — loading is handled separately by loaders.

| Class | Source |
|---|---|
| `FileIngester` | Local file system (recursive BFS) |
| `WebIngester` | Web crawler (BFS, respects `robots.txt`) |
| `SharePointIngester` | Microsoft SharePoint via Graph API |

```python
from local_llm_toolkit.ingesters import FileIngester, WebIngester

# Collect files from one or more directories
ingester = FileIngester(roots=["./docs", "./data"])
items = ingester.collect()   # list[FileItem]

# Crawl a website up to 2 levels deep
ingester = WebIngester(root_url="https://example.com", max_depth=2)
items = ingester.collect()   # list[WebItem]
```

---

## Loaders

Loaders accept a `BaseItem` and return a `Document` with content populated.

| Class | Formats |
|---|---|
| `UniversalLoader` | Auto-selects by extension — use this by default |
| `PdfLoader` | `.pdf` |
| `DocxLoader` | `.docx` |
| `TextLoader` | `.txt` |
| `MarkdownLoader` | `.md` |
| `HTMLLoader` | `.html`, `WebItem` |
| `JSONLoader` | `.json` |
| `CSVLoader` | `.csv` |
| `ExcelLoader` | `.xlsx` |

```python
from local_llm_toolkit.loaders import UniversalLoader

loader = UniversalLoader()
document = loader.load(file_item)

print(document.metadata)   # source, doctype, hash_id, etc.
print(document.content)    # extracted text
```

`Document.content` automatically computes a SHA-256 `hash_id` on every assignment, which downstream vector stores use to skip re-embedding unchanged content.

---

## Chunkers

Chunkers split a `Document` into a list of `Chunk` objects for embedding.

| Class | Strategy |
|---|---|
| `UniversalChunker` | Auto-selects by file extension — use this by default |
| `FixedSizeChunk` | Fixed character-count windows with overlap |
| `RecursiveChunk` | Recursive separator-based splitting; separator list resolved from doctype |
| `SemanticChunk` | Groups sentences by cosine similarity; requires an embedder |
| `LLMChunk` | LLM-guided boundaries *(not yet implemented)* |
| `TableChunk` | Row-level splitting for tabular data *(not yet implemented)* |

```python
from local_llm_toolkit.chunkers import RecursiveChunk, SemanticChunk

# Separator list is auto-selected from document doctype
chunker = RecursiveChunk(max_tokens=500, overlap_tokens=50)
chunks = chunker.chunk(document)

# Semantic chunking requires an embedder
chunker = SemanticChunk(embedder=embedder, breakpoint_threshold=0.7)
chunks = chunker.chunk(document)
```

---

## Vector Stores

| Class | Backend |
|---|---|
| `ChromaVectorStore` | ChromaDB (ephemeral or persistent) |

Use the named constructors — do not instantiate directly.

```python
from local_llm_toolkit.vectorstores import ChromaVectorStore

# In-memory (lost on process exit)
store = ChromaVectorStore.ephemeral(embedder, collection_name="my_collection")

# Persistent (survives restarts)
store = ChromaVectorStore.persistent(embedder, path="./my_db", collection_name="my_collection")

# Query
results = store.query("What is RAG?", top_k=5, max_distance=0.8)

# Upsert (skips unchanged chunks, deletes stale ones)
store.upsert(chunks)
```

---

## Pipelines

Pipelines combine an ingester, loader, chunker, embedder, and vector store into a single `run()` call.

### FilePipeline

```python
from local_llm_toolkit.pipelines import FilePipeline
from local_llm_toolkit.embedders import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")

# Ephemeral (in-memory):
pipeline = FilePipeline.create("./docs", embedder)

# Persistent (on disk):
pipeline = FilePipeline.create(
    source="./docs",
    embedder=embedder,
    destination="./my_db",
    collection="my_docs"
)

pipeline.run()
```

`FilePipeline.create()` uses `UniversalLoader` and `UniversalChunker` by default. Pass `loader=` or `chunker=` to override.

### Manual pipeline

```python
from local_llm_toolkit.ingesters import FileIngester
from local_llm_toolkit.loaders import UniversalLoader
from local_llm_toolkit.chunkers import RecursiveChunk
from local_llm_toolkit.vectorstores import ChromaVectorStore
from local_llm_toolkit.embedders import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")
store = ChromaVectorStore.persistent(embedder, "./my_db", "docs")
loader = UniversalLoader()
chunker = RecursiveChunk(max_tokens=500, overlap_tokens=50)

for item in FileIngester(roots=["./docs"]).collect():
    document = loader.load(item)
    chunks = chunker.chunk(document)
    store.upsert(chunks)
```
