import uuid
import pytest
from unittest.mock import MagicMock
from local_llm_toolkit.chunkers.Chunk import Chunk
from local_llm_toolkit.vectorstores.ChromaVectorStore import ChromaVectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_embedder(dim=4):
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * dim
    embedder.embed_documents.return_value = [[0.1] * dim]
    return embedder

def make_chunk(content, source="file.txt", hash_id="abc123", **extra_meta):
    return Chunk(
        content=content,
        metadata={"source": source, "hash_id": hash_id, **extra_meta}
    )

def make_store(embedder=None):
    """Each call gets a unique collection name to avoid shared EphemeralClient state."""
    embedder = embedder or make_embedder()
    return ChromaVectorStore.ephemeral(embedder, collection_name=str(uuid.uuid4()))


# ── upsert: basic add ─────────────────────────────────────────────────────────

def test_upsert_returns_added_count():
    store = make_store()
    chunks = [make_chunk("hello world sentence one", source="doc.txt", hash_id="h1")]
    result = store.upsert(chunks)
    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["skipped"] == 0
    assert result["deleted"] == 0

def test_upsert_empty_list_returns_zeros():
    store = make_store()
    result = store.upsert([])
    assert result == {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}

def test_upsert_adds_chunks_to_collection():
    store = make_store()
    chunks = [make_chunk("content one", source="doc.txt", hash_id="h1")]
    store.upsert(chunks)
    assert store.count() == 1

def test_upsert_multiple_chunks_same_source():
    embedder = make_embedder()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    store = make_store(embedder)
    chunks = [
        make_chunk("first chunk content", source="doc.txt", hash_id="h1"),
        make_chunk("second chunk content", source="doc.txt", hash_id="h2"),
    ]
    result = store.upsert(chunks)
    assert result["added"] == 2
    assert store.count() == 2


# ── upsert: skip unchanged ────────────────────────────────────────────────────

def test_upsert_skips_unchanged_chunk():
    store = make_store()
    chunks = [make_chunk("same content", source="doc.txt", hash_id="same_hash")]
    store.upsert(chunks)
    result = store.upsert(chunks)
    assert result["skipped"] == 1
    assert result["added"] == 0

def test_upsert_updates_changed_chunk():
    store = make_store()
    chunk_v1 = make_chunk("original content", source="doc.txt", hash_id="hash_v1")
    store.upsert([chunk_v1])

    chunk_v2 = make_chunk("updated content", source="doc.txt", hash_id="hash_v2")
    result = store.upsert([chunk_v2])
    assert result["updated"] == 1
    assert result["added"] == 0


# ── upsert: stale chunk deletion ─────────────────────────────────────────────

def test_upsert_deletes_stale_chunks():
    embedder = make_embedder()
    embedder.embed_documents.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    store = make_store(embedder)

    # First upsert: 2 chunks
    store.upsert([
        make_chunk("chunk one", source="doc.txt", hash_id="h1"),
        make_chunk("chunk two", source="doc.txt", hash_id="h2"),
    ])
    assert store.count() == 2

    # Second upsert: only 1 chunk — stale second chunk should be deleted
    result = store.upsert([make_chunk("chunk one", source="doc.txt", hash_id="h1")])
    assert result["deleted"] == 1
    assert store.count() == 1


# ── upsert: ID scoping by source ──────────────────────────────────────────────

def test_upsert_different_sources_do_not_interfere():
    embedder = make_embedder()
    embedder.embed_documents.return_value = [[0.1] * 4]
    store = make_store(embedder)

    store.upsert([make_chunk("content from doc a", source="a.txt", hash_id="ha")])
    result = store.upsert([make_chunk("content from doc b", source="b.txt", hash_id="hb")])

    assert result["added"] == 1
    assert result["deleted"] == 0
    assert store.count() == 2
