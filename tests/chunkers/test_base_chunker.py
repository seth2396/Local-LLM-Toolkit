import pytest
from unittest.mock import MagicMock
from local_llm_toolkit.chunkers.BaseChunker import BaseChunker
from local_llm_toolkit.chunkers.Chunk import Chunk
from local_llm_toolkit.chunkers.ChunkFilter import ChunkFilter


# ── Minimal concrete implementation ───────────────────────────────────────────

class PassthroughChunker(BaseChunker):
    """Returns one chunk per call with fixed content."""
    def __init__(self, chunks):
        self._chunks = chunks

    def _chunk(self, document):
        return self._chunks


def make_document(content="hello world this is a test document"):
    doc = MagicMock()
    doc.content = content
    doc.metadata = {"source": "test.txt"}
    return doc

def make_chunk(content):
    return Chunk(content=content, metadata={"source": "test.txt"})


# ── Template method ───────────────────────────────────────────────────────────

def test_chunk_calls_inner_chunk():
    raw = [make_chunk("hello world sentence here")]
    chunker = PassthroughChunker(raw)
    result = chunker.chunk(make_document(), chunk_filter=None)
    assert result == raw

def test_chunk_applies_default_filter():
    # empty chunk should be dropped by DEFAULT_CHUNK_FILTER
    chunks = [make_chunk(""), make_chunk("valid content passes filter easily")]
    chunker = PassthroughChunker(chunks)
    result = chunker.chunk(make_document())
    assert len(result) == 1
    assert result[0].content == "valid content passes filter easily"

def test_chunk_applies_custom_filter():
    chunks = [make_chunk("short"), make_chunk("long enough content to pass the filter")]
    chunker = PassthroughChunker(chunks)
    custom = ChunkFilter(min_length=20, min_alpha_ratio=None, max_digit_ratio=None,
                         max_symbol_ratio=None, min_word_count=None,
                         min_avg_word_length=None, max_avg_word_length=None)
    result = chunker.chunk(make_document(), chunk_filter=custom)
    assert len(result) == 1
    assert "long enough" in result[0].content

def test_chunk_no_filter_when_false():
    chunks = [make_chunk(""), make_chunk("x")]
    chunker = PassthroughChunker(chunks)
    result = chunker.chunk(make_document(), chunk_filter=False)
    assert len(result) == 2

def test_chunk_none_uses_default_filter():
    chunks = [make_chunk(""), make_chunk("valid content passes filter easily")]
    chunker = PassthroughChunker(chunks)
    result = chunker.chunk(make_document(), chunk_filter=None)
    assert len(result) == 1


# ── chunk_documents ───────────────────────────────────────────────────────────

def test_chunk_documents_returns_one_list_per_document():
    raw = [make_chunk("valid sentence content here")]
    chunker = PassthroughChunker(raw)
    docs = [make_document(), make_document()]
    result = chunker.chunk_documents(docs, chunk_filter=None)
    assert len(result) == 2

def test_chunk_documents_applies_filter_to_each():
    chunks = [make_chunk(""), make_chunk("good content passes filter check")]
    chunker = PassthroughChunker(chunks)
    docs = [make_document(), make_document()]
    result = chunker.chunk_documents(docs)
    assert all(len(r) == 1 for r in result)


# ── Abstract enforcement ──────────────────────────────────────────────────────

def test_cannot_instantiate_base_chunker_directly():
    with pytest.raises(TypeError):
        BaseChunker()
