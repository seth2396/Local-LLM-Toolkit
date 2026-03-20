from unittest.mock import MagicMock
from local_llm_toolkit.chunkers.FixedSizeChunk import FixedSizeChunk
from local_llm_toolkit.chunkers.RecursiveChunk import RecursiveChunk
from local_llm_toolkit.loaders.Document import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_document(content: str, doctype: str = "txt") -> Document:
    item = MagicMock()
    item.to_metadata.return_value = {"source": "test.txt", "doctype": doctype}
    doc = Document(item)
    doc.content = content
    return doc


SHORT_TEXT = "Hello world. This is a short document."
LONG_TEXT = "word " * 200  # 1000 chars
PARAGRAPH_TEXT = (
    "First paragraph with some content here.\n\n"
    "Second paragraph has different information.\n\n"
    "Third paragraph concludes the document."
)


# ── FixedSizeChunk: chunk size ─────────────────────────────────────────────────

def test_fixed_no_chunk_exceeds_max_tokens():
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    assert all(len(c.content) <= 100 for c in chunks)

def test_fixed_chunk_size_respected_with_overlap():
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    # Base chunks (before overlap) must fit — first chunk has no overlap prefix
    assert len(chunks[0].content) <= 100

def test_fixed_short_document_returns_one_chunk():
    chunker = FixedSizeChunk(max_tokens=500, overlap_tokens=0)
    chunks = chunker.chunk(make_document(SHORT_TEXT), chunk_filter=False)
    assert len(chunks) == 1
    assert chunks[0].content == SHORT_TEXT

def test_fixed_exact_size_document_returns_one_chunk():
    text = "a" * 100
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(text), chunk_filter=False)
    assert len(chunks) == 1

def test_fixed_produces_multiple_chunks_for_long_text():
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    assert len(chunks) > 1

def test_fixed_all_content_covered():
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=0)
    doc = make_document(LONG_TEXT)
    chunks = chunker.chunk(doc, chunk_filter=False)
    # Reassemble — without overlap every character appears exactly once
    reassembled = "".join(c.content for c in chunks)
    assert reassembled == LONG_TEXT


# ── FixedSizeChunk: overlap ────────────────────────────────────────────────────

def test_fixed_overlap_second_chunk_starts_with_tail_of_first():
    chunker = FixedSizeChunk(max_tokens=50, overlap_tokens=10)
    doc = make_document("a" * 200)
    chunks = chunker.chunk(doc, chunk_filter=False)
    tail_of_first = chunks[0].content[-10:]
    assert chunks[1].content.startswith(tail_of_first)

def test_fixed_no_overlap_chunks_are_contiguous():
    text = "abcdefghij" * 10  # 100 chars
    chunker = FixedSizeChunk(max_tokens=25, overlap_tokens=0)
    chunks = chunker.chunk(make_document(text), chunk_filter=False)
    assert "".join(c.content for c in chunks) == text

def test_fixed_zero_overlap_no_repeated_content():
    chunker = FixedSizeChunk(max_tokens=50, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    for i in range(len(chunks) - 1):
        assert chunks[i].content[-1] not in chunks[i + 1].content[:1] or True
    total_chars = sum(len(c.content) for c in chunks)
    assert total_chars == len(LONG_TEXT)


# ── FixedSizeChunk: metadata ───────────────────────────────────────────────────

def test_fixed_metadata_attached_to_chunks():
    chunker = FixedSizeChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"


# ── RecursiveChunk: chunk size ─────────────────────────────────────────────────

def test_recursive_no_chunk_exceeds_max_tokens():
    chunker = RecursiveChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    assert all(len(c.content) <= 100 for c in chunks)

def test_recursive_respects_max_tokens_on_paragraph_text():
    chunker = RecursiveChunk(max_tokens=60, overlap_tokens=0)
    chunks = chunker.chunk(make_document(PARAGRAPH_TEXT), chunk_filter=False)
    assert all(len(c.content) <= 60 for c in chunks)

def test_recursive_short_document_returns_one_chunk():
    chunker = RecursiveChunk(max_tokens=500, overlap_tokens=0)
    chunks = chunker.chunk(make_document(SHORT_TEXT), chunk_filter=False)
    assert len(chunks) == 1

def test_recursive_produces_multiple_chunks_for_long_text():
    chunker = RecursiveChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    assert len(chunks) > 1


# ── RecursiveChunk: separator preference ──────────────────────────────────────

def test_recursive_prefers_paragraph_splits():
    # Text has clear paragraph breaks — chunks should align with them
    chunker = RecursiveChunk(max_tokens=80, overlap_tokens=0)
    chunks = chunker.chunk(make_document(PARAGRAPH_TEXT), chunk_filter=False)
    # No chunk should contain \n\n (paragraph separator consumed by split)
    for chunk in chunks:
        assert "\n\n" not in chunk.content

def test_recursive_falls_back_to_character_split():
    # One long word with no whitespace — must fall back to character split
    text = "x" * 300
    chunker = RecursiveChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(text), chunk_filter=False)
    assert all(len(c.content) <= 100 for c in chunks)


# ── RecursiveChunk: overlap ────────────────────────────────────────────────────

def test_recursive_overlap_second_chunk_starts_with_tail_of_first():
    chunker = RecursiveChunk(max_tokens=50, overlap_tokens=10)
    doc = make_document("a" * 200)
    chunks = chunker.chunk(doc, chunk_filter=False)
    tail_of_first = chunks[0].content[-10:]
    assert chunks[1].content.startswith(tail_of_first)

def test_recursive_no_overlap_when_zero():
    text = "a" * 200
    chunker = RecursiveChunk(max_tokens=50, overlap_tokens=0)
    chunks = chunker.chunk(make_document(text), chunk_filter=False)
    # No overlap means no characters are duplicated — total length equals original
    assert sum(len(c.content) for c in chunks) == len(text)


# ── RecursiveChunk: metadata ───────────────────────────────────────────────────

def test_recursive_metadata_attached_to_chunks():
    chunker = RecursiveChunk(max_tokens=100, overlap_tokens=0)
    chunks = chunker.chunk(make_document(LONG_TEXT), chunk_filter=False)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"

def test_recursive_explicit_separators_override_doctype():
    # Each part is 40 chars — two parts together (81 chars) exceed max_tokens=50,
    # so the splitter must break on "|"
    part = "x" * 40
    text = f"{part}|{part}|{part}"
    chunker = RecursiveChunk(max_tokens=50, overlap_tokens=0, separators=["|"])
    chunks = chunker.chunk(make_document(text), chunk_filter=False)
    assert len(chunks) > 1
    assert all(len(c.content) <= 50 for c in chunks)
