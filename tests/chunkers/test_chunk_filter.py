import pytest
from local_llm_toolkit.chunkers.Chunk import Chunk
from local_llm_toolkit.chunkers.ChunkFilter import ChunkFilter, DEFAULT_CHUNK_FILTER


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(content: str) -> Chunk:
    return Chunk(content=content, metadata={"source": "test"})

def make_chunks(*contents) -> list[Chunk]:
    return [make_chunk(c) for c in contents]


# ── DEFAULT_CHUNK_FILTER ───────────────────────────────────────────────────────

def test_default_filter_is_chunk_filter_instance():
    assert isinstance(DEFAULT_CHUNK_FILTER, ChunkFilter)

def test_default_filter_passes_normal_text():
    chunks = make_chunks("This is a normal sentence with enough content to pass.")
    assert len(DEFAULT_CHUNK_FILTER.filter(chunks)) == 1


# ── min_length / max_length ───────────────────────────────────────────────────

def test_drops_chunk_below_min_length():
    f = ChunkFilter(min_length=20, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    assert f.filter(make_chunks("short")) == []

def test_keeps_chunk_at_min_length():
    f = ChunkFilter(min_length=5, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("hello"))) == 1

def test_drops_chunk_above_max_length():
    f = ChunkFilter(max_length=10, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    assert f.filter(make_chunks("this is way too long for the max")) == []

def test_no_max_length_keeps_long_chunks():
    f = ChunkFilter(max_length=None, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("x" * 10_000))) == 1


# ── strip_whitespace ──────────────────────────────────────────────────────────

def test_strips_whitespace_from_content():
    f = ChunkFilter(min_length=3, strip_whitespace=True, min_alpha_ratio=None,
                    max_digit_ratio=None, max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    result = f.filter(make_chunks("  hello  "))
    assert result[0].content == "hello"

def test_no_strip_preserves_whitespace():
    f = ChunkFilter(min_length=3, strip_whitespace=False, min_alpha_ratio=None,
                    max_digit_ratio=None, max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    result = f.filter(make_chunks("  hello  "))
    assert result[0].content == "  hello  "

def test_whitespace_only_chunk_dropped():
    assert DEFAULT_CHUNK_FILTER.filter(make_chunks("     ")) == []


# ── alpha_ratio ───────────────────────────────────────────────────────────────

def test_drops_chunk_below_min_alpha_ratio():
    f = ChunkFilter(min_alpha_ratio=0.8, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    # "12345 abc" — mostly digits, low alpha ratio
    assert f.filter(make_chunks("12345 abc")) == []

def test_keeps_chunk_above_min_alpha_ratio():
    f = ChunkFilter(min_alpha_ratio=0.5, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("hello world one two three"))) == 1

def test_alpha_ratio_none_skips_check():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("1234567890"))) == 1


# ── digit_ratio ───────────────────────────────────────────────────────────────

def test_drops_chunk_above_max_digit_ratio():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=0.2, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert f.filter(make_chunks("1234567890 ab")) == []

def test_keeps_chunk_below_max_digit_ratio():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=0.5, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("hello world 123"))) == 1


# ── symbol_ratio ──────────────────────────────────────────────────────────────

def test_drops_chunk_above_max_symbol_ratio():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=0.1,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert f.filter(make_chunks("!!!???###$$$%%%^^^"[:18])) == []

def test_keeps_chunk_below_max_symbol_ratio():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=0.5,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("hello world!"))) == 1


# ── word_count ────────────────────────────────────────────────────────────────

def test_drops_chunk_below_min_word_count():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=5, min_avg_word_length=None, max_avg_word_length=None)
    assert f.filter(make_chunks("only three words")) == []

def test_keeps_chunk_at_min_word_count():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=3, min_avg_word_length=None, max_avg_word_length=None)
    assert len(f.filter(make_chunks("one two three"))) == 1


# ── avg_word_length ───────────────────────────────────────────────────────────

def test_drops_chunk_below_min_avg_word_length():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=4.0, max_avg_word_length=None)
    # single-char tokens: avg word length = 1
    assert f.filter(make_chunks("a b c d e f g h")) == []

def test_drops_chunk_above_max_avg_word_length():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=None, max_avg_word_length=5.0)
    # very long words
    assert f.filter(make_chunks("superlongword anotherlongword concatenatedtext")) == []

def test_keeps_chunk_within_avg_word_length_bounds():
    f = ChunkFilter(min_alpha_ratio=None, max_digit_ratio=None, max_symbol_ratio=None,
                    min_word_count=None, min_avg_word_length=2.0, max_avg_word_length=10.0)
    assert len(f.filter(make_chunks("hello world foo bar"))) == 1


# ── metadata preserved ────────────────────────────────────────────────────────

def test_metadata_preserved_after_filter():
    chunk = Chunk(content="  valid content here  ", metadata={"source": "doc.pdf", "page": 1})
    f = ChunkFilter(min_length=5, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    result = f.filter([chunk])
    assert result[0].metadata == {"source": "doc.pdf", "page": 1}


# ── multiple chunks ───────────────────────────────────────────────────────────

def test_filter_keeps_only_passing_chunks():
    f = ChunkFilter(min_length=10, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    chunks = make_chunks("short", "this one is long enough to pass", "tiny")
    result = f.filter(chunks)
    assert len(result) == 1
    assert result[0].content == "this one is long enough to pass"

def test_filter_preserves_order():
    f = ChunkFilter(min_length=3, min_alpha_ratio=None, max_digit_ratio=None,
                    max_symbol_ratio=None, min_word_count=None,
                    min_avg_word_length=None, max_avg_word_length=None)
    chunks = make_chunks("aaa", "bbb", "ccc")
    result = f.filter(chunks)
    assert [c.content for c in result] == ["aaa", "bbb", "ccc"]

def test_filter_empty_input_returns_empty():
    assert DEFAULT_CHUNK_FILTER.filter([]) == []
