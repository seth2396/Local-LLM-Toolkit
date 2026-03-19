import pytest
from unittest.mock import MagicMock, mock_open, patch
from local_llm_toolkit.loaders.text_loaders import (
    _clean,
    _clean_for_doctype,
    _ARTIFACTS,
    _CLEANING_PATTERNS,
    _TRANSFORMS,
    PdfLoader,
    DocxLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_file_item(name="doc", source="doc.pdf", doctype="pdf", path="doc.pdf", ext=".pdf"):
    item = MagicMock()
    item.name = name
    item.source = source
    item.doctype = doctype
    item.path = path
    item.ext = ext
    item.to_metadata.return_value = {
        "name": name, "source": source, "doctype": doctype,
        "path": path, "ext": ext,
    }
    return item


# ── _clean: artifacts ─────────────────────────────────────────────────────────

def test_clean_replaces_artifacts():
    result = _clean("hello\xa0world", artifacts={"\xa0": " "})
    assert result == "hello world"

def test_clean_multiple_artifacts():
    result = _clean("a\xa0b\fc", artifacts={"\xa0": " ", "\f": "\n"})
    assert result == "a b\nc"

def test_clean_no_artifacts_leaves_text():
    result = _clean("hello world", artifacts=None)
    assert result == "hello world"


# ── _clean: patterns ──────────────────────────────────────────────────────────

def test_clean_applies_regex_pattern():
    result = _clean("hello   world", patterns=[(r"[ \t]+", " ")])
    assert result == "hello world"

def test_clean_collapses_blank_lines():
    result = _clean("a\n\n\n\nb", patterns=[(r"\n{3,}", "\n\n")])
    assert result == "a\n\nb"

def test_clean_rejoins_hyphenated_line_breaks():
    result = _clean("hyphen-\nated", patterns=[(r"-\n", "")])
    assert result == "hyphénated".replace("é", "e") or result == "hyphénated".replace("é", "e") or result == "hyphenated"

def test_clean_multiple_patterns_applied_in_order():
    result = _clean(
        "hello   world\n\n\n\nbye",
        patterns=[(r"[ \t]+", " "), (r"\n{3,}", "\n\n")]
    )
    assert result == "hello world\n\nbye"

def test_clean_no_patterns_leaves_text():
    result = _clean("hello world", patterns=None)
    assert result == "hello world"


# ── _clean: transforms ────────────────────────────────────────────────────────

def test_clean_applies_transform():
    result = _clean("hello world", transforms=[str.upper])
    assert result == "HELLO WORLD"

def test_clean_multiple_transforms_in_order():
    result = _clean("  hello  ", transforms=[str.strip, str.upper])
    assert result == "HELLO"

def test_clean_no_transforms_leaves_text():
    result = _clean("hello world", transforms=None)
    assert result == "hello world"


# ── _clean: combined and strip ────────────────────────────────────────────────

def test_clean_strips_result():
    result = _clean("  hello world  ")
    assert result == "hello world"

def test_clean_all_steps_applied_in_order():
    # artifact replaces \xa0, then pattern collapses spaces, then transform uppercases
    result = _clean(
        "hello\xa0 world",
        artifacts={"\xa0": " "},
        patterns=[(r"[ \t]+", " ")],
        transforms=[str.upper],
    )
    assert result == "HELLO WORLD"

def test_clean_empty_string_returns_empty():
    assert _clean("") == ""


# ── _clean_for_doctype ────────────────────────────────────────────────────────

def test_clean_for_doctype_pdf_removes_nonbreaking_space():
    result = _clean_for_doctype("hello\xa0world", "pdf")
    assert "\xa0" not in result

def test_clean_for_doctype_pdf_rejoins_hyphens():
    result = _clean_for_doctype("hyphen-\nated word", "pdf")
    assert "hyphenated" in result

def test_clean_for_doctype_pdf_removes_form_feed():
    result = _clean_for_doctype("page1\fpage2", "pdf")
    assert "\f" not in result

def test_clean_for_doctype_md_preserves_structure():
    md = "## Heading\n\n- item one\n- item two\n\n> blockquote"
    result = _clean_for_doctype(md, "md")
    assert result == md.strip()

def test_clean_for_doctype_collapses_excessive_newlines():
    result = _clean_for_doctype("a\n\n\n\n\nb", "txt")
    assert "\n\n\n" not in result

def test_clean_for_doctype_unknown_falls_back_to_default():
    result = _clean_for_doctype("hello\xa0world", "unknown_type")
    assert "\xa0" not in result


# ── PdfLoader ─────────────────────────────────────────────────────────────────

def test_pdf_loader_extracts_text_from_pages():
    item = make_file_item(doctype="pdf", ext=".pdf")
    page = MagicMock()
    page.extract_text.return_value = "page one content"

    mock_pdf = MagicMock()
    mock_pdf.pages = [page]
    mock_pdf.metadata = {}
    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("local_llm_toolkit.loaders.text_loaders.pdfplumber.open", return_value=mock_pdf):
        doc = PdfLoader().load(item)

    assert "page one content" in doc.content

def test_pdf_loader_skips_image_only_pages():
    item = make_file_item(doctype="pdf", ext=".pdf")
    text_page = MagicMock()
    text_page.extract_text.return_value = "real content here"
    image_page = MagicMock()
    image_page.extract_text.return_value = None

    mock_pdf = MagicMock()
    mock_pdf.pages = [text_page, image_page]
    mock_pdf.metadata = {}
    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("local_llm_toolkit.loaders.text_loaders.pdfplumber.open", return_value=mock_pdf):
        doc = PdfLoader().load(item)

    assert "real content here" in doc.content

def test_pdf_loader_merges_primitive_metadata():
    item = make_file_item(doctype="pdf", ext=".pdf")
    page = MagicMock()
    page.extract_text.return_value = "content"

    mock_pdf = MagicMock()
    mock_pdf.pages = [page]
    mock_pdf.metadata = {"Author": "Alice", "Pages": 10, "nested": {"key": "val"}}
    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("local_llm_toolkit.loaders.text_loaders.pdfplumber.open", return_value=mock_pdf):
        doc = PdfLoader().load(item)

    assert doc.metadata["Author"] == "Alice"
    assert doc.metadata["Pages"] == 10
    assert "nested" not in doc.metadata


# ── TextLoader ────────────────────────────────────────────────────────────────

def test_text_loader_reads_content():
    item = make_file_item(doctype="txt", ext=".txt")
    with patch("builtins.open", mock_open(read_data="hello world content")):
        doc = TextLoader().load(item)
    assert "hello world content" in doc.content

def test_text_loader_cleans_nonbreaking_space():
    item = make_file_item(doctype="txt", ext=".txt")
    with patch("builtins.open", mock_open(read_data="hello\xa0world")):
        doc = TextLoader().load(item)
    assert "\xa0" not in doc.content


# ── MarkdownLoader ────────────────────────────────────────────────────────────

def test_markdown_loader_reads_content():
    item = make_file_item(doctype="md", ext=".md")
    md = "## Title\n\n- item one\n- item two"
    with patch("builtins.open", mock_open(read_data=md)):
        doc = MarkdownLoader().load(item)
    assert "## Title" in doc.content
    assert "- item one" in doc.content

def test_markdown_loader_preserves_structure():
    item = make_file_item(doctype="md", ext=".md")
    md = "## Heading\n\n> quote\n\n- list"
    with patch("builtins.open", mock_open(read_data=md)):
        doc = MarkdownLoader().load(item)
    assert doc.content == md.strip()


# ── HTMLLoader ────────────────────────────────────────────────────────────────

def test_html_loader_strips_tags():
    item = make_file_item(doctype="html", ext=".html")
    item.html_content = None  # force file path branch
    html = "<html><body><p>Hello world</p></body></html>"
    with patch("builtins.open", mock_open(read_data=html)):
        doc = HTMLLoader().load(item)
    assert "<p>" not in doc.content
    assert "Hello world" in doc.content

def test_html_loader_uses_html_content_from_web_item():
    item = MagicMock()
    item.to_metadata.return_value = {"source": "http://example.com", "doctype": "html"}
    item.html_content = "<html><body><p>Web content</p></body></html>"
    doc = HTMLLoader().load(item)
    assert "Web content" in doc.content

def test_html_loader_falls_back_to_file_when_no_html_content():
    item = make_file_item(doctype="html", ext=".html")
    item.html_content = None
    html = "<html><body><p>File content</p></body></html>"
    with patch("builtins.open", mock_open(read_data=html)):
        doc = HTMLLoader().load(item)
    assert "File content" in doc.content


# ── DocxLoader ────────────────────────────────────────────────────────────────

def test_docx_loader_extracts_paragraphs():
    item = make_file_item(doctype="docx", ext=".docx")
    para = MagicMock()
    para.text = "This is a paragraph"

    mock_doc = MagicMock()
    mock_doc.paragraphs = [para]
    mock_doc.tables = []

    with patch("local_llm_toolkit.loaders.text_loaders.DocxDocument", return_value=mock_doc):
        doc = DocxLoader().load(item)

    assert "This is a paragraph" in doc.content

def test_docx_loader_cleans_nonbreaking_space():
    item = make_file_item(doctype="docx", ext=".docx")
    para = MagicMock()
    para.text = "hello\xa0world"

    mock_doc = MagicMock()
    mock_doc.paragraphs = [para]
    mock_doc.tables = []

    with patch("local_llm_toolkit.loaders.text_loaders.DocxDocument", return_value=mock_doc):
        doc = DocxLoader().load(item)

    assert "\xa0" not in doc.content
