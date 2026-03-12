"""
Default separator lists for RecursiveChunk, keyed by document doctype.

Separators are tried in order — broader splits first, falling back to
finer-grained ones. The "default" entry is used when no doctype-specific
list is found.
"""

SEPARATORS_BY_DOCTYPE: dict[str, list[str]] = {
    "default":    ["\n\n", "\n", ". ", "? ", "! ", " ", ""],

    # Plain text / prose
    "txt":        ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    "pdf":        ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    "docx":       ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    "doc":        ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    "rtf":        ["\n\n", "\n", ". ", "? ", "! ", " ", ""],

    # Markdown — split on headings first
    "md":         ["## ", "### ", "#### ", "\n\n", "\n", ". ", " ", ""],
    "markdown":   ["## ", "### ", "#### ", "\n\n", "\n", ". ", " ", ""],

    # HTML (post-parsed plain text)
    "html":       ["\n\n", "\n", ". ", " ", ""],

    # Structured data — row/field level
    "json":       ["\n", ", ", " ", ""],
    "csv":        ["\n", ",", " ", ""],
    "xlsx":       ["\n", "\t", " ", ""],
    "xls":        ["\n", "\t", " ", ""],

    # Code — split on top-level declarations first
    "python":     ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    "py":         ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    "js":         ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
    "javascript": ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
    "ts":         ["\nfunction ", "\nclass ", "\nconst ", "\n\n", "\n", " ", ""],
    "java":       ["\npublic ", "\nprivate ", "\nprotected ", "\nclass ", "\n\n", "\n", " ", ""],
    "cpp":        ["\nvoid ", "\nint ", "\nclass ", "\n\n", "\n", " ", ""],
    "cs":         ["\npublic ", "\nprivate ", "\nclass ", "\n\n", "\n", " ", ""],
}
