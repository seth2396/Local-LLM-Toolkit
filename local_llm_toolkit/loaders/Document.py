import hashlib

from ..ingesters import BaseItem


class Document:
    """
    Represents a loaded document with content and associated metadata.

    Created by a loader from a BaseItem. The content setter automatically
    recomputes a SHA-256 hash of the content and stores it in
    metadata["hash_id"], allowing downstream systems to detect changes
    without re-reading the source.

    Attributes:
        metadata (dict): Key-value pairs derived from the source item, plus
            "hash_id" which is updated on every content assignment.
        content (str): The document's text content.
    """

    def __init__(self, item: BaseItem):
        self.metadata = item.to_metadata()
        self.content = ""  # triggers the setter, adds hash_id

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str):
        self._content = value
        self.metadata["hash_id"] = hashlib.sha256(value.encode("utf-8")).hexdigest()

    def __str__(self):
        trim_len = min(len(self.content), 100)
        return f"Document(doctype={self.metadata.get('doctype')}, metadata={self.metadata}, content='{self.content[:trim_len]}...')"

    def to_dict(self, max_content_len: int = 100) -> dict:
        """Return a summary dict, truncating content to max_content_len characters."""
        return {
            "doctype": self.metadata.get("doctype"),
            "metadata": self.metadata,
            "content": self.content[:max_content_len],
            "length": len(self.content),
        }
