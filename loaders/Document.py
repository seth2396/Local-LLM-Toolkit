import os
import hashlib


class Document:
    def __init__(self, file):
        raw = {
            "file_name": file.name,
            "extension": file.ext,
            "doctype": file.doctype,
            "base_path": os.path.dirname(file.path),
            "last_modified": file.modified_ts,
            "file_size": file.size_bytes,
        }
        self.metadata = {k: v for k, v in raw.items() if v is not None}
        self.content = ""  # triggers the setter, adds hash_id

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, value: str):
        self._content = value
        self.metadata["hash_id"] = hashlib.sha256(value.encode("utf-8")).hexdigest()

    def __str__(self):
        """
        Return a trimmed string representation of the Document.
        """
        trim_len = min(len(self.content), 100)
        return f"Document(doctype={self.metadata['doctype']}, metadata={self.metadata}, content='{self.content[:trim_len]}...')"

    def to_dict(self, max_content_len: int = 100) -> dict:
        """
            Return a dict representation with trimmed content.
        """
        return {
            "extension": self.metadata['doctype'],
            "metadata": self.metadata,
            "content": self.content[:max_content_len],
            "length": len(self.content)
        }
