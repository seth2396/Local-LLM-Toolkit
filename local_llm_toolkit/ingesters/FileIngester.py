import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .BaseIngester import BaseIngester
from .BaseItem import BaseItem


DEFAULT_DOCTYPE_EXTENSIONS: Dict[str, Set[str]] = {
    "txt": {".txt"},
    "pdf": {".pdf"},
    "docx": {".docx"},
    "doc": {".doc"},
    "xlsx": {".xlsx"},
    "xls": {".xls"},
    "pptx": {".pptx"},
    "ppt": {".ppt"},
    "md": {".md"},
    "rtf": {".rtf"},
    "json": {".json"},
    "csv": {".csv"},
}


@dataclass(frozen=True)
class FileItem(BaseItem):
    """
    Structured representation of a discovered local file.

    Attributes:
        name:        File name (stem + extension).
        source:      Absolute path to the file as a string.
        doctype:     Logical type (e.g., 'pdf', 'docx'); 'unknown' if unmapped.
        path:        Absolute path as a Path object.
        ext:         File extension including the dot (e.g., '.pdf').
        size_bytes:  File size in bytes.
        modified_ts: Last modified timestamp (Unix epoch seconds).
    """
    name: str
    source: str
    doctype: str
    path: Path
    ext: str
    size_bytes: Optional[int] = None
    modified_ts: Optional[float] = None

    def to_metadata(self) -> dict:
        return {k: v for k, v in {
            "name": self.name,
            "source": self.source,
            "doctype": self.doctype,
            "path": str(self.path),
            "ext": self.ext,
            "size_bytes": self.size_bytes,
            "modified_ts": self.modified_ts,
        }.items() if v is not None}


class FileIngester(BaseIngester):
    """
    Ingests files from local folders using glob with optional doctype filtering.

    Args:
        roots:              One or more root folders to scan.
        doctype_filter:     If provided, only files whose extensions map to these
                            doctypes will be returned. If None, no filtering is applied.
        doctype_extensions: Custom doctype-to-extension mapping.
                            Defaults to DEFAULT_DOCTYPE_EXTENSIONS.
        include_hidden:     If True, include files in hidden (dot-prefixed) folders.
        follow_symlinks:    If True, include files referenced by symlinks.

    Raises:
        ValueError: If roots is empty or contains a non-existent path.
    """

    def __init__(
        self,
        roots: Iterable[str | Path],
        doctype_filter: Optional[Set[str]] = None,
        doctype_extensions: Optional[Dict[str, Set[str]]] = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> None:
        self.roots: List[Path] = [Path(r).resolve() for r in roots]
        if not self.roots:
            raise ValueError("At least one root folder must be provided.")

        for r in self.roots:
            if not r.exists() or not r.is_dir():
                raise ValueError(f"Root does not exist or is not a directory: {r}")

        self.include_hidden = include_hidden
        self.follow_symlinks = follow_symlinks
        self.doctype_filter = {dt.lower() for dt in doctype_filter} if doctype_filter else None

        base_map = doctype_extensions or DEFAULT_DOCTYPE_EXTENSIONS
        self.ext_to_doctype: Dict[str, str] = {
            ext.lower(): dt.lower()
            for dt, exts in base_map.items()
            for ext in exts
        }

    def collect(self) -> List[FileItem]:
        """
        Discover files under roots, apply filtering, and return file metadata.

        Returns:
            List of FileItem sorted by modified time descending.
        """
        results: List[FileItem] = []

        for root in self.roots:
            pattern = str(root / "**" / "*")
            for entry in glob.iglob(pattern, recursive=True):
                p = Path(entry)

                if not p.is_file():
                    continue
                if p.is_symlink() and not self.follow_symlinks:
                    continue
                if not self.include_hidden and self._is_hidden(p):
                    continue

                ext = p.suffix.lower()
                doctype = self.ext_to_doctype.get(ext, "unknown")

                if self.doctype_filter is not None:
                    if doctype == "unknown" or doctype not in self.doctype_filter:
                        continue

                try:
                    stat = p.stat()
                except (FileNotFoundError, PermissionError, OSError):
                    continue

                results.append(FileItem(
                    name=p.name,
                    source=str(p),
                    doctype=doctype,
                    path=p,
                    ext=ext,
                    size_bytes=stat.st_size,
                    modified_ts=stat.st_mtime,
                ))

        results.sort(key=lambda fi: fi.modified_ts or 0, reverse=True)
        return results

    @staticmethod
    def _is_hidden(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)
