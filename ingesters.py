import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


# Default mapping from logical doctypes to file extensions.
# Extend this as needed (e.g., add .md, .rtf, .json, .csv, etc.).
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


@dataclass(frozen = True)
class FileItem:
    """
        Structured representation of a discovered file. 

     
        Attributes:
            path (Path): Absolute path to the file.
            name (str): File name (stem + extension).
            size_bytes (int): File size in bytes.
            modified_ts (float): Last modified timestamp (Unix epoch seconds).
            doctype (str): Logical type (e.g., 'pdf', 'docx', 'txt'); 'unknown' if unmapped.
            ext (str): File extension including the dot (e.g., '.pdf').
    """
    path: Path
    name: str
    size_bytes: int
    modified_ts: float
    doctype: str
    ext: str


class GlobIngester:
    """Ingests files from local folders using glob with optional doctype filtering.

    This class discovers files recursively under one or more root directories,
    optionally filtering by logical 'doctype' categories (e.g., 'pdf', 'docx').
    It returns rich metadata per file for downstream loaders/chunkers.

    Example:
        ingester = GlobIngester(
            roots=["/data/knowledge_base", "/data/shared"],
            doctype_filter={"pdf", "docx"},
        )
        files = ingester.collect()
        # Pass `files` to your loader module.

    Args:
        roots (Iterable[str | Path]): One or more root folders to scan.
        doctype_filter (Optional[Set[str]]): If provided, only files whose extensions
            map to these doctypes will be returned (case-insensitive). If None, no
            filtering is applied.
        doctype_extensions (Optional[Dict[str, Set[str]]]): Custom mapping of doctypes
            to file extensions. If None, DEFAULT_DOCTYPE_EXTENSIONS is used.
        include_hidden (bool): If True, include files inside hidden folders/files (dot-prefixed).
            Defaults to False.
        follow_symlinks (bool): If True, include files referenced by symlinks.
            Defaults to False.

    Raises:
        ValueError: If `roots` is empty or contains a non-existent path.
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

        # Normalize the filter to lower case if provided.
        self.doctype_filter = {dt.lower() for dt in doctype_filter} if doctype_filter else None

        # Build the extension map (lowercase ext keys, lowercase doctypes).
        base_map = doctype_extensions or DEFAULT_DOCTYPE_EXTENSIONS
        self.ext_to_doctype: Dict[str, str] = {}
        for dt, exts in base_map.items():
            for ext in exts:
                self.ext_to_doctype[ext.lower()] = dt.lower()

    def collect(self) -> List[FileItem]:
        """Discover files under roots using glob, apply filtering, and return metadata.

        Returns:
            List[FileItem]: A list of discovered files with metadata.

        Notes:
            - Recursively searches using patterns like '**/*'.
            - Skips hidden files/folders unless `include_hidden=True`.
            - Applies doctype filtering if `doctype_filter` is provided.
            - Symlink handling controlled by `follow_symlinks`.
        """
        results: List[FileItem] = []

        for root in self.roots:
            # '**/*' yields all paths recursively; we'll filter down to files.
            pattern = str(root / "**" / "*")
            for entry in glob.iglob(pattern, recursive=True):
                p = Path(entry)

                # Skip directories; only files.
                if not p.is_file():
                    continue

                # Skip symlinks if not allowed.
                if p.is_symlink() and not self.follow_symlinks:
                    continue

                # Skip hidden files or files in hidden dirs unless allowed.
                if not self.include_hidden and self._is_hidden(p):
                    continue

                ext = p.suffix.lower()
                doctype = self.ext_to_doctype.get(ext, "unknown")

                # If filtering is requested, keep only mapped doctypes in the filter.
                if self.doctype_filter is not None:
                    if doctype == "unknown":
                        # Unmapped extension => exclude when filtering.
                        continue
                    if doctype not in self.doctype_filter:
                        continue

                # Collect metadata safely; handle permissions edge cases.
                try:
                    stat = p.stat()
                    size_bytes = stat.st_size
                    modified_ts = stat.st_mtime
                except (FileNotFoundError, PermissionError, OSError):
                    # Skip files we can't stat (e.g., permission issues).
                    continue

                results.append(
                    FileItem(
                        path=p,
                        name=p.name,
                        size_bytes=size_bytes,
                        modified_ts=modified_ts,
                        doctype=doctype,
                        ext=ext,
                    )
                )

        # Optional: sort by modified time descending to prioritize recent content.
        results.sort(key=lambda fi: fi.modified_ts, reverse=True)
        return results

    @staticmethod
    def _is_hidden(path: Path) -> bool:
        """Determine whether a path is hidden (dot-prefixed file or inside dot-prefixed dir).

        Args:
            path (Path): The file path to check.

        Returns:
            bool: True if hidden, False otherwise.
        """
        # Any part of the path starting with '.' indicates hidden on Unix-like systems.
        return any(part.startswith(".") for part in path.parts)
