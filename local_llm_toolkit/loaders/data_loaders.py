from pathlib import Path
import json
import pandas as pd

from .Document import Document
from .BaseLoader import BaseLoader
from ..ingesters import FileItem


class JSONLoader(BaseLoader):
    """Loads JSON files, storing the parsed Python object as document content."""

    def load(self, item: FileItem) -> Document:
        """Parse a JSON file and store the resulting object in document.content."""
        document = Document(item)
        with open(item.path, 'r', encoding='utf-8') as f:
            document.content = json.load(f)
        return document


class CSVLoader(BaseLoader):
    """Loads CSV files into a pandas DataFrame stored as document content."""

    def load(self, item: FileItem) -> Document:
        """Read a CSV file into a DataFrame and store it in document.content."""
        document = Document(item)
        document.content = pd.read_csv(item.path)
        return document


class ExcelLoader(BaseLoader):
    """Loads Excel files (.xlsx) into a pandas DataFrame stored as document content."""

    def load(self, item: FileItem) -> Document:
        """Read the first sheet of an Excel file into a DataFrame and store it in document.content."""
        document = Document(item)
        document.content = pd.read_excel(item.path)
        return document


_DATA_LOADERS = {
    '.json': JSONLoader,
    '.csv': CSVLoader,
    '.xlsx': ExcelLoader
}
SUPPORTED_DATA_FORMATS = _DATA_LOADERS.keys()


class DataLoader:
    """
    Dispatcher that routes a file to the appropriate data loader by extension.

    Supported formats are defined in LOADERS (.json, .csv, .xlsx).
    """
    LOADERS = _DATA_LOADERS

    def load(self, file) -> Document:
        ext = Path(file.path).suffix.lower()
        if ext in self.LOADERS:
            loader = self.LOADERS[ext]
            return loader.load(file.path)

        raise NotImplementedError(
            f"No data loader implemented for file type '{ext}'. "
            "Please register a loader in DataLoader.LOADERS."
        )
