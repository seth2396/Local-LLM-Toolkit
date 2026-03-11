from pathlib import Path
import json
import pandas as pd

from .Document import Document
from .BaseLoader import BaseLoader


class JSONLoader(BaseLoader):
    def load(self, file) -> Document:
        document = Document(file)
        with open(file.path, 'r', encoding='utf-8') as f:
            document.content = json.load(f)
        return document


class CSVLoader(BaseLoader):
    def load(self, file) -> Document:
        document = Document(file)
        document.content = pd.read_csv(file.path)
        return document


class ExcelLoader(BaseLoader):
    def load(self, file) -> Document:
        document = Document(file)
        document.content = pd.read_excel(file.path)
        return document


_DATA_LOADERS = {
    '.json': JSONLoader,
    '.csv': CSVLoader,
    '.xlsx': ExcelLoader
}
SUPPORTED_DATA_FORMATS = _DATA_LOADERS.keys()


class DataLoader:
    f"""
    A loader class to handle loading of various data document formats.
    Currently supported formats: {', '.join(SUPPORTED_DATA_FORMATS)}
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
