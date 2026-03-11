from pathlib import Path

from .Document import Document
from .BaseLoader import BaseLoader


class JPEGLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .jpeg not yet implemented.")


class PNGLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .png not yet implemented.")


class GIFLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .gif not yet implemented.")


class BMPLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .bmp not yet implemented.")


_IMAGE_LOADERS = {}  # {'.jpeg': JPEGLoader, '.png': PNGLoader, '.gif': GIFLoader, '.bmp': BMPLoader}
SUPPORTED_IMAGE_FORMATS = _IMAGE_LOADERS.keys()


class ImageLoader:
    f"""
    A loader class to handle loading of various image document formats.
    Currently supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}
    """
    LOADERS = _IMAGE_LOADERS

    def load(self, file) -> Document:
        ext = Path(file.path).suffix.lower()
        if ext in self.LOADERS:
            loader = self.LOADERS[ext]
            return loader.load(file.path)

        raise NotImplementedError(
            f"No Image loader implemented for file type '{ext}'. "
            "Please register a loader in ImageLoader.LOADERS."
        )
