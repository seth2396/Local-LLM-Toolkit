from pathlib import Path

from .Document import Document
from .BaseLoader import BaseLoader


class JPEGLoader(BaseLoader):
    """Loads JPEG image files (.jpeg). Not yet implemented."""

    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .jpeg not yet implemented.")


class PNGLoader(BaseLoader):
    """Loads PNG image files (.png). Not yet implemented."""

    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .png not yet implemented.")


class GIFLoader(BaseLoader):
    """Loads GIF image files (.gif). Not yet implemented."""

    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .gif not yet implemented.")


class BMPLoader(BaseLoader):
    """Loads BMP image files (.bmp). Not yet implemented."""

    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .bmp not yet implemented.")


_IMAGE_LOADERS = {}  # {'.jpeg': JPEGLoader, '.png': PNGLoader, '.gif': GIFLoader, '.bmp': BMPLoader}
SUPPORTED_IMAGE_FORMATS = _IMAGE_LOADERS.keys()


class ImageLoader:
    """
    Dispatcher that routes an image file to the appropriate loader by extension.

    Supported formats are defined in LOADERS (.jpeg, .png, .gif, .bmp).
    Note: all individual loaders are stubs and not yet implemented.
    """
    LOADERS = _IMAGE_LOADERS

    def load(self, file) -> Document:
        extension = Path(file.path).suffix.lower()
        if extension in self.LOADERS:
            loader = self.LOADERS[extension]
            return loader.load(file.path)

        raise NotImplementedError(
            f"No Image loader implemented for file type '{extension}'. "
            "Please register a loader in ImageLoader.LOADERS."
        )
