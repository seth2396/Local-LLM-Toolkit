from .Document import Document
from ..ingesters import FileItem

from .BaseLoader import BaseLoader

from .text_loaders import _TEXT_LOADERS, SUPPORTED_TEXT_FORMATS
from .data_loaders import _DATA_LOADERS, SUPPORTED_DATA_FORMATS
from .image_loaders import _IMAGE_LOADERS, SUPPORTED_IMAGE_FORMATS
from .code_loaders import _CODE_LOADERS, SUPPORTED_CODE_FORMATS


ALL_SUPPORTED_FORMATS = list(SUPPORTED_TEXT_FORMATS) + list(SUPPORTED_DATA_FORMATS) + list(SUPPORTED_IMAGE_FORMATS) + list(SUPPORTED_CODE_FORMATS)


class UniversalLoader(BaseLoader):
    """
    Unified loader that dispatches to the appropriate loader based on file extension.

    Merges all registered loaders from text, data, image, and code modules.
    Supported formats are the union of all LOADERS dicts — see each submodule
    for what is currently implemented vs. stubbed.
    """
    LOADERS = _TEXT_LOADERS | _DATA_LOADERS | _IMAGE_LOADERS | _CODE_LOADERS

    def load(self, file: FileItem) -> Document:
        """Select and invoke the appropriate loader for the file's extension."""
        extension = file.extension
        if extension in self.LOADERS:
            loader_class = self.LOADERS[extension]
            loader = loader_class()
            return loader.load(file)

        raise NotImplementedError(
            f"No loader implemented for file type '{extension}'. "
            "Please register a loader in Loader.loaders."
        )

    