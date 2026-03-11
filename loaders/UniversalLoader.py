from .Document import Document
from .text_loaders import _TEXT_LOADERS, SUPPORTED_TEXT_FORMATS
from .data_loaders import _DATA_LOADERS, SUPPORTED_DATA_FORMATS
from .image_loaders import _IMAGE_LOADERS, SUPPORTED_IMAGE_FORMATS
from .code_loaders import _CODE_LOADERS, SUPPORTED_CODE_FORMATS


ALL_SUPPORTED_FORMATS = list(SUPPORTED_TEXT_FORMATS) + list(SUPPORTED_DATA_FORMATS) + list(SUPPORTED_IMAGE_FORMATS) + list(SUPPORTED_CODE_FORMATS)


class UniversalLoader:
    f"""
    A unified loader class to handle loading of various document formats.
    Currently supported formats: {', '.join(ALL_SUPPORTED_FORMATS)}
    """
    LOADERS = _TEXT_LOADERS | _DATA_LOADERS | _IMAGE_LOADERS | _CODE_LOADERS

    def load(self, file) -> Document:
        ext = file.ext
        if ext in self.LOADERS:
            loader_class = self.LOADERS[ext]
            loader = loader_class()
            return loader.load(file)

        raise NotImplementedError(
            f"No loader implemented for file type '{ext}'. "
            "Please register a loader in Loader.loaders."
        )
