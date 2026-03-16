from pathlib import Path

from .Document import Document
from .BaseLoader import BaseLoader
from ..ingesters import FileItem


class PythonLoader(BaseLoader):
    """Loads Python source files (.py). Not yet implemented."""

    def load(self, item: FileItem) -> Document:
        raise NotImplementedError("Doc type ending with .py not yet implemented.")


class JavaScriptLoader(BaseLoader):
    """Loads JavaScript source files (.js). Not yet implemented."""

    def load(self, item: FileItem) -> Document:
        raise NotImplementedError("Doc type ending with .js not yet implemented.")


class JavaLoader(BaseLoader):
    """Loads Java source files (.java). Not yet implemented."""

    def load(self, item: FileItem) -> Document:
        raise NotImplementedError("Doc type ending with .java not yet implemented.")


class CppLoader(BaseLoader):
    """Loads C++ source files (.cpp). Not yet implemented."""

    def load(self, item: FileItem) -> Document:
        raise NotImplementedError("Doc type ending with .cpp not yet implemented.")


class CSharpLoader(BaseLoader):
    """Loads C# source files (.cs). Not yet implemented."""

    def load(self, item: FileItem) -> Document:
        raise NotImplementedError("Doc type ending with .cs not yet implemented.")


_CODE_LOADERS = {}  # {'.py': PythonLoader, '.js': JavaScriptLoader, '.java': JavaLoader, '.cpp': CppLoader, '.cs': CSharpLoader}
SUPPORTED_CODE_FORMATS = _CODE_LOADERS.keys()


class CodeLoader:
    """
    Dispatcher that routes a source file to the appropriate code loader by extension.

    Supported formats are defined in LOADERS (.py, .js, .java, .cpp, .cs).
    Note: all individual loaders are stubs and not yet implemented.
    """
    LOADERS = {
        '.py': PythonLoader,
        '.js': JavaScriptLoader,
        '.java': JavaLoader,
        '.cpp': CppLoader,
        '.cs': CSharpLoader
    }

    def load(self, file) -> Document:
        extension = Path(file.path).suffix.lower()
        if extension in self.LOADERS:
            loader = self.LOADERS[extension]
            return loader.load(file.path)

        raise NotImplementedError(
            f"No code loader implemented for file type '{extension}'. "
            "Please register a loader in CodeLoader.LOADERS."
        )
