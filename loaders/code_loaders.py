from pathlib import Path

from .Document import Document
from .BaseLoader import BaseLoader


class PythonLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .py not yet implemented.")


class JavaScriptLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .js not yet implemented.")


class JavaLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .java not yet implemented.")


class CppLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .cpp not yet implemented.")


class CSharpLoader(BaseLoader):  # Not implemented yet
    def load(self, file) -> Document:
        raise NotImplementedError("Doc type ending with .cs not yet implemented.")


_CODE_LOADERS = {}  # {'.py': PythonLoader, '.js': JavaScriptLoader, '.java': JavaLoader, '.cpp': CppLoader, '.cs': CSharpLoader}
SUPPORTED_CODE_FORMATS = _CODE_LOADERS.keys()


class CodeLoader:
    f"""
    A loader class to handle loading of various code document formats.
    Currently supported formats: {', '.join(SUPPORTED_CODE_FORMATS)}
    """
    LOADERS = {
        '.py': PythonLoader,
        '.js': JavaScriptLoader,
        '.java': JavaLoader,
        '.cpp': CppLoader,
        '.cs': CSharpLoader
    }

    def load(self, file) -> Document:
        ext = Path(file.path).suffix.lower()
        if ext in self.LOADERS:
            loader = self.LOADERS[ext]
            return loader.load(file.path)

        raise NotImplementedError(
            f"No code loader implemented for file type '{ext}'. "
            "Please register a loader in CodeLoader.LOADERS."
        )
