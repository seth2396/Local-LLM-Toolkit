from .Document import Document


class BaseLoader:
    def load(self, file) -> Document:
        raise NotImplementedError("Load method not implemented.")
