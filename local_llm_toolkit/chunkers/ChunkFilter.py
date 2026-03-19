from .Chunk import Chunk


class ChunkFilter:
    """
    Filters a list of Chunk objects, removing chunks that don't meet quality criteria.

    Applied automatically by BaseChunker.chunk() using DEFAULT_CHUNK_FILTER unless
    a custom filter is passed. Pass chunk_filter=None to skip filtering entirely.

    Attributes:
        min_length (int): Minimum character length after stripping. [Default: 10]
        max_length (int | None): Maximum character length — None means no upper limit. [Default: None]
        strip_whitespace (bool): Strip leading/trailing whitespace before all checks. [Default: True]
        min_alpha_ratio (float | None): Minimum ratio of alphabetic characters — filters chunks that are mostly numbers or symbols. [Default: 0.5]
        max_digit_ratio (float | None): Maximum ratio of digit characters — filters predominantly numeric chunks e.g. raw data tables. [Default: 0.3]
        max_symbol_ratio (float | None): Maximum ratio of non-alphanumeric non-whitespace characters — filters noisy or garbled text. [Default: 0.2]
        min_word_count (int | None): Minimum number of whitespace-delimited words. [Default: 3]
        min_avg_word_length (float | None): Minimum average word length — filters single-character token noise. [Default: 2.0]
        max_avg_word_length (float | None): Maximum average word length — filters unspaced or concatenated text. [Default: 20.0]
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = None,
        strip_whitespace: bool = True,
        min_alpha_ratio: float = 0.5,
        max_digit_ratio: float = 0.3,
        max_symbol_ratio: float = 0.2,
        min_word_count: int = 3,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 20.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.strip_whitespace = strip_whitespace
        self.min_alpha_ratio = min_alpha_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.min_word_count = min_word_count
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    def _metrics(self, content: str) -> dict:
        """Compute quality metrics for a chunk's content string."""
        total = len(content)
        if total == 0:
            return {
                "alpha_ratio": 0.0,
                "digit_ratio": 0.0,
                "symbol_ratio": 0.0,
                "word_count": 0,
                "avg_word_length": 0.0,
            }

        alpha = sum(c.isalpha() for c in content)
        digit = sum(c.isdigit() for c in content)
        symbol = sum(not c.isalnum() and not c.isspace() for c in content)
        words = content.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0

        return {
            "alpha_ratio": alpha / total,
            "digit_ratio": digit / total,
            "symbol_ratio": symbol / total,
            "word_count": len(words),
            "avg_word_length": avg_word_length,
        }

    def _passes(self, content: str) -> bool:
        """Return True if content passes all configured thresholds."""
        if not content:
            return False
        if len(content) < self.min_length:
            return False
        if self.max_length is not None and len(content) > self.max_length:
            return False

        m = self._metrics(content)

        if self.min_alpha_ratio is not None and m["alpha_ratio"] < self.min_alpha_ratio:
            return False
        if self.max_digit_ratio is not None and m["digit_ratio"] > self.max_digit_ratio:
            return False
        if self.max_symbol_ratio is not None and m["symbol_ratio"] > self.max_symbol_ratio:
            return False
        if self.min_word_count is not None and m["word_count"] < self.min_word_count:
            return False
        if self.min_avg_word_length is not None and m["avg_word_length"] < self.min_avg_word_length:
            return False
        if self.max_avg_word_length is not None and m["avg_word_length"] > self.max_avg_word_length:
            return False

        return True

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Filter chunks, returning only those that pass all configured criteria.

        Args:
            chunks: List of Chunk objects to filter.

        Returns:
            list[Chunk]: Chunks that pass all criteria, in original order.
        """
        result = []
        for chunk in chunks:
            content = chunk.content.strip() if self.strip_whitespace else chunk.content
            if not self._passes(content):
                continue
            if self.strip_whitespace and content != chunk.content:
                result.append(Chunk(content=content, metadata=chunk.metadata))
            else:
                result.append(chunk)
        return result


DEFAULT_CHUNK_FILTER = ChunkFilter()
