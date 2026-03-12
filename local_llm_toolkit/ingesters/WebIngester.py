from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from typing import Optional

from .BaseIngester import BaseIngester
from .BaseItem import BaseItem


@dataclass(frozen=True)
class WebItem(BaseItem):
    """
    Structured representation of a discovered web page.

    Attributes:
        name:         Page title, or URL path if title is unavailable.
        source:       Full URL of the page.
        doctype:      Always 'html' for web pages.
        url:          Full URL of the page.
        content_type: MIME type returned by the server (e.g., 'text/html').
        depth:        Crawl depth from the seed URL (0 = seed).
        title:        Page title tag content, if available.
    """
    name: str
    source: str
    doctype: str
    url: str
    content_type: str
    depth: int
    title: Optional[str] = None
    html_content: Optional[str] = None

    def to_metadata(self) -> dict:
        # html_content is excluded — it goes into Document.content, not metadata
        return {k: v for k, v in {
            "name": self.name,
            "source": self.source,
            "doctype": self.doctype,
            "url": self.url,
            "content_type": self.content_type,
            "depth": self.depth,
            "title": self.title,
        }.items() if v is not None}


class WebIngester(BaseIngester):
    """
    Ingests pages by crawling from one or more seed URLs using BFS.

    Follows links up to max_depth, stays within the seed domains by default,
    and respects robots.txt. HTTP errors are skipped rather than raised.

    Args:
        seeds:          One or more URLs to start crawling from.
        max_depth:      Maximum link-follow depth from the seed. Defaults to 2.
        max_pages:      Maximum total pages to collect. Defaults to 100.
        same_domain:    If True, only follow links within the same domain as each seed.
        respect_robots: If True, honour robots.txt exclusions.
        timeout:        HTTP request timeout in seconds. Defaults to 10.
    """

    def __init__(
        self,
        seeds: list[str],
        max_depth: int = 2,
        max_pages: int = 100,
        same_domain: bool = True,
        respect_robots: bool = True,
        timeout: int = 10,
    ):
        self.seeds = seeds
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.same_domain = same_domain
        self.respect_robots = respect_robots
        self.timeout = timeout

    def collect(self) -> list[WebItem]:
        """
        Crawl from seed URLs using BFS and return discovered pages.

        Returns:
            List of WebItem, one per successfully fetched page.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required: pip install requests beautifulsoup4")

        allowed_domains = {urlparse(s).netloc for s in self.seeds} if self.same_domain else None
        robots_cache: dict[str, RobotFileParser] = {}
        session = requests.Session()
        session.headers["User-Agent"] = "LocalLLMToolkit-WebIngester/1.0"

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((url, 0) for url in self.seeds)
        items: list[WebItem] = []

        while queue and len(items) < self.max_pages:
            url, depth = queue.popleft()
            url, _ = urldefrag(url)  # strip #fragment

            if url in visited:
                continue
            visited.add(url)

            if allowed_domains and urlparse(url).netloc not in allowed_domains:
                continue

            if self.respect_robots and not self._is_allowed(url, robots_cache, session):
                continue

            try:
                resp = session.get(url, timeout=self.timeout)
                resp.raise_for_status()
            except Exception:
                continue

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
            if "html" not in content_type:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else None

            items.append(WebItem(
                name=title or urlparse(url).path or url,
                source=url,
                doctype="html",
                url=url,
                content_type=content_type,
                depth=depth,
                title=title,
                html_content=resp.text,
            ))

            if depth < self.max_depth:
                for tag in soup.find_all("a", href=True):
                    href = urljoin(url, tag["href"])
                    href, _ = urldefrag(href)
                    if href.startswith("http") and href not in visited:
                        queue.append((href, depth + 1))

        return items

    def _is_allowed(self, url: str, cache: dict, session) -> bool:
        """Check robots.txt for the given URL, caching the parser per domain."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        if origin not in cache:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp.allow_all = True
            cache[origin] = rp

        return cache[origin].can_fetch(session.headers["User-Agent"], url)
