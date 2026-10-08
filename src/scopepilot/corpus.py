"""Fetch public guides into a local cache and pull their screenshots out.

The guides belong to their publishers, so the repository holds only pointers
(a URL and, for each test case, a page and image number). The files themselves
live in a cache outside the repository and are never committed.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


def cache_dir() -> Path:
    root = os.environ.get("SCOPEPILOT_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "scopepilot"


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    profile: str
    publisher: str
    sha256: str = ""

    @property
    def directory(self) -> Path:
        return cache_dir() / "corpus" / self.id

    @property
    def pdf(self) -> Path:
        return self.directory / "guide.pdf"

    def image_path(self, page: int, image: int) -> Path:
        """Screenshot `image` (0-based) embedded on `page` (1-based)."""
        return self.directory / "images" / f"p{page:03d}-{image:02d}.png"


def load_sources(path: Path) -> dict[str, Source]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: Source(**entry) for entry in raw.get("source", [])}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source: Source) -> Path:
    """Download the guide if it is not cached, and check it against its recorded hash."""
    if not source.pdf.exists():
        source.directory.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(source.url, headers={"User-Agent": "scopepilot"})
        with urllib.request.urlopen(request, timeout=120) as response:
            source.pdf.write_bytes(response.read())
    actual = sha256_of(source.pdf)
    if source.sha256 and actual != source.sha256:
        raise ValueError(
            f"{source.id}: the guide has changed since the test cases were labelled "
            f"(expected sha256 {source.sha256[:12]}, got {actual[:12]}); page and image "
            f"numbers may no longer line up"
        )
    return source.pdf


def extract_images(source: Source, min_width: int = 500) -> list[Path]:
    """Write every embedded image at least `min_width` wide to the cache as PNG."""
    from pypdf import PdfReader

    out_dir = source.directory / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page_number, page in enumerate(PdfReader(source.pdf).pages, start=1):
        for index, embedded in enumerate(page.images):
            try:
                image = embedded.image
            except Exception:  # pypdf raises several types on images it cannot decode
                continue
            if image is None or image.width < min_width:
                continue
            path = source.image_path(page_number, index)
            image.convert("RGB").save(path)
            written.append(path)
    return written
