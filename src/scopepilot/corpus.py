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


class CorpusError(RuntimeError):
    """A guide could not be fetched or read."""


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
    """Make sure the cached guide is the one the test cases were labelled against.

    Downloads when the guide is missing or does not match its recorded hash. A
    download only replaces the cached file once it has been checked, so a bad
    response never gets stuck in the cache.
    """
    if source.pdf.exists() and (not source.sha256 or sha256_of(source.pdf) == source.sha256):
        return source.pdf

    source.directory.mkdir(parents=True, exist_ok=True)
    partial = source.directory / "guide.pdf.part"
    try:
        request = urllib.request.Request(source.url, headers={"User-Agent": "scopepilot"})
        with urllib.request.urlopen(request, timeout=120) as response:
            partial.write_bytes(response.read())
        if not partial.read_bytes().startswith(b"%PDF-"):
            raise CorpusError(f"{source.id}: {source.url} did not return a PDF")
        actual = sha256_of(partial)
        if source.sha256 and actual != source.sha256:
            raise CorpusError(
                f"{source.id}: the guide at {source.url} is not the file the test cases were "
                f"labelled against (expected sha256 {source.sha256[:12]}, got {actual[:12]}); "
                f"page and image numbers may no longer line up"
            )
        os.replace(partial, source.pdf)
    finally:
        partial.unlink(missing_ok=True)
    return source.pdf


def extract_images(source: Source, min_width: int = 500) -> list[Path]:
    """Write every embedded image at least `min_width` wide to the cache as PNG."""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as error:
        raise CorpusError(
            'reading guides needs pypdf; install it with: pip install "scopepilot[corpus]"'
        ) from error

    out_dir = source.directory / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page_number, page in enumerate(PdfReader(source.pdf).pages, start=1):
        images = page.images
        # By index, not by iterating: pypdf decodes an image when it is looked
        # up, and one it cannot decode must be skipped without losing its number.
        for index in range(len(images)):
            try:
                image = images[index].image
            except Exception:  # pypdf and Pillow raise several types here
                continue
            if image is None or image.width < min_width:
                continue
            path = source.image_path(page_number, index)
            image.convert("RGB").save(path)
            written.append(path)
    return written
