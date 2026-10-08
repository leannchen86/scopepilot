import hashlib
from types import SimpleNamespace

import pypdf
import pytest
from PIL import Image

from scopepilot import corpus
from scopepilot.corpus import CorpusError, Source

PDF = b"%PDF-1.7 pretend guide"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SCOPEPILOT_CACHE", str(tmp_path / "cache"))
    return tmp_path


def source_for(tmp_path, body, sha256=""):
    served = tmp_path / "served.pdf"
    served.write_bytes(body)
    return Source("guide", served.as_uri(), "lasx-industry", "someone", sha256)


def test_fetch_caches_a_guide_that_matches_its_hash(cache):
    source = source_for(cache, PDF, hashlib.sha256(PDF).hexdigest())

    assert corpus.fetch(source).read_bytes() == PDF


def test_fetch_rejects_a_response_that_is_not_a_pdf_and_caches_nothing(cache):
    source = source_for(cache, b"<html>moved</html>")

    with pytest.raises(CorpusError, match="did not return a PDF"):
        corpus.fetch(source)
    assert not source.pdf.exists()
    assert not list(source.directory.iterdir())


def test_fetch_rejects_a_changed_guide_and_caches_nothing(cache):
    source = source_for(cache, PDF, "0" * 64)

    with pytest.raises(CorpusError, match="not the file the test cases were labelled against"):
        corpus.fetch(source)
    assert not source.pdf.exists()


def test_fetch_replaces_a_cached_file_that_does_not_match(cache):
    source = source_for(cache, PDF, hashlib.sha256(PDF).hexdigest())
    source.directory.mkdir(parents=True)
    source.pdf.write_bytes(b"%PDF- an older or broken download")

    assert corpus.fetch(source).read_bytes() == PDF


class FakeImages:
    """Like pypdf's page.images: decoding happens on lookup and can fail."""

    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if entry is None:
            raise OSError("cannot identify image file")
        return SimpleNamespace(image=entry)


def test_an_image_that_cannot_be_decoded_is_skipped_and_keeps_its_number(cache, monkeypatch):
    wide, narrow = Image.new("RGB", (800, 600)), Image.new("RGB", (100, 80))
    page = SimpleNamespace(images=FakeImages([wide, None, narrow, wide]))
    monkeypatch.setattr(pypdf, "PdfReader", lambda path: SimpleNamespace(pages=[page]))
    source = Source("guide", "unused", "lasx-industry", "someone")

    written = corpus.extract_images(source)

    assert [path.name for path in written] == ["p001-00.png", "p001-03.png"]
