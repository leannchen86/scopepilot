"""Image helpers: resize for the model, crop for a closer look, draw the answer."""

from __future__ import annotations

from PIL import Image, ImageDraw

from scopepilot.types import Box

HIGHLIGHT = (255, 45, 85)


def fit_within(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink so the long edge is at most `max_edge`. Never enlarges."""
    long_edge = max(image.size)
    if long_edge <= max_edge:
        return image
    ratio = max_edge / long_edge
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def zoom_region(image: Image.Image, box: Box, factor: float = 4.0, min_edge: int = 320) -> Box:
    """A region around `box` to crop for a second, closer look.

    The region is `factor` times the box, at least `min_edge` pixels a side, and
    is slid back inside the image when it would cross an edge so the crop keeps
    its full size wherever the image allows.
    """
    width = min(image.width, max(box.width * factor, min_edge))
    height = min(image.height, max(box.height * factor, min_edge))
    cx, cy = box.center
    left = min(max(cx - width / 2, 0), image.width - width)
    top = min(max(cy - height / 2, 0), image.height - height)
    return Box(left, top, left + width, top + height)


def crop(image: Image.Image, region: Box) -> Image.Image:
    return image.crop(region.clamped(image.width, image.height).as_ints())


def annotate(
    image: Image.Image, box: Box, pad: int = 6, color: tuple[int, int, int] = HIGHLIGHT
) -> Image.Image:
    """Return a copy of the screenshot with the target outlined."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    outline = box.padded(pad).clamped(out.width - 1, out.height - 1)
    stroke = max(3, round(max(out.size) / 500))
    draw.rectangle(outline.as_ints(), outline=color, width=stroke)
    return out
