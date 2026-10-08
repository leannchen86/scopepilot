"""Shared value types.

Coordinates are pixels in the original screenshot, origin top-left, unless a
name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"box has negative size: {self}")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def scaled(self, sx: float, sy: float) -> Box:
        return Box(self.left * sx, self.top * sy, self.right * sx, self.bottom * sy)

    def shifted(self, dx: float, dy: float) -> Box:
        return Box(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    def padded(self, pad: float) -> Box:
        return Box(self.left - pad, self.top - pad, self.right + pad, self.bottom + pad)

    def clamped(self, width: float, height: float) -> Box:
        """Clip to an image of the given size."""
        left = min(max(self.left, 0), width)
        top = min(max(self.top, 0), height)
        return Box(left, top, min(max(self.right, left), width), min(max(self.bottom, top), height))

    def as_ints(self) -> tuple[int, int, int, int]:
        return (round(self.left), round(self.top), round(self.right), round(self.bottom))


@dataclass(frozen=True)
class Guidance:
    """The answer to "where is the control for X?" on one screenshot."""

    visible: bool
    """Whether the control is on this screen. When False, `steps` says how to reach it."""
    label: str
    """What the control is called on screen, or the best description of it."""
    box: Box | None
    """Where the control is. None when it is not visible."""
    steps: tuple[str, ...]
    """What the user should do, in order."""
    confidence: str
    """high, medium or low: the model's own estimate, not a measured accuracy."""
    note: str = ""
    """Anything the user should know first, such as a known trap."""
