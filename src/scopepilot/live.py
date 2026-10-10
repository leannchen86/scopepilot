"""Live guide mode: where on the user's own screen the answer is drawn.

`locate()` answers in pixels of the screenshot it was given. A window toolkit
places things in logical units, and the two differ whenever the display is
scaled: a Retina Mac or Windows at 150% hands back a capture that is larger
than the screen's logical size. This module is the arithmetic between the two,
and the choice of where the caption card goes and whether the prompt bar has to
move out of its way. It has no toolkit imports, so it can be tested exactly and
imported without the `live` extra; everything that draws is in `overlay.py`.

Three coordinate spaces are in play, and names say which one a value is in:

- image: pixels of the captured screenshot, as `locate()` returns them.
- screen: logical units measured from the top-left corner of the captured
  screen. The overlay window covers that screen, so this is what it draws in.
- desktop: logical units across all monitors. A second monitor's corner is not
  (0, 0) here, and is negative when it sits left of or above the main one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from scopepilot.types import Box, Guidance

PAD = 6
"""Space between the control and the ring, so the ring never hides its edge."""
RING_INSET = 2
"""How far inside the screen edge a ring stops, so its stroke is not cut off."""
GAP = 12
"""Space between the ring and the caption card."""
MARGIN = 8
"""Closest the caption card comes to a screen edge."""
BAR_MARGIN = 48
"""Space under the prompt bar."""

BESIDE_CONTROL = ("right", "left", "below", "above")
"""Controls in this kind of software sit in columns in side panels, so a card
to the side covers the image viewer rather than the neighbouring controls."""
BY_PROMPT_BAR = ("above", "below", "left", "right")


class GuideUnavailable(OSError):
    """Guide mode cannot start on this machine.

    An OSError so the command line reports it as one line, the way it reports a
    missing file, instead of a traceback.
    """


@dataclass(frozen=True)
class Screen:
    """One monitor: its top-left corner on the desktop and its logical size."""

    left: float
    top: float
    width: float
    height: float

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)


@dataclass(frozen=True)
class Caption:
    """The words on the caption card."""

    title: str
    steps: tuple[str, ...]
    note: str
    confidence: str


@dataclass(frozen=True)
class Placement:
    """Where the overlay draws, in screen coordinates."""

    ring: Box | None
    """Outline around the control. None when the control is not on this screen."""
    card: Box
    bar: Box
    """Where the prompt bar belongs: where it was, unless it sat on the ring or the card."""


def load_overlay() -> ModuleType:
    """Import the Qt half, or say how to get it.

    Imported on demand so `import scopepilot` and the other commands work
    without PySide6 installed.
    """
    try:
        from scopepilot import overlay
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "PySide6":
            raise
        raise GuideUnavailable(
            "guide mode needs PySide6; install it with: pip install 'scopepilot[live]'"
        ) from error
    return overlay


def image_to_screen(box: Box, image_size: tuple[int, int], screen: Screen) -> Box:
    """Map a box in image pixels to screen coordinates.

    The scale comes from the capture itself (image size over logical size)
    rather than from the device pixel ratio the system reports, which can
    disagree with what was actually captured. Each axis uses its own ratio so a
    capture that was rounded by a pixel still lines up at the far edge.
    """
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"captured image has no size: {image_size}")
    scaled = box.scaled(screen.width / image_width, screen.height / image_height)
    return scaled.clamped(screen.width, screen.height)


def screen_to_desktop(box: Box, screen: Screen) -> Box:
    return box.shifted(screen.left, screen.top)


def desktop_to_screen(box: Box, screen: Screen) -> Box:
    return box.shifted(-screen.left, -screen.top)


def ring_around(
    target: Box, screen_size: tuple[float, float], pad: float = PAD, inset: float = RING_INSET
) -> Box:
    """The outline to draw around `target`: padded, and stopped just inside the screen."""
    width, height = screen_size
    box = target.padded(pad)
    return Box(
        max(box.left, inset),
        max(box.top, inset),
        min(box.right, width - inset),
        min(box.bottom, height - inset),
    )


def _slide(position: float, size: float, limit: float, margin: float) -> float:
    """Move a span of `size` so it lies within `limit`, keeping its start when it cannot fit."""
    return max(margin, min(position, limit - margin - size))


def _beside(
    avoid: Box,
    card_size: tuple[float, float],
    screen_size: tuple[float, float],
    side: str,
    gap: float,
    margin: float,
) -> Box:
    """The card on one side of `avoid`, centred on it, then slid back on screen."""
    width, height = card_size
    cx, cy = avoid.center
    left, top = {
        "right": (avoid.right + gap, cy - height / 2),
        "left": (avoid.left - gap - width, cy - height / 2),
        "below": (cx - width / 2, avoid.bottom + gap),
        "above": (cx - width / 2, avoid.top - gap - height),
    }[side]
    left = _slide(left, width, screen_size[0], margin)
    top = _slide(top, height, screen_size[1], margin)
    return Box(left, top, left + width, top + height)


def _overlap(a: Box, b: Box) -> float:
    width = min(a.right, b.right) - max(a.left, b.left)
    height = min(a.bottom, b.bottom) - max(a.top, b.top)
    return max(width, 0) * max(height, 0)


def place_card(
    avoid: Box,
    card_size: tuple[float, float],
    screen_size: tuple[float, float],
    sides: tuple[str, ...] = BESIDE_CONTROL,
    gap: float = GAP,
    margin: float = MARGIN,
    also: Box | None = None,
) -> Box:
    """Put the caption card next to `avoid` without covering it or leaving the screen.

    Tries each side in order and takes the first where the card fits. Staying
    on screen wins over staying clear: when `avoid` is so large that no side
    has room, the card goes where it covers the least of it.

    `also` is the prompt bar, which is drawn on top of the card. Between sides
    that are equally clear of `avoid`, the card takes the one where the bar
    hides the least of it.
    """
    candidates = [_beside(avoid, card_size, screen_size, side, gap, margin) for side in sides]

    def covered(card: Box) -> tuple[float, float]:
        return (_overlap(card, avoid), _overlap(card, also) if also is not None else 0)

    # min() keeps the first of equals, so the order of `sides` is the preference.
    return min(candidates, key=covered)


def bar_clear_of(
    ring: Box,
    card: Box,
    bar: Box,
    screen_size: tuple[float, float],
    gap: float = GAP,
    margin: float = MARGIN,
) -> Box:
    """The prompt bar, moved straight up or down if it was sitting on the answer.

    The bar is opaque and comes back on top once the answer is in. Left on the
    control it would hide the thing being pointed at and take the click meant
    for it; left on the card it would hide the steps. It moves clear of both
    when there is room above or below them, otherwise clear of the control
    alone, and stays where it is when it is not in the way or cannot get out.
    """
    if _overlap(bar, ring) == 0 and _overlap(bar, card) == 0:
        return bar
    both = Box(
        min(ring.left, card.left),
        min(ring.top, card.top),
        max(ring.right, card.right),
        max(ring.bottom, card.bottom),
    )
    for keep_clear in (both, ring):
        if _overlap(bar, keep_clear) == 0:
            break
        for top in (keep_clear.top - gap - bar.height, keep_clear.bottom + gap):
            if margin <= top <= screen_size[1] - margin - bar.height:
                return bar.shifted(0, top - bar.top)
    return bar


def bar_position(
    bar_size: tuple[float, float], area: Screen, margin: float = BAR_MARGIN
) -> tuple[float, float]:
    """Top-left corner of the prompt bar in desktop coordinates: bottom centre of `area`."""
    width, height = bar_size
    return (area.left + (area.width - width) / 2, area.top + area.height - height - margin)


def arrange(
    box: Box | None,
    image_size: tuple[int, int],
    screen: Screen,
    card_size: tuple[float, float],
    bar: Box,
) -> Placement:
    """Decide what the overlay draws for one answer.

    `box` is in image pixels, as `locate()` returns it. `bar` is the prompt bar
    in desktop coordinates; with no control to point at, the card goes by the
    bar, where the user is already looking. With one, the card keeps out from
    under the bar where it can, and the bar moves if it is still in the way.
    """
    bar = desktop_to_screen(bar, screen)
    if box is None:
        return Placement(None, place_card(bar, card_size, screen.size, BY_PROMPT_BAR), bar)
    ring = ring_around(image_to_screen(box, image_size, screen), screen.size)
    card = place_card(ring, card_size, screen.size, also=bar)
    return Placement(ring, card, bar_clear_of(ring, card, bar, screen.size))


def caption(guidance: Guidance) -> Caption:
    # The model leaves the label empty when the question is not about a control
    # at all, or the screen is not showing the software.
    if guidance.box is not None:
        title = guidance.label or "This control"
    elif guidance.label:
        title = f"Not on this screen: {guidance.label}"
    else:
        title = "Nothing to point at on this screen"
    steps = tuple(f"{number}. {step}" for number, step in enumerate(guidance.steps, start=1))
    note = f"Note: {guidance.note}" if guidance.note else ""
    return Caption(title, steps, note, f"Confidence: {guidance.confidence}")
