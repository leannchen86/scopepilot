"""Find the control for a task on a screenshot.

Two passes. The first looks at the whole screen and says which control to use
and roughly where it is. The second looks at a native-resolution crop around
that spot and tightens the box, because controls in this kind of software are a
few pixels across once a full screenshot has been shrunk for the model.

The model reports positions as integers from 0 to 1000, as fractions of the
image it was shown. That keeps the answer independent of any resizing between
here and the model.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from scopepilot.imaging import crop, fit_within, zoom_region
from scopepilot.model import Backend
from scopepilot.profiles import Profile
from scopepilot.types import Box, Guidance

NORM = 1000

_EDGES = ("left", "top", "right", "bottom")
_EDGE_PROPERTIES = {edge: {"type": "integer"} for edge in _EDGES}

FIND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "label": {"type": "string"},
        **_EDGE_PROPERTIES,
        "steps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "note": {"type": "string"},
    },
    "required": ["visible", "label", *_EDGES, "steps", "confidence", "note"],
    "additionalProperties": False,
}

REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"found": {"type": "boolean"}, **_EDGE_PROPERTIES},
    "required": ["found", *_EDGES],
    "additionalProperties": False,
}

_COORDINATES = (
    f"Give positions as integers from 0 to {NORM}: left and right as fractions of the image "
    f"width, top and bottom as fractions of its height, with 0,0 at the top-left corner."
)

_SYSTEM = """\
You help a person operate microscope software by looking at a screenshot of it. \
They ask how to do something; you find the one control they should use next and \
say where it is. You never operate the software yourself.

Work from what is on the screenshot. The notes below describe how this software \
is usually laid out; trust the screenshot over the notes when they disagree, \
because layouts change between versions and instruments.

If the control is on screen, set visible to true and box it tightly: the button, \
field, slider or tab itself, not the panel around it. If it is not on screen, set \
visible to false, set the four positions to 0, and use steps to say what to open \
first to get to it. If the screenshot does not let you tell, say so in note and \
set confidence to low; a wrong pointer costs the user more than an honest "not sure".

Write steps as short instructions a person at the microscope can follow, one \
action each, naming controls by the text or icon shown on screen.

{profile}"""

_FIND = """\
The screenshot is attached. The user asks: {question}

Find the control they should use next. {coordinates}"""

_REFINE = """\
This is a zoomed-in crop of a larger screenshot. Find this control in it: {label}
It is the control for: {question}

Box the control itself as tightly as you can. {coordinates} \
If the control is not in this crop, set found to false and the four positions to 0."""


def _box_from(answer: dict[str, Any], width: int, height: int) -> Box | None:
    """Turn the model's 0-1000 positions into pixels, or None if they make no box."""
    try:
        left, top, right, bottom = (float(answer[edge]) for edge in _EDGES)
    except (KeyError, TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    box = Box(left, top, right, bottom).clamped(NORM, NORM)
    if box.width == 0 or box.height == 0:
        return None
    return box.scaled(width / NORM, height / NORM)


def locate(
    screenshot: Image.Image,
    question: str,
    profile: Profile,
    backend: Backend,
    *,
    refine: bool = True,
    max_edge: int = 1920,
) -> Guidance:
    """Answer "where is the control for `question`?" on one screenshot."""
    system = _SYSTEM.format(profile=profile.prompt_text())
    shown = fit_within(screenshot, max_edge)
    answer = backend.ask(
        system, _FIND.format(question=question, coordinates=_COORDINATES), [shown], FIND_SCHEMA
    )

    label = str(answer.get("label", ""))
    box = _box_from(answer, screenshot.width, screenshot.height) if answer.get("visible") else None
    if box is not None and refine:
        box = _refine(screenshot, box, label, question, system, backend, max_edge)

    return Guidance(
        visible=box is not None,
        label=label,
        box=box,
        steps=tuple(str(step) for step in answer.get("steps", [])),
        confidence=str(answer.get("confidence", "low")),
        note=str(answer.get("note", "")),
    )


def _refine(
    screenshot: Image.Image,
    rough: Box,
    label: str,
    question: str,
    system: str,
    backend: Backend,
    max_edge: int,
) -> Box:
    """Look again at a native-resolution crop around `rough`. Keeps `rough` if that fails."""
    region = zoom_region(screenshot, rough)
    close_up = fit_within(crop(screenshot, region), max_edge)
    answer = backend.ask(
        system,
        _REFINE.format(label=label, question=question, coordinates=_COORDINATES),
        [close_up],
        REFINE_SCHEMA,
    )
    if not answer.get("found"):
        return rough
    tight = _box_from(answer, round(region.width), round(region.height))
    if tight is None:
        return rough
    return tight.shifted(region.left, region.top)
