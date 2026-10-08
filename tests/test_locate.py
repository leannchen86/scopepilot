import pytest
from PIL import Image

from scopepilot.locate import FIND_SCHEMA, REFINE_SCHEMA, locate
from scopepilot.profiles import load_profile
from scopepilot.types import Box


class ScriptedBackend:
    """Answers from a list, and records what it was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def ask(self, system, prompt, images, schema):
        self.calls.append({"system": system, "prompt": prompt, "images": images, "schema": schema})
        return self.answers.pop(0)


def found(left, top, right, bottom, **extra):
    return {
        "visible": True,
        "label": "Live Image",
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "steps": ["Click Live Image."],
        "confidence": "high",
        "note": "",
        **extra,
    }


PROFILE = load_profile("lasx-industry")


def test_positions_are_scaled_to_screenshot_pixels():
    screenshot = Image.new("RGB", (2000, 1000))
    backend = ScriptedBackend([found(100, 200, 150, 260)])

    guidance = locate(screenshot, "start live view", PROFILE, backend, refine=False)

    assert guidance.visible
    assert guidance.box == Box(200, 200, 300, 260)
    assert guidance.steps == ("Click Live Image.",)


def test_large_screenshot_is_shrunk_for_the_model_but_answer_is_in_original_pixels():
    screenshot = Image.new("RGB", (3840, 2160))
    backend = ScriptedBackend([found(500, 500, 510, 510)])

    guidance = locate(screenshot, "q", PROFILE, backend, refine=False, max_edge=1920)

    assert max(backend.calls[0]["images"][0].size) == 1920
    assert guidance.box.center == pytest.approx((1939.2, 1090.8))


def test_refine_maps_the_crop_answer_back_to_the_screenshot():
    screenshot = Image.new("RGB", (2000, 1000))
    # Rough box 40x20 px centred on (220, 110). The crop is 320x320, slid to the
    # top-left corner of the screenshot because the box is near it.
    backend = ScriptedBackend(
        [found(100, 100, 120, 120), {"found": True, "left": 500, "top": 250, "right": 750, "bottom": 500}]
    )

    guidance = locate(screenshot, "q", PROFILE, backend)

    assert backend.calls[0]["schema"] is FIND_SCHEMA
    assert backend.calls[1]["schema"] is REFINE_SCHEMA
    assert backend.calls[1]["images"][0].size == (320, 320)
    assert guidance.box == Box(220, 80, 300, 160)


def test_refine_keeps_the_rough_box_when_the_closer_look_fails():
    screenshot = Image.new("RGB", (2000, 1000))
    rough = found(100, 100, 120, 120)
    not_found = {"found": False, "left": 0, "top": 0, "right": 0, "bottom": 0}
    nonsense = {"found": True, "left": 600, "top": 600, "right": 400, "bottom": 400}

    for second in (not_found, nonsense):
        guidance = locate(screenshot, "q", PROFILE, ScriptedBackend([rough, second]))
        assert guidance.box == Box(200, 100, 240, 120)


def test_control_not_on_screen_gives_steps_and_no_box():
    backend = ScriptedBackend(
        [
            {
                "visible": False,
                "label": "Scale Bar Properties",
                "left": 0,
                "top": 0,
                "right": 0,
                "bottom": 0,
                "steps": ["Open the annotation tools."],
                "confidence": "medium",
                "note": "",
            }
        ]
    )

    guidance = locate(Image.new("RGB", (800, 600)), "scale bar", PROFILE, backend)

    assert not guidance.visible
    assert guidance.box is None
    assert guidance.steps == ("Open the annotation tools.",)
    assert len(backend.calls) == 1


def test_a_visible_answer_with_no_usable_box_is_reported_as_not_visible():
    backend = ScriptedBackend([found(300, 300, 300, 300)])

    guidance = locate(Image.new("RGB", (800, 600)), "q", PROFILE, backend)

    assert not guidance.visible
    assert guidance.box is None


def test_the_profile_notes_reach_the_model():
    backend = ScriptedBackend([found(1, 1, 2, 2)])

    locate(Image.new("RGB", (800, 600)), "q", PROFILE, backend, refine=False)

    assert "Scale Bar Properties" in backend.calls[0]["system"]
