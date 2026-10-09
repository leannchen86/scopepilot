"""Live guide mode: the coordinate arithmetic first, then the Qt windows.

The Qt tests run on Qt's "offscreen" platform, which draws into memory. They
never open a window on the real display and never capture it: the capture
function is always a fake that returns a blank image of a chosen size.
"""

import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

# Qt chooses its platform when the application object is created, and reads
# this then. Set before anything can import PySide6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image

from scopepilot import cli, live
from scopepilot.imaging import HIGHLIGHT
from scopepilot.live import GAP, MARGIN, PAD, RING_INSET, Placement, Screen
from scopepilot.model import ModelError
from scopepilot.profiles import load_profile
from scopepilot.types import Box, Guidance

PROFILE = load_profile("lasx-industry")
FULL_HD = Screen(0, 0, 1920, 1080)
CARD = (340, 200)


def overlap(a: Box, b: Box) -> float:
    width = min(a.right, b.right) - max(a.left, b.left)
    height = min(a.bottom, b.bottom) - max(a.top, b.top)
    return max(width, 0) * max(height, 0)


def on_screen(box: Box, size: tuple[float, float], margin: float = 0) -> bool:
    return (
        box.left >= margin
        and box.top >= margin
        and box.right <= size[0] - margin
        and box.bottom <= size[1] - margin
    )


# --- image pixels to screen coordinates ------------------------------------


def test_scale_1_leaves_the_box_where_it_is():
    box = Box(200, 100, 400, 160)

    assert live.image_to_screen(box, (1920, 1080), FULL_HD) == box


def test_scale_2_halves_the_box():
    # A Retina capture: twice the pixels of the logical screen.
    box = Box(200, 100, 400, 160)

    assert live.image_to_screen(box, (3840, 2160), FULL_HD) == Box(100, 50, 200, 80)


def test_fractional_scale():
    # Windows at 150%.
    box = Box(300, 150, 600, 300)

    assert live.image_to_screen(box, (2880, 1620), FULL_HD) == Box(200, 100, 400, 200)


def test_scale_comes_from_the_image_size_on_each_axis():
    # A capture one pixel short of exactly 1.5x still maps edge to edge.
    whole_image = Box(0, 0, 2879, 1619)

    mapped = live.image_to_screen(whole_image, (2879, 1619), FULL_HD)

    assert mapped.as_ints() == (0, 0, 1920, 1080)


def test_width_and_height_are_scaled_separately_and_not_swapped():
    # Half scale across, quarter scale down. No real display does this, but
    # with the two ratios equal a mix-up between the axes would go unnoticed.
    stretched = Screen(0, 0, 1000, 250)

    mapped = live.image_to_screen(Box(200, 100, 400, 160), (2000, 1000), stretched)

    assert mapped == Box(100, 25, 200, 40)


def test_a_capture_with_a_slightly_different_shape_still_maps_edge_to_edge():
    # 16:10 pixels handed back for a 16:9 logical screen: the far corner of the
    # image is the far corner of the screen, and the middle is the middle.
    corner = live.image_to_screen(Box(3800, 2360, 3840, 2400), (3840, 2400), FULL_HD)
    middle = live.image_to_screen(Box(1900, 1190, 1940, 1210), (3840, 2400), FULL_HD)

    assert corner == Box(1900, 1062, 1920, 1080)
    assert middle.center == (960, 540)
    assert (middle.width, middle.height) == (20, 9)


@pytest.mark.parametrize("origin", [(0, 0), (1920, 0), (-2560, -300), (0, 1080)])
def test_screen_coordinates_do_not_depend_on_where_the_monitor_sits(origin):
    screen = Screen(*origin, 2560, 1440)
    box = Box(5000, 2800, 5120, 2880)  # bottom-right corner of a 2x capture

    assert live.image_to_screen(box, (5120, 2880), screen) == Box(2500, 1400, 2560, 1440)


def test_desktop_coordinates_add_the_monitor_origin():
    second = Screen(-2560, -300, 2560, 1440)
    on_second = Box(100, 50, 200, 80)

    on_desktop = live.screen_to_desktop(on_second, second)

    assert on_desktop == Box(-2460, -250, -2360, -220)
    assert live.desktop_to_screen(on_desktop, second) == on_second


def test_an_image_with_no_size_is_an_error_not_a_division_by_zero():
    with pytest.raises(ValueError, match="no size"):
        live.image_to_screen(Box(0, 0, 1, 1), (0, 0), FULL_HD)


def test_a_box_past_the_image_edge_is_clipped_to_the_screen():
    mapped = live.image_to_screen(Box(3800, 2100, 3900, 2200), (3840, 2160), FULL_HD)

    assert mapped == Box(1900, 1050, 1920, 1080)


# --- the ring ---------------------------------------------------------------


def test_ring_is_the_target_plus_padding():
    assert live.ring_around(Box(500, 400, 550, 420), FULL_HD.size) == Box(494, 394, 556, 426)


# Where a 40x20 control sits on a 3840x2160 capture of a 1920x1080 screen.
EDGES_AND_CORNERS = {
    "top-left": (0, 0),
    "top": (1900, 0),
    "top-right": (3800, 0),
    "right": (3800, 1070),
    "bottom-right": (3800, 2140),
    "bottom": (1900, 2140),
    "bottom-left": (0, 2140),
    "left": (0, 1070),
}


@pytest.mark.parametrize("place", EDGES_AND_CORNERS)
def test_control_at_a_screen_edge_or_corner(place):
    left, top = EDGES_AND_CORNERS[place]
    target = live.image_to_screen(Box(left, top, left + 40, top + 20), (3840, 2160), FULL_HD)

    placement = live.arrange(
        Box(left, top, left + 40, top + 20), (3840, 2160), FULL_HD, CARD, Box(700, 950, 1220, 1030)
    )
    ring, card = placement.ring, placement.card

    assert target == Box(left / 2, top / 2, left / 2 + 20, top / 2 + 10)
    # The ring stays where its stroke can be seen, and is padded on every side
    # that is not against the screen edge.
    assert on_screen(ring, FULL_HD.size, RING_INSET)
    assert ring.left == max(target.left - PAD, RING_INSET)
    assert ring.top == max(target.top - PAD, RING_INSET)
    assert ring.right == min(target.right + PAD, 1920 - RING_INSET)
    assert ring.bottom == min(target.bottom + PAD, 1080 - RING_INSET)
    assert on_screen(card, FULL_HD.size, MARGIN)
    assert overlap(card, ring) == 0
    assert (card.width, card.height) == CARD


# --- the caption card -------------------------------------------------------


def test_card_goes_to_the_right_centred_on_the_target_when_there_is_room():
    target = Box(400, 500, 460, 540)

    card = live.place_card(target, CARD, FULL_HD.size)

    assert card == Box(460 + GAP, 420, 460 + GAP + 340, 620)


def test_card_that_would_fall_off_the_right_edge_goes_to_the_left():
    target = Box(1700, 500, 1760, 540)

    card = live.place_card(target, CARD, FULL_HD.size)

    assert card == Box(1700 - GAP - 340, 420, 1700 - GAP, 620)


def test_card_that_would_fall_off_the_top_is_slid_down():
    target = Box(400, 4, 460, 30)

    card = live.place_card(target, CARD, FULL_HD.size)

    assert card == Box(472, MARGIN, 812, MARGIN + 200)


def test_card_that_would_fall_off_the_bottom_is_slid_up():
    target = Box(400, 1050, 460, 1076)

    card = live.place_card(target, CARD, FULL_HD.size)

    assert card == Box(472, 1080 - MARGIN - 200, 812, 1080 - MARGIN)


def test_card_goes_below_a_target_that_spans_the_width():
    toolbar = Box(0, 40, 1920, 80)

    card = live.place_card(toolbar, CARD, FULL_HD.size)

    assert card == Box(790, 80 + GAP, 1130, 80 + GAP + 200)


def test_card_goes_above_a_target_that_spans_the_width_at_the_bottom():
    status_bar = Box(0, 1040, 1920, 1080)

    card = live.place_card(status_bar, CARD, FULL_HD.size)

    assert card == Box(790, 1040 - GAP - 200, 1130, 1040 - GAP)


@pytest.mark.parametrize(
    ("target", "edge"),
    [
        (Box(0, 40, 300, 80), "left"),  # card below would hang off the left edge
        (Box(1700, 40, 1920, 80), "right"),
    ],
)
def test_card_above_or_below_is_slid_sideways_onto_the_screen(target, edge):
    card = live.place_card(target, CARD, FULL_HD.size, sides=("below", "above"))

    assert card.top == 80 + GAP
    assert on_screen(card, FULL_HD.size, MARGIN)
    assert card.left == MARGIN if edge == "left" else card.right == 1920 - MARGIN


def test_card_keeps_clear_even_when_it_has_to_sit_closer_than_the_gap():
    # 353 units to the right of the target: room for the card and its margin
    # (348), but not for the full gap as well (360).
    target = Box(100, 500, 1567, 540)

    card = live.place_card(target, CARD, FULL_HD.size, sides=("right",))

    assert card.left == 1920 - MARGIN - 340
    assert overlap(card, target) == 0


def test_target_too_large_to_avoid_gets_the_card_where_it_covers_least():
    # A control the size of the image viewer. Nothing fits beside it, so the
    # card stays on screen and overlaps as little as it can: along the bottom.
    viewer = Box(20, 20, 1900, 1060)

    card = live.place_card(viewer, CARD, FULL_HD.size)

    assert card == Box(790, 872, 1130, 1072)
    assert on_screen(card, FULL_HD.size, MARGIN)
    for side in ("right", "left", "below", "above"):
        other = live.place_card(viewer, CARD, FULL_HD.size, sides=(side,))
        assert overlap(card, viewer) <= overlap(other, viewer)


def test_target_covering_the_whole_screen_still_gets_a_card_on_screen():
    card = live.place_card(Box(0, 0, 1920, 1080), CARD, FULL_HD.size)

    assert on_screen(card, FULL_HD.size, MARGIN)
    assert (card.width, card.height) == CARD


def test_card_larger_than_the_screen_keeps_its_top_left_corner_visible():
    card = live.place_card(Box(100, 100, 140, 120), (340, 900), (640, 480))

    assert (card.left, card.top) == (140 + GAP, MARGIN)


# --- the prompt bar and the answer as a whole --------------------------------


def test_bar_sits_bottom_centre_of_the_usable_area_of_its_monitor():
    # A second monitor to the right, with a menu bar taking the top 23 units.
    area = Screen(1920, 23, 1280, 700)

    assert live.bar_position((520, 78), area) == (1920 + 380, 23 + 700 - 78 - live.BAR_MARGIN)


def test_arrange_maps_scales_and_places_in_one_go():
    second = Screen(-2560, -300, 2560, 1440)

    placement = live.arrange(
        Box(1000, 800, 1100, 840), (5120, 2880), second, CARD, Box(-1540, 1000, -1020, 1078)
    )

    # The bar is nowhere near the control, so it stays: at left 1020, top 1300
    # in that monitor's own coordinates.
    assert placement == Placement(
        ring=Box(494, 394, 556, 426),
        card=Box(556 + GAP, 310, 556 + GAP + 340, 510),
        bar=Box(1020, 1300, 1540, 1378),
    )


def test_control_not_on_screen_gets_no_ring_and_a_card_above_the_bar():
    second = Screen(-2560, -300, 2560, 1440)
    bar = Box(-1540, 1000, -1020, 1078)  # desktop coordinates, on the second monitor

    placement = live.arrange(None, (5120, 2880), second, CARD, bar)

    # In that monitor's own coordinates the bar is at left 1020, top 1300.
    assert placement.ring is None
    assert placement.card == Box(1110, 1300 - GAP - 200, 1450, 1300 - GAP)
    assert placement.bar == Box(1020, 1300, 1540, 1378)


def test_control_not_on_screen_with_the_bar_dragged_to_the_top():
    placement = live.arrange(None, (1920, 1080), FULL_HD, CARD, Box(700, 10, 1220, 88))

    assert placement.card == Box(790, 88 + GAP, 1130, 88 + GAP + 200)


# --- keeping the prompt bar off the answer -----------------------------------

BAR = Box(700, 954, 1220, 1032)  # bottom centre of a full-HD screen, where it starts


def test_bar_that_is_clear_of_the_answer_stays_where_it_is():
    placement = live.arrange(Box(400, 500, 460, 540), (1920, 1080), FULL_HD, CARD, BAR)

    assert placement.bar == BAR
    assert placement.card == Box(466 + GAP, 420, 466 + GAP + 340, 620)


def test_bar_sitting_on_the_control_moves_above_the_control_and_the_card():
    # A slider at the bottom of the viewer, right where the bar starts out.
    placement = live.arrange(Box(900, 980, 960, 1000), (1920, 1080), FULL_HD, CARD, BAR)
    ring, card, bar = placement.ring, placement.card, placement.bar

    assert overlap(BAR, ring) > 0  # left alone, the bar would hide the control
    assert bar == Box(700, card.top - GAP - 78, 1220, card.top - GAP)
    assert overlap(bar, ring) == 0 and overlap(bar, card) == 0
    assert overlap(card, ring) == 0
    assert on_screen(bar, FULL_HD.size, MARGIN) and on_screen(card, FULL_HD.size, MARGIN)


def test_bar_on_another_monitor_is_moved_in_that_monitors_coordinates():
    second = Screen(-1920, 200, 1920, 1080)

    placement = live.arrange(
        Box(900, 980, 960, 1000), (1920, 1080), second, CARD, live.screen_to_desktop(BAR, second)
    )

    assert placement == live.arrange(Box(900, 980, 960, 1000), (1920, 1080), FULL_HD, CARD, BAR)


def test_card_takes_the_side_where_the_bar_does_not_hide_it():
    # A control just left of the bar. The card's first choice, to the right,
    # is under the bar; to the left it is in the clear, and the bar can stay.
    placement = live.arrange(Box(600, 980, 660, 1000), (1920, 1080), FULL_HD, CARD, BAR)

    assert placement.card == Box(594 - GAP - 340, 872, 594 - GAP, 1072)
    assert placement.bar == BAR


def test_bar_moves_off_the_control_alone_when_it_cannot_clear_the_card_too():
    small = (1000, 400)
    ring, bar = Box(450, 300, 550, 330), Box(240, 290, 760, 368)
    tall_card = Box(562, 60, 902, 392)  # no 78 units free above or below it

    moved = live.bar_clear_of(ring, tall_card, bar, small)

    assert moved == Box(240, 300 - GAP - 78, 760, 300 - GAP)
    assert overlap(moved, ring) == 0


@pytest.mark.parametrize(
    "ring",
    [
        Box(450, 50, 550, 380),  # under the bar, and too tall to get above or below
        Box(50, 300, 90, 330),  # not under the bar: only the card is
    ],
)
def test_bar_that_cannot_get_out_of_the_way_stays_where_it_is(ring):
    bar, tall_card = Box(240, 290, 760, 368), Box(562, 60, 902, 392)

    assert live.bar_clear_of(ring, tall_card, bar, (1000, 400)) == bar


def test_bar_in_the_space_between_the_control_and_the_card_stays_where_it_is():
    # Inside the rectangle that holds both, but touching neither.
    ring, card, bar = Box(50, 600, 110, 630), Box(600, 300, 940, 500), Box(130, 560, 590, 640)

    assert live.bar_clear_of(ring, card, bar, FULL_HD.size) == bar


def test_caption_for_a_control_on_screen():
    guidance = Guidance(
        visible=True,
        label="Live Image",
        box=Box(1, 2, 3, 4),
        steps=("Click Live Image.", "Wait for the image."),
        confidence="high",
        note="Exposure is not kept between projects.",
    )

    assert live.caption(guidance) == live.Caption(
        title="Live Image",
        steps=("1. Click Live Image.", "2. Wait for the image."),
        note="Note: Exposure is not kept between projects.",
        confidence="Confidence: high",
    )


def test_caption_for_a_control_not_on_screen():
    guidance = Guidance(
        visible=False, label="Scale Bar", box=None, steps=("Open Annotations.",), confidence="low"
    )

    caption = live.caption(guidance)

    assert caption.title == "Not on this screen: Scale Bar"
    assert caption.steps == ("1. Open Annotations.",)
    assert caption.note == ""


# --- the command line, and running without PySide6 ---------------------------


class FakeOverlayModule:
    def __init__(self):
        self.calls = []

    def run(self, profile, backend, **options):
        self.calls.append((profile, backend, options))
        return 0


def test_guide_command_passes_its_options_to_the_overlay(monkeypatch):
    fake = FakeOverlayModule()
    monkeypatch.setattr(live, "load_overlay", lambda: fake)
    monkeypatch.setattr(cli, "_backend", lambda name, model: (name, model))

    assert cli.main(["guide", "--profile", "lasx-industry"]) == 0
    assert cli.main(
        ["guide", "--profile", "lasx-widefield", "--ask", "where is live?", "--seconds", "3",
         "--backend", "api", "--model", "some-model", "--no-refine"]
    ) == 0  # fmt: skip

    (profile, _, interactive), (other_profile, backend, one_shot) = fake.calls
    assert profile.id == "lasx-industry"
    assert interactive == {"question": None, "seconds": 15.0, "refine": True}
    assert other_profile.id == "lasx-widefield"
    assert backend == ("api", "some-model")
    assert one_shot == {"question": "where is live?", "seconds": 3.0, "refine": False}


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter, where blocking an import cannot leak into other tests."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )


# None in sys.modules makes `import PySide6` fail exactly as if it were not installed.
WITHOUT_PYSIDE6 = "import sys; sys.modules['PySide6'] = None\n"


def test_guide_without_pyside6_says_what_to_install():
    done = run_python(
        WITHOUT_PYSIDE6
        + "from scopepilot import cli\n"
        + "sys.exit(cli.main(['guide', '--profile', 'lasx-industry', '--ask', 'q']))\n"
    )

    assert done.returncode == 1
    assert done.stderr.strip().splitlines() == [
        "scopepilot: guide mode needs PySide6; install it with: pip install 'scopepilot[live]'"
    ]


def test_everything_else_imports_and_runs_without_pyside6():
    done = run_python(
        WITHOUT_PYSIDE6
        + "import scopepilot, scopepilot.live\n"
        + "from scopepilot import cli\n"
        + "try:\n"
        + "    cli.main(['where', '--help'])\n"
        + "except SystemExit as stop:\n"
        + "    assert stop.code == 0\n"
        + "assert not [name for name in sys.modules if name.startswith('PySide6.')]\n"
        + "assert 'scopepilot.overlay' not in sys.modules\n"
    )

    assert done.returncode == 0, done.stderr


def test_a_different_missing_module_is_not_blamed_on_pyside6():
    # Some other import failing inside overlay.py is a bug, and must stay a traceback.
    done = run_python(
        "import sys\n"
        + "from scopepilot import live\n"
        + "sys.modules['PIL'] = None\n"
        + "live.load_overlay()\n"
    )

    assert done.returncode == 1
    assert "ModuleNotFoundError" in done.stderr
    assert "scopepilot[live]" not in done.stderr


# --- the Qt windows, headless -------------------------------------------------

# Two monitors: the main one, and a high-DPI one up and to the left of it, so
# its corner on the desktop is negative.
SCREENS = [
    {"name": "main", "x": 0, "y": 0, "width": 1000, "height": 800, "dpr": 1},
    {"name": "side", "x": -1280, "y": 120, "width": 1280, "height": 720, "dpr": 2},
]


def found(left, top, right, bottom):
    """A model answer for a control on screen, in the 0-1000 positions the model uses."""
    return {
        "visible": True,
        "label": "Live Image",
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "steps": ["Click Live Image.", "Wait for the camera image to appear in the viewer."],
        "confidence": "high",
        "note": "Exposure is not kept between projects.",
    }


NOT_ON_SCREEN = {
    "visible": False,
    "label": "Scale Bar",
    "left": 0,
    "top": 0,
    "right": 0,
    "bottom": 0,
    "steps": ["Open the Annotations tab."],
    "confidence": "medium",
    "note": "",
}


class GatedBackend:
    """Gives one scripted answer, records the thread that asked, and can be held shut."""

    def __init__(self, answer):
        self.answer = answer
        self.threads = []
        self.gate = threading.Event()
        self.gate.set()

    def ask(self, system, prompt, images, schema):
        self.threads.append(threading.get_ident())
        assert self.gate.wait(10), "the test never released the backend"
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _platform(config) -> str:
    """The offscreen platform with our two monitors, when Qt can be told where they are.

    Qt splits the platform string on colons, so a path with a drive letter
    cannot be passed; the tests that need the second monitor are then skipped.
    """
    try:
        path = os.path.relpath(config)
    except ValueError:
        return "offscreen"
    return "offscreen" if ":" in path else f"offscreen:configfile={path}"


@pytest.fixture(scope="module")
def qt(tmp_path_factory):
    """The Qt application and modules; skips the test when PySide6 is not installed."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from scopepilot import overlay

    config = tmp_path_factory.mktemp("qt") / "screens.json"
    screens = [dict(screen, logicalDpi=96, logicalBaseDpi=96) for screen in SCREENS]
    config.write_text(json.dumps({"screens": screens}))
    # -platform wins over the environment, so even a stray QT_QPA_PLATFORM
    # cannot make these tests open real windows.
    app = QApplication.instance() or QApplication(["tests", "-platform", _platform(config)])
    if app.platformName() != "offscreen":
        pytest.skip("the Qt tests only run on the offscreen platform")

    def wait_for(condition, timeout=5.0):
        """Run the event loop until `condition()` holds."""
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, "timed out waiting for the GUI"
            app.processEvents()
            time.sleep(0.002)

    named = {screen.name(): screen for screen in app.screens()}
    return SimpleNamespace(
        app=app,
        overlay=overlay,
        main=named.get("main", app.primaryScreen()),
        side=named.get("side"),
        wait_for=wait_for,
        QImage=QImage,
        QPoint=QPoint,
        QTest=QTest,
        QTimer=QTimer,
        Qt=Qt,
    )


@pytest.fixture
def side(qt):
    if qt.side is None:
        pytest.skip("this Qt cannot be given a second offscreen monitor")
    return qt.side


@pytest.fixture
def start_guide(qt):
    """Make a Guide whose bar is on a chosen screen, with a fake backend and fake capture."""
    started = []

    def start(answer, screen, *, scale=1, capture=None, show_bar=True):
        backend = GatedBackend(answer)
        seen = SimpleNamespace(shots=[], answers=[], failures=[], closed=[])
        # The fakes below reach the windows through this, never through the
        # guide itself: a guide that holds a function that holds the guide is
        # only freed by the garbage collector, possibly after Qt has shut down.
        windows = SimpleNamespace(bar=None, overlay=None)

        def blank_capture(target):
            # What was on screen at the moment of capture, and an image `scale`
            # times the logical size, as a high-DPI display would give.
            bar_up, overlay_up = windows.bar.isVisible(), windows.overlay.isVisible()
            seen.shots.append(SimpleNamespace(screen=target, bar_up=bar_up, overlay_up=overlay_up))
            size = target.geometry().size()
            return Image.new("RGB", (round(size.width() * scale), round(size.height() * scale)))

        guide = qt.overlay.Guide(
            PROFILE, backend, refine=False, capture=capture or blank_capture, settle_ms=0
        )
        windows.bar, windows.overlay = guide.bar, guide.overlay
        guide.answered.connect(lambda answer: seen.answers.append((answer, threading.get_ident())))
        guide.failed.connect(seen.failures.append)
        guide.closed.connect(lambda: seen.closed.append(True))
        area = screen.geometry()
        x, y = live.bar_position(
            (guide.bar.width(), guide.bar.height()),
            Screen(area.x(), area.y(), area.width(), area.height()),
        )
        guide.bar.move(round(x), round(y))
        if show_bar:
            guide.start()
        started.append((guide, backend))
        return SimpleNamespace(
            guide=guide, bar=guide.bar, overlay=guide.overlay, backend=backend, seen=seen
        )

    yield start
    for guide, backend in started:
        backend.gate.set()
        guide.shutdown()
        guide.close()
    # Deliver whatever is still queued for these guides before they go.
    qt.app.processEvents()


def type_question(qt, session, text="how do I start live view?"):
    session.bar.field.setText(text)
    qt.QTest.keyClick(session.bar.field, qt.Qt.Key.Key_Return)


def ask_and_wait(qt, session, text="how do I start live view?"):
    before = len(session.seen.answers) + len(session.seen.failures)
    type_question(qt, session, text)
    qt.wait_for(lambda: len(session.seen.answers) + len(session.seen.failures) > before)


def rect_box(rect) -> Box:
    return Box(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())


def edges(box: Box) -> tuple[float, float, float, float]:
    """A box as numbers, to compare with pytest.approx after float scaling."""
    return (box.left, box.top, box.right, box.bottom)


def test_bar_starts_bottom_centre_with_a_hint(qt, start_guide):
    session = start_guide(found(1, 1, 2, 2), qt.main)
    bar, screen = session.bar.geometry(), qt.main.geometry()

    assert session.bar.isVisible()
    assert screen.contains(bar)
    assert bar.center().x() == pytest.approx(screen.center().x(), abs=1)
    assert bar.bottom() < screen.bottom()
    assert session.bar.field.placeholderText() == "Ask how to do something..."
    assert "Esc" in session.bar.status.text()
    assert not session.overlay.isVisible()


def test_bar_and_overlay_are_the_kinds_of_window_the_design_needs(qt, start_guide):
    session = start_guide(found(1, 1, 2, 2), qt.main)
    window, attribute = qt.Qt.WindowType, qt.Qt.WidgetAttribute
    bar, overlay = session.bar.windowFlags(), session.overlay.windowFlags()

    assert bar & window.FramelessWindowHint and bar & window.WindowStaysOnTopHint
    for flag in (
        window.FramelessWindowHint,
        window.WindowStaysOnTopHint,
        window.WindowTransparentForInput,  # clicks reach the software underneath
        window.WindowDoesNotAcceptFocus,
    ):
        assert overlay & flag
    # A tool window stays out of the taskbar and the window switcher.
    assert overlay & window.WindowType_Mask == window.Tool
    assert session.overlay.testAttribute(attribute.WA_TranslucentBackground)
    assert session.overlay.testAttribute(attribute.WA_ShowWithoutActivating)
    assert session.overlay.testAttribute(attribute.WA_MacAlwaysShowToolWindow)
    assert session.overlay.focusPolicy() == qt.Qt.FocusPolicy.NoFocus
    # The card is drawn inside the overlay's own window, not one of its own,
    # so the same flags let clicks through the card too.
    assert session.overlay.card.window() is session.overlay
    assert session.overlay.card.windowHandle() is None


def test_the_screen_is_captured_with_our_own_windows_out_of_the_way(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    ask_and_wait(qt, session)
    assert session.overlay.isVisible()

    ask_and_wait(qt, session, "and how do I stop it?")

    assert [shot.screen for shot in session.seen.shots] == [qt.main, qt.main]
    assert not any(shot.bar_up or shot.overlay_up for shot in session.seen.shots)


def test_the_model_call_runs_on_another_thread_and_the_bar_stays_alive(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    session.backend.gate.clear()

    type_question(qt, session)
    assert not session.bar.isVisible()  # hidden at once, before the capture
    qt.wait_for(lambda: session.backend.threads)

    # The backend is still held shut, yet control is back here on the GUI thread.
    assert session.backend.threads != [threading.get_ident()]
    assert session.guide.busy
    assert session.seen.answers == []
    assert session.bar.isVisible()
    assert session.bar.status.text() == qt.overlay.LOOKING
    assert not session.bar.field.isEnabled()
    assert not session.overlay.isVisible()

    # A second question while the first is out is ignored, not queued.
    session.guide.ask("something else")
    qt.app.processEvents()
    assert len(session.seen.shots) == 1

    session.backend.gate.set()
    qt.wait_for(lambda: session.seen.answers)

    guidance, thread = session.seen.answers[0]
    assert thread == threading.get_ident()  # delivered on the GUI thread
    assert guidance.label == "Live Image"
    assert not session.guide.busy
    assert session.bar.field.isEnabled()
    assert session.bar.status.text() == "Highlighted: Live Image"
    assert session.overlay.isVisible()


def test_highlight_and_card_on_a_scale_2_capture(qt, start_guide):
    screen = qt.main.geometry()
    session = start_guide(found(500, 500, 550, 525), qt.main, scale=2)
    bar_before = session.bar.geometry()

    ask_and_wait(qt, session)

    # The model's box is the same fraction of any image, so on screen it is
    # that fraction of the logical size whatever the capture's resolution.
    width, height = screen.width(), screen.height()
    target = Box(width * 0.5, height * 0.5, width * 0.55, height * 0.525)
    placement, card = session.overlay.placement, session.overlay.card
    assert session.overlay.geometry() == screen
    assert edges(placement.ring) == pytest.approx(edges(target.padded(PAD)))
    assert placement.card.left == placement.ring.right + GAP  # to the right of the control
    assert rect_box(card.geometry()).as_ints() == placement.card.as_ints()
    assert overlap(rect_box(card.geometry()), placement.ring) == 0
    assert on_screen(rect_box(card.geometry()), (screen.width(), screen.height()), MARGIN)
    assert card.title.text() == "Live Image"
    assert card.steps.text().splitlines() == [
        "1. Click Live Image.",
        "2. Wait for the camera image to appear in the viewer.",
    ]
    assert card.note.text() == "Note: Exposure is not kept between projects."
    assert card.confidence.text() == "Confidence: high"
    assert session.bar.geometry() == bar_before  # not in the way, so not moved


def test_the_same_answer_lands_in_the_same_place_at_scale_1(qt, start_guide):
    at_1 = start_guide(found(500, 500, 550, 525), qt.main, scale=1)
    at_2 = start_guide(found(500, 500, 550, 525), qt.main, scale=2)

    ask_and_wait(qt, at_1)
    ask_and_wait(qt, at_2)

    one, two = at_1.overlay.placement, at_2.overlay.placement
    assert edges(one.ring) == pytest.approx(edges(two.ring))
    assert edges(one.card) == pytest.approx(edges(two.card))


def test_the_ring_is_drawn_around_the_control_and_leaves_it_uncovered(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    ask_and_wait(qt, session)
    ring, card = session.overlay.placement.ring, session.overlay.placement.card

    # QWidget.grab() renders the widget into memory; it is not a screen capture.
    drawn = session.overlay.grab().toImage()
    left, top, right, bottom = ring.as_ints()
    x, y = round(ring.center[0]), round(ring.center[1])
    on_control = drawn.pixelColor(x, y)
    on_card = drawn.pixelColor(round(card.center[0]), round(card.center[1]))

    # All four sides, so a ring drawn with the wrong width or height is caught.
    for side in ((x, top), (x, bottom), (left, y), (right, y)):
        on_ring = drawn.pixelColor(*side)
        assert (on_ring.red(), on_ring.green(), on_ring.blue()) == HIGHLIGHT
        assert on_ring.alpha() == 255
    # And nothing past it: 8 units out is beyond the halo and short of the card.
    for outside in ((x, top - 8), (x, bottom + 8), (left - 8, y), (right + 8, y)):
        assert drawn.pixelColor(*outside).alpha() == 0
    assert on_control.alpha() == 0
    assert on_card.alpha() > 200


def test_second_monitor_with_a_negative_origin(qt, side, start_guide):
    screen = side.geometry()
    assert (screen.x(), screen.y()) == (-1280, 120)
    session = start_guide(found(0, 0, 50, 50), side, scale=2)

    ask_and_wait(qt, session)

    placement = session.overlay.placement
    assert session.seen.shots[0].screen is side
    assert session.overlay.geometry() == screen
    # A control in the top-left corner of that monitor: 64 x 36 logical units.
    assert edges(placement.ring) == pytest.approx((RING_INSET, RING_INSET, 64 + PAD, 36 + PAD))
    assert placement.card.left == pytest.approx(64 + PAD + GAP)
    assert placement.card.top == MARGIN
    assert rect_box(session.overlay.card.geometry()).as_ints() == placement.card.as_ints()
    # On the desktop, that is on the second monitor and nowhere near the first.
    corner = session.overlay.mapToGlobal(qt.QPoint(RING_INSET, RING_INSET))
    assert (corner.x(), corner.y()) == (-1280 + RING_INSET, 120 + RING_INSET)


def test_control_not_on_screen_shows_the_steps_by_the_bar(qt, side, start_guide):
    session = start_guide(NOT_ON_SCREEN, side, scale=2)
    assert side.geometry().contains(session.bar.geometry())

    ask_and_wait(qt, session)

    placement, card = session.overlay.placement, session.overlay.card
    bar = session.bar.geometry()
    card_on_desktop = rect_box(card.geometry()).shifted(-1280, 120)
    assert placement.ring is None
    assert session.overlay.isVisible()
    assert card_on_desktop.bottom == bar.top() - GAP
    assert card_on_desktop.center[0] == pytest.approx(bar.x() + bar.width() / 2, abs=1)
    assert card.title.text() == "Not on this screen: Scale Bar"
    assert card.steps.text() == "1. Open the Annotations tab."
    assert not card.note.isVisible()
    assert session.bar.status.text() == "Not on this screen: Scale Bar"
    # Nothing in the highlight colour is drawn anywhere.
    drawn = qt.overlay.to_pil(session.overlay.grab().toImage())
    assert HIGHLIGHT not in {color for _, color in drawn.getcolors(maxcolors=1 << 24)}


def test_a_card_with_fewer_lines_is_shorter(qt, start_guide):
    with_note = start_guide(found(500, 500, 550, 525), qt.main)
    without = start_guide({**found(500, 500, 550, 525), "note": "", "steps": ["Click."]}, qt.main)

    ask_and_wait(qt, with_note)
    ask_and_wait(qt, without)

    assert without.overlay.card.height() < with_note.overlay.card.height()
    assert without.overlay.card.width() == with_note.overlay.card.width()


def test_esc_clears_the_highlight_and_a_second_esc_quits(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    ask_and_wait(qt, session)

    qt.QTest.keyClick(session.bar.field, qt.Qt.Key.Key_Escape)
    assert not session.overlay.isVisible()
    assert session.bar.isVisible()
    assert session.seen.closed == []

    qt.QTest.keyClick(session.bar.field, qt.Qt.Key.Key_Escape)
    assert session.seen.closed == [True]
    assert not session.bar.isVisible()


def test_close_button_quits(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    ask_and_wait(qt, session)

    session.bar.close_button.click()

    assert session.seen.closed == [True]
    assert not session.bar.isVisible()
    assert not session.overlay.isVisible()


def test_an_empty_question_does_nothing(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)

    type_question(qt, session, "   ")
    qt.app.processEvents()

    assert session.bar.isVisible()
    assert not session.guide.busy
    assert session.seen.shots == []


def test_a_failed_model_call_brings_the_bar_back_with_the_reason(qt, start_guide):
    session = start_guide(ModelError("the API returned 401: invalid key\nmore detail"), qt.main)

    ask_and_wait(qt, session)

    assert session.seen.failures == ["the API returned 401: invalid key"]
    assert session.bar.isVisible()
    assert session.bar.field.isEnabled()
    assert session.bar.status.text() == "the API returned 401: invalid key"
    assert not session.overlay.isVisible()
    assert not session.guide.busy


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (RuntimeError("the connection dropped"), "the connection dropped"),
        (TypeError("'NoneType' object is not iterable"), "'NoneType' object is not iterable"),
        (KeyError("label"), "'label'"),
        (ValueError(), "ValueError"),  # no message of its own, so its name
        # Not an Exception. Let out of the worker thread, it aborts the process.
        (SystemExit("the backend gave up"), "the backend gave up"),
    ],
)
def test_any_other_error_from_the_backend_is_shown_the_same_way(qt, start_guide, error, shown):
    session = start_guide(error, qt.main)

    ask_and_wait(qt, session)

    assert session.backend.threads != [threading.get_ident()]
    assert session.seen.failures == [shown]
    assert session.bar.isVisible()
    assert session.bar.field.isEnabled()
    assert session.bar.status.text() == shown
    assert not session.overlay.isVisible()
    assert not session.guide.busy

    # The guide is still alive: the next question gets asked and answered.
    session.backend.answer = found(500, 500, 550, 525)
    ask_and_wait(qt, session, "and again?")
    assert [guidance.label for guidance, _ in session.seen.answers] == ["Live Image"]
    assert session.overlay.isVisible()


def test_an_error_while_drawing_the_answer_brings_the_bar_back(qt, start_guide, monkeypatch):
    def broken_point(self, *args):
        raise RuntimeError("the screen went away")

    session = start_guide(found(500, 500, 550, 525), qt.main)
    monkeypatch.setattr(qt.overlay.Overlay, "point", broken_point)

    ask_and_wait(qt, session)

    assert session.seen.failures == ["the screen went away"]
    assert session.seen.answers == []
    assert session.bar.isVisible()
    assert session.bar.field.isEnabled()
    assert session.bar.status.text() == "the screen went away"
    assert not session.overlay.isVisible()
    assert not session.guide.busy


def test_a_long_reason_is_wrapped_instead_of_cut_off(qt, start_guide):
    reason = "the API client could not send the request: " + "no credentials were found. " * 8
    session = start_guide(ModelError(reason), qt.main)
    before, one_line = session.bar.geometry(), session.bar.status.height()

    ask_and_wait(qt, session)
    qt.app.processEvents()  # the layout gives the status line its new height

    after, status = session.bar.geometry(), session.bar.status
    assert status.text() == reason.strip()
    assert status.height() >= status.heightForWidth(status.width()) > one_line
    # The bar grew to hold it, upward: its bottom edge is near the screen's.
    assert after.height() - before.height() == status.height() - one_line
    assert (after.x(), after.width()) == (before.x(), before.width())
    assert after.bottom() == before.bottom()
    assert qt.main.geometry().contains(after)

    session.bar.set_state(qt.overlay.HINT, busy=False)
    assert session.bar.geometry() == before


def test_bar_sitting_on_the_control_is_moved_off_it(qt, side, start_guide):
    # The bar starts bottom centre of the second monitor; so does this control.
    session = start_guide(found(480, 870, 520, 890), side, scale=2)
    before = rect_box(session.bar.geometry())
    control = Box(1280 * 0.48, 720 * 0.87, 1280 * 0.52, 720 * 0.89).shifted(-1280, 120)
    assert overlap(before, control) == pytest.approx(control.width * control.height)

    ask_and_wait(qt, session)

    after = rect_box(session.bar.geometry())
    ring = live.screen_to_desktop(session.overlay.placement.ring, Screen(-1280, 120, 1280, 720))
    card = rect_box(session.overlay.card.geometry()).shifted(-1280, 120)
    assert ring.contains(*control.center)
    assert overlap(after, ring) == 0 and overlap(after, card) == 0
    assert after.bottom <= min(ring.top, card.top) - GAP + 1  # straight up, by whole units
    assert (after.left, after.width, after.height) == (before.left, before.width, before.height)
    assert side.geometry().contains(session.bar.geometry())
    assert session.bar.isVisible()
    assert session.bar.field.isEnabled()


def test_a_failed_capture_brings_the_bar_back_with_the_reason(qt, start_guide):
    def broken_capture(screen):
        raise OSError("the screen could not be captured")

    session = start_guide(found(1, 1, 2, 2), qt.main, capture=broken_capture)

    ask_and_wait(qt, session)

    assert session.seen.failures == ["the screen could not be captured"]
    assert session.backend.threads == []
    assert session.bar.isVisible()
    assert session.bar.field.isEnabled()


def test_one_shot_never_shows_the_bar(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main, show_bar=False)

    session.guide.ask("how do I start live view?")
    qt.wait_for(lambda: session.seen.answers)

    assert session.overlay.isVisible()
    assert session.overlay.placement.ring is not None
    assert not session.bar.isVisible()


def test_quitting_mid_question_waits_for_the_model_call_to_end(qt, start_guide):
    session = start_guide(found(500, 500, 550, 525), qt.main)
    session.backend.gate.clear()
    type_question(qt, session)
    qt.wait_for(lambda: session.backend.threads)

    qt.QTest.keyClick(session.bar, qt.Qt.Key.Key_Escape)  # no highlight up, so this quits
    assert session.seen.closed == [True]
    threading.Timer(0.05, session.backend.gate.set).start()
    session.guide.shutdown()

    assert session.backend.gate.is_set()
    # The answer that was on its way does not put the windows back.
    qt.app.processEvents()
    assert session.seen.answers == []
    assert not session.overlay.isVisible() and not session.bar.isVisible()


def test_captured_pixels_survive_the_trip_to_pil(qt):
    # Five pixels wide, so each three-byte row is padded and the padding must be skipped.
    image = qt.QImage(5, 3, qt.QImage.Format.Format_ARGB32)
    image.fill(0xFF102030)
    image.setPixel(4, 2, 0xFFFF2D55)

    converted = qt.overlay.to_pil(image)

    assert converted.mode == "RGB"
    assert converted.size == (5, 3)
    assert converted.getpixel((0, 0)) == (0x10, 0x20, 0x30)
    assert converted.getpixel((3, 2)) == (0x10, 0x20, 0x30)
    assert converted.getpixel((4, 2)) == (0xFF, 0x2D, 0x55)


def test_a_capture_that_comes_back_empty_is_an_error(qt):
    with pytest.raises(OSError, match="could not be captured"):
        qt.overlay.to_pil(qt.QImage())


def fake_grab(screen):
    size = screen.geometry().size()
    return Image.new("RGB", (size.width(), size.height()))


def test_run_one_shot_shows_the_answer_for_the_given_time_then_exits(qt, monkeypatch):
    monkeypatch.setattr(qt.overlay, "grab", fake_grab)
    backend = GatedBackend(found(500, 500, 550, 525))
    pointed, bar_shown = [], []
    point = qt.overlay.Overlay.point

    def spy(self, guidance, *rest):
        pointed.append((guidance.label, time.monotonic()))
        return point(self, guidance, *rest)

    monkeypatch.setattr(qt.overlay.Overlay, "point", spy)
    monkeypatch.setattr(qt.overlay.PromptBar, "show", lambda self: bar_shown.append(True))

    code = qt.overlay.run(PROFILE, backend, question="start live view", seconds=0.3, refine=False)

    assert code == 0
    assert [label for label, _ in pointed] == ["Live Image"]
    # Qt's default timers may fire a few percent early, so not exactly 0.3.
    assert time.monotonic() - pointed[0][1] >= 0.25
    assert backend.threads != [threading.get_ident()]
    assert bar_shown == []


def test_run_one_shot_reports_a_failure_in_one_line(qt, monkeypatch, capsys):
    monkeypatch.setattr(qt.overlay, "grab", fake_grab)
    backend = GatedBackend(ModelError("the model declined the request"))

    code = qt.overlay.run(PROFILE, backend, question="start live view", refine=False)

    assert code == 1
    assert capsys.readouterr().err == "scopepilot: the model declined the request\n"


def test_run_one_shot_ends_when_the_answer_cannot_be_drawn(qt, monkeypatch, capsys):
    def broken_point(self, *args):
        raise RuntimeError("the screen went away")

    monkeypatch.setattr(qt.overlay, "grab", fake_grab)
    monkeypatch.setattr(qt.overlay.Overlay, "point", broken_point)
    # With nothing on screen there is no way to quit by hand, so a run that
    # does not end by itself would hang. This turns a hang into a failure.
    watchdog = qt.QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(qt.app.quit)
    watchdog.start(5000)

    backend = GatedBackend(found(500, 500, 550, 525))
    code = qt.overlay.run(PROFILE, backend, question="start live view", refine=False)

    assert watchdog.isActive(), "run() only ended because the test stopped it"
    watchdog.stop()
    assert code == 1
    assert capsys.readouterr().err == "scopepilot: the screen went away\n"


def test_run_interactive_shows_the_bar_until_esc(qt, monkeypatch):
    monkeypatch.setattr(qt.overlay, "grab", fake_grab)
    statuses = []

    def press_escape():
        bars = [w for w in qt.app.topLevelWidgets() if isinstance(w, qt.overlay.PromptBar)]
        statuses.extend(bar.status.text() for bar in bars if bar.isVisible())
        for bar in bars:
            qt.QTest.keyClick(bar.field, qt.Qt.Key.Key_Escape)

    qt.QTimer.singleShot(50, press_escape)
    qt.QTimer.singleShot(5000, qt.app.quit)  # so a broken Esc fails the test instead of hanging it

    assert qt.overlay.run(PROFILE, GatedBackend(NOT_ON_SCREEN)) == 0
    assert statuses == [qt.overlay.HINT]


def test_run_rejects_a_blank_question_and_a_negative_time(qt):
    with pytest.raises(ValueError, match="--ask"):
        qt.overlay.run(PROFILE, GatedBackend(NOT_ON_SCREEN), question="  ")
    with pytest.raises(ValueError, match="--seconds"):
        qt.overlay.run(PROFILE, GatedBackend(NOT_ON_SCREEN), question="q", seconds=-1)
