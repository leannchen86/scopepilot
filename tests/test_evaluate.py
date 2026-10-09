from PIL import Image

import pytest

from scopepilot.evaluate import Case, Outcome, load_cases, summarize
from scopepilot.imaging import fit_within, zoom_region
from scopepilot.profiles import available_profiles, load_profile
from scopepilot.types import Box, Guidance


def case(status="verified"):
    return Case("c1", "src", 1, 0, "start live view", Box(100, 100, 140, 120), status=status)


def answer(box):
    return Guidance(box is not None, "Live Image", box, (), "high")


def test_hit_when_the_answer_centre_is_inside_the_label():
    assert Outcome(case(), answer(Box(110, 105, 150, 125))).hit
    assert not Outcome(case(), answer(Box(140, 100, 180, 120))).hit
    assert not Outcome(case(), answer(None)).hit
    assert not Outcome(case(), None, "boom").hit


def test_summary_flags_unchecked_labels():
    outcomes = [Outcome(case("draft"), answer(Box(110, 105, 130, 115)))]

    text = summarize(outcomes)

    assert "1 of 1 answers were right" in text
    assert "Only 0 of 1 labels have been checked" in text


def test_zoom_region_stays_inside_the_image_at_full_size():
    image = Image.new("RGB", (1000, 600))

    corner = zoom_region(image, Box(980, 580, 1000, 600))
    assert corner == Box(680, 280, 1000, 600)

    huge = zoom_region(image, Box(0, 0, 900, 500))
    assert huge == Box(0, 0, 1000, 600)


def test_every_profile_loads_and_cites_a_source_for_each_note():
    for profile_id in available_profiles():
        profile = load_profile(profile_id)
        assert profile.controls
        for entry in (*profile.controls, *profile.traps):
            assert entry["source"], f"{profile_id}: {entry} has no source"


def test_palette_screenshots_are_shrunk_with_real_resampling():
    # One-pixel black and white stripes: nearest-neighbour would keep them pure,
    # proper resampling blends them to grey.
    stripes = Image.new("L", (400, 4))
    stripes.putdata([255 * (x % 2) for _ in range(4) for x in range(400)])

    shrunk = fit_within(stripes.convert("P"), 200)

    assert shrunk.mode == "RGB"
    assert 100 < shrunk.getpixel((100, 1))[0] < 160


def absent_case():
    return Case("c2", "src", 1, 0, "add a scale bar", expect="absent", status="verified")


def test_an_absent_case_is_right_only_when_no_control_is_pointed_at():
    assert Outcome(absent_case(), answer(None)).hit
    assert not Outcome(absent_case(), answer(Box(1, 1, 2, 2))).hit
    assert not Outcome(absent_case(), None, "boom").hit


def test_summary_splits_on_screen_and_not_on_screen():
    outcomes = [
        Outcome(case(), answer(Box(110, 105, 130, 115))),
        Outcome(absent_case(), answer(None)),
        Outcome(absent_case(), answer(Box(1, 1, 2, 2))),
    ]

    text = summarize(outcomes)

    assert "2 of 3 answers were right" in text
    assert "control on screen: 1 of 1 pointed at it; control not on screen: 1 of 2 said so" in text


def test_cases_must_say_consistently_whether_the_control_is_there(tmp_path):
    base = 'id = "x"\nsource = "s"\npage = 1\nimage = 0\nquestion = "q"\n'
    good = tmp_path / "good.toml"
    good.write_text(f"[[case]]\n{base}box = [1, 2, 3, 4]\n\n[[case]]\n{base}expect = \"absent\"\n")
    present, absent = load_cases(good)
    assert present.box == Box(1, 2, 3, 4) and absent.box is None

    for body in (base, f'{base}box = [1, 2, 3, 4]\nexpect = "absent"\n', f'{base}expect = "maybe"\n'):
        bad = tmp_path / "bad.toml"
        bad.write_text(f"[[case]]\n{body}")
        with pytest.raises(ValueError):
            load_cases(bad)
