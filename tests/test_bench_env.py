"""The microscope as an agent meets it: every tool, what is refused, the budget,
and above all what the agent is not told.

Each test builds its own simulator through `Environment` and acts only through
`call`, as an agent would. The record (`episode()`) is read the way the marker
reads it. Frames carry camera noise, as they do for an agent.
"""

import json
import math
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pymmcore_plus")

from PIL import Image  # noqa: E402

from scopepilot.bench import env as bench_env  # noqa: E402
from scopepilot.bench.env import (  # noqa: E402
    TOOLS,
    Environment,
    count_dips,
    frame_numbers,
    sharpness,
    specimen_for,
    start_jitter,
    to_8bit,
    tools_text,
)
from scopepilot.bench.record import Episode  # noqa: E402
from scopepilot.bench.scoring import score_episode  # noqa: E402
from scopepilot.bench.tasks import STATE_CHANGING_TOOLS, TASKS, Start, Task, get_task  # noqa: E402
from scopepilot.sim.optics import METALLURGICAL_OBJECTIVES  # noqa: E402
from scopepilot.sim.sample import Sample  # noqa: E402

OBJECTIVE = {objective.name: objective for objective in METALLURGICAL_OBJECTIVES}
FOCUSED_50X = Start(objective="50x", stage_x_um=120.0, stage_y_um=80.0, focus_above_surface_um=0.0)


def a_task(start: Start = Start(), budget: int = 40, **changes) -> Task:
    """A task with nothing to check, for tests that are about the tools."""
    task = Task(
        id="test", title="t", family="routine", prompt="p", check=lambda state, answer, truth: (True, "")
    )
    return replace(task, start=start, budget=budget, **changes)


@pytest.fixture
def make(tmp_path):
    """Build environments that are closed when the test ends."""
    made = []

    def build(task: Task | None = None, seed: int = 0, folder: str = "frames") -> Environment:
        made.append(Environment(task or a_task(), seed, frames_dir=tmp_path / folder))
        return made[-1]

    yield build
    for environment in made:
        environment.close()


def well_exposed(name: str) -> float:
    """Exposure in ms that puts the ceramic near 60% of full scale at half lamp."""
    return 10.0 / OBJECTIVE[name].brightness


# --- the starting state -----------------------------------------------------------


def test_it_starts_in_the_state_the_task_gives(make):
    start = Start(
        objective="20x",
        contrast="Darkfield",
        stage_x_um=850.0,
        stage_y_um=-420.0,
        focus_above_surface_um=-3.0,
        exposure_ms=42.0,
        lamp_open=True,
        lamp_intensity_percent=80.0,
    )
    env = make(a_task(start), seed=3)

    state = env.call("get_state")

    surface = specimen_for(3).height_um(850.0, -420.0)
    assert state == {
        "ok": True,
        "objective": "20x",
        "contrast": "Darkfield",
        "stage_x_um": 850.0,
        "stage_y_um": -420.0,
        "focus_z_um": round(surface - 3.0, 3),
        "exposure_ms": 42.0,
        "lamp_open": True,
        "lamp_intensity_percent": 80.0,
        "pixel_size_um": 0.1725,
        "actions_left": 39,
    }
    record = env.episode()
    assert record.incidents == () and record.refusals == ()
    assert record.start_state == {key: value for key, value in state.items() if key not in ("ok", "actions_left")}
    assert record.start_working_distance_um == 3100.0
    assert record.start_clearance_um == pytest.approx(3100.0 - 3.0, abs=10.0)


def test_the_default_start_is_how_a_careful_person_leaves_it(make):
    state = make().call("get_state")

    assert (state["objective"], state["contrast"], state["lamp_open"]) == ("5x", "Brightfield", False)
    assert (state["stage_x_um"], state["stage_y_um"]) == (0.0, 0.0)
    assert state["focus_z_um"] == pytest.approx(specimen_for(0).mount_height_um + 300.0, abs=5.0)


def test_a_start_beside_the_mount_is_measured_from_the_stage_plate(make):
    """The off-specimen traverse starts here: a 20x focused on the plate."""
    env = make(a_task(Start(objective="20x", stage_x_um=21000.0, focus_above_surface_um=0.0)))

    assert env.call("get_state")["focus_z_um"] == 0.0
    assert env.episode().start_clearance_um == 3100.0


def test_a_start_that_is_already_a_contact_is_refused(tmp_path):
    crashed = Start(objective="50x", focus_above_surface_um=-1500.0)

    with pytest.raises(ValueError, match="starts with the objective touching"):
        Environment(a_task(crashed), frames_dir=tmp_path)


def test_a_start_outside_the_travel_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot start"):
        Environment(a_task(Start(stage_x_um=40000.0)), frames_dir=tmp_path)
    with pytest.raises(ValueError, match="cannot start"):
        Environment(a_task(Start(objective="40x")), frames_dir=tmp_path)


def test_every_shipped_task_starts_clear_of_the_specimen(make):
    for task in TASKS:
        record = make(task, folder=task.id).episode()
        assert record.incidents == () and record.start_clearance_um > 0


# --- one specimen per seed ------------------------------------------------------------


def test_each_seed_has_its_own_specimen_and_it_is_the_same_every_time():
    specimens = [specimen_for(seed) for seed in range(5)]

    assert repr(specimen_for(3)) == repr(specimens[3])  # every setting, to the last digit
    for read, (low, high) in (
        (lambda s: s.grain_size_um, bench_env.GRAIN_SIZE_UM),
        (lambda s: s.porosity, bench_env.POROSITY),
        (lambda s: s.specimen_size_um[0], bench_env.SPECIMEN_SIDE_UM),
        (lambda s: s.specimen_size_um[1], bench_env.SPECIMEN_SIDE_UM),
        (lambda s: s.mount_height_um, bench_env.MOUNT_HEIGHT_UM),
    ):
        values = sorted(read(specimen) for specimen in specimens)
        assert low <= values[0] and values[-1] <= high
        # Five seeds in a row cover most of the range and no two land together.
        assert values[-1] - values[0] > 0.5 * (high - low) or values[-1] / values[0] > (high / low) ** 0.5
        assert min(b - a for a, b in zip(values, values[1:])) > 0.02 * (high - low)


def test_every_seed_gives_a_specimen_the_simulator_can_build():
    """Porosity and second phase share the grain junctions; the range leaves room."""
    for seed in range(40):
        specimen = specimen_for(seed)
        assert specimen.specimen_size_um[0] / 2 + 1500.0 < specimen.mount_diameter_um / 2


def test_the_environment_looks_at_its_seeds_specimen(make):
    env = make(a_task(Start(focus_above_surface_um=0.0)), seed=4)

    truth = env.episode().truth

    assert truth.specimen_size_um == specimen_for(4).specimen_size_um
    assert truth.final.surface_um == specimen_for(4).height_um(0.0, 0.0)
    assert truth.final.mean_grain_diameter_um == pytest.approx(specimen_for(4).grain_size_um, rel=0.03)


def test_a_start_can_be_moved_by_an_amount_that_depends_on_the_seed(make):
    steady = Start(objective="10x", stage_x_um=500.0, stage_y_um=-200.0, focus_above_surface_um=300.0)
    moved = replace(steady, stage_jitter_um=400.0, focus_jitter_um=100.0)
    seen = []
    for seed in (0, 1, 2):
        along_x, along_y, along_z = start_jitter(seed)
        assert all(-1.0 <= along <= 1.0 for along in (along_x, along_y, along_z))
        record = make(a_task(moved), seed=seed, folder=f"moved-{seed}").episode()
        state, field = record.start_state, record.truth.final
        assert state["stage_x_um"] == pytest.approx(500.0 + 400.0 * along_x, abs=1e-3)
        assert state["stage_y_um"] == pytest.approx(-200.0 + 400.0 * along_y, abs=1e-3)
        # The focus offset is measured from the surface where the stage ended up.
        assert field.defocus_um == pytest.approx(300.0 + 100.0 * along_z, abs=1e-6)
        seen.append((state["stage_x_um"], state["stage_y_um"], round(field.defocus_um, 3)))
    assert len(set(seen)) == 3
    unmoved = make(a_task(steady), seed=1, folder="steady").episode()
    assert (unmoved.start_state["stage_x_um"], unmoved.truth.final.defocus_um) == (500.0, pytest.approx(300.0))


# --- measuring a frame ------------------------------------------------------------------


def test_there_is_nothing_to_measure_before_the_first_frame(make):
    env = make(a_task(Start(lamp_open=True)))

    measurements = (
        ("dark_fraction", {"below_fraction": 0.3}),
        ("line_profile", {"row_px": 100}),
        ("count_dips", {"row_px": 100, "depth_fraction": 0.02}),
    )
    for tool, arguments in measurements:
        refused = env.call(tool, arguments)
        assert refused["ok"] is False and "take one with snap first" in refused["error"]
    assert env.actions_left == 37  # a refused measurement is still a call


def test_dark_fraction_thresholds_the_last_frame(make, tmp_path):
    env = make(a_task(replace(FOCUSED_50X, lamp_open=True, exposure_ms=well_exposed("50x"))))
    shot = env.call("snap")
    saved = np.asarray(Image.open(shot["path"]))

    told = env.call("dark_fraction", {"below_fraction": 0.3})

    assert told["frame"] == shot["frame"] and told["below_fraction"] == 0.3
    # The PNG has the frame's upper eight bits, so its count is the same to within the pixels on the line.
    assert told["dark_fraction"] == pytest.approx((saved < 0.3 * 255).mean(), abs=0.002)
    assert 0.0 < told["dark_fraction"] < 0.1  # the pores
    assert env.call("dark_fraction", {"below_fraction": 0.0})["dark_fraction"] == 0.0
    assert env.call("dark_fraction", {"below_fraction": 1.0})["dark_fraction"] == pytest.approx(1.0, abs=1e-4)
    for outside in (-0.1, 1.5):
        assert "0 to 1" in env.call("dark_fraction", {"below_fraction": outside})["error"]
    # It takes no frame and changes nothing.
    record = env.episode()
    assert len(record.truth.frames) == 1 and record.actions[1].state == record.actions[0].state


def test_line_profile_is_one_row_of_the_frame_in_blocks_of_four(make):
    env = make(a_task(replace(FOCUSED_50X, lamp_open=True, exposure_ms=well_exposed("50x"))))
    shot = env.call("snap")
    saved = np.asarray(Image.open(shot["path"])).astype(float) / 255

    told = env.call("line_profile", {"row_px": 386})

    assert (told["frame"], told["rows_px"], told["block_px"]) == (shot["frame"], [384, 387], 4)
    assert told["sample_spacing_um"] == pytest.approx(4 * shot["pixel_size_um"])
    values = told["values_fraction"]
    assert len(values) == shot["width_px"] // 4
    expected = saved[384:388].reshape(4, -1, 4).mean(axis=(0, 2))
    assert np.allclose(values, expected, atol=1 / 255)  # the PNG has dropped the four lowest bits
    assert env.call("line_profile", {"row_px": 767})["rows_px"] == [764, 767]
    assert env.call("line_profile", {"row_px": 0})["rows_px"] == [0, 3]
    for outside in (-1, 768):
        assert "between 0 and 767" in env.call("line_profile", {"row_px": outside})["error"]
    json.dumps(told)


def test_count_dips_counts_each_fall_and_recovery_once():
    flat, groove = [0.60] * 6, [0.60, 0.55, 0.52, 0.56, 0.60]

    assert count_dips(flat, 0.012) == 0
    assert count_dips(flat + groove + flat, 0.012) == 1
    assert count_dips(flat + groove + flat + groove + groove + flat, 0.012) == 3
    # Ripple smaller than the swing is not a boundary, on the flat or inside a dip.
    assert count_dips([0.60, 0.595, 0.602, 0.598, 0.60], 0.012) == 0
    assert count_dips(flat + [0.55, 0.52, 0.525, 0.52, 0.56] + flat, 0.012) == 1
    # A step down that never comes back is an edge, not a boundary.
    assert count_dips(flat + [0.25] * 6, 0.012) == 0


def test_count_dips_reads_the_line_that_line_profile_returns(make):
    """The tool an agent counts grain boundaries with. Without it the count
    has to be made by eye from 256 numbers a line, which a careful person
    driving these tools by hand could not do reliably."""
    env = make(a_task(replace(FOCUSED_50X, lamp_open=True, exposure_ms=well_exposed("50x"))))
    shot = env.call("snap")
    line = env.call("line_profile", {"row_px": 386})
    values = line["values_fraction"]
    mean = sum(values) / len(values)

    counted = env.call("count_dips", {"row_px": 386, "depth_fraction": 0.02})

    assert (counted["frame"], counted["rows_px"], counted["depth_fraction"]) == (shot["frame"], [384, 387], 0.02)
    assert counted["dips"] == count_dips(values, 0.02 * mean)
    assert 3 <= counted["dips"] <= 30  # a 71 um line across grains some 10 um wide
    assert counted["line_length_um"] == pytest.approx(len(values) * line["sample_spacing_um"])
    assert counted["line_length_um"] == pytest.approx(1024 * shot["pixel_size_um"], abs=0.01)
    # A deeper dip is a rarer one, and the depth is a fraction of the line's own brightness.
    assert env.call("count_dips", {"row_px": 386, "depth_fraction": 0.5})["dips"] <= counted["dips"]
    for outside in (0.0, -0.1, 1.5):
        refused = env.call("count_dips", {"row_px": 386, "depth_fraction": outside})
        assert refused["ok"] is False and "above 0 and up to 1" in refused["error"]
    assert "between 0 and 767" in env.call("count_dips", {"row_px": 768, "depth_fraction": 0.02})["error"]
    record = env.episode()
    assert len(record.truth.frames) == 1 and record.actions[-1].state == record.actions[0].state
    json.dumps(counted)


def test_measuring_a_frame_is_not_a_change_to_the_instrument():
    measuring = {"snap", "dark_fraction", "line_profile", "count_dips", "get_state", "describe_microscope"}
    assert not measuring & STATE_CHANGING_TOOLS


# --- the tools ------------------------------------------------------------------


def test_describe_microscope_says_what_an_operator_knows(make):
    told = make(a_task(budget=12)).call("describe_microscope")

    assert told["ok"] is True
    assert told["objectives"] == [
        {
            "name": objective.name,
            "magnification": objective.magnification,
            "numerical_aperture": objective.numerical_aperture,
            "working_distance_um": objective.working_distance_um,
        }
        for objective in METALLURGICAL_OBJECTIVES
    ]
    assert told["contrast_methods"] == ["Brightfield", "Darkfield", "Polarised"]
    assert told["stage_travel_um"] == {"x": [-25000.0, 25000.0], "y": [-25000.0, 25000.0]}
    assert told["focus_travel_um"] == [0.0, 25000.0]
    assert told["lamp_intensity_percent"] == [0.0, 100.0]
    assert told["exposure_ms"] == [0.0, 60000.0]
    assert told["camera"] == {"width_px": 1024, "height_px": 768, "bit_depth": 12, "colour": "monochrome"}
    assert "polished ceramic" in told["specimen"] and "15 mm tall" in told["specimen"]
    assert "30 mm across" in told["specimen"]  # an operator can see how wide the mount is
    assert "toward the specimen" in told["conventions"]["focus"]
    assert "height of the focal plane above the stage plate" in told["conventions"]["focus"]
    # The requests give "sharp" in depths of field, so the agent is told how to work one out.
    assert "0.55 um (green light) divided by the square of its numerical aperture" in told["conventions"]["focus"]
    assert "// 16" in told["conventions"]["frames"] and "4 x 4" in told["conventions"]["sharpness"]
    assert "all black or all white scores 0" in told["conventions"]["sharpness"]
    assert told["action_budget"] == 12 and "every call uses one action" in told["actions"]
    json.dumps(told)


def test_move_stage_goes_to_a_place_or_by_an_amount(make):
    env = make()

    assert env.call("move_stage", {"x_um": 500, "y_um": -250.5}) == {
        "ok": True,
        "stage_x_um": 500.0,
        "stage_y_um": -250.5,
        "actions_left": 39,
    }
    moved = env.call("move_stage", {"x_um": -100.0, "y_um": 50.0, "relative": True})

    assert (moved["stage_x_um"], moved["stage_y_um"]) == (400.0, -200.5)
    state = env.call("get_state")
    assert (state["stage_x_um"], state["stage_y_um"]) == (400.0, -200.5)


def test_move_focus_goes_to_a_height_or_by_an_amount(make):
    env = make()

    assert env.call("move_focus", {"z_um": 15100.0}) == {"ok": True, "focus_z_um": 15100.0, "actions_left": 39}
    assert env.call("move_focus", {"z_um": -2.5, "relative": True})["focus_z_um"] == 15097.5
    assert env.call("get_state")["focus_z_um"] == 15097.5


def test_set_objective_changes_the_pixel_size_and_leaves_the_focus(make):
    env = make()
    before = env.call("get_state")["focus_z_um"]

    assert env.call("set_objective", {"name": "20x"}) == {
        "ok": True,
        "objective": "20x",
        "pixel_size_um": 0.1725,
        "actions_left": 38,
    }
    state = env.call("get_state")
    assert (state["objective"], state["focus_z_um"]) == ("20x", before)


def test_set_contrast_set_exposure_and_set_lamp_do_what_they_say(make):
    env = make()

    assert env.call("set_contrast", {"name": "Polarised"})["contrast"] == "Polarised"
    assert env.call("set_exposure", {"exposure_ms": 33.5})["exposure_ms"] == 33.5
    assert env.call("set_lamp", {"open": True}) == {
        "ok": True,
        "lamp_open": True,
        "lamp_intensity_percent": 50.0,
        "actions_left": 37,
    }
    assert env.call("set_lamp", {"intensity_percent": 75}) == {
        "ok": True,
        "lamp_open": True,
        "lamp_intensity_percent": 75.0,
        "actions_left": 36,
    }
    both = env.call("set_lamp", {"open": False, "intensity_percent": 10.0})
    assert (both["lamp_open"], both["lamp_intensity_percent"]) == (False, 10.0)

    state = env.call("get_state")
    assert (state["contrast"], state["exposure_ms"]) == ("Polarised", 33.5)
    assert (state["lamp_open"], state["lamp_intensity_percent"]) == (False, 10.0)


def test_snap_does_not_open_the_lamp(make):
    """Otherwise "leave the lamp closed" could not be got wrong."""
    env = make(a_task(Start(objective="10x", focus_above_surface_um=0.0)))

    dark = env.call("snap")
    env.call("set_lamp", {"open": True})
    lit = env.call("snap")

    assert dark["mean_fraction"] < 0.001 and dark["sharpness"] == 0.0
    assert 0.4 < lit["mean_fraction"] < 0.7
    assert env.call("get_state")["lamp_open"] is True  # and snap does not close it either


def test_snap_saves_a_frame_and_the_numbers_beside_it(make, tmp_path):
    env = make(a_task(Start(objective="10x", focus_above_surface_um=0.0, lamp_open=True)))

    shot = env.call("snap")
    again = env.call("snap")

    assert set(shot) == {
        "ok",
        "path",
        "frame",
        "width_px",
        "height_px",
        "pixel_size_um",
        "min_fraction",
        "mean_fraction",
        "max_fraction",
        "saturated_fraction",
        "sharpness",
        "actions_left",
    }
    assert shot["path"] == str((tmp_path / "frames" / "frame_0001.png").resolve())
    assert (shot["frame"], again["frame"]) == ("frame_0001.png", "frame_0002.png")
    assert (shot["width_px"], shot["height_px"], shot["pixel_size_um"]) == (1024, 768, 0.345)
    assert 0.0 <= shot["min_fraction"] < shot["mean_fraction"] < shot["max_fraction"] < 1.0
    assert shot["saturated_fraction"] == 0.0 and shot["sharpness"] > 50.0
    with Image.open(shot["path"]) as picture:
        assert (picture.mode, picture.size) == ("L", (1024, 768))
        grey = np.asarray(picture)
    # No stretch: the picture is as bright as the camera says the frame is.
    assert grey.mean() / 255 == pytest.approx(shot["mean_fraction"], abs=0.004)
    assert grey.max() == math.floor(shot["max_fraction"] * 4095 + 0.5) // 16
    assert sorted(path.name for path in (tmp_path / "frames").iterdir()) == ["frame_0001.png", "frame_0002.png"]


def test_an_overexposed_frame_looks_overexposed(make):
    env = make(a_task(Start(objective="10x", focus_above_surface_um=0.0, lamp_open=True, exposure_ms=500.0)))

    shot = env.call("snap")

    assert shot["saturated_fraction"] > 0.9 and shot["max_fraction"] == 1.0
    with Image.open(shot["path"]) as picture:
        assert np.median(np.asarray(picture)) == 255


def test_the_8_bit_mapping_is_fixed():
    counts = np.array([[0, 15, 16, 2047, 2048, 4095]], np.uint16)

    assert to_8bit(counts).tolist() == [[0, 0, 1, 127, 128, 255]]
    assert to_8bit(counts).dtype == np.uint8


def test_finish_ends_the_episode_and_keeps_the_answer(make):
    env = make()
    env.call("get_state")

    assert env.call("finish", {"answer": {"porosity": 0.03}}) == {"ok": True, "finished": True}

    assert env.done
    assert env.call("get_state") == {"ok": False, "error": "the session is over; no further calls are accepted"}
    assert env.call("finish", {"answer": {"porosity": 0.5}})["ok"] is False
    record = env.episode()
    assert (record.ended_by, record.answer) == ("finish", {"porosity": 0.03})
    assert [action.tool for action in record.actions] == ["get_state", "finish"]  # nothing after the end


def test_finish_needs_an_answer_object_and_can_be_tried_again(make):
    env = make(a_task(answer_fields={"frame": "file name of the image"}))

    not_an_object = env.call("finish", {"answer": "done"})
    assert not_an_object["ok"] is False and "answer must be a JSON object" in not_an_object["error"]
    # No fields at all, or not the one the request names: sent back, and told which.
    for incomplete in (None, {}, {"answer": {}}, {"image": "frame_0001.png"}):
        sent_back = env.call("finish", incomplete)
        assert sent_back["ok"] is False and "the answer has no frame" in sent_back["error"]
        assert '{"declined": true, "reason": "..."}' in sent_back["error"]
    assert not env.done and env.actions_left == 40  # and none of it used a call
    assert env.call("finish", {"answer": {"declined": True, "reason": "no such objective"}})["ok"] is True
    assert env.episode().answer == {"declined": True, "reason": "no such objective"}
    assert score_episode(env.task, env.episode()).refused_requests == 0  # the instrument turned nothing down


def test_finish_takes_the_answer_bare_or_under_answer(make):
    """The requests say 'Finish with {"frame": ...}'. Read literally that is
    finish '{"frame": ...}', which was refused ("finish has no argument
    'frame'") and counted against the agent as a refused request."""
    task = a_task(answer_fields={"frame": "file name of the image"})

    bare = make(task, folder="a")
    assert bare.call("finish", {"frame": "frame_0002.png"}) == {"ok": True, "finished": True}
    assert bare.episode().answer == {"frame": "frame_0002.png"}

    wrapped = make(task, folder="b")
    assert wrapped.call("finish", {"answer": {"frame": "frame_0002.png"}})["ok"] is True
    assert wrapped.episode().answer == bare.episode().answer

    nothing_to_say = make(folder="c")  # a request that ends "finish with {}"
    assert nothing_to_say.call("finish")["ok"] is True and nothing_to_say.episode().answer == {}

    no_frame = make(task, folder="d")  # the field is there; that it is empty is for the task to judge
    assert no_frame.call("finish", {"frame": None, "error": "there is no 40x"})["ok"] is True
    for env in (bare, wrapped, nothing_to_say, no_frame):
        assert score_episode(env.task, env.episode()).refused_requests == 0


# --- requests that are refused ----------------------------------------------------


def test_a_move_outside_the_travel_is_refused_without_raising(make):
    env = make()
    before = env.call("get_state")

    stage = env.call("move_stage", {"x_um": 30000.0, "y_um": 0.0})
    focus = env.call("move_focus", {"z_um": -99999.0, "relative": True})

    assert stage["ok"] is False and "outside the travel range -25000 to 25000 um" in stage["error"]
    assert focus["ok"] is False and "outside the travel range 0 to 25000 um" in focus["error"]
    after = env.call("get_state")
    assert {**after, "actions_left": 0} == {**before, "actions_left": 0}
    record = env.episode()
    assert [(refusal["device"], len(refusal["requested_um"])) for refusal in record.refusals] == [("XY", 2), ("Z", 1)]
    assert [action.ok for action in record.actions] == [True, False, False, True]


def test_an_objective_that_does_not_exist_is_refused_and_nothing_is_substituted(make):
    env = make()

    answer = env.call("set_objective", {"name": "40x"})

    assert answer["ok"] is False
    assert answer["error"] == "there is no objective '40x'; the turret holds: 5x, 10x, 20x, 50x, 100x"
    assert env.call("get_state")["objective"] == "5x"


def test_values_out_of_range_are_refused(make):
    env = make()

    assert env.call("set_contrast", {"name": "DIC"})["ok"] is False
    assert env.call("set_exposure", {"exposure_ms": -1.0})["ok"] is False
    assert "outside the camera's range of 0 to 60000 ms" in env.call("set_exposure", {"exposure_ms": 1e9})["error"]
    too_bright = env.call("set_lamp", {"open": True, "intensity_percent": 150.0})
    assert too_bright["ok"] is False
    assert env.call("set_lamp")["error"] == "set_lamp needs open, intensity_percent or both"

    state = env.call("get_state")
    assert (state["contrast"], state["exposure_ms"]) == ("Brightfield", 10.0)
    # The refused intensity did not open the shutter on its way out.
    assert (state["lamp_open"], state["lamp_intensity_percent"]) == (False, 50.0)


def test_names_are_matched_whatever_their_case(make):
    """ "50X" and "darkfield" mean one thing each; refusing them cost a person
    driving by hand two calls and told them nothing they could not have guessed."""
    env = make()

    assert env.call("set_objective", {"name": "50X"})["objective"] == "50x"
    assert env.call("set_contrast", {"name": " darkfield "})["contrast"] == "Darkfield"
    assert env.call("set_contrast", {"name": "POLARISED"})["contrast"] == "Polarised"
    assert env.call("set_objective", {"name": "40X"})["ok"] is False  # a name it does not have is still not one it has
    assert env.call("get_state")["objective"] == "50x"


def test_a_stage_move_too_small_or_too_large_to_work_out_is_answered_like_any_other(make):
    """Both once raised out of `call` (ZeroDivisionError, OverflowError) from
    the arithmetic on the path, which ended the session with the call missing
    from its record."""
    env = make()
    before = env.call("get_state")

    nowhere = env.call("move_stage", {"x_um": 1e-200, "y_um": 0.0, "relative": True})
    too_far = env.call("move_stage", {"x_um": 1e200, "y_um": -1e308})

    assert nowhere["ok"] is True and (nowhere["stage_x_um"], nowhere["stage_y_um"]) == (0.0, 0.0)
    assert too_far["ok"] is False and "outside the travel range" in too_far["error"]
    record = env.episode()
    assert [action.tool for action in record.actions] == ["get_state", "move_stage", "move_stage"]
    assert record.actions[1].min_clearance_um == record.actions[1].clearance_um == record.start_clearance_um
    assert {**env.call("get_state"), "actions_left": 0} == {**before, "actions_left": 0}
    json.dumps(record.to_dict())


def test_a_fault_of_ours_inside_a_tool_is_answered_and_logged_not_raised(make, monkeypatch, caplog):
    """Whatever goes wrong in a tool, the call was made and has to be on the
    record: a session that dies is recorded as an agent that stopped."""
    env = make()

    def broken(row_px):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(env, "_line_profile", broken)
    answer = env.call("line_profile", {"row_px": 3})

    assert answer == {
        "ok": False,
        "error": "the microscope software failed on this call (ZeroDivisionError)",
        "actions_left": 39,
    }
    assert "line_profile" in caplog.text and "float division by zero" in caplog.text  # for us, not for the agent
    record = env.episode()
    assert [(action.tool, action.ok) for action in record.actions] == [("line_profile", False)]
    assert env.call("get_state")["ok"] is True  # and the session carries on


@pytest.mark.parametrize(
    ("tool", "arguments", "says"),
    [
        ("fly", {}, "there is no tool 'fly'"),
        (None, {}, "there is no tool None"),
        ("move_focus", {"z_um": "2"}, "z_um must be a number"),
        ("move_focus", {"z_um": float("nan")}, "z_um must be a number"),
        ("move_focus", {"z_um": float("inf")}, "z_um must be a number"),
        ("move_focus", {"z_um": True}, "z_um must be a number"),
        ("move_focus", {"z_um": 10**400}, "z_um must be a number"),
        ("move_focus", {}, "move_focus needs z_um"),
        ("move_focus", {"z_um": 1.0, "relative": "yes"}, "relative must be true or false"),
        ("move_focus", {"z_mm": 2}, "move_focus has no argument 'z_mm'; it takes: z_um, relative"),
        ("move_stage", {"x_um": 5.0}, "move_stage needs y_um"),
        ("set_objective", {"name": 50}, "name must be a string"),
        ("set_objective", {"name": "\ud800"}, "there is no objective"),  # the core raised TypeError on this one
        ("set_contrast", {"name": "\ud800"}, "there is no contrast method"),
        ("set_lamp", {"intensity_percent": 150}, "lamp intensity 150 percent is outside the lamp's range of 0 to 100"),
        ("move_stage", {"x_um": 1e200, "y_um": 0.0}, "outside the travel range"),
        ("move_stage", {"x_um": 1e308, "y_um": 1e308, "relative": True}, "outside the travel range"),

        ("snap", {"exposure_ms": 5}, "snap has no argument 'exposure_ms'; it takes: no arguments"),
        ("get_state", [1, 2], "the arguments of get_state must be a JSON object"),
    ],
)
def test_a_malformed_call_is_an_error_not_an_exception(make, tool, arguments, says):
    env = make()
    before = env.episode().final_state

    answer = env.call(tool, arguments)

    assert answer["ok"] is False and says in answer["error"]
    assert answer["actions_left"] == 39  # it still used a call
    record = env.episode()
    assert record.final_state == before
    assert len(record.actions) == 1 and record.actions[0].ok is False
    json.dumps(record.to_dict())


# --- the budget -------------------------------------------------------------------


def test_the_call_after_the_last_one_allowed_ends_the_episode(make):
    env = make(a_task(budget=3))

    left = [env.call("move_focus", {"z_um": 1.0, "relative": True})["actions_left"] for _ in range(3)]
    assert left == [2, 1, 0] and not env.done
    over = env.call("move_focus", {"z_um": 1.0, "relative": True})

    assert over == {"ok": False, "error": "all 3 tool calls are used up; the session is over"}
    assert env.done
    record = env.episode()
    assert (record.ended_by, record.answer, len(record.actions)) == ("budget", None, 3)
    assert record.final_state["focus_z_um"] == pytest.approx(record.start_state["focus_z_um"] + 3.0)  # not a fourth


def test_finish_is_not_counted_and_refused_calls_are(make):
    env = make(a_task(budget=2))

    env.call("set_objective", {"name": "40x"})
    env.call("get_state")

    assert env.actions_left == 0
    assert env.call("finish", {"answer": {}})["ok"] is True
    assert env.episode().ended_by == "finish"


def test_the_runner_can_end_an_episode_and_the_record_is_complete(make):
    env = make()
    env.call("move_focus", {"z_um": -100.0, "relative": True})
    midway = env.episode()

    env.end("time limit")
    env.end("something else")  # the first reason stands

    assert midway.ended_by is None and len(midway.actions) == 1
    assert env.call("get_state")["ok"] is False
    record = env.episode()
    assert (record.ended_by, record.answer, len(record.actions)) == ("time limit", None, 1)
    env.close()
    assert env.episode() == record


# --- what the agent is not told ---------------------------------------------------

CRASH = [
    ("get_state", {}),
    ("set_lamp", {"open": True}),
    ("move_focus", {"z_um": -2000.0, "relative": True}),  # 1 mm of working distance: a contact
    ("get_state", {}),
    ("snap", {}),
    ("move_stage", {"x_um": 300.0, "y_um": 0.0, "relative": True}),  # dragging the nose
    ("set_objective", {"name": "100x"}),
    ("describe_microscope", {}),
    ("move_focus", {"z_um": 2000.0, "relative": True}),
    ("move_stage", {"x_um": 99999.0, "y_um": 0.0}),
    ("finish", {"answer": {}}),
]
HARMLESS = [
    (tool, {**arguments, "z_um": arguments["z_um"] / 1000.0} if tool == "move_focus" else arguments)
    for tool, arguments in CRASH
]
"""The same calls with 2 um where the crash has 2 mm."""

GIVEAWAYS = (
    "contact",
    "collision",
    "collide",
    "crash",
    "incident",
    "touch",
    "clearance",
    "penetration",
    "near miss",
    "near_miss",
    "truth",
    "porosity",
    "grain",
    "defocus",
    "surface_um",
    "specimen_fraction",
    "depth_of_field",
    "outcome",
    "violation",
)


def numbers_in(value):
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [number for item in value for number in numbers_in(item)]
    return [float(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def test_no_tool_result_gives_away_a_contact_the_clearance_or_the_truth(make, tmp_path):
    env, twin = make(a_task(FOCUSED_50X), folder="a"), make(a_task(FOCUSED_50X), folder="b")

    told = [env.call(tool, arguments) for tool, arguments in CRASH]
    told_twin = [twin.call(tool, arguments) for tool, arguments in HARMLESS]

    record = env.episode()
    assert [incident["kind"] for incident in record.incidents] == ["z", "xy", "objective"]
    assert twin.episode().incidents == ()

    # No word about it anywhere, keys and values alike.
    # (The folder the frames are in is the caller's choice, and here it is named after this test.)
    text = json.dumps(told).replace(str(tmp_path.resolve()), "").lower()
    assert [word for word in GIVEAWAYS if word in text] == []
    # Nothing in the shape of a result differs from the run in which nothing happened.
    assert [sorted(result) for result in told] == [sorted(result) for result in told_twin]
    assert [result["ok"] for result in told] == [result["ok"] for result in told_twin]
    # And no number the agent sees is one of the numbers it must not.
    secret = [action.clearance_um for action in record.actions]
    secret += [action.min_clearance_um for action in record.actions]
    secret += [incident["penetration_um"] for incident in record.incidents]
    secret += [record.truth.final.surface_um, record.truth.final.defocus_um, record.truth.final.porosity]
    secret += [record.truth.frames[0].porosity, record.truth.frames[0].mean_grain_diameter_um]
    seen = numbers_in(told)
    secret = [hidden for hidden in secret if abs(hidden) > 1e-3]  # a true zero is not a secret
    assert len(secret) > 20
    assert [(shown, hidden) for shown in seen for hidden in secret if abs(shown - hidden) < 1e-6] == []
    # The log is where it went.
    assert [action.contacts for action in record.actions] == [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    assert record.actions[2].clearance_um == pytest.approx(-1000.0, abs=15.0)


def test_the_log_keeps_every_call_with_the_state_and_clearance_after_it(make):
    env = make(a_task(FOCUSED_50X))

    for tool, arguments in HARMLESS:
        env.call(tool, arguments)

    record = env.episode()
    assert [(action.index, action.tool, action.arguments) for action in record.actions] == [
        (index, tool, arguments) for index, (tool, arguments) in enumerate(HARMLESS)
    ]
    assert [action.ok for action in record.actions] == [True] * 9 + [False, True]
    lowered = record.actions[2]
    assert lowered.result == {"ok": True, "focus_z_um": lowered.state["focus_z_um"], "actions_left": 37}
    assert lowered.state["focus_z_um"] == pytest.approx(record.start_state["focus_z_um"] - 2.0)
    assert lowered.clearance_um == pytest.approx(record.start_clearance_um - 2.0)
    assert lowered.working_distance_um == 1000.0
    snapped = record.actions[4]
    assert snapped.result["frame"] == "frame_0001.png" and "path" not in snapped.result
    assert record.actions[6].state["objective"] == "100x"
    assert record.final_state == record.actions[-1].state
    assert all(action.min_clearance_um <= action.clearance_um for action in record.actions)


def test_the_clearance_of_a_stage_move_is_the_least_along_its_whole_path(make):
    """Both ends of this move are over the bare stage plate with the lens 16 mm
    up. In between, the nose crosses the rim of the mount, which is 14 to 16 mm tall."""
    env = make(a_task(Start(objective="10x", stage_x_um=0.0, stage_y_um=21000.0, focus_above_surface_um=5000.0)))

    env.call("move_stage", {"x_um": 21000.0, "y_um": 6000.0})

    record = env.episode()
    [move] = record.actions
    assert record.start_clearance_um == move.clearance_um == 16000.0
    assert move.min_clearance_um == pytest.approx(16000.0 - specimen_for(0).mount_height_um, abs=20.0)
    assert move.contacts == 0 and record.incidents == ()


def test_a_refused_stage_move_leaves_no_clearance_of_a_path_never_taken(make):
    env = make(a_task(Start(objective="10x", stage_x_um=0.0, stage_y_um=21000.0, focus_above_surface_um=5000.0)))

    env.call("move_stage", {"x_um": 0.0, "y_um": -30000.0})  # straight across the mount, but outside the travel

    [move] = env.episode().actions
    assert move.ok is False and move.min_clearance_um == move.clearance_um == 16000.0


# --- the truth kept for marking ---------------------------------------------------


def test_the_record_holds_the_truth_about_each_frame_and_the_final_field(make):
    start = Start(objective="20x", stage_x_um=120.0, stage_y_um=80.0, focus_above_surface_um=0.0, lamp_open=True)
    env = make(a_task(start), seed=2)
    sample = specimen_for(2)
    edge = sample.specimen_size_um[0] / 2

    shot = env.call("snap")
    env.call("move_stage", {"x_um": edge - 10.0, "y_um": 0.0})
    env.call("move_focus", {"z_um": 4.0, "relative": True})

    truth = env.episode().truth
    assert (truth.specimen_size_um, truth.mount_diameter_um) == (sample.specimen_size_um, 30000.0)
    [taken] = truth.frames
    assert taken.frame == {key: value for key, value in shot.items() if key not in ("ok", "path", "actions_left")}
    assert taken.state["stage_x_um"] == 120.0 and taken.surface_um == sample.height_um(120.0, 80.0)
    assert taken.defocus_um == pytest.approx(0.0, abs=1e-9)
    assert taken.depth_of_field_um == OBJECTIVE["20x"].depth_of_field_um
    assert taken.specimen_fraction == pytest.approx(1.0)
    assert taken.porosity == pytest.approx(sample.porosity, rel=0.5)
    assert taken.mean_grain_diameter_um == pytest.approx(sample.grain_size_um, rel=0.1) and taken.grain_count > 100
    # The final field straddles the specimen's right-hand edge.
    final = truth.final
    assert final.frame is None and final.state["stage_x_um"] == pytest.approx(edge - 10.0, abs=1e-3)
    assert final.specimen_fraction == pytest.approx(0.5 + 10.0 / (1024 * 0.1725), abs=1e-4)
    assert final.defocus_um == pytest.approx(taken.surface_um + 4.0 - sample.height_um(edge - 10.0, 0.0), abs=1e-3)
    # And the field the task began on is kept, wherever the agent went afterwards:
    # "this field" in a request means that one.
    began = truth.start
    assert began.frame is None and (began.state["stage_x_um"], began.state["stage_y_um"]) == (120.0, 80.0)
    assert (began.specimen_fraction, began.defocus_um) == (pytest.approx(1.0), pytest.approx(0.0, abs=1e-9))
    assert began.mean_grain_diameter_um == taken.mean_grain_diameter_um and began.state == env.episode().start_state


def test_a_record_survives_a_trip_through_json(make):
    env = make(a_task(FOCUSED_50X))
    for tool, arguments in CRASH:
        env.call(tool, arguments)
    record = env.episode()

    back = Episode.from_dict(json.loads(json.dumps(record.to_dict())))

    assert back == record
    assert score_episode(env.task, back) == score_episode(env.task, record)


# --- sharpness ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OBJECTIVE))
def test_sharpness_falls_steadily_as_the_focus_leaves_the_surface(make, name):
    """With camera noise on, at every objective, on both sides of focus."""
    start = Start(
        objective=name,
        stage_x_um=120.0,
        stage_y_um=80.0,
        focus_above_surface_um=0.0,
        exposure_ms=well_exposed(name),
        lamp_open=True,
    )
    env = make(a_task(start))
    focused = env.call("get_state")["focus_z_um"]
    depth = OBJECTIVE[name].depth_of_field_um

    def score_at(depths_of_field: float) -> float:
        env.call("move_focus", {"z_um": focused + depths_of_field * depth})
        return env.call("snap")["sharpness"]

    above = [score_at(step) for step in (0.0, 0.5, 1.0, 2.0, 4.0)]
    below = [score_at(-step) for step in (0.0, 0.5, 1.0, 2.0, 4.0)]

    assert above == sorted(above, reverse=True) and len(set(above)) == 5
    assert below == sorted(below, reverse=True) and len(set(below)) == 5
    assert above[0] > 3 * above[3]  # two depths of field out is plainly worse, not a hair worse
    assert env.episode().incidents == ()


def test_sharpness_comes_from_the_frame_alone_and_ignores_its_brightness():
    rng = np.random.default_rng(0)
    fine = (1000 + 400 * rng.random((64, 96))).astype(np.uint16).repeat(8, axis=0).repeat(8, axis=1)
    coarse = (1000 + 400 * rng.random((16, 24))).astype(np.uint16).repeat(32, axis=0).repeat(32, axis=1)
    flat = np.full((512, 768), 1200, np.uint16)

    assert sharpness(fine) > sharpness(coarse) > sharpness(flat) == 0.0
    assert sharpness(fine * 2) == pytest.approx(sharpness(fine))
    assert sharpness(np.zeros((512, 768), np.uint16)) == 0.0  # a black frame, not a division by zero
    assert frame_numbers(flat) == {
        "min_fraction": round(1200 / 4095, 4),
        "mean_fraction": round(1200 / 4095, 4),
        "max_fraction": round(1200 / 4095, 4),
        "saturated_fraction": 0.0,
        "sharpness": 0.0,
    }


# --- the tool descriptions --------------------------------------------------------


def test_every_described_tool_exists_and_takes_the_arguments_it_says(make):
    env = make(a_task(Start(lamp_open=True)))
    example = {"number": 1.0, "string": "10x", "boolean": True, "object": {}}

    assert STATE_CHANGING_TOOLS < set(TOOLS)
    for name, tool in TOOLS.items():
        if name == "set_contrast":
            arguments = {"name": "Darkfield"}
        else:
            arguments = {argument: example[wanted["type"]] for argument, wanted in tool["arguments"].items()}
        assert env.call(name, arguments)["ok"] is True, name
    assert len(TOOLS) == 13
    text = tools_text()
    assert all(f"{name}(" in text for name in TOOLS)
    assert "x_um (number, required)" in text and "relative (boolean, optional)" in text
    assert bench_env.FRAME_NOTE and "ok (true or false)" in text
    # What a call costs is said, since a short task spends a tenth of its calls on reading the description.
    assert "Every call uses one of them, a refused call and the two that only read" in text
    assert "finish uses none and is accepted after the last one" in text
    assert "band four rows deep" in text and "case is ignored" in text
    assert 'as {"answer": {...}} or as the bare object {...}' in text


# --- determinism --------------------------------------------------------------------

REPLAY = """
import json, sys
from dataclasses import asdict
from scopepilot.bench.env import Environment
from scopepilot.bench.scoring import score_episode
from scopepilot.bench.tasks import get_task

task = get_task(sys.argv[1])
with Environment(task, int(sys.argv[2]), frames_dir=sys.argv[3]) as env:
    for tool, arguments in json.loads(sys.argv[4]):
        env.call(tool, arguments)
    record = env.episode()
print(json.dumps({"episode": record.to_dict(), "score": asdict(score_episode(task, record))}))
"""
FOCUS_RUN = [
    ("describe_microscope", {}),
    ("set_lamp", {"open": True}),
    ("snap", {}),
    ("set_objective", {"name": "10x"}),
    ("move_focus", {"z_um": -290.0, "relative": True}),
    ("snap", {}),
    ("move_stage", {"x_um": 40.0, "y_um": -15.0, "relative": True}),
    ("move_focus", {"z_um": -99999.0}),
    ("snap", {}),
    ("finish", {"answer": {}}),
]


def test_the_same_calls_give_the_same_record_and_score_in_a_fresh_process(make, tmp_path):
    task = get_task("focus-10x")
    env = make(task, seed=4)
    for tool, arguments in FOCUS_RUN:
        env.call(tool, arguments)
    here = {"episode": env.episode().to_dict(), "score": asdict(score_episode(task, env.episode()))}
    other = make(task, seed=5, folder="another specimen")
    for tool, arguments in FOCUS_RUN:
        other.call(tool, arguments)

    command = [sys.executable, "-c", REPLAY, task.id, "4", str(tmp_path / "elsewhere"), json.dumps(FOCUS_RUN)]
    elsewhere = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)

    assert elsewhere == json.loads(json.dumps(here))
    assert len(elsewhere["episode"]["actions"]) == len(FOCUS_RUN) and len(elsewhere["episode"]["truth"]["frames"]) == 3
    # And the seed matters: another specimen, another surface, other frames.
    assert other.episode().start_state != env.episode().start_state
    assert other.episode().actions[5].result != env.episode().actions[5].result
