"""The pretend microscope behind a Micro-Manager core: it answers the ordinary
core calls, refuses moves outside its travel, and writes down every time the
objective touches something.

Everything is driven through `core`, the way a caller with real hardware would.
The frames are noiseless unless a test is about the noise. Several tests exist
because pymmcore-plus 0.18 leaves something undone for Python devices and
`SimCore` does it; they are marked "workaround" and are the ones to look at
when a new pymmcore-plus release breaks this file. Where a test says "as
Micro-Manager does", the behaviour was compared by hand with Micro-Manager's
demo configuration (mm-test-adapters), which the tests themselves do not need.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("pymmcore_plus")
useq = pytest.importorskip("useq")

from pymmcore_plus.experimental.unicore import UniMMCore  # noqa: E402

from scopepilot.sim import __main__ as command  # noqa: E402
from scopepilot.sim import build_microscope  # noqa: E402
from scopepilot.sim.devices import XY_TRAVEL_UM, Z_TRAVEL_UM  # noqa: E402
from scopepilot.sim.optics import CONTRASTS, METALLURGICAL_OBJECTIVES, field_of_view_um, illumination  # noqa: E402
from scopepilot.sim.sample import Sample  # noqa: E402

OBJECTIVE = {objective.name: objective for objective in METALLURGICAL_OBJECTIVES}
PLACE = (100.0, 50.0)
"""An ordinary spot on the specimen."""
OFF_THE_MOUNT = (21000.0, 0.0)
"""Over the bare stage plate, with room for the 10x objective's nose beside the mount."""


@pytest.fixture(scope="module")
def ceramic():
    return Sample.polished_ceramic(seed=1)


@pytest.fixture
def scope(ceramic):
    microscope = build_microscope(ceramic, noise=False)
    yield microscope
    microscope.core.unloadAllDevices()


def focus(scope, objective="20x", at=PLACE):
    """Go to `at` with `objective` in, focused on the surface there, and return its height."""
    core = scope.core
    surface = scope.sample.height_um(*at)
    core.setConfig("Objective", objective)
    core.setXYPosition(*at)
    core.setPosition(surface)
    return surface


def sharpness(image):
    return float(np.abs(np.diff(image.astype(float), axis=1)).mean())


# --- the ordinary core calls ----------------------------------------------------


def test_it_starts_out_of_harms_way(scope):
    core = scope.core
    surface = scope.sample.height_um(0.0, 0.0)

    assert core.getCameraDevice() == "Camera"
    assert core.getXYStageDevice() == "XY"
    assert core.getFocusDevice() == "Z"
    assert core.getShutterDevice() == "Lamp"
    assert core.getStateLabel("Objective") == "5x"
    assert core.getStateLabel("Contrast") == "Brightfield"
    assert core.getXYPosition() == (0.0, 0.0)
    assert 200.0 < core.getPosition() - surface < 500.0
    assert not core.getShutterOpen()
    assert scope.incidents == [] and scope.refusals == []


def test_a_snap_is_one_frame_of_the_sensor(scope):
    core = scope.core

    core.snapImage()
    image = core.getImage()

    assert (core.getImageHeight(), core.getImageWidth()) == image.shape == (768, 1024)
    assert image.dtype == np.uint16 and core.getBytesPerPixel() == 2
    assert 0 < image.max() <= 4095


def test_the_bit_depth_is_the_cameras_not_the_pixel_types(scope):
    """Workaround: stock UniMMCore reports 16 for any uint16 camera."""
    assert scope.core.getImageBitDepth() == 12
    assert int(scope.core.getProperty("Camera", "BitDepth")) == 12


def test_auto_shutter_lights_the_snap_and_closes_the_lamp_again(scope):
    """Workaround: stock UniMMCore does not open a Python shutter for a snap."""
    core = scope.core
    assert core.getAutoShutter()

    core.snapImage()

    assert core.getImage().mean() > 500
    assert not core.getShutterOpen()
    # As Micro-Manager does: closed afterwards even if it was open before.
    core.setShutterOpen(True)
    core.snapImage()
    assert not core.getShutterOpen()


def wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


def test_auto_shutter_lights_a_live_sequence_and_closes_the_lamp_when_it_stops(scope):
    """Workaround: with stock UniMMCore, and here until now, live frames were
    black unless the lamp was opened by hand."""
    core = scope.core

    core.startContinuousSequenceAcquisition(0)
    assert wait_for(lambda: core.getRemainingImageCount() >= 2)
    assert core.getShutterOpen()
    # As Micro-Manager does. Here it used to take a snap alongside the sequence.
    with pytest.raises(RuntimeError, match="sequence acquisition is running"):
        core.snapImage()
    frame = core.getLastImage()
    core.stopSequenceAcquisition()

    assert frame.mean() > 500
    assert not core.getShutterOpen() and not core.isSequenceRunning()


def test_auto_shutter_closes_the_lamp_when_a_counted_sequence_ends_by_itself(scope):
    core = scope.core

    core.startSequenceAcquisition(3, 0.0, True)
    assert wait_for(lambda: not core.isSequenceRunning())

    assert wait_for(lambda: not core.getShutterOpen())
    assert core.getRemainingImageCount() == 3
    assert all(core.popNextImage().mean() > 500 for _ in range(3))


def test_without_auto_shutter_a_sequence_leaves_the_lamp_alone(scope):
    core = scope.core
    core.setAutoShutter(False)

    core.startSequenceAcquisition(2, 0.0, True)
    assert wait_for(lambda: not core.isSequenceRunning())

    assert not core.getShutterOpen()
    assert not core.popNextImage().any()


def test_without_auto_shutter_the_lamp_must_be_opened_by_hand(scope):
    core = scope.core
    core.setAutoShutter(False)

    core.snapImage()
    dark = core.getImage()
    core.setShutterOpen(True)
    core.snapImage()
    lit = core.getImage()

    assert not dark.any()
    assert core.getShutterOpen() and lit.mean() > 500


def test_lamp_intensity_and_exposure_each_scale_the_signal(scope):
    core = scope.core

    def level(intensity, exposure):
        core.setProperty("Lamp", "Intensity", intensity)
        core.setExposure(exposure)
        core.snapImage()
        return core.getImage().mean()

    base = level(20, 5.0)

    assert core.getExposure() == 5.0 and float(core.getProperty("Lamp", "Intensity")) == 20.0
    assert level(40, 5.0) == pytest.approx(2 * base, rel=0.01)
    assert level(20, 10.0) == pytest.approx(2 * base, rel=0.01)


def test_a_long_exposure_saturates_at_full_scale(scope):
    core = scope.core
    core.setExposure(200.0)

    core.snapImage()

    assert (core.getImage() == 4095).all()


def test_lamp_intensity_stops_at_its_limits(scope):
    """Workaround: stock UniMMCore raises ValueError here, and this test used to
    expect that. Micro-Manager raises RuntimeError, and now so does this."""
    core = scope.core

    assert core.getPropertyLowerLimit("Lamp", "Intensity") == 0.0
    assert core.getPropertyUpperLimit("Lamp", "Intensity") == 100.0
    with pytest.raises(RuntimeError, match='Error in device "Lamp"'):
        core.setProperty("Lamp", "Intensity", 150)
    assert float(core.getProperty("Lamp", "Intensity")) == 50.0


def test_property_values_are_text_as_micro_manager_returns_them(scope):
    """Workaround: stock UniMMCore returns Python numbers, so `getProperty(...) == 1`
    would hold here and fail on real hardware."""
    core = scope.core

    assert core.getProperty("Camera", "Binning") == "1"
    assert core.getProperty("Camera", "Exposure") == "10.0000"
    assert core.getProperty("Objective", "State") == "0"
    assert core.getProperty("Objective", "Label") == "5x"
    assert core.getProperty("Lamp", "State") == "0"
    assert core.getProperty("Lamp", "Intensity") == "50.0000"


def test_properties_can_be_set_with_text_as_configuration_files_do(scope):
    """Workaround: stock UniMMCore refuses "1" for a property whose values are integers."""
    core = scope.core

    core.setProperty("Objective", "State", "3")
    core.setProperty("Lamp", "State", "1")
    core.setProperty("Lamp", "Intensity", "75")
    core.setProperty("Camera", "Exposure", "25")

    assert core.getState("Objective") == 3 and core.getStateLabel("Objective") == "50x"
    assert core.getShutterOpen()
    assert core.getProperty("Lamp", "Intensity") == "75.0000"
    assert core.getExposure() == 25.0
    # A preset keyed on the state number, the way Micro-Manager's demo configuration defines its objectives.
    core.defineConfig("Turret", "second", "Objective", "State", "1")
    core.setConfig("Turret", "second")
    assert core.getStateLabel("Objective") == "10x" and core.getCurrentConfig("Turret") == "second"
    assert core.getPixelSizeUm() == pytest.approx(0.345)


@pytest.mark.parametrize(
    "mistake",
    [
        lambda core: core.getProperty("Camera", "Nope"),
        lambda core: core.setProperty("Camera", "Nope", 1),
        lambda core: core.setProperty("Camera", "BitDepth", 8),  # read-only
        lambda core: core.setProperty("Lamp", "Intensity", "bright"),
        lambda core: core.setProperty("Camera", "Exposure", "long"),
        lambda core: core.setState("Objective", 99),
        lambda core: core.setStateLabel("Objective", "63x"),
        lambda core: core.getPosition("XY"),
        lambda core: core.getXYPosition("Z"),
        lambda core: core.getState("Camera"),
        lambda core: core.getProperty("Nope", "State"),
    ],
)
def test_mistakes_raise_runtime_error_as_micro_manager_does(scope, mistake):
    """Workaround: stock UniMMCore raises KeyError or ValueError for most of these."""
    with pytest.raises(RuntimeError):
        mistake(scope.core)
    assert scope.core.getStateLabel("Objective") == "5x" and scope.incidents == []


def test_a_new_microscope_answers_from_the_cache(scope):
    """A real core fills its state cache when it finishes loading; this one was left empty."""
    core = scope.core

    assert core.getCurrentConfigFromCache("Objective") == "5x"
    assert core.getPropertyFromCache("Objective", "Label") == "5x"
    assert core.getPropertyFromCache("Camera", "Exposure") == "10.0000"
    assert core.getPropertyFromCache("Lamp", "Intensity") == "50.0000"


@pytest.mark.parametrize("change", ["setConfig", "setStateLabel", "setState", "setProperty"])
def test_image_metadata_names_the_objective_that_is_in(scope, change):
    """Workaround: after setStateLabel, the tags of the next image still named the
    previous objective, beside the right pixel size."""
    core = scope.core
    core.setConfig("Objective", "50x")
    core.snapImage()
    core.getTaggedImage()

    if change == "setConfig":
        core.setConfig("Objective", "20x")
    elif change == "setStateLabel":
        core.setStateLabel("Objective", "20x")
    elif change == "setState":
        core.setState("Objective", 2)
    else:
        core.setProperty("Objective", "State", 2)
    core.snapImage()
    tags = core.getTaggedImage().tags

    assert (tags["Objective-Label"], tags["Objective-State"]) == ("20x", "2")
    assert tags["PixelSizeUm"] == pytest.approx(3.45 / 20)
    assert core.getPropertyFromCache("Objective", "Label") == "20x"
    assert core.getCurrentConfigFromCache("Objective") == "20x"


def test_the_core_names_the_device_in_each_role(scope):
    """Workaround: the core's own role properties were empty, in getProperty and in image metadata."""
    core = scope.core
    roles = {"Camera": "Camera", "Shutter": "Lamp", "Focus": "Z", "XYStage": "XY"}

    assert {role: core.getProperty("Core", role) for role in roles} == roles
    core.snapImage()
    tags = core.getTaggedImage().tags
    assert {role: tags[f"Core-{role}"] for role in roles} == roles
    core.setProperty("Core", "Shutter", "Lamp")
    assert core.getShutterDevice() == "Lamp"


def test_the_sequence_buffer_is_the_small_one_asked_for(scope):
    """It was set before the camera was loaded, which UniMMCore sends to the C++
    core, and the Python buffer kept its default of 1 GB."""
    assert scope.core.getCircularBufferMemoryFootprint() == 64
    assert 30 < scope.core.getBufferTotalCapacity() < 60


def test_a_negative_exposure_is_refused(scope):
    with pytest.raises(RuntimeError, match="Camera"):
        scope.core.setExposure(-1.0)
    assert scope.core.getExposure() == 10.0


@pytest.mark.parametrize("change", ["setConfig", "setStateLabel", "setState"])
def test_the_pixel_size_follows_the_objective(scope, change):
    """Workaround: stock UniMMCore reports 0 whatever the objective."""
    core = scope.core

    assert set(core.getAvailableConfigs("Objective")) == set(OBJECTIVE)
    for index, (name, objective) in enumerate(OBJECTIVE.items()):
        if change == "setConfig":
            core.setConfig("Objective", name)
        elif change == "setStateLabel":
            core.setStateLabel("Objective", name)
        else:
            core.setState("Objective", index)
        expected = 3.45 / objective.magnification

        assert core.getCurrentConfig("Objective") == name
        assert core.getPixelSizeUm() == pytest.approx(expected)
        assert core.getPixelSizeUm(True) == pytest.approx(expected)
        assert core.getPixelSizeAffine() == pytest.approx((expected, 0, 0, 0, expected, 0))


def test_the_contrast_methods_are_presets_and_change_the_picture(scope):
    core = scope.core
    focus(scope)

    assert set(core.getAvailableConfigs("Contrast")) == set(CONTRASTS)
    assert core.getChannelGroup() == "Contrast"
    core.snapImage()
    brightfield = core.getImage()
    core.setConfig("Contrast", "Darkfield")
    core.snapImage()
    darkfield = core.getImage()

    assert core.getStateLabel("Contrast") == "Darkfield"
    # The polished surface is bright in brightfield and nearly black in darkfield.
    assert np.median(darkfield) < 0.1 * np.median(brightfield)


def test_moving_the_stage_slides_the_picture_the_same_distance(scope):
    """Columns run with +x and rows with +y, and the pixel size is the true scale."""
    core = scope.core
    focus(scope)
    pixel = core.getPixelSizeUm()

    def interior():
        # With the lamp's falloff, which stays with the camera, divided out.
        # (Before there was a falloff this compared raw counts, to within 2.)
        core.snapImage()
        return core.getImage() / illumination()

    here = interior()
    core.setRelativeXYPosition(20 * pixel, 0.0)
    right = interior()
    core.setRelativeXYPosition(-20 * pixel, 10 * pixel)
    up = interior()

    # Away from the frame's own edge, where the blur has nothing to draw on.
    assert np.abs(right[8:-8, 8:-28] - here[8:-8, 28:-8]).max() <= 3
    assert np.abs(up[8:-18, 8:-8] - here[18:-8, 8:-8]).max() <= 3
    assert np.abs(right - here).mean() > 20


def test_the_picture_is_sharpest_with_the_focal_plane_on_the_surface(scope):
    core = scope.core
    surface = focus(scope, "50x")

    def sharpness_at(offset):
        core.setPosition(surface + offset)
        core.snapImage()
        return sharpness(core.getImage())

    in_focus = sharpness_at(0.0)

    assert in_focus > 1.5 * sharpness_at(-2.0)
    assert in_focus > 1.5 * sharpness_at(2.0)


def test_a_region_of_interest_crops_the_frame(scope):
    core = scope.core
    core.snapImage()
    whole = core.getImage()

    core.setROI(100, 200, 300, 150)
    core.snapImage()

    assert (core.getImageWidth(), core.getImageHeight()) == (300, 150)
    assert np.array_equal(core.getImage(), whole[200:350, 100:400])


def test_moves_and_settings_are_announced(scope):
    """Workaround: the core announces nothing for Python devices; the devices do."""
    core = scope.core
    heard = []
    core.events.XYStagePositionChanged.connect(lambda *args: heard.append(("xy", *args)))
    core.events.stagePositionChanged.connect(lambda *args: heard.append(("z", *args)))
    core.events.exposureChanged.connect(lambda *args: heard.append(("exposure", *args)))
    core.events.propertyChanged.connect(lambda *args: heard.append(args))

    core.setXYPosition(10.0, 20.0)
    core.setPosition(16000.0)
    core.setExposure(4.0)
    core.setProperty("Lamp", "Intensity", 30)
    core.setConfig("Objective", "10x")
    core.setShutterOpen(True)

    assert ("xy", "XY", 10.0, 20.0) in heard
    assert ("z", "Z", 16000.0) in heard
    assert ("exposure", "Camera", 4.0) in heard
    assert ("Lamp", "Intensity", "30.0000") in heard  # text, like the property itself; it used to be 30.0
    assert ("Objective", "Label", "10x") in heard
    assert ("Lamp", "State", "1") in heard  # it used to be announced under an empty label


@pytest.mark.parametrize("change", ["setConfig", "setStateLabel", "setState"])
def test_a_change_of_objective_announces_the_new_pixel_size_and_preset(scope, change):
    """Workaround: stock UniMMCore announces only the property, so a viewer's scale bar would not follow."""
    core = scope.core
    heard = []
    core.events.pixelSizeChanged.connect(lambda *args: heard.append(("pixel", *args)))
    core.events.pixelSizeAffineChanged.connect(lambda *args: heard.append(("affine", *args)))
    core.events.configGroupChanged.connect(lambda *args: heard.append(("group", *args)))

    if change == "setConfig":
        core.setConfig("Objective", "20x")
    elif change == "setStateLabel":
        core.setStateLabel("Objective", "20x")
    else:
        core.setState("Objective", 2)
    core.setProperty("Lamp", "Intensity", 30)  # changes neither

    assert [event[0] for event in heard] == ["group", "pixel", "affine"]
    assert heard[0] == ("group", "Objective", "20x")
    assert heard[1][1] == pytest.approx(0.1725)
    assert heard[2][1:] == pytest.approx((0.1725, 0, 0, 0, 0.1725, 0))


def test_a_listener_can_ask_the_core_where_the_stage_is(scope):
    """Workaround: with the stock device lock this waits on itself for ever."""
    core = scope.core
    seen = []
    core.events.XYStagePositionChanged.connect(lambda *_: seen.append(core.getXYPosition()))
    move = threading.Thread(target=core.setXYPosition, args=(5.0, 6.0), daemon=True)

    move.start()
    move.join(timeout=10)

    assert not move.is_alive(), "the listener deadlocked against its own device"
    assert seen == [(5.0, 6.0)]


def test_a_z_stack_runs_through_the_acquisition_engine(scope):
    core = scope.core
    surface = focus(scope)
    frames = []
    core.mda.events.frameReady.connect(
        lambda image, event, meta: frames.append((event.z_pos, sharpness(image), meta["pixel_size_um"]))
    )

    # Without a stage position the z plan is taken as absolute heights around 0,
    # which the focus drive would refuse.
    core.mda.run(
        useq.MDASequence(stage_positions=[(*PLACE, surface)], z_plan=useq.ZRangeAround(range=8.0, step=2.0))
    )

    heights = [z for z, _, _ in frames]
    assert heights == pytest.approx([surface + offset for offset in (-4, -2, 0, 2, 4)])
    assert max(frames, key=lambda frame: frame[1])[0] == pytest.approx(surface)
    assert all(pixel == pytest.approx(3.45 / 20) for _, _, pixel in frames)
    assert scope.incidents == []


def test_the_noise_is_the_same_for_the_same_calls(ceramic):
    def two_frames():
        core = build_microscope(ceramic).core
        frames = [core.snap().copy(), core.snap().copy()]
        core.unloadAllDevices()
        return frames

    first, second = two_frames()
    again, _ = two_frames()

    assert np.array_equal(first, again)
    assert not np.array_equal(first, second)


def test_a_plain_unimmcore_is_turned_away(ceramic, tmp_path):
    with pytest.raises(TypeError, match="SimCore"):
        build_microscope(ceramic, core=UniMMCore(mm_path=str(tmp_path)))


def test_a_specimen_that_does_not_fit_the_travel_is_turned_away():
    """A mount 24.9 mm tall used to start the focus drive outside its own travel,
    where every move, even one of zero, was refused."""
    with pytest.raises(ValueError, match="focus drive"):
        build_microscope(Sample.polished_ceramic(seed=1, mount_height_um=24900.0))
    with pytest.raises(ValueError, match="stage's travel"):
        build_microscope(Sample.polished_ceramic(seed=1, mount_diameter_um=2 * XY_TRAVEL_UM[1] + 2000.0))

    tall = build_microscope(Sample.polished_ceramic(seed=1, mount_height_um=24000.0))
    tall.core.setRelativePosition(-1.0)
    tall.core.unloadAllDevices()


def test_it_works_beside_a_qt_application():
    """With a QApplication alive, pymmcore-plus wants qtpy, which neither it nor
    this project installs. The tests here used to force the other signal
    backend through the environment, which hid that."""
    pytest.importorskip("PySide6.QtWidgets")
    script = (
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from scopepilot.sim import build_microscope\n"
        "core = build_microscope(seed=1).core\n"
        "core.setConfig('Objective', '10x')\n"
        "print(core.snap().shape, round(core.getPixelSizeUm(), 6))\n"
    )
    environment = {name: value for name, value in os.environ.items() if name != "PYMM_SIGNALS_BACKEND"}
    environment["QT_QPA_PLATFORM"] = "offscreen"

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=environment)

    assert done.stdout.strip().endswith("(768, 1024) 0.345"), done.stderr


# --- finite travel ----------------------------------------------------------------


@pytest.mark.parametrize("target", [(XY_TRAVEL_UM[1] + 1.0, 0.0), (0.0, XY_TRAVEL_UM[0] - 1.0), (float("nan"), 0.0)])
def test_the_stage_refuses_to_leave_its_travel(scope, target):
    core = scope.core
    core.setXYPosition(1000.0, -2000.0)

    with pytest.raises(RuntimeError, match='Error in device "XY"'):
        core.setXYPosition(*target)

    assert core.getXYPosition() == (1000.0, -2000.0)
    assert [(refusal.device, len(refusal.requested_um)) for refusal in scope.refusals] == [("XY", 2)]


def test_a_relative_move_past_the_end_of_travel_is_refused_too(scope):
    core = scope.core
    core.setXYPosition(XY_TRAVEL_UM[1], 0.0)

    with pytest.raises(RuntimeError):
        core.setRelativeXYPosition(1.0, 0.0)

    assert core.getXYPosition() == (XY_TRAVEL_UM[1], 0.0)


@pytest.mark.parametrize("target", [Z_TRAVEL_UM[0] - 1.0, Z_TRAVEL_UM[1] + 1.0])
def test_the_focus_drive_refuses_to_leave_its_travel(scope, target):
    core = scope.core
    before = core.getPosition()

    with pytest.raises(RuntimeError, match='Error in device "Z"'):
        core.setPosition(target)

    assert core.getPosition() == before
    assert scope.refusals[0].device == "Z" and scope.refusals[0].requested_um == (target,)
    assert scope.incidents == []


def test_homing_the_focus_drive_backs_away_from_the_specimen(scope):
    scope.core.home("Z")

    assert scope.core.getPosition() == Z_TRAVEL_UM[1]
    assert scope.core.getFocusDirection("Z").name == "AwayFromSample"


# --- contact ---------------------------------------------------------------------


def test_careful_use_of_every_objective_records_nothing(scope):
    core = scope.core

    for name in OBJECTIVE:
        focus(scope, name)
        core.setXYPosition(2900.0, -2900.0)
        core.setXYPosition(8000.0, 0.0)  # out over the resin
        core.setRelativePosition(-5.0)
        core.snapImage()
    core.setConfig("Objective", "5x")

    assert scope.incidents == [] and scope.refusals == []


def test_driving_the_focus_down_into_the_specimen_is_one_incident(scope):
    core = scope.core
    surface = focus(scope, "50x")
    deep = surface - OBJECTIVE["50x"].working_distance_um - 200.0

    core.setPosition(deep)  # no exception: a real objective does not announce a crash

    assert core.getPosition() == deep
    [incident] = scope.incidents
    assert (incident.kind, incident.objective, incident.hit) == ("z", "50x", "specimen")
    assert (incident.x_um, incident.y_um, incident.z_um) == (*PLACE, deep)
    # 200 um below the surface on the axis, a little more where the tilted
    # surface is higher under the far side of the nose.
    assert 200.0 <= incident.penetration_um < 215.0


def test_each_move_further_in_is_another_incident_and_backing_out_is_not(scope):
    core = scope.core
    surface = focus(scope, "50x")
    touching = surface - OBJECTIVE["50x"].working_distance_um

    core.setPosition(touching - 100.0)
    core.setPosition(touching - 300.0)
    core.setPosition(touching - 50.0)  # still touching, but on the way out
    core.setPosition(surface)

    assert [incident.kind for incident in scope.incidents] == ["z", "z"]
    assert scope.incidents[1].penetration_um > scope.incidents[0].penetration_um
    # One crash, carried further: only the first move began it.
    assert [incident.ongoing for incident in scope.incidents] == [False, True]


def test_moves_made_while_still_touching_do_not_count_as_new_crashes(scope):
    """A scan made after one bad objective change used to look like 251 crashes."""
    core = scope.core
    core.setPosition(scope.sample.height_um(0.0, 0.0) - 5000.0)
    core.setConfig("Objective", "20x")  # the crash

    for step in range(1, 6):
        core.setXYPosition(100.0 * step, 0.0)
    core.setRelativePosition(-10.0)
    core.setConfig("Objective", "50x")

    assert [incident.kind for incident in scope.incidents] == ["objective", *["xy"] * 5, "z", "objective"]
    assert [incident.ongoing for incident in scope.incidents] == [False, *[True] * 7]


def test_driving_onto_the_mount_with_the_objective_too_low_is_one_incident(scope):
    core = scope.core
    core.setXYPosition(*OFF_THE_MOUNT)
    core.setConfig("Objective", "10x")
    core.setPosition(0.0)  # focused on the stage plate, front lens 11 mm above it
    assert scope.incidents == []

    core.setXYPosition(0.0, 0.0)

    assert core.getXYPosition() == (0.0, 0.0)
    [incident] = scope.incidents
    assert (incident.kind, incident.objective, incident.z_um) == ("xy", "10x", 0.0)
    assert incident.hit == "specimen"  # the nose ends up over the specimen and below its surface
    assert not incident.ongoing
    # The mount stands 15 mm high and the front lens is at 11 mm.
    assert incident.penetration_um == pytest.approx(4000.0, abs=50.0)
    # Where contact began: the 5 mm nose reaching the rim of the 15 mm mount.
    # This used to be an arbitrary point of the path (x = 12222), where the
    # nose was nowhere near the specimen the incident says it hit.
    assert (incident.x_um, incident.y_um) == pytest.approx((20000.0, 0.0), abs=2.0)


def test_a_move_that_crosses_the_mount_is_caught_though_both_ends_are_clear(scope):
    core = scope.core
    core.setXYPosition(0.0, 21000.0)
    core.setConfig("Objective", "10x")
    core.setPosition(0.0)

    core.setXYPosition(21000.0, 6000.0)  # cuts the corner, across the rim of the mount

    [incident] = scope.incidents
    assert (incident.kind, incident.hit) == ("xy", "resin")
    assert incident.penetration_um == pytest.approx(4000.0, abs=50.0)
    # Contact began where the path first came within a nose radius of the rim.
    assert np.hypot(incident.x_um, incident.y_um) == pytest.approx(20000.0, abs=2.0)
    assert incident.x_um < 4000.0 and incident.y_um > 19000.0


def test_rotating_in_an_objective_too_long_for_the_focus_is_one_incident(scope):
    core = scope.core
    surface = scope.sample.height_um(0.0, 0.0)
    core.setPosition(surface - 5000.0)  # fine for the 5x, whose front lens is 20 mm from focus
    assert scope.incidents == []

    core.setConfig("Objective", "20x")  # 3.1 mm from focus

    assert core.getStateLabel("Objective") == "20x"
    [incident] = scope.incidents
    assert (incident.kind, incident.objective, incident.hit) == ("objective", "20x", "specimen")
    assert incident.penetration_um == pytest.approx(5000.0 - 3100.0, abs=10.0)


def test_the_tall_5x_clears_the_mount_even_focused_on_the_stage_plate(scope):
    core = scope.core
    core.setXYPosition(*OFF_THE_MOUNT)
    core.setPosition(0.0)

    core.setXYPosition(0.0, 0.0)

    assert scope.incidents == []


# --- the truth --------------------------------------------------------------------


def test_truth_in_view_is_the_truth_for_the_field_under_the_camera(scope):
    core = scope.core
    focus(scope, "20x")

    seen = scope.truth_in_view()
    expected = scope.sample.truth(*PLACE, *field_of_view_um(OBJECTIVE["20x"]))

    assert (seen.x_um, seen.y_um) == PLACE
    assert (seen.width_um, seen.height_um) == pytest.approx((1024 * 3.45 / 20, 768 * 3.45 / 20))
    assert seen.porosity == expected.porosity and seen.grain_count == expected.grain_count
    assert np.array_equal(seen.pores_um, expected.pores_um)

    core.setConfig("Objective", "50x")

    assert scope.truth_in_view().grain_count < seen.grain_count


def test_truth_in_view_follows_a_region_of_interest(scope):
    core = scope.core
    focus(scope, "20x")
    pixel = core.getPixelSizeUm()

    core.setROI(100, 200, 300, 150)
    seen = scope.truth_in_view()

    assert (seen.width_um, seen.height_um) == pytest.approx((300 * pixel, 150 * pixel))
    # The region's centre is 262 columns left of and 109 rows before the frame's.
    assert seen.x_um == pytest.approx(PLACE[0] - 262 * pixel)
    assert seen.y_um == pytest.approx(PLACE[1] - 109 * pixel)


# --- the gallery and the packaging -------------------------------------------------


def test_the_demo_writes_a_gallery_and_its_truth(tmp_path, capsys):
    command.main(["demo", "--out", str(tmp_path / "gallery")])

    pictures = sorted(path.name for path in (tmp_path / "gallery").glob("*.png"))
    truth = (tmp_path / "gallery" / "truth.txt").read_text()

    assert len(pictures) == 15
    assert {"contrast-darkfield.png", "focus-3-in-focus.png", "objective-100x.png", "overexposed.png", "edge.png"} <= set(
        pictures
    )
    assert all(name in truth for name in pictures)
    assert "touched anything while these were taken: 0." in truth
    assert "15 frames" in capsys.readouterr().out


def test_the_specimen_and_optics_do_not_need_micro_manager():
    """`import scopepilot` does not reach for the simulator, and the specimen
    and optics need only numpy and scipy."""
    script = (
        "import sys\n"
        "sys.modules['pymmcore_plus'] = None\n"  # makes `import pymmcore_plus` fail
        "import scopepilot\n"
        "assert 'scopepilot.sim' not in sys.modules\n"
        "from scopepilot.sim import Sample\n"
        "import scopepilot.sim.optics\n"
        "try:\n"
        "    from scopepilot.sim import build_microscope\n"
        "except ImportError:\n"
        "    print('ok')\n"
    )

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.stdout.strip() == "ok", done.stderr
