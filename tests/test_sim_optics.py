"""The pretend optics: frames behave the way frames from a reflected-light
microscope do when the stage, the focus, the lamp and the contrast method change.

Most frames here are small crops of the sensor so the file runs in seconds.
None of the thresholds comes from a real instrument.
"""

import pytest

np = pytest.importorskip("numpy")
ndimage = pytest.importorskip("scipy.ndimage")

from dataclasses import replace  # noqa: E402

from scopepilot.sim import optics  # noqa: E402
from scopepilot.sim.optics import (  # noqa: E402
    METALLURGICAL_OBJECTIVES,
    exposure_for,
    field_of_view_um,
    illumination,
    render,
)
from scopepilot.sim.sample import CERAMIC, Region, Sample  # noqa: E402

OBJECTIVE = {objective.name: objective for objective in METALLURGICAL_OBJECTIVES}
SMALL = (256, 320)


@pytest.fixture(scope="module")
def ceramic():
    return Sample.polished_ceramic(seed=1)


@pytest.fixture(scope="module")
def pore(ceramic):
    """Centre and diameter of a large pore."""
    pores = ceramic.truth(200.0, -300.0, 150.0, 150.0).pores_um
    return pores[np.argmax(pores[:, 2])]


def frame(sample, x, y, objective="50x", contrast="Brightfield", *, defocus=0.0, shape=SMALL, exposure=None, **settings):
    """A noiseless, well-exposed frame focused `defocus` above the surface at (x, y)."""
    objective = OBJECTIVE[objective]
    settings.setdefault("lamp_fraction", 1.0)
    settings.setdefault("shutter_open", True)
    return render(
        sample,
        x_um=x,
        y_um=y,
        z_um=sample.height_um(x, y) + defocus,
        objective=objective,
        contrast=contrast,
        exposure_ms=exposure_for(sample, objective, contrast) if exposure is None else exposure,
        sensor_shape=shape,
        **settings,
    )


def view(sample, x, y, objective="50x", shape=SMALL):
    """What the specimen holds under each pixel of such a frame."""
    return sample.fields(x, y, *field_of_view_um(OBJECTIVE[objective], shape), shape)


def well_inside(mask, pixels):
    """The part of a mask at least `pixels` from anything outside it."""
    return ndimage.minimum_filter(mask.astype(np.uint8), size=2 * pixels + 1).astype(bool)


def sharpness(image):
    image = image.astype(np.float64)
    return float(np.mean(np.diff(image, axis=0) ** 2) + np.mean(np.diff(image, axis=1) ** 2))


# --- objectives ------------------------------------------------------------------


def test_the_objective_set():
    assert [o.name for o in METALLURGICAL_OBJECTIVES] == ["5x", "10x", "20x", "50x", "100x"]
    for low, high in zip(METALLURGICAL_OBJECTIVES, METALLURGICAL_OBJECTIVES[1:]):
        assert low.magnification < high.magnification
        assert low.numerical_aperture < high.numerical_aperture
        assert low.working_distance_um >= high.working_distance_um
        assert low.depth_of_field_um > high.depth_of_field_um
        assert low.resolution_um > high.resolution_um
    assert OBJECTIVE["100x"].resolution_um == pytest.approx(0.61 * 0.55 / 0.9)
    assert OBJECTIVE["5x"].depth_of_field_um == pytest.approx(0.55 / 0.15**2)
    assert OBJECTIVE["50x"].depth_of_field_um < 1.0  # a micrometre of focus error shows at 50x


def test_the_field_of_view_follows_the_magnification():
    assert field_of_view_um(OBJECTIVE["5x"]) == pytest.approx((1024 * 0.69, 768 * 0.69))
    assert field_of_view_um(OBJECTIVE["100x"], (100, 200), pixel_pitch_um=5.0) == pytest.approx((10.0, 5.0))


def test_a_full_frame_has_the_sensor_shape(ceramic):
    image = render(
        ceramic, x_um=0.0, y_um=0.0, z_um=ceramic.height_um(0.0, 0.0), objective=OBJECTIVE["5x"],
        contrast="Brightfield", lamp_fraction=0.5, exposure_ms=10.0, shutter_open=True,
    )  # fmt: skip

    assert image.shape == (768, 1024) and image.dtype == np.uint16
    # A 12% reflector at half lamp and 10 ms is 60% of full scale, less what pores and grooves take.
    assert 0.5 * 4095 < np.median(image) < 0.62 * 4095


def test_unknown_settings_are_refused(ceramic):
    with pytest.raises(ValueError, match="unknown contrast"):
        frame(ceramic, 0.0, 0.0, contrast="Phase")
    with pytest.raises(ValueError, match="lamp_fraction"):
        frame(ceramic, 0.0, 0.0, lamp_fraction=1.5)
    with pytest.raises(ValueError, match="exposure_ms"):
        frame(ceramic, 0.0, 0.0, exposure=-1.0)


# --- where things land in the frame ------------------------------------------------


def test_the_stage_position_is_the_centre_of_the_frame_and_columns_run_with_x(ceramic, pore):
    x, y, _ = pore
    pixel = 3.45 / 50

    centred = frame(ceramic, x, y)
    moved_right = frame(ceramic, x + 40 * pixel, y)
    moved_up = frame(ceramic, x, y + 30 * pixel)

    bright = np.median(centred)
    assert centred[128, 160] < 0.3 * bright
    # Moving the stage to larger x brings the pore to a smaller column; same for y and rows.
    assert moved_right[128, 120] < 0.3 * bright and moved_right[128, 160] > 0.5 * bright
    assert moved_up[98, 160] < 0.3 * bright


def test_the_same_pore_is_there_at_every_magnification(ceramic, pore):
    x, y, diameter = pore
    rows, cols = np.mgrid[-64:65, -64:65]
    for name in ("5x", "20x", "100x"):
        image = frame(ceramic, x, y, name, shape=(129, 129)).astype(float)
        pixels_across = diameter / (3.45 / OBJECTIVE[name].magnification)
        darkness = np.clip(np.median(image) - image, 0, None) * (np.hypot(rows, cols) < max(0.8 * pixels_across, 3))

        assert image[64, 64] < 0.8 * np.median(image)
        for axis in (rows, cols):
            assert abs((darkness * axis).sum() / darkness.sum()) < max(1.0, 0.1 * pixels_across)


# --- the three contrast methods ------------------------------------------------------


def test_pores_are_dark_in_brightfield_and_bright_in_darkfield(ceramic, pore):
    x, y, _ = pore
    fields = view(ceramic, x, y)
    plain = (fields.pore == 0) & (fields.boundary == 0) & (fields.second_phase == 0) & (fields.scratch == 0)
    flat = well_inside(plain, 8)
    in_pore = well_inside(fields.pore == 1, 4)
    assert flat.sum() > 500 and in_pore.sum() > 50

    brightfield = frame(ceramic, x, y)
    darkfield = frame(ceramic, x, y, contrast="Darkfield")

    assert brightfield[in_pore].mean() < 0.25 * brightfield[flat].mean()
    assert brightfield[flat].mean() == pytest.approx(0.65 * 4095, rel=0.05)
    assert darkfield[fields.pore > 0.5].mean() > 5 * darkfield[flat].mean()
    assert darkfield[flat].mean() < 0.06 * 4095
    # The wall of the pore is brighter than its floor.
    rim = (fields.pore_rim > 0.9) & (fields.pore == 1)
    assert darkfield[rim].mean() > 1.5 * darkfield[in_pore].mean()


def test_etched_grain_boundaries_are_dark_lines_in_brightfield_and_bright_in_darkfield(ceramic):
    fields = view(ceramic, 30.0, 40.0)
    clear = (fields.pore == 0) & (fields.second_phase == 0) & (fields.scratch == 0)
    groove = well_inside(clear, 6) & (fields.boundary > 0.8)  # the strongly etched ones
    faint = well_inside(clear, 6) & (fields.boundary > 0) & (fields.boundary < 0.15)
    flat = well_inside(clear & (fields.boundary == 0), 8)
    assert groove.sum() > 500 and faint.sum() > 100 and flat.sum() > 5000
    even = illumination(SMALL)

    brightfield = frame(ceramic, 30.0, 40.0) / even
    darkfield = frame(ceramic, 30.0, 40.0, contrast="Darkfield") / even
    unetched = frame(Sample.polished_ceramic(seed=1, etched=False), 30.0, 40.0) / even

    assert brightfield[groove].mean() < 0.85 * brightfield[flat].mean()
    assert darkfield[groove].mean() > 3 * darkfield[flat].mean()
    assert unetched[groove].mean() == pytest.approx(unetched[flat].mean(), rel=0.02)
    # A boundary the etch barely touched is barely there.
    assert brightfield[faint].mean() > 0.9 * brightfield[flat].mean()


@pytest.mark.parametrize("name", ["50x", "100x"])
def test_a_threshold_on_an_etched_brightfield_frame_measures_the_porosity(name):
    """With grooves as dark as pore edges, as they were, a threshold halfway
    between pore and matrix counted grooves as pores: 1371 blobs for 5 pores
    in one 100x field. Grooves now stay well above that threshold."""
    sample = Sample.polished_ceramic(seed=0)
    objective = OBJECTIVE[name]
    halfway = (optics.PORE_DARKNESS + 1.0) / 2
    for seed, (x, y) in enumerate([(120.0, 80.0), (-900.0, 400.0), (1500.0, -1200.0)]):
        image = frame(sample, x, y, name, shape=(768, 1024), noise_seed=seed).astype(float)
        truth = sample.truth(x, y, *field_of_view_um(objective))
        dark = image < halfway * np.median(image)
        fields = view(sample, x, y, name, shape=(768, 1024))
        near_a_pore = ndimage.maximum_filter(fields.pore, size=2 * int(1.0 / (3.45 / objective.magnification)) + 5) > 0

        assert dark.mean() == pytest.approx(truth.porosity, rel=0.08)
        # Pores cut by the frame's edge are seen but not listed, hence the slack.
        assert abs(ndimage.label(dark)[1] - len(truth.pores_um)) <= 5
        assert image[~near_a_pore].min() > 0.62 * np.median(image), "nothing but a pore comes near the threshold"


def grain_levels(sample, contrast):
    """Mean level of each grain's interior in a flat-fielded frame, and the largest spread within one."""
    fields = view(sample, 30.0, 40.0)
    image = frame(sample, 30.0, 40.0, contrast=contrast) / illumination(SMALL)
    labels = fields.grain_id
    one_grain = ndimage.minimum_filter(labels, size=31) == ndimage.maximum_filter(labels, size=31)
    clear = (fields.pore == 0) & (fields.second_phase == 0) & (fields.scratch == 0)
    interior = one_grain & well_inside(clear, 15)
    grains = [image[interior & (labels == grain)] for grain in np.unique(labels[interior])]
    grains = [pixels for pixels in grains if len(pixels) > 50]
    assert len(grains) >= 5
    return np.array([pixels.mean() for pixels in grains]), max(np.ptp(pixels) for pixels in grains)


def test_between_crossed_polarisers_a_ceramic_glows_dimly_and_its_pores_are_the_brightest_thing(ceramic, pore):
    """This test used to demand strong grain-to-grain contrast from the
    ceramic. That is how a hexagonal metal looks; a translucent, weakly
    birefringent ceramic shows a dim glow with bright pores and hardly any
    grain contrast, so its grains cannot be read off an unetched specimen."""
    x, y, _ = pore
    fields = view(ceramic, x, y)
    image = frame(ceramic, x, y, contrast="Polarised") / illumination(SMALL)
    in_pore = well_inside(fields.pore == 1, 4)
    matrix = well_inside((fields.pore == 0) & (fields.scratch == 0), 8)

    assert image[in_pore].mean() == pytest.approx(0.65 * 4095, rel=0.05)
    assert 0.15 * 4095 < image[matrix].mean() < 0.3 * 4095
    assert image[matrix].max() < image[in_pore].min()

    # What the grains add is a few percent of that glow, less than its own unevenness.
    isotropic = Sample(seed=1, material=replace(CERAMIC, anisotropy=0.0))
    without_grains = frame(isotropic, x, y, contrast="Polarised") / illumination(SMALL)
    from_the_grains = (image - without_grains)[matrix]
    assert 0.0 < from_the_grains.max() < 0.05 * image[matrix].mean()
    assert from_the_grains.max() < np.ptp(without_grains[matrix])
    # In brightfield the same grains are all alike.
    assert np.ptp(grain_levels(ceramic, "Brightfield")[0]) < 0.05 * 4095


def test_between_crossed_polarisers_a_strongly_anisotropic_material_shows_its_grains():
    """What the polarised mode is for on a hexagonal metal: one flat tone per
    grain, set by its orientation. No such material ships; this one is made up."""
    metal = replace(CERAMIC, name="made-up hexagonal metal", anisotropy=0.04, subsurface_glow=0.0)
    specimen = Sample(seed=1, material=metal)

    levels, spread_within_a_grain = grain_levels(specimen, "Polarised")

    assert spread_within_a_grain <= 4, "one grain, one brightness"
    assert np.ptp(levels) > 0.2 * 4095


def test_resin_is_duller_than_the_specimen_and_the_stage_plate_is_darker_still(ceramic):
    edge = frame(ceramic, 3000.0, 100.0, "10x")
    region = view(ceramic, 3000.0, 100.0, "10x").region
    assert np.all(region[:, :150] == Region.SPECIMEN) and np.all(region[:, 170:] == Region.RESIN)
    assert np.median(edge[:, 170:]) < 0.6 * np.median(edge[:, :150])

    rim = frame(ceramic, 15000.0, 0.0, "5x", exposure=exposure_for(ceramic, OBJECTIVE["5x"]))
    assert rim[:, -20:].mean() < rim[:, :20].mean()


def test_at_the_rim_of_the_mount_the_resin_stays_even_and_the_stage_plate_is_nearly_black(ceramic):
    """The plate's blurred light used to be added on top of the resin, as if
    the mount were glass: the resin brightened by 40% towards its rim and the
    plate, 15 mm out of focus, came out almost as bright as the resin."""
    objective = OBJECTIVE["5x"]
    rim = render(
        ceramic, x_um=15000.0, y_um=0.0, z_um=ceramic.height_um(14700.0, 0.0), objective=objective,
        contrast="Brightfield", lamp_fraction=1.0, exposure_ms=exposure_for(ceramic, objective), shutter_open=True,
    )  # fmt: skip
    even = rim / illumination() / 4095
    region = ceramic.fields(15000.0, 0.0, *field_of_view_um(objective), (768, 1024)).region
    resin, stage = region == Region.RESIN, region == Region.STAGE
    far_from_the_rim = even[:, :100][resin[:, :100]].mean()
    at_the_rim = even[well_inside(resin, 3) & ~well_inside(resin, 40)].mean()

    assert at_the_rim == pytest.approx(far_from_the_rim, rel=0.03)
    assert even[well_inside(stage, 3)].max() < 0.02
    assert far_from_the_rim > 0.2


def test_the_specimens_edge_is_a_dark_line_in_brightfield_and_a_bright_one_in_darkfield(ceramic):
    """It used not to show in darkfield at all, where a real cut edge is among the brightest lines."""
    x, y = 3000.0, 100.0
    fields = view(ceramic, x, y, "20x")
    resin = fields.region == Region.RESIN
    gap, plain_resin = fields.edge_gap > 0.9, well_inside(resin & (fields.edge_gap == 0), 12)
    assert gap.sum() > 500 and plain_resin.sum() > 5000
    even = illumination(SMALL)
    on_resin = ceramic.height_um(3050.0, y) - ceramic.height_um(x, y)

    brightfield = frame(ceramic, x, y, "20x", defocus=on_resin) / even
    darkfield = frame(ceramic, x, y, "20x", "Darkfield", defocus=on_resin) / even

    assert brightfield[gap].mean() < 0.8 * brightfield[plain_resin].mean()
    assert darkfield[gap].mean() > 2 * darkfield[plain_resin].mean()
    flat = well_inside((fields.pore == 0) & (fields.boundary == 0) & (fields.region == Region.SPECIMEN), 8)
    assert darkfield[gap].mean() > 5 * darkfield[flat].mean()


# --- focus -----------------------------------------------------------------------------


def test_the_frame_is_sharpest_when_the_focal_plane_is_on_the_surface(ceramic):
    offsets = (-3.0, -1.5, -0.7, -0.3, 0.0, 0.3, 0.7, 1.5, 3.0)
    scores = [sharpness(frame(ceramic, 30.0, 40.0, "50x", defocus=d)) for d in offsets]

    assert int(np.argmax(scores)) == offsets.index(0.0)
    assert scores[:5] == sorted(scores[:5])
    assert scores[4:] == sorted(scores[4:], reverse=True)
    assert scores[0] == pytest.approx(scores[-1], rel=0.05)  # the same above and below


def test_focus_is_lost_faster_at_higher_numerical_aperture(ceramic):
    kept = {}
    for name in ("5x", "20x", "50x", "100x"):
        in_focus = sharpness(frame(ceramic, 30.0, 40.0, name))
        kept[name] = sharpness(frame(ceramic, 30.0, 40.0, name, defocus=2.0)) / in_focus

    assert kept["5x"] > 0.9
    assert kept["5x"] > kept["20x"] > kept["50x"] > kept["100x"]
    assert kept["50x"] < 0.1


def test_blur_follows_the_depth_of_field():
    objective = OBJECTIVE["20x"]
    in_focus = float(optics.blur_sigma_um(objective, 0.0))

    assert in_focus * 2.3548 == pytest.approx(objective.resolution_um, rel=1e-3)
    assert float(optics.blur_sigma_um(objective, objective.depth_of_field_um / 2)) == pytest.approx(in_focus * 2**0.5)
    assert float(optics.blur_sigma_um(objective, -40.0)) == pytest.approx(float(optics.blur_sigma_um(objective, 40.0)))


def test_the_fast_wide_blur_matches_a_plain_gaussian():
    rng = np.random.default_rng(0)
    image = ndimage.gaussian_filter(rng.random((200, 260)).astype(np.float32), 1.0)
    for sigma in (4.5, 9.0, 23.0):
        exact = ndimage.gaussian_filter(image, sigma, mode="nearest")
        assert np.abs(optics._blur(image, sigma) - exact).max() < 0.02 * np.ptp(image)


def test_specimen_and_resin_cannot_both_be_in_focus_at_high_magnification(ceramic):
    """Resin sits a micrometre or so below the specimen's rounded edge: more than a 100x objective forgives."""
    x, y = 2999.0, 100.0
    fields = view(ceramic, x, y, "100x")
    specimen = fields.region == Region.SPECIMEN
    assert specimen[:, :180].all() and not specimen[:, 200:].any()
    on_specimen = float(fields.surface_um[:, :180].mean())
    on_resin = float(fields.surface_um[:, 200:].mean())
    assert 0.5 < on_specimen - on_resin < 2.5

    here = ceramic.height_um(x, y)
    focused = frame(ceramic, x, y, "100x", defocus=on_specimen - here)
    too_low = frame(ceramic, x, y, "100x", defocus=on_resin - here)

    assert sharpness(focused[:, 20:160]) > 3 * sharpness(too_low[:, 20:160])


# --- light ---------------------------------------------------------------------------------


def test_brightness_follows_lamp_times_exposure(ceramic):
    base = frame(ceramic, 30.0, 40.0, "20x", exposure=4.0, lamp_fraction=0.5).astype(float)
    longer = frame(ceramic, 30.0, 40.0, "20x", exposure=8.0, lamp_fraction=0.5).astype(float)
    brighter = frame(ceramic, 30.0, 40.0, "20x", exposure=4.0, lamp_fraction=1.0)
    traded = frame(ceramic, 30.0, 40.0, "20x", exposure=16.0, lamp_fraction=0.25)

    assert base.max() < 2000
    assert longer.mean() == pytest.approx(2 * base.mean(), rel=0.002)
    assert np.array_equal(brighter, longer)
    assert np.array_equal(traded, longer)


def test_too_much_light_saturates_at_full_scale(ceramic):
    fair = exposure_for(ceramic, OBJECTIVE["20x"])
    over = frame(ceramic, 30.0, 40.0, "20x", exposure=3 * fair)
    way_over = frame(ceramic, 30.0, 40.0, "20x", exposure=300 * fair, noise_seed=1)

    assert over.max() == 4095 and np.mean(over == 4095) > 0.5 and over.min() < 4095  # pores still show
    assert np.all(way_over == 4095)
    assert frame(ceramic, 30.0, 40.0, "20x", exposure=300 * fair, bit_depth=8).max() == 255
    assert frame(ceramic, 30.0, 40.0, "20x", exposure=300 * fair, bit_depth=16).max() == 65535


def test_higher_magnification_needs_more_light(ceramic):
    levels = [np.median(frame(ceramic, 30.0, 40.0, name, exposure=5.0)) for name in ("5x", "20x", "100x")]

    assert levels[0] > 1.5 * levels[1] > 1.5 * 1.5 * levels[2]
    # exposure_for makes up for it. A specimen with nothing on it shows the level it aims for.
    plain = Sample.polished_ceramic(seed=1, etched=False, porosity=0.0, second_phase=0.0, scratch_count=0)
    for name in ("5x", "10x", "20x", "50x", "100x"):
        assert np.median(frame(plain, 30.0, 40.0, name)) == pytest.approx(0.65 * 4095, rel=0.03)


def test_darkfield_and_polarised_need_far_more_exposure_than_brightfield(ceramic):
    objective = OBJECTIVE["20x"]
    brightfield = exposure_for(ceramic, objective)

    assert exposure_for(ceramic, objective, "Darkfield") > 8 * brightfield
    assert exposure_for(ceramic, objective, "Polarised") > 8 * brightfield
    assert exposure_for(ceramic, objective, lamp_fraction=0.5) == pytest.approx(2 * brightfield)
    # Switching to darkfield without touching the exposure gives a nearly
    # black frame. It used to give a usable picture, a fifth as bright as
    # brightfield in places.
    unadjusted = frame(ceramic, 30.0, 40.0, "20x", "Darkfield", exposure=brightfield)
    assert unadjusted.mean() < 0.01 * 4095
    assert np.percentile(unadjusted, 99) < 0.05 * 4095


def test_the_lamp_lights_the_corners_of_the_frame_less_than_the_centre():
    """Illumination used to be perfectly even, which no real frame is without a flat-field correction."""
    plain = Sample.polished_ceramic(seed=1, etched=False, porosity=0.0, second_phase=0.0, scratch_count=0)
    image = frame(plain, 30.0, 40.0, "20x", shape=(768, 1024)).astype(float)
    corners = np.mean([image[:8, :8].mean(), image[:8, -8:].mean(), image[-8:, :8].mean(), image[-8:, -8:].mean()])
    centre = image[380:388, 508:516].mean()
    even = illumination()

    assert corners / centre == pytest.approx(1 - optics.ILLUMINATION_FALLOFF, abs=0.01)
    assert 0.03 < optics.ILLUMINATION_FALLOFF < 0.2
    assert even.shape == (768, 1024) and even.max() == pytest.approx(1.0, abs=1e-4)
    assert np.ptp(image / even) < 0.04 * centre  # what is left is the specimen's own slow mottle
    # The falloff belongs to the camera: it does not move with the stage.
    moved = frame(plain, 300.0, -200.0, "20x", shape=(768, 1024)).astype(float)
    assert moved[:8, :8].mean() / moved[380:388, 508:516].mean() == pytest.approx(0.92, abs=0.02)


def test_a_closed_shutter_gives_a_dark_frame(ceramic):
    silent = frame(ceramic, 30.0, 40.0, shutter_open=False)
    noisy = frame(ceramic, 30.0, 40.0, shutter_open=False, noise_seed=5)

    assert not silent.any()
    assert noisy.mean() < 2 and noisy.max() < 12
    assert not frame(ceramic, 30.0, 40.0, lamp_fraction=0.0).any()
    assert not frame(ceramic, 30.0, 40.0, exposure=0.0).any()


# --- noise ---------------------------------------------------------------------------------


def test_noise_comes_from_its_seed(ceramic):
    first = frame(ceramic, 30.0, 40.0, noise_seed=11)
    again = frame(ceramic, 30.0, 40.0, noise_seed=11)
    other = frame(ceramic, 30.0, 40.0, noise_seed=12)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_without_a_seed_there_is_no_noise(ceramic):
    # The stage plate beside the mount is featureless, so once the lamp's
    # falloff is divided out, anything left is noise. The plate is dim, hence
    # the long exposure. (This used 15 ms when the plate was 20 times brighter
    # and the lamp perfectly even.)
    stage_plate = dict(defocus=15000.0, exposure=300.0)
    clean = frame(ceramic, 20000.0, 0.0, "5x", **stage_plate)
    noisy = frame(ceramic, 20000.0, 0.0, "5x", **stage_plate, noise_seed=2).astype(float)

    assert np.ptp(clean / illumination(SMALL)) < 1.5 and clean[128, 160] > 1000
    assert np.array_equal(clean, frame(ceramic, 20000.0, 0.0, "5x", **stage_plate))
    # Shot noise: about sqrt(electrons), here around 1% of the level.
    electrons = clean.mean() / 4095 * optics.FULL_WELL_ELECTRONS
    assert noisy.mean() == pytest.approx(clean.mean(), rel=0.002)
    assert (noisy - clean).std() / clean.mean() == pytest.approx(1 / np.sqrt(electrons), rel=0.1)
