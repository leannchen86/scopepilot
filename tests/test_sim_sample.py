"""The pretend specimen: the same every time, the same at every scale, and
honest about itself.

Windows here are small so the whole file runs in a few seconds. Nothing is
compared with a real micrograph; the checks are that the generator agrees with
itself and with what it was asked for.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from scopepilot.sim import sample as sample_module  # noqa: E402
from scopepilot.sim.sample import RESIN_RECESS_UM, Region, Sample  # noqa: E402


@pytest.fixture(scope="module")
def ceramic():
    return Sample.polished_ceramic(seed=1)


def biggest_pore(sample, x=200.0, y=-300.0, size=150.0):
    pores = sample.truth(x, y, size, size).pores_um
    return pores[np.argmax(pores[:, 2])]


# --- the same every time -------------------------------------------------------


def test_the_same_seed_gives_the_same_specimen():
    window = (40.0, -25.0, 90.0, 60.0, (120, 180))
    first = Sample.polished_ceramic(seed=7).fields(*window)
    again = Sample.polished_ceramic(seed=7).fields(*window)
    other = Sample.polished_ceramic(seed=8).fields(*window)

    for name in ("grain_id", "orientation", "boundary", "pore", "pore_rim", "second_phase", "scratch", "height_um"):
        assert np.array_equal(getattr(first, name), getattr(again, name)), name
    assert not np.array_equal(first.grain_id, other.grain_id)
    assert not np.array_equal(first.pore, other.pore)


def test_truth_is_the_same_every_time():
    first = Sample.polished_ceramic(seed=7).truth(0.0, 0.0, 300.0, 200.0)
    again = Sample.polished_ceramic(seed=7).truth(0.0, 0.0, 300.0, 200.0)

    assert first.porosity == again.porosity
    assert first.grain_count == again.grain_count
    assert np.array_equal(first.pores_um, again.pores_um)
    assert first.scratches == again.scratches


def test_a_repeated_window_comes_from_the_cache_and_cannot_be_written_to(ceramic):
    first = ceramic.fields(0.0, 0.0, 50.0, 50.0, (64, 64))

    assert ceramic.fields(0.0, 0.0, 50.0, 50.0, (64, 64)) is first
    with pytest.raises(ValueError):
        first.pore[0, 0] = 1.0


# --- the same at every scale ----------------------------------------------------


def test_a_point_has_the_same_grain_whichever_window_it_is_seen_through(ceramic):
    # Samples of the small window coincide with samples of the big one, which
    # also crosses several tiles.
    big = ceramic.fields(0.0, 0.0, 600.0, 600.0, (600, 600))
    small = ceramic.fields(100.0, -150.0, 40.0, 40.0, (40, 40))

    rows = np.searchsorted(big.y_um, small.y_um[0] - 1e-6)
    cols = np.searchsorted(big.x_um, small.x_um[0] - 1e-6)
    assert np.allclose(big.x_um[cols : cols + 40], small.x_um)
    same_place = big.grain_id[rows : rows + 40, cols : cols + 40]
    assert np.array_equal(same_place, small.grain_id)
    assert np.array_equal(big.orientation[rows : rows + 40, cols : cols + 40], small.orientation)


def test_every_sample_gets_its_nearest_grain(ceramic):
    """The row-walking shortcut must agree with looking every sample up directly."""
    fields = ceramic.fields(-2990.0, 228.0, 60.0, 60.0, (150, 150))  # over the specimen's edge and a tile border
    on = fields.region == Region.SPECIMEN
    found = ceramic._gather(-3100.0, -2900.0, 150.0, 300.0)
    x, y = np.meshgrid(fields.x_um, fields.y_um)
    distance = np.hypot(x[on][:, None] - found.cells.x[None, :], y[on][:, None] - found.cells.y[None, :])

    assert np.array_equal(fields.grain_id[on], found.cells.id[np.argmin(distance, axis=1)])
    assert np.all(fields.grain_id[~on] == -1)


def test_a_pore_is_the_same_pore_at_low_and_high_magnification(ceramic):
    x, y, diameter = biggest_pore(ceramic)
    coarse = ceramic.fields(x + 3.0, y - 2.0, 707.0, 530.0, (768, 1024))  # a 5x frame, 0.69 um a pixel
    fine = ceramic.fields(x + 3.0, y - 2.0, 70.7, 53.0, (768, 1024))  # a 50x frame

    for fields, tolerance in ((coarse, 0.69), (fine, 0.069)):
        gx, gy = np.meshgrid(fields.x_um, fields.y_um)
        near = np.hypot(gx - x, gy - y) < diameter / 2 + 0.7
        weight = fields.pore * near
        assert abs((weight * gx).sum() / weight.sum() - x) < tolerance
        assert abs((weight * gy).sum() / weight.sum() - y) < tolerance
        pixel_area = (fields.x_um[1] - fields.x_um[0]) * (fields.y_um[1] - fields.y_um[0])
        assert weight.sum() * pixel_area == pytest.approx(np.pi * diameter**2 / 4, rel=0.03)


def test_the_amount_of_pore_and_second_phase_does_not_depend_on_the_sampling(ceramic):
    window = (150.0, 80.0, 140.0, 105.0)
    coarse = ceramic.fields(*window, (150, 200))  # 0.7 um a sample: many pores are smaller than that
    fine = ceramic.fields(*window, (1500, 2000))

    assert coarse.pore.mean() == pytest.approx(fine.pore.mean(), rel=0.03)
    assert coarse.second_phase.mean() == pytest.approx(fine.second_phase.mean(), rel=0.03)
    assert coarse.boundary.mean() == pytest.approx(fine.boundary.mean(), rel=0.05)


def test_a_scratch_runs_where_truth_says_at_low_and_high_magnification(ceramic):
    scratch = ceramic.scratches[0]
    along = np.array([scratch.x1_um - scratch.x0_um, scratch.y1_um - scratch.y0_um])
    length = np.hypot(*along)
    mid_x, mid_y = (scratch.x0_um + scratch.x1_um) / 2, (scratch.y0_um + scratch.y1_um) / 2

    for width, shape in ((353.0, (384, 512)), (35.0, (384, 512))):
        fields = ceramic.fields(mid_x, mid_y, width, width * 0.75, shape)
        pitch = width / shape[1]
        gx, gy = np.meshgrid(fields.x_um, fields.y_um)
        off_line = np.abs((gx - scratch.x0_um) * along[1] - (gy - scratch.y0_um) * along[0]) / length
        # Other scratches may cross the frame; this one must be there and be thin.
        assert fields.scratch[(off_line < pitch / 2) & (fields.pore == 0)].min() > 0
        ours = (fields.scratch > 0) & (off_line < 20 * scratch.width_um)
        assert off_line[ours].max() <= scratch.width_um / 2 + pitch


# --- truth agrees with what a camera would be shown --------------------------------


def test_porosity_and_second_phase_in_truth_match_the_masks(ceramic):
    window = (-420.0, 310.0, 80.0, 60.0)
    truth = ceramic.truth(*window)
    fields = ceramic.fields(*window, (1200, 1600))  # 0.05 um a sample

    assert truth.specimen_fraction == 1.0
    assert 0.005 < truth.porosity < 0.08
    assert fields.pore_mask.mean() == pytest.approx(truth.porosity, rel=0.01)
    assert fields.second_phase_mask.mean() == pytest.approx(truth.second_phase_fraction, rel=0.02)


def test_pores_and_pockets_never_overlap(ceramic):
    fields = ceramic.fields(-420.0, 310.0, 80.0, 60.0, (1200, 1600))

    assert not np.any(fields.pore_mask & fields.second_phase_mask)
    assert fields.pore.max() <= 1.0 and fields.second_phase.max() <= 1.0


def test_each_listed_pore_is_a_pore_of_that_size(ceramic):
    window = (-420.0, 310.0, 80.0, 60.0)
    pores = ceramic.truth(*window).pores_um
    fields = ceramic.fields(*window, (1200, 1600))
    gx, gy = np.meshgrid(fields.x_um, fields.y_um)

    assert len(pores) > 10
    for x, y, diameter in pores[np.argsort(pores[:, 2])[-5:]]:
        inside = np.hypot(gx - x, gy - y) < diameter / 2 - 0.1
        outside_ring = np.abs(np.hypot(gx - x, gy - y) - (diameter / 2 + 0.15)) < 0.03
        assert fields.pore_mask[inside].all()
        assert not fields.pore_mask[outside_ring].any()


def test_grain_areas_and_intercepts_in_truth_match_the_grain_map(ceramic):
    window = (500.0, 500.0, 150.0, 150.0)
    truth = ceramic.truth(*window)
    fields = ceramic.fields(*window, (600, 600))  # 0.25 um a sample
    labels = fields.grain_id

    # Grains wholly inside the window: their pixel count is their area.
    touching = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    ids, counts = np.unique(labels, return_counts=True)
    whole = ~np.isin(ids, touching)
    cells = ceramic._gather(400.0, 600.0, 400.0, 600.0).cells
    listed = dict(zip(cells.id.tolist(), cells.area.tolist()))
    measured = counts[whole] * 0.25**2
    expected = np.array([listed[i] for i in ids[whole].tolist()])
    assert whole.sum() > 200
    assert np.allclose(measured, expected, rtol=0.03, atol=0.5)

    # Mean lineal intercept, the way a person measures it: crossings per length of test line.
    crossings = np.count_nonzero(labels[:, 1:] != labels[:, :-1]) + np.count_nonzero(labels[1:] != labels[:-1])
    test_line = 2 * 600 * 150.0
    assert test_line / crossings == pytest.approx(truth.mean_intercept_um, rel=0.04)
    assert truth.grain_count == pytest.approx(150.0 * 150.0 / np.mean(list(listed.values())), rel=0.1)


def test_truth_counts_only_the_part_of_the_window_on_the_specimen(ceramic):
    inside = ceramic.truth(2900.0, 0.0, 200.0, 200.0)  # covers x from 2800 to 3000
    straddling = ceramic.truth(3000.0, 0.0, 400.0, 200.0)  # the same specimen, plus resin
    off = ceramic.truth(4000.0, 0.0, 200.0, 200.0)

    assert straddling.specimen_fraction == pytest.approx(0.5)
    assert straddling.porosity == pytest.approx(inside.porosity)
    assert straddling.grain_count == inside.grain_count
    assert off.specimen_fraction == 0.0 and off.grain_count == 0 and len(off.pores_um) == 0


def test_scratches_in_truth_are_cut_to_the_window(ceramic):
    whole = ceramic.scratches[0]
    mid_x, mid_y = (whole.x0_um + whole.x1_um) / 2, (whole.y0_um + whole.y1_um) / 2
    scratches = ceramic.truth(mid_x, mid_y, 100.0, 100.0).scratches

    assert len(ceramic.scratches) == ceramic.scratch_count
    assert whole.id in [s.id for s in scratches]
    for s in scratches:
        assert all(abs(v - mid_x) <= 50.0 + 1e-6 for v in (s.x0_um, s.x1_um))
        assert all(abs(v - mid_y) <= 50.0 + 1e-6 for v in (s.y0_um, s.y1_um))
        assert 0.3 <= s.width_um <= 0.9 and 0.05 <= s.depth_um <= 0.3
    cut = next(s for s in scratches if s.id == whole.id)
    assert np.hypot(cut.x1_um - cut.x0_um, cut.y1_um - cut.y0_um) < np.hypot(whole.x1_um - whole.x0_um, whole.y1_um - whole.y0_um)


# --- the specimen is what was asked for ----------------------------------------------


def test_porosity_and_grain_size_come_out_as_requested(ceramic):
    truth = ceramic.truth(-800.0, 600.0, 1000.0, 1000.0)

    assert truth.grain_count > 15000
    assert truth.mean_grain_diameter_um == pytest.approx(8.0, rel=0.02)
    assert truth.porosity == pytest.approx(0.03, rel=0.06)
    assert truth.second_phase_fraction == pytest.approx(0.02, rel=0.06)
    assert 0.6 < truth.mean_intercept_um / truth.mean_grain_diameter_um < 0.8


def test_other_requests_are_honoured_too():
    fine = Sample.polished_ceramic(seed=3, grain_size_um=3.0, porosity=0.01, second_phase=0.0)
    truth = fine.truth(0.0, 0.0, 400.0, 400.0)

    assert truth.mean_grain_diameter_um == pytest.approx(3.0, rel=0.02)
    assert truth.porosity == pytest.approx(0.01, rel=0.08)
    assert truth.second_phase_fraction == 0.0
    assert truth.pores_um[:, 2].max() < 3.0  # pores scale with the grains


def test_asking_for_more_than_the_junctions_can_hold_is_refused():
    with pytest.raises(ValueError, match="cannot fit"):
        Sample.polished_ceramic(porosity=0.25)
    with pytest.raises(ValueError, match="does not fit"):
        Sample.polished_ceramic(specimen_size_um=(25000.0, 25000.0))


def test_an_unetched_specimen_has_the_same_grains_and_no_grooves():
    etched = Sample.polished_ceramic(seed=2).fields(0.0, 0.0, 60.0, 60.0, (200, 200))
    unetched = Sample.polished_ceramic(seed=2, etched=False).fields(0.0, 0.0, 60.0, 60.0, (200, 200))

    assert 0.8 < etched.boundary.max() <= 1.0
    assert not unetched.boundary.any()
    assert np.array_equal(etched.grain_id, unetched.grain_id)
    assert np.array_equal(etched.pore, unetched.pore)


def test_grooves_lie_on_the_lines_between_grains(ceramic):
    fields = ceramic.fields(0.0, 0.0, 30.0, 30.0, (600, 600))  # 0.05 um a sample
    labels = fields.grain_id
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]

    clear = (fields.pore == 0) & (fields.second_phase == 0)
    on_lines = fields.boundary[edge & clear]
    assert on_lines.min() > 0.0
    assert not fields.boundary[fields.pore == 1].any()
    assert 0.03 < fields.boundary.mean() < 0.15  # 0.4 um grooves around 8 um grains, not all at full strength


def test_the_etch_reveals_some_boundaries_better_than_others(ceramic):
    """A groove's value is its boundary's strength: a tenth are faint, the rest anywhere from 0.25 to 1."""
    fields = ceramic.fields(0.0, 0.0, 60.0, 60.0, (1200, 1200))  # 0.05 um a sample
    labels = fields.grain_id
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]
    on_lines = fields.boundary[edge & (fields.pore == 0) & (fields.second_phase == 0)]

    assert 0.03 < np.mean(on_lines < 0.15) < 0.25
    # Nothing in between, but for the odd sample where a strong groove half covers a faint one.
    assert np.mean((on_lines > 0.151) & (on_lines < 0.249)) < 0.005
    assert on_lines.max() > 0.9
    assert 0.5 < on_lines[on_lines > 0.2].mean() < 0.75
    # The same boundary has the same strength however it is sampled.
    coarse = ceramic.fields(0.0, 0.0, 60.0, 60.0, (600, 600))
    assert np.array_equal(coarse.boundary, ceramic.fields(0.0, 0.0, 60.0, 60.0, (600, 600)).boundary)
    assert coarse.boundary.mean() == pytest.approx(fields.boundary.mean(), rel=0.05)


# --- no trace of how it was made ----------------------------------------------------


def test_the_grid_the_grains_are_drawn_on_does_not_show(ceramic):
    """Each grain's point comes from one cell of a square grid. With the points
    held near their cells the grid's period stood 200 to 500 times above the
    rest of a frame's Fourier transform and gave the grain size away."""
    step = 0.69  # a 5x frame
    fields = ceramic.fields(1000.0, 1000.0, 768 * step, 768 * step, (768, 768))
    picture = fields.boundary + 3.0 * fields.pore
    power = np.abs(np.fft.fftshift(np.fft.fft2(picture - picture.mean()))) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(768, step))
    fx, fy = np.meshgrid(frequency, frequency)
    grid = 1.0 / ceramic._pitch
    ring = np.abs(np.hypot(fx, fy) - grid) < 1.5 * (frequency[1] - frequency[0])
    centre, at = 384, int(np.argmin(np.abs(frequency - grid)))
    along_x = power[centre - 1 : centre + 2, at - 1 : at + 2].max()
    along_y = power[at - 1 : at + 2, centre - 1 : centre + 2].max()

    assert max(along_x, along_y) < 20 * np.median(power[ring])


def test_grains_come_in_a_range_of_sizes(ceramic):
    cells = ceramic._gather(-400.0, 400.0, -400.0, 400.0).cells
    diameter = np.sqrt(4 * cells.area / np.pi)

    assert len(diameter) > 9000
    assert 0.17 < diameter.std() / diameter.mean() < 0.3
    assert diameter.max() > 1.5 * diameter.mean()


def test_a_block_is_what_a_far_wider_triangulation_would_give(ceramic, monkeypatch):
    """The padding round a block is argued to be enough in sample.py. This is
    a spot check of one block, not a proof: the argument is about the worst
    case, which random points rarely come near."""

    def block():
        cells, edges, junctions = ceramic._block(-32, 0, 64, 96)
        own = junctions.inside
        order = np.lexsort(((edges.ay + edges.by), (edges.ax + edges.bx)))
        return (
            (cells.id, cells.x, cells.y, cells.area, cells.perimeter),
            tuple(v[order] for v in (edges.ax + edges.bx, edges.ay + edges.by, edges.left, edges.right)),
            np.sort(junctions.x[own] + 1e4 * junctions.y[own]),
        )

    usual = block()
    monkeypatch.setattr(sample_module, "_PAD", 16)
    generous = block()

    for ours, theirs in zip(usual, generous):
        if isinstance(ours, tuple):
            assert all(np.allclose(a, b, rtol=1e-9, atol=1e-6) for a, b in zip(ours, theirs, strict=True))
        else:
            assert np.allclose(ours, theirs, rtol=1e-9, atol=1e-6)


def test_most_pores_are_small_and_a_few_are_large(ceramic):
    diameter = ceramic.truth(0.0, 0.0, 600.0, 600.0).pores_um[:, 2]
    middle = (diameter.min() + diameter.max()) / 2

    assert len(diameter) > 2000
    assert np.median(diameter) < 0.75 * middle
    assert np.sum(diameter < middle) > 1.5 * np.sum(diameter >= middle)
    assert diameter.max() > 3.0 and diameter.min() < 0.6


# --- the sampling grid --------------------------------------------------------------


def test_long_thin_samples_are_refused_rather_than_losing_pores(ceramic):
    """With samples 140 times longer than wide, pores fell between sample
    centres and three quarters of the porosity went missing without a word."""
    truth = ceramic.truth(0.0, 0.0, 400.0, 50.0)
    square = ceramic.fields(0.0, 0.0, 400.0, 50.0, (100, 800))
    four_to_one = ceramic.fields(0.0, 0.0, 400.0, 50.0, (100, 200))
    one_to_four = ceramic.fields(0.0, 0.0, 400.0, 50.0, (25, 800))

    for fields in (four_to_one, one_to_four):
        assert fields.pore.mean() == pytest.approx(truth.porosity, rel=0.03)
        assert fields.second_phase.mean() == pytest.approx(truth.second_phase_fraction, rel=0.03)
        assert fields.boundary.mean() == pytest.approx(square.boundary.mean(), rel=0.08)
        assert fields.scratch.mean() == pytest.approx(square.scratch.mean(), rel=0.05)
    with pytest.raises(ValueError, match="too far from square"):
        ceramic.fields(0.0, 0.0, 4000.0, 50.0, (400, 400))
    with pytest.raises(ValueError, match="too far from square"):
        ceramic.fields(0.0, 0.0, 400.0, 50.0, (800, 100))


def test_the_tile_cache_holds_two_low_magnification_fields_of_fine_grains():
    """With a fixed 128 tiles, a 5x field of 2 um grains was rebuilt on every frame."""
    for grain_size in (8.0, 2.0):
        fine = Sample.polished_ceramic(seed=4, grain_size_um=grain_size)
        span = sample_module._TILE * fine._pitch
        tiles_in_a_5x_field = (np.ceil(707.0 / span) + 1) * (np.ceil(530.0 / span) + 1)

        assert fine._tile.cache_info().maxsize >= 2 * tiles_in_a_5x_field


def test_the_gap_round_the_specimen_follows_its_outline(ceramic):
    corner = ceramic.fields(3000.0, 3000.0, 20.0, 20.0, (200, 200))  # 0.1 um a sample
    inside = ceramic.fields(0.0, 0.0, 20.0, 20.0, (200, 200))
    gx, gy = np.meshgrid(corner.x_um, corner.y_um)
    beyond = np.maximum(gx - 3000.0, gy - 3000.0)

    assert not inside.edge_gap.any()
    assert np.all(corner.edge_gap[(beyond > 0.1) & (beyond < 0.9)] == 1.0)
    assert not corner.edge_gap[(beyond < -0.1) | (beyond > 1.1)].any()
    assert not corner.edge_gap[corner.region == Region.SPECIMEN].any()


# --- heights ------------------------------------------------------------------------


def test_tilt_is_the_slope_that_was_asked_for():
    flat = Sample.polished_ceramic(seed=5, tilt_um_per_mm=0.0)
    tilted = Sample.polished_ceramic(seed=5, tilt_um_per_mm=4.0)

    def rise(dx, dy):
        return (tilted.height_um(dx, dy) - tilted.height_um(0, 0)) - (flat.height_um(dx, dy) - flat.height_um(0, 0))

    assert np.hypot(rise(1000.0, 0.0), rise(0.0, 1000.0)) == pytest.approx(4.0)
    # Without tilt the face is still not perfectly flat, but nearly.
    heights = flat.height_um(np.linspace(-2500, 2500, 50)[:, None], np.linspace(-2500, 2500, 50)[None, :])
    assert 0.05 < np.ptp(heights) < 3.0


def test_focus_drifts_across_the_specimen(ceramic):
    assert abs(ceramic.height_um(2500.0, 2500.0) - ceramic.height_um(-2500.0, -2500.0)) > 1.0
    assert ceramic.height_um(0.0, 0.0) == pytest.approx(ceramic.mount_height_um, abs=1.0)


def test_resin_sits_below_the_specimen_and_the_stage_plate_is_at_zero(ceramic):
    x, y = np.array([2900.0, 3100.0]), np.array([10.0, 10.0])
    on_specimen, on_resin = ceramic.height_um(x, y)

    assert 0.5 * RESIN_RECESS_UM < on_specimen - on_resin < 1.5 * RESIN_RECESS_UM
    assert ceramic.height_um(14000.0, 0.0) > 14900.0
    assert ceramic.height_um(15001.0, 0.0) == 0.0
    assert isinstance(ceramic.height_um(0.0, 0.0), float)


def test_fields_report_what_is_under_each_sample(ceramic):
    edge = ceramic.fields(3000.0, 0.0, 100.0, 50.0, (50, 100))
    rim = ceramic.fields(15000.0, 0.0, 100.0, 50.0, (50, 100))

    assert np.all(edge.region[:, :50] == Region.SPECIMEN) and np.all(edge.region[:, 50:] == Region.RESIN)
    assert np.all(edge.grain_id[:, 50:] == -1) and np.all(edge.grain_id[:, :50] >= 0)
    assert not edge.pore[:, 50:].any() and not edge.boundary[:, 50:].any()
    assert np.all(rim.region[:, :49] == Region.RESIN) and np.all(rim.region[:, 51:] == Region.STAGE)
    assert np.all(rim.height_um[:, 51:] == 0.0)
    gx, gy = np.meshgrid(edge.x_um, edge.y_um)
    assert np.allclose(edge.surface_um, ceramic.height_um(gx, gy), atol=1e-9)


def test_pores_and_grooves_lie_below_the_polished_surface(ceramic):
    x, y, diameter = biggest_pore(ceramic)
    fields = ceramic.fields(x, y, 20.0, 20.0, (400, 400))
    flat = (fields.pore == 0) & (fields.boundary == 0) & (fields.second_phase == 0) & (fields.scratch == 0)
    centre = fields.height_um[200, 200] - fields.surface_um[200, 200]

    assert np.array_equal(fields.height_um[flat], fields.surface_um[flat])
    assert np.all(fields.height_um <= fields.surface_um)
    assert centre == pytest.approx(-sample_module.PORE_DEPTH_PER_RADIUS * diameter / 2, rel=0.05)
    # A groove is as deep as its boundary was strongly etched.
    groove = (fields.boundary > 0) & (fields.pore == 0) & (fields.second_phase == 0) & (fields.scratch == 0)
    assert groove.any()
    depth = (fields.surface_um - fields.height_um)[groove]
    assert np.allclose(depth, sample_module.GROOVE_DEPTH_UM * fields.boundary[groove], atol=1e-3)


def test_the_highest_point_under_an_objective(ceramic):
    centre = ceramic.max_height_um(0.0, 0.0, 3500.0)
    ring = [ceramic.height_um(3500.0 * np.cos(a), 3500.0 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60)]

    assert centre >= ceramic.height_um(0.0, 0.0)
    assert centre >= max(ring) - 0.1
    assert centre < ceramic.mount_height_um + 30.0
    # Beside the mount: nothing, until the nose reaches over its rim.
    assert ceramic.max_height_um(18000.0, 0.0, 2000.0) == 0.0
    assert ceramic.max_height_um(18000.0, 0.0, 3500.0) > 14900.0
    assert ceramic.max_height_um(15000.5, 0.0, 1.0) > 14900.0
