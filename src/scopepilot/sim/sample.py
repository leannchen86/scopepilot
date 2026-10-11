"""The pretend specimen: a polished ceramic in a resin mount, and the truth about it.

Everything is procedural and nothing is calibrated against a real specimen or
instrument. The numbers below are plausible choices for a sintered ceramic of
the aluminium nitride kind, picked so the pictures look like what a materials
lab sees; none of them is a measurement.

Units and frames
    Lengths are micrometres. x and y lie on the specimen surface with the origin
    at the centre of the specimen. Heights are above the stage plate, which is
    at 0. A window is given by its centre and its size, and its samples sit at
    pixel centres: columns run with +x and rows with +y.

How the specimen is made
    Grains are the Voronoi cells of one point per cell of a square grid, each
    point free to wander over its own cell and half of each neighbour. That is
    enough to leave no trace of the grid in a picture or its Fourier transform,
    and it keeps the bound (one point per cell) that lets a piece of the
    specimen be built without the rest. Every random choice is an integer hash
    of the seed and of the thing being chosen (a grid cell, a grain triple, a
    scratch number), so any part of the specimen can be built on its own and
    always comes out the same: a pore seen at 5x is the same pore at 50x. Pores
    and second-phase pockets sit on grain junctions and no two of them overlap,
    which is what lets `truth` add up areas exactly instead of estimating them
    from pixels.

How it falls short of a real ceramic
    Grain sizes are more alike than real ones: the equivalent diameters have a
    coefficient of variation of 0.20 and the largest grain is about 1.8 times
    the mean, where a sintered ceramic is usually nearer 0.3 to 0.45 and 2 to
    2.5 (figures from general knowledge, not from a source checked for this
    code). Boundaries are straight. Every pore is a disc on a grain junction
    and every pocket a triangle; real pores are irregular and some lie inside
    grains. The specimen's outline is a ruler-straight rectangle with no
    chipping, and the resin is featureless.

    The specimen is built in tiles of 32 x 32 grid cells, cached, so the several
    hundred thousand grains of a 6 mm specimen never exist at once.

Determinism
    The hashes are exact integer arithmetic and identical on every machine. The
    geometry after them is ordinary floating point; two machines can differ in
    the last bit of a height, which does not move a grain or a pore.

Not modelled: pull-outs, cracks, comet tails, grain-size gradients, texture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import IntEnum
from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay, cKDTree

# --- illustrative constants -------------------------------------------------
# None of these is measured. Each is a plausible value for a polished, etched,
# liquid-phase-sintered ceramic in an epoxy mount.

GROOVE_WIDTH_UM = 0.4
"""Width of an etched grain-boundary groove."""
GROOVE_DEPTH_UM = 0.08
PORE_RIM_UM = 0.25
"""Width of the steep wall just inside a pore's edge, which is what lights up in darkfield."""
PORE_DEPTH_PER_RADIUS = 0.8
"""A pore is a bowl about this deep at its centre, in units of its radius."""
SCRATCH_MAX_DEPTH_UM = 0.3
SECOND_PHASE_RELIEF_UM = 0.05
"""The softer second phase polishes this much lower than the grains."""
RESIN_RECESS_UM = 2.0
"""Resin wears faster than ceramic, so it ends up this far below the specimen."""
EDGE_GAP_UM = 1.0
"""Width of the gap where the resin has shrunk away from the specimen's cut edge."""
EDGE_ROUNDING_UM = 1.0
EDGE_ROUNDING_LENGTH_UM = 15.0
"""The specimen surface drops by EDGE_ROUNDING_UM at its edge, recovering over this length."""
DOME_SAG_UM = 4.0
"""Polishing leaves the mount slightly domed: its rim is this much lower than its centre."""
ROLL_OFF_UM = 5.0
ROLL_OFF_WIDTH_UM = 1000.0
"""The outer millimetre of the mount rolls off by a further ROLL_OFF_UM."""

_PORE_RADIUS_PER_PITCH = (0.03, 0.25)
_POCKET_RADIUS_PER_PITCH = (0.07, 0.30)
"""Radius range of a pore, and of the circle a pocket's corners lie on, in grid cells.

With 8 um grains that is pores of about 0.4 to 3.6 um across. Pore radii are
drawn towards the small end (the square of a uniform draw), so most pores are
small and a few are large; before that, sizes were spread evenly, which no
sintered ceramic shows.
"""
_FAINT_BOUNDARIES = 0.1
"""Share of grain boundaries the etch barely reveals (strength under 0.15). The
rest are spread evenly from 0.25 to 1."""
_CLEARANCE = 1.1
"""Two of them clash when their centres are closer than this times the sum of
their radii, so survivors never touch."""

_JITTER = 2.0
"""Side of the square a grain's generating point may wander over, in grid cells,
centred on its own cell. At 2 each point ranges over its cell and half of every
neighbour, which is what hides the grid: below about 1.5 the grid's period shows
as a sharp peak in a frame's Fourier transform and gives the grain size away."""
_OVERHANG = (_JITTER - 1.0) / 2.0
"""How far outside its own cell a generating point can lie, in grid cells."""
_ECD_PER_PITCH = 1.106
"""Mean equivalent circle diameter of a grain, in grid cells. Measured from this
generator (three seeds of about 120,000 grains each, which agreed to 0.0002)
and checked by a test; it depends on _JITTER."""
_TILE = 32
"""Grid cells along one side of a cached tile."""
_PAD = 6
_SAFE = 3
"""Cells triangulated around a block, and how many of them can be trusted.

Any disc of radius 2.13 cells contains an axis-aligned square of side 3, which
contains the whole 2 x 2 square some grain's point wanders over, and so
contains that point. No empty circle among the points is larger than that,
then, and no grain reaches further than 2.13 cells from its point. Where every
point that could lie within 2.13 cells is among those triangulated, which is
from 2.63 cells inside the padding's edge (2.13 plus the half cell a point can
overhang), the triangulation is exactly what an infinite one would give. That
leaves 3.37 trustworthy cells around a block; 3 are used. They hold every
corner of the block's grains (a grain's point is at most half a cell outside
the block and its corners 2.13 beyond that), both ends of its boundaries, and
every neighbour close enough to clash with one of its pores. A test compares a
block with the same block triangulated far more generously."""
_CACHED_AREA_UM2 = 1.5e6
_MOST_TILES = 512
"""Tiles are kept for this much specimen, about three 5x fields of the default
camera, so that going back and forth between two fields builds nothing twice;
but never fewer than 128 tiles nor more than 512 (about 150 MB)."""
_ID_OFFSET = 64
"""Grid cells outside the specimen that still get a valid grain number."""
_CALIBRATION_CELLS = 96

_CH_SEED_X, _CH_SEED_Y, _CH_KIND, _CH_SIZE, _CH_ANGLE, _CH_TILT, _CH_WAVE, _CH_SCRATCH, _CH_GROOVE = range(9)


class Region(IntEnum):
    """What is under a point."""

    STAGE = 0
    RESIN = 1
    SPECIMEN = 2


@dataclass(frozen=True)
class Material:
    """How the specimen answers light. Fractions of the incident light, at normal incidence."""

    name: str
    reflectance: float
    """Of the dense, polished matrix."""
    second_phase_reflectance: float
    resin_reflectance: float
    anisotropy: float
    """Between crossed polarisers, the brightest grain returns this fraction of
    what the matrix returns in brightfield, by reflection at its surface. Tiny
    for a ceramic, a few percent for a hexagonal metal."""
    subsurface_glow: float
    """Between crossed polarisers, what a translucent matrix returns from below
    its surface, where scattering has scrambled the polarisation; the same
    fraction. Zero for a metal."""


CERAMIC = Material(
    name="aluminium-nitride-like ceramic",
    # (n-1)^2/(n+1)^2 for n = 2.05 is 0.12. The yttrium aluminate second phase
    # has a lower index, so it should be a little darker; that is an inference,
    # not something seen in a micrograph. Epoxy at n = 1.55 gives 0.05.
    reflectance=0.12,
    second_phase_reflectance=0.105,
    resin_reflectance=0.05,
    # Aluminium nitride is only weakly birefringent, so its grains barely
    # differ between crossed polarisers; a rough Fresnel estimate from
    # remembered indices puts the true figure nearer 0.0002. What shows instead
    # is the milky glow of a translucent ceramic. Both numbers are guesses.
    anisotropy=0.001,
    subsurface_glow=0.015,
)


@dataclass(frozen=True)
class Scratch:
    """One polishing scratch: a straight groove from (x0, y0) to (x1, y1)."""

    id: int
    x0_um: float
    y0_um: float
    x1_um: float
    y1_um: float
    width_um: float
    depth_um: float


@dataclass(frozen=True, eq=False)
class Fields:
    """The specimen sampled on a grid of rows x cols points.

    Each sample stands for the small rectangle around it (one pixel pitch on a
    side), so thin or small things are reported as the fraction of that
    rectangle they cover. That keeps a pore's total darkness the same whether it
    spans forty samples or a third of one.

    The arrays are shared with a cache and are read-only.
    """

    x_um: np.ndarray
    """Sample centres along x, one per column."""
    y_um: np.ndarray
    """Sample centres along y, one per row."""
    region: np.ndarray
    """A `Region` code per sample (uint8)."""
    grain_id: np.ndarray
    """The grain under each sample (int64), -1 off the specimen. The same grain
    has the same number in every window."""
    orientation: np.ndarray
    """In-plane crystal angle of that grain, radians in [0, pi). 0 off the specimen."""
    boundary: np.ndarray
    """Fraction covered by an etched grain-boundary groove, times how strongly
    the etch revealed that boundary, 0 to 1. Strength differs from boundary to
    boundary and one in ten is nearly invisible. All zero on an unetched
    specimen, where the grains are still there but nothing marks them."""
    pore: np.ndarray
    """Fraction covered by pore, 0 to 1."""
    pore_rim: np.ndarray
    """Fraction covered by the steep wall just inside a pore's edge, 0 to 1."""
    pore_depth_um: np.ndarray
    """How far the pores put the surface below the polished plane, averaged over the sample."""
    second_phase: np.ndarray
    """Fraction covered by second phase, 0 to 1."""
    scratch: np.ndarray
    """Scratch strength, 0 to 1: the fraction covered times the scratch's depth
    relative to SCRATCH_MAX_DEPTH_UM."""
    edge_gap: np.ndarray
    """Fraction covered by the gap between specimen and resin, 0 to 1: a line
    EDGE_GAP_UM wide in the resin, all round the specimen's outline. It marks
    the outline for the optics and has no depth of its own in `height_um`."""
    surface_um: np.ndarray
    """Height of the polished plane with pores, grooves and scratches filled in:
    what a person focuses on. The same as `Sample.height_um`."""
    height_um: np.ndarray
    """Height of the actual top surface, relief included."""

    @property
    def pore_mask(self) -> np.ndarray:
        """True where a sample is more pore than not."""
        return self.pore >= 0.5

    @property
    def second_phase_mask(self) -> np.ndarray:
        return self.second_phase >= 0.5


@dataclass(frozen=True, eq=False)
class Truth:
    """The true answers for one window. Fractions are of the specimen inside the window."""

    x_um: float
    y_um: float
    width_um: float
    height_um: float
    specimen_fraction: float
    """How much of the window lies on the specimen."""
    porosity: float
    """Pore area over specimen area."""
    pores_um: np.ndarray
    """One row per pore whose centre is in the window: centre x, centre y,
    equivalent circle diameter. A pore cut by the window edge is listed whole."""
    grain_count: int
    """Grains whose generating point is in the window. On average this is the
    window area over the mean grain area; it is not the number of grains the
    window touches."""
    mean_grain_diameter_um: float
    """Mean equivalent circle diameter, sqrt(4 * area / pi), of those grains,
    from each grain's whole area. Pores and pockets count as part of the grains
    they sit between."""
    mean_intercept_um: float
    """Mean lineal intercept of those grains, pi * area / perimeter, which is
    what counting boundary crossings along random test lines converges to."""
    second_phase_fraction: float
    scratches: tuple[Scratch, ...]
    """Scratches that cross the window, each cut down to the part inside it."""


# --- hashing ------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)


def _hash(*keys) -> np.ndarray:
    """Mix integer keys (splitmix64 finaliser) into uint64. Exact on every machine."""
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(k)).astype(np.uint64) for k in keys])
    h = np.zeros(arrays[0].shape, np.uint64)
    for key in arrays:
        h = h + key + _GOLDEN
        h = (h ^ (h >> np.uint64(30))) * _MIX_1
        h = (h ^ (h >> np.uint64(27))) * _MIX_2
        h = h ^ (h >> np.uint64(31))
    return h


def _unit(*keys) -> np.ndarray:
    """A float in [0, 1) for each combination of keys."""
    return (_hash(*keys) >> np.uint64(11)).astype(np.float64) * 2.0**-53


# --- pieces of geometry -------------------------------------------------------


@dataclass(frozen=True)
class _Cells:
    """Grains: number, generating point, area and perimeter of the Voronoi cell."""

    id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    perimeter: np.ndarray


@dataclass(frozen=True)
class _Edges:
    """Grain boundaries: a segment a-b with the grain on its -x side and on its +x side."""

    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class _Junctions:
    """Points where three grains meet, before anything is placed on them."""

    x: np.ndarray
    y: np.ndarray
    kind: np.ndarray
    """Uniform draw deciding what, if anything, sits here."""
    size: np.ndarray
    """Uniform draw deciding how big it would like to be."""
    inside: np.ndarray
    """True for junctions that belong to this block rather than its padding."""
    normal_x: np.ndarray
    normal_y: np.ndarray
    offset: np.ndarray
    """Edge normals and offsets, shape (n, 3), of the triangle whose corners lie
    one unit along each of the three boundaries leaving the junction."""
    area: np.ndarray
    """Area of that unit triangle."""


@dataclass(frozen=True)
class _Features:
    """Pores (discs) and second-phase pockets (triangles) on junctions."""

    is_pore: np.ndarray
    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    """Of the pore, or of the circle the pocket's corners lie on."""
    area: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    offset: np.ndarray
    """Pocket edges, shape (n, 3): a point q from the centre is inside when
    normal . q < offset for all three. Unused for pores."""


@dataclass(frozen=True)
class _Tile:
    cells: _Cells
    edges: _Edges
    features: _Features


def _take(group, keep):
    return type(group)(*[getattr(group, f.name)[keep] for f in dataclass_fields(group)])


def _join(groups):
    kind = type(groups[0])
    names = [f.name for f in dataclass_fields(kind)]
    return kind(*[np.concatenate([getattr(group, name) for group in groups]) for name in names])


@dataclass(frozen=True)
class _Grid:
    """A regular grid of sample centres."""

    x0: float
    y0: float
    dx: float
    dy: float
    rows: int
    cols: int

    @property
    def footprint(self) -> float:
        """Side of the square with the same area as one sample."""
        return math.sqrt(self.dx * self.dy)


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For counts [2, 0, 3] return owners [0, 0, 2, 2, 2] and offsets [0, 1, 0, 1, 2]."""
    owners = np.repeat(np.arange(len(counts)), counts)
    starts = np.cumsum(counts) - counts
    return owners, np.arange(len(owners)) - starts[owners]


def _spans(lo, hi, origin, step, count):
    """First index and number of grid lines whose centre lies in [lo, hi]."""
    first = np.clip(np.ceil((lo - origin) / step), 0, count).astype(np.int64)
    last = np.clip(np.floor((hi - origin) / step), -1, count - 1).astype(np.int64)
    return first, np.maximum(last - first + 1, 0)


def _overlap(lo, hi, a, b):
    """Length of the overlap of [lo, hi] with [a, b]."""
    return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)


def _label_image(cells: _Cells, edges: _Edges, grid: _Grid) -> np.ndarray:
    """The grain under every sample, exactly, in time proportional to the samples.

    Walking along a row, the grain changes only where the row crosses a
    boundary, and there it changes from the grain on the boundary's -x side to
    the one on its +x side. So: look up the grain at the start of each row, drop
    (right - left) at every crossing, and sum along the row.
    """
    rows, cols = grid.rows, grid.cols
    ys = grid.y0 + grid.dy * np.arange(rows)
    tree = cKDTree(np.column_stack([cells.x, cells.y]))

    def nearest(x, y):
        return cells.id[tree.query(np.column_stack([x, y]))[1]]

    steps = np.zeros(rows * cols)
    upward = edges.by >= edges.ay
    lo_x, lo_y = np.where(upward, edges.ax, edges.bx), np.where(upward, edges.ay, edges.by)
    hi_x, hi_y = np.where(upward, edges.bx, edges.ax), np.where(upward, edges.by, edges.ay)
    # A row belongs to an edge when lo_y <= y < hi_y, so a row through a junction
    # crosses exactly the edges a row just above it would.
    first = np.clip(np.ceil((lo_y - grid.y0) / grid.dy), 0, rows).astype(np.int64)
    stop = np.clip(np.ceil((hi_y - grid.y0) / grid.dy), 0, rows).astype(np.int64)
    edge, offset = _expand(np.maximum(stop - first, 0))
    row = first[edge] + offset
    along = (ys[row] - lo_y[edge]) / (hi_y[edge] - lo_y[edge])
    cross_x = lo_x[edge] + along * (hi_x[edge] - lo_x[edge])
    col = np.floor((cross_x - grid.x0) / grid.dx).astype(np.int64) + 1
    seen = (col >= 1) & (col < cols)
    np.add.at(
        steps,
        row[seen] * cols + col[seen],
        (edges.right[edge[seen]] - edges.left[edge[seen]]).astype(np.float64),
    )
    steps = steps.reshape(rows, cols)
    steps[:, 0] = nearest(np.full(rows, grid.x0), ys)
    labels = np.rint(np.cumsum(steps, axis=1)).astype(np.int64)

    # A crossing that lands within rounding error of a sample centre could be
    # counted on the wrong side of the row's first sample. The far end of the row
    # shows it, and such a row is looked up sample by sample instead.
    x_last = grid.x0 + grid.dx * (cols - 1)
    for bad in np.flatnonzero(labels[:, -1] != nearest(np.full(rows, x_last), ys)):
        labels[bad] = nearest(grid.x0 + grid.dx * np.arange(cols), np.full(cols, ys[bad]))
    return labels


def _splat_segments(ax, ay, bx, by, half_width, value, grid: _Grid) -> np.ndarray:
    """Coverage of each sample by lines of the given half-width, times `value`.

    Only samples near a line are visited: for each line, the rows it spans, and
    in each row the columns within reach of it.
    """
    out = np.zeros(grid.rows * grid.cols, np.float32)
    foot = grid.footprint
    reach = half_width + foot / 2
    first_row, row_count = _spans(
        np.minimum(ay, by) - reach, np.maximum(ay, by) + reach, grid.y0, grid.dy, grid.rows
    )
    seg, offset = _expand(row_count)
    if len(seg) == 0:
        return out.reshape(grid.rows, grid.cols)
    row = first_row[seg] + offset
    y = grid.y0 + grid.dy * row
    dx, dy = (bx - ax)[seg], (by - ay)[seg]
    length = np.maximum(np.hypot(dx, dy), 1e-12)
    x_lo = np.minimum(ax, bx)[seg] - reach[seg]
    x_hi = np.maximum(ax, bx)[seg] + reach[seg]
    # Within a row, the samples near the infinite line are a run centred on
    # where the line crosses the row. A near-horizontal line has no such run and
    # falls back to its whole extent in x.
    steep = np.abs(dy) > 1e-6 * length
    safe_dy = np.where(steep, dy, 1.0)
    centre = ax[seg] + (y - ay[seg]) * dx / safe_dy
    half_run = reach[seg] * length / np.abs(safe_dy)
    x_lo = np.where(steep, np.maximum(x_lo, centre - half_run), x_lo)
    x_hi = np.where(steep, np.minimum(x_hi, centre + half_run), x_hi)
    first_col, col_count = _spans(x_lo, x_hi, grid.x0, grid.dx, grid.cols)
    run, offset = _expand(col_count)
    if len(run) == 0:
        return out.reshape(grid.rows, grid.cols)
    col = first_col[run] + offset
    seg, row = seg[run], row[run]
    px = grid.x0 + grid.dx * col - ax[seg]
    py = grid.y0 + grid.dy * row - ay[seg]
    dx, dy, length = dx[run], dy[run], length[run]
    along = np.clip((px * dx + py * dy) / length**2, 0.0, 1.0)
    distance = np.hypot(px - along * dx, py - along * dy)
    half = half_width[seg]
    covered = _overlap(distance - foot / 2, distance + foot / 2, -half, half) / foot
    covered = (np.minimum(covered, 1.0) * value[seg]).astype(np.float32)
    np.maximum.at(out, row * grid.cols + col, covered)
    return out.reshape(grid.rows, grid.cols)


def _near_samples(x, y, radius, grid: _Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (item, row, col) for which the sample's rectangle touches the item's bounding square."""
    reach_x, reach_y = radius + grid.dx / 2, radius + grid.dy / 2
    first_row, row_count = _spans(y - reach_y, y + reach_y, grid.y0, grid.dy, grid.rows)
    first_col, col_count = _spans(x - reach_x, x + reach_x, grid.x0, grid.dx, grid.cols)
    item, row_offset = _expand(np.where(col_count > 0, row_count, 0))
    run, col_offset = _expand(col_count[item])
    item = item[run]
    return item, first_row[item] + row_offset[run], first_col[item] + col_offset


def _splat_features(features: _Features, grid: _Grid) -> dict[str, np.ndarray]:
    """Coverage of each sample by pores and pockets, visiting only samples near one."""
    foot = grid.footprint
    out = {
        name: np.zeros(grid.rows * grid.cols, np.float32)
        for name in ("pore", "pore_rim", "pore_depth_um", "second_phase")
    }
    item, row, col = _near_samples(features.x, features.y, features.radius, grid)
    flat = row * grid.cols + col
    qx = grid.x0 + grid.dx * col - features.x[item]
    qy = grid.y0 + grid.dy * row - features.y[item]
    radius = features.radius[item]
    # Two ways to turn a shape into coverage. A shape much larger than a sample
    # is cut by an edge that is locally straight. A shape near or below the size
    # of a sample is treated as a square of the same area, whose overlap with
    # the sample is exact, so its total coverage is right however coarse the grid.
    large = radius >= 1.5 * max(grid.dx, grid.dy)
    side = np.sqrt(features.area[item])
    square = (
        _overlap(qx - grid.dx / 2, qx + grid.dx / 2, -side / 2, side / 2)
        * _overlap(qy - grid.dy / 2, qy + grid.dy / 2, -side / 2, side / 2)
        / (grid.dx * grid.dy)
    )

    def add(name, keep, values):
        np.add.at(out[name], flat[keep], values.astype(np.float32))

    pore = np.flatnonzero(features.is_pore[item])
    distance = np.hypot(qx[pore], qy[pore])
    r, big, small = radius[pore], large[pore], square[pore]
    outside = distance - r
    covered = np.where(big, np.clip(0.5 - outside / foot, 0.0, 1.0), small)
    rim = np.minimum(PORE_RIM_UM, r)
    bowl = PORE_DEPTH_PER_RADIUS * np.sqrt(np.clip(r**2 - distance**2, 0.0, None))
    wall = _overlap(outside - foot / 2, outside + foot / 2, -rim, 0.0) / foot
    add("pore", pore, covered)
    add("pore_rim", pore, np.where(big, wall, small * (1 - (1 - rim / r) ** 2)))
    # A small pore gets the mean depth of the bowl, two thirds of its deepest.
    add("pore_depth_um", pore, np.where(big, covered * bowl, small * PORE_DEPTH_PER_RADIUS * r * 2 / 3))

    pocket = np.flatnonzero(~features.is_pore[item])
    which = item[pocket]
    outside = (
        features.normal_x[which] * qx[pocket, None]
        + features.normal_y[which] * qy[pocket, None]
        - features.offset[which]
    )
    # A triangle is where three half-planes overlap. Multiplying the three
    # coverages is exact along an edge and close at a corner, where taking the
    # nearest edge alone would count the sliver beyond the tip as covered.
    by_edges = np.prod(np.clip(0.5 - outside / foot, 0.0, 1.0), axis=1)
    add("second_phase", pocket, np.where(large[pocket], by_edges, square[pocket]))

    for name in ("pore", "pore_rim", "second_phase"):
        np.minimum(out[name], 1.0, out=out[name])
    return {name: values.reshape(grid.rows, grid.cols) for name, values in out.items()}


def _area_inside(features: _Features, x_lo, x_hi, y_lo, y_hi) -> np.ndarray:
    """Area of each pore or pocket that lies inside a rectangle.

    Whole when the shape is clear of the rectangle's edges, which is nearly
    always. A shape that straddles an edge is measured on a 48 x 48 lattice over
    its bounding square, good to about a percent of that one shape.
    """
    x, y, r = features.x, features.y, features.radius
    whole = (x - r >= x_lo) & (x + r <= x_hi) & (y - r >= y_lo) & (y + r <= y_hi)
    apart = (x + r <= x_lo) | (x - r >= x_hi) | (y + r <= y_lo) | (y - r >= y_hi)
    area = np.where(whole, features.area, 0.0)
    cut = np.flatnonzero(~whole & ~apart)
    if len(cut):
        part = _take(features, cut)
        steps = (np.arange(48) + 0.5) / 24 - 1.0
        qx = part.radius[:, None, None] * steps[None, None, :]
        qy = part.radius[:, None, None] * steps[None, :, None]
        in_disc = qx**2 + qy**2 < part.radius[:, None, None] ** 2
        in_triangle = np.ones(in_disc.shape, bool)
        for k in range(3):
            in_triangle &= (
                part.normal_x[:, k, None, None] * qx + part.normal_y[:, k, None, None] * qy
                < part.offset[:, k, None, None]
            )
        in_shape = np.where(part.is_pore[:, None, None], in_disc, in_triangle)
        x, y = part.x[:, None, None] + qx, part.y[:, None, None] + qy
        in_window = (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
        area[cut] = (in_shape & in_window).sum(axis=(1, 2)) * (part.radius / 24) ** 2
    return area


def _clip_segment(x0, y0, x1, y1, x_lo, x_hi, y_lo, y_hi):
    """The part of a segment inside a rectangle (Liang-Barsky), or None."""
    t0, t1 = 0.0, 1.0
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 - x_lo), (dx, x_hi - x0), (-dy, y0 - y_lo), (dy, y_hi - y0)):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    if t0 >= t1:
        return None
    return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


# --- the specimen ---------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Sample:
    """A rectangular polished specimen set in a round resin mount, on a stage plate.

    Build one with `Sample.polished_ceramic`. The same arguments always give the
    same specimen.
    """

    seed: int = 0
    grain_size_um: float = 8.0
    """Mean equivalent circle diameter of the grains."""
    porosity: float = 0.03
    """Area fraction of pores."""
    second_phase: float = 0.02
    """Area fraction of the minority phase, in pockets where three grains meet."""
    etched: bool = True
    """Etching is what makes grain boundaries visible. Unetched, only pores,
    second phase and scratches show."""
    material: Material = CERAMIC
    specimen_size_um: tuple[float, float] = (6000.0, 6000.0)
    """Extent of the specimen along x and y."""
    mount_diameter_um: float = 30000.0
    mount_height_um: float = 15000.0
    """Height of the polished face above the stage plate, at the centre of the mount."""
    tilt_um_per_mm: float = 1.5
    """Slope of the polished face, as left by a levelling press. Its direction comes from the seed."""
    scratch_count: int = 40
    scratches: tuple[Scratch, ...] = field(init=False, repr=False)
    """Every polishing scratch on the specimen, whole."""

    @classmethod
    def polished_ceramic(
        cls,
        seed: int = 0,
        grain_size_um: float = 8.0,
        porosity: float = 0.03,
        etched: bool = True,
        *,
        second_phase: float = 0.02,
        specimen_size_um: tuple[float, float] = (6000.0, 6000.0),
        mount_diameter_um: float = 30000.0,
        mount_height_um: float = 15000.0,
        tilt_um_per_mm: float = 1.5,
        scratch_count: int = 40,
    ) -> Sample:
        """A dense sintered ceramic: equiaxed grains, a few percent of pores at
        grain junctions, a little second phase, a few polishing scratches.

        The default 3% porosity is on the porous side for a fully dense
        sintered part; ask for 0.01 or less for one of those. Porosity can go
        up to about 0.07 and second phase to about 0.05.

        Time and memory go with the number of grains in view, so they grow as
        grains get finer: a 5x field (0.7 x 0.5 mm) of 8 um grains takes about
        0.2 s the first time it is seen, of 4 um grains 0.5 s, of 2 um grains
        over a second, of 1 um grains about four (on the laptop this was
        written on). Below about 2 um a 5x field also no longer fits in the
        tile cache and is rebuilt on every visit.
        """
        return cls(
            seed=seed,
            grain_size_um=grain_size_um,
            porosity=porosity,
            second_phase=second_phase,
            etched=etched,
            material=CERAMIC,
            specimen_size_um=specimen_size_um,
            mount_diameter_um=mount_diameter_um,
            mount_height_um=mount_height_um,
            tilt_um_per_mm=tilt_um_per_mm,
            scratch_count=scratch_count,
        )

    def __post_init__(self) -> None:
        width, depth = self.specimen_size_um
        if self.grain_size_um <= 0 or width <= 0 or depth <= 0:
            raise ValueError("grain size and specimen size must be positive")
        if math.hypot(width, depth) >= self.mount_diameter_um:
            raise ValueError("the specimen does not fit inside the mount")
        if self.porosity < 0 or self.second_phase < 0:
            raise ValueError("porosity and second phase cannot be negative")

        def put(name, value):  # the dataclass is frozen; these are worked out once, here
            object.__setattr__(self, name, value)

        put("_key", self.seed & 0xFFFFFFFFFFFFFFFF)
        put("_pitch", self.grain_size_um / _ECD_PER_PITCH)
        put("_stride", math.ceil(max(width, depth) / self._pitch) + 2 * _ID_OFFSET)
        tiles = math.ceil(_CACHED_AREA_UM2 / (_TILE * self._pitch) ** 2)
        put("_tile", lru_cache(maxsize=min(max(tiles, 128), _MOST_TILES))(self._build_tile))
        put("_fields", lru_cache(maxsize=2)(self._build_fields))

        direction = 2 * math.pi * _unit(self._key, _CH_TILT)[0]
        slope = self.tilt_um_per_mm / 1000.0
        put("_tilt", (slope * math.cos(direction), slope * math.sin(direction)))
        # Slow waviness: four plane waves 0.7 to 2.5 mm long and about 0.2 um high.
        n = np.arange(4)
        wavelength = 700.0 * (2500.0 / 700.0) ** _unit(self._key, _CH_WAVE, n, 0)
        heading = 2 * np.pi * _unit(self._key, _CH_WAVE, n, 1)
        put(
            "_waves",
            (
                2 * np.pi * np.cos(heading) / wavelength,
                2 * np.pi * np.sin(heading) / wavelength,
                2 * np.pi * _unit(self._key, _CH_WAVE, n, 2),
                0.1 + 0.2 * _unit(self._key, _CH_WAVE, n, 3),
            ),
        )
        put("scratches", self._make_scratches())
        put("_fill", self._calibrate())

    # --- shape of the whole thing ---------------------------------------------

    def height_um(self, x_um, y_um):
        """Height of the polished surface at a point: the focus position that
        makes that point sharp.

        Pores, grooves and scratches are filled in here; `fields(...).height_um`
        has them. Takes numbers or arrays. Off the mount this is the stage
        plate, at 0.
        """
        out = self._surface(np.asarray(x_um, float), np.asarray(y_um, float))
        return float(out) if out.ndim == 0 else out

    def max_height_um(self, x_um: float, y_um: float, radius_um: float) -> float:
        """The highest point of anything within `radius_um` of a point: what an
        objective's front lens would touch first.

        Found by sampling the polished surface on 64 rings of 256 points, plus
        the nearest point of the mount when the circle only clips its rim. With
        the default slopes that is good to about 0.1 um, far finer than the
        hundreds of micrometres a collision is decided by.
        """
        ring = radius_um * np.sqrt(np.arange(65) / 64.0)[:, None]
        angle = 2 * np.pi * np.arange(256)[None, :] / 256
        x = (x_um + ring * np.cos(angle)).ravel()
        y = (y_um + ring * np.sin(angle)).ravel()
        from_centre = math.hypot(x_um, y_um)
        mount_radius = self.mount_diameter_um / 2
        if mount_radius < from_centre <= mount_radius + radius_um:
            inward = (mount_radius * (1 - 1e-9)) / from_centre
            x, y = np.append(x, x_um * inward), np.append(y, y_um * inward)
        return float(self._surface(x, y).max())

    def _surface(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Height of the polished surface. x and y broadcast against each other.

        Written so that, for the two axes of a grid, everything that depends on
        one axis is worked out on that axis alone and only a handful of
        operations touch the whole grid.
        """
        mount_radius = self.mount_diameter_um / 2
        radius_sq = x * x + y * y
        plane = (self.mount_height_um + self._tilt[0] * x - DOME_SAG_UM * x * x / mount_radius**2) + (
            self._tilt[1] * y - DOME_SAG_UM * y * y / mount_radius**2
        )
        # Each wave is sin(kx x + ky y + phase), split by the angle-sum rule
        # into products of a factor in x and a factor in y.
        kx, ky, phase, amplitude = (w.reshape((-1,) + (1,) * max(x.ndim, y.ndim)) for w in self._waves)
        in_x = np.concatenate([amplitude * np.sin(kx * x + phase), amplitude * np.cos(kx * x + phase)])
        in_y = np.concatenate([np.cos(ky * y), np.sin(ky * y)])
        plane = plane + np.einsum("k...,k...->...", in_x, in_y)
        if np.any(radius_sq > (mount_radius - ROLL_OFF_WIDTH_UM) ** 2):
            rim = np.clip((np.sqrt(radius_sq) - (mount_radius - ROLL_OFF_WIDTH_UM)) / ROLL_OFF_WIDTH_UM, 0.0, 1.0)
            plane = plane - ROLL_OFF_UM * rim**2
        inset_x = self.specimen_size_um[0] / 2 - np.abs(x)
        inset_y = self.specimen_size_um[1] / 2 - np.abs(y)
        # exp(-min(a, b)) is max(exp(-a), exp(-b)): the rounding of the nearer edge.
        rounding = EDGE_ROUNDING_UM * np.maximum(
            np.exp(-np.clip(inset_x, 0.0, None) / EDGE_ROUNDING_LENGTH_UM),
            np.exp(-np.clip(inset_y, 0.0, None) / EDGE_ROUNDING_LENGTH_UM),
        )
        on_specimen = (inset_x > 0) & (inset_y > 0)
        on_mount = plane - np.where(on_specimen, rounding, RESIN_RECESS_UM)
        return np.where(radius_sq < mount_radius**2, on_mount, 0.0)

    def _make_scratches(self) -> tuple[Scratch, ...]:
        """Straight grooves in random directions, 0.3 to 4 mm long, mostly short."""
        n = np.arange(self.scratch_count)
        draw = lambda channel: _unit(self._key, _CH_SCRATCH, n, channel)  # noqa: E731
        cx = (draw(0) - 0.5) * self.specimen_size_um[0]
        cy = (draw(1) - 0.5) * self.specimen_size_um[1]
        heading = np.pi * draw(2)
        half = 0.5 * 300.0 * (4000.0 / 300.0) ** draw(3)
        ux, uy = half * np.cos(heading), half * np.sin(heading)
        width = 0.3 + 0.6 * draw(4)
        depth = 0.05 + (SCRATCH_MAX_DEPTH_UM - 0.05) * draw(5)
        rows = zip(n, cx - ux, cy - uy, cx + ux, cy + uy, width, depth)
        return tuple(Scratch(int(i), *(float(v) for v in values)) for i, *values in rows)

    # --- grains, boundaries and junctions --------------------------------------

    def _block(self, ix0: int, ix1: int, iy0: int, iy1: int) -> tuple[_Cells, _Edges, _Junctions]:
        """Geometry of the grid cells [ix0, ix1) x [iy0, iy1).

        Triangulates those cells plus padding, and keeps the grains, boundaries
        and junctions that belong to the block. Every block computes a shared
        boundary or junction from the same numbers in the same order, so
        neighbouring blocks agree to the last bit.
        """
        pitch = self._pitch
        x_origin, y_origin = -self.specimen_size_um[0] / 2, -self.specimen_size_um[1] / 2
        ix, iy = np.meshgrid(np.arange(ix0 - _PAD, ix1 + _PAD), np.arange(iy0 - _PAD, iy1 + _PAD))
        ix, iy = ix.ravel(), iy.ravel()
        sx = x_origin + pitch * (ix - _OVERHANG + _JITTER * _unit(self._key, _CH_SEED_X, ix, iy))
        sy = y_origin + pitch * (iy - _OVERHANG + _JITTER * _unit(self._key, _CH_SEED_Y, ix, iy))
        grain = (iy + _ID_OFFSET) * self._stride + (ix + _ID_OFFSET)

        triangulation = Delaunay(np.column_stack([sx, sy]))
        # Corners in grid order, which is the same in every block.
        a, b, c = np.sort(triangulation.simplices, axis=1).T
        abx, aby, acx, acy = sx[b] - sx[a], sy[b] - sy[a], sx[c] - sx[a], sy[c] - sy[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            twice = 2 * (abx * acy - aby * acx)
            ab_sq, ac_sq = abx**2 + aby**2, acx**2 + acy**2
            # The circumcentre of each triangle is the junction of its three grains.
            jx = sx[a] + (acy * ab_sq - aby * ac_sq) / twice
            jy = sy[a] + (abx * ac_sq - acx * ab_sq) / twice

        def within(x, y, margin):
            return (
                (x >= x_origin + pitch * (ix0 - margin))
                & (x < x_origin + pitch * (ix1 + margin))
                & (y >= y_origin + pitch * (iy0 - margin))
                & (y < y_origin + pitch * (iy1 + margin))
            )

        with np.errstate(invalid="ignore"):
            sound = within(jx, jy, _SAFE)

        # A grain boundary joins the junctions of two triangles that share a side.
        triangle = np.repeat(np.arange(len(a)), 3)
        corner = np.tile(np.arange(3), len(a))
        other = triangulation.neighbors.ravel()
        once = (other > triangle) & sound[triangle] & sound[np.maximum(other, 0)]
        triangle, corner, other = triangle[once], corner[once], other[once]
        p = triangulation.simplices[triangle, (corner + 1) % 3]
        q = triangulation.simplices[triangle, (corner + 2) % 3]
        ax, ay, bx, by = jx[triangle], jy[triangle], jx[other], jy[other]
        length = np.hypot(bx - ax, by - ay)
        apart = np.hypot(sx[p] - sx[q], sy[p] - sy[q])
        # A Voronoi cell is a fan of triangles from its point to each of its
        # sides, every one of height apart / 2.
        area = np.bincount(p, length * apart / 4, len(sx)) + np.bincount(q, length * apart / 4, len(sx))
        perimeter = np.bincount(p, length, len(sx)) + np.bincount(q, length, len(sx))
        p_is_left = sx[p] < sx[q]
        mine = within((ax + bx) / 2, (ay + by) / 2, 0)
        edges = _Edges(
            ax[mine],
            ay[mine],
            bx[mine],
            by[mine],
            np.where(p_is_left, grain[p], grain[q])[mine],
            np.where(p_is_left, grain[q], grain[p])[mine],
        )

        own = (ix >= ix0) & (ix < ix1) & (iy >= iy0) & (iy < iy1)
        cells = _Cells(grain[own], sx[own], sy[own], area[own], perimeter[own])

        a, b, c, jx, jy = a[sound], b[sound], c[sound], jx[sound], jy[sound]
        # The three boundaries leaving a junction: each is square to the line
        # between two of the grains and points away from the third.
        rays = []
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            rx, ry = -(sy[v] - sy[u]), sx[v] - sx[u]
            away = np.sign(rx * ((sx[u] + sx[v]) / 2 - sx[w]) + ry * ((sy[u] + sy[v]) / 2 - sy[w]))
            norm = np.hypot(rx, ry)
            rays.append((away * rx / norm, away * ry / norm))
        ray_x, ray_y = np.array([r[0] for r in rays]).T, np.array([r[1] for r in rays]).T
        order = np.argsort(np.arctan2(ray_y, ray_x), axis=1)
        ray_x, ray_y = np.take_along_axis(ray_x, order, 1), np.take_along_axis(ray_y, order, 1)
        next_x, next_y = np.roll(ray_x, -1, axis=1), np.roll(ray_y, -1, axis=1)
        side = np.hypot(next_x - ray_x, next_y - ray_y)
        normal_x, normal_y = (next_y - ray_y) / side, -(next_x - ray_x) / side
        junctions = _Junctions(
            x=jx,
            y=jy,
            kind=_unit(self._key, _CH_KIND, grain[a], grain[b], grain[c]),
            size=_unit(self._key, _CH_SIZE, grain[a], grain[b], grain[c]),
            inside=within(jx, jy, 0),
            normal_x=normal_x,
            normal_y=normal_y,
            offset=normal_x * ray_x + normal_y * ray_y,
            area=0.5 * np.abs(np.sum(ray_x * next_y - ray_y * next_x, axis=1)),
        )
        return cells, edges, junctions

    def _place(self, junctions: _Junctions, pore_share: float, pocket_share: float) -> _Features:
        """Put pores and pockets on a share of the junctions.

        Each candidate has a size drawn for it. Where two would overlap, the
        smaller gives way, whatever becomes of the larger: a rule that needs
        only a junction's near neighbours, so every tile reaches the same
        verdict, and that leaves no two survivors touching.
        """
        is_pore = junctions.kind < pore_share
        chosen = np.flatnonzero(is_pore | (junctions.kind > 1.0 - pocket_share))
        is_pore = is_pore[chosen]
        x, y = junctions.x[chosen], junctions.y[chosen]
        lo, hi = np.where(is_pore[:, None], _PORE_RADIUS_PER_PITCH, _POCKET_RADIUS_PER_PITCH).T
        size = junctions.size[chosen]
        radius = self._pitch * (lo + (hi - lo) * np.where(is_pore, size**2, size))
        keep = junctions.inside[chosen]
        if len(chosen) > 1:
            reach = 2 * _CLEARANCE * self._pitch * max(_PORE_RADIUS_PER_PITCH[1], _POCKET_RADIUS_PER_PITCH[1])
            i, j = cKDTree(np.column_stack([x, y])).query_pairs(reach, output_type="ndarray").T
            clash = np.hypot(x[i] - x[j], y[i] - y[j]) < _CLEARANCE * (radius[i] + radius[j])
            i, j = i[clash], j[clash]
            keep[np.where(radius[i] < radius[j], i, j)] = False
        scale = radius[:, None]
        features = _Features(
            is_pore=is_pore,
            x=x,
            y=y,
            radius=radius,
            area=np.where(is_pore, np.pi * radius**2, junctions.area[chosen] * radius**2),
            normal_x=junctions.normal_x[chosen],
            normal_y=junctions.normal_y[chosen],
            offset=junctions.offset[chosen] * scale,
        )
        return _take(features, keep)

    def _calibrate(self) -> tuple[float, float]:
        """The share of junctions to offer to pores and to pockets.

        How much area a given share ends up as depends on how many candidates
        give way to a neighbour, which is hard to write down, so it is
        measured: place them on a reference patch, compare with what was asked
        for, adjust, repeat. The patch holds a few thousand pores, so the
        answer is good to a couple of percent of the request.
        """
        junctions = self._block(0, _CALIBRATION_CELLS, 0, _CALIBRATION_CELLS)[2]
        patch_area = (_CALIBRATION_CELLS * self._pitch) ** 2
        wanted = np.array([self.porosity, self.second_phase])
        asked = wanted > 0
        share = np.where(asked, 0.2, 0.0)
        best, best_miss = share, np.inf
        for _ in range(16):
            placed = self._place(junctions, *share)
            pores, pockets = placed.area[placed.is_pore].sum(), placed.area[~placed.is_pore].sum()
            got = np.array([pores, pockets]) / patch_area
            miss = float(np.max(np.abs(got[asked] / wanted[asked] - 1.0), initial=0.0))
            if share.sum() <= 1.0 and miss < best_miss:
                best, best_miss = share, miss
            if best_miss <= 0.003 or np.any(asked & (got == 0)):
                break
            share = np.minimum(np.where(asked, share * wanted / np.where(got > 0, got, 1.0), 0.0), 1.0)
        if best_miss > 0.03:
            raise ValueError(
                f"cannot fit porosity {self.porosity} and second phase {self.second_phase}: pores and "
                "pockets sit on grain junctions and there are only so many. Porosity alone can reach "
                "about 0.07 and second phase alone about 0.05, and each takes room from the other"
            )
        return float(best[0]), float(best[1])

    def _build_tile(self, tx: int, ty: int) -> _Tile:
        x0, y0 = tx * _TILE, ty * _TILE
        cells, edges, junctions = self._block(x0, x0 + _TILE, y0, y0 + _TILE)
        placed = self._place(junctions, *self._fill)
        # Nothing may poke out over the edge of the specimen.
        half_x, half_y = self.specimen_size_um[0] / 2, self.specimen_size_um[1] / 2
        fits = (np.abs(placed.x) + placed.radius < half_x) & (np.abs(placed.y) + placed.radius < half_y)
        return _Tile(cells, edges, _take(placed, fits))

    def _gather(self, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> _Tile:
        """Everything in the tiles that a rectangle touches, limited to tiles near the specimen."""
        half_x, half_y = self.specimen_size_um[0] / 2, self.specimen_size_um[1] / 2
        reach = _SAFE * self._pitch
        span = _TILE * self._pitch

        def tiles(lo, hi, half):
            lo, hi = max(lo, -half - reach), min(hi, half + reach)
            return range(math.floor((lo + half) / span), math.floor((hi + half) / span) + 1)

        found = [self._tile(tx, ty) for ty in tiles(y_lo, y_hi, half_y) for tx in tiles(x_lo, x_hi, half_x)]
        return _Tile(*[_join([getattr(tile, part) for tile in found]) for part in ("cells", "edges", "features")])

    # --- what a camera or a scorer asks for --------------------------------------

    def fields(
        self, x_um: float, y_um: float, width_um: float, height_um: float, shape: tuple[int, int]
    ) -> Fields:
        """The specimen over a window centred on (x_um, y_um), sampled on a
        (rows, cols) grid.

        Takes around 0.1 s for a camera frame once the tiles under it are
        cached, and the last two windows asked for are remembered, so a focus
        series at one position pays once. The two sample spacings must be
        within a factor of 4 of each other.
        """
        rows, cols = int(shape[0]), int(shape[1])
        if rows < 1 or cols < 1 or width_um <= 0 or height_um <= 0:
            raise ValueError("a window needs a positive size and at least one sample each way")
        # Grooves and scratches are spread over samples as if samples were
        # square. At 4 to 1 that overstates a groove by about 5%; much beyond,
        # a line can fall between sample centres and vanish.
        if not 1 / 4 <= (width_um / cols) / (height_um / rows) <= 4:
            raise ValueError(
                f"samples of {width_um / cols:.3g} x {height_um / rows:.3g} um are too far from square; "
                "keep the two spacings within a factor of 4 of each other"
            )
        return self._fields(float(x_um), float(y_um), float(width_um), float(height_um), rows, cols)

    def _build_fields(self, x_um, y_um, width_um, height_um, rows, cols) -> Fields:
        dx, dy = width_um / cols, height_um / rows
        xs = x_um - width_um / 2 + dx * (np.arange(cols) + 0.5)
        ys = y_um - height_um / 2 + dy * (np.arange(rows) + 0.5)
        surface = self._surface(xs[None, :], ys[:, None])
        on_x = np.abs(xs) < self.specimen_size_um[0] / 2
        on_y = np.abs(ys) < self.specimen_size_um[1] / 2
        on_mount = xs[None, :] ** 2 + ys[:, None] ** 2 < (self.mount_diameter_um / 2) ** 2
        region = on_mount.astype(np.uint8)  # Region.RESIN is 1, Region.STAGE is 0
        region[np.ix_(on_y, on_x)] = Region.SPECIMEN

        layers = {}
        if on_x.any() and on_y.any():
            # The specimen is a rectangle, so the samples on it are a block of
            # the grid, and only that block is worked on.
            c0, r0 = int(np.argmax(on_x)), int(np.argmax(on_y))
            block_rows, block_cols = int(on_y.sum()), int(on_x.sum())
            layers = self._microstructure(_Grid(xs[c0], ys[r0], dx, dy, block_rows, block_cols))
            if (block_rows, block_cols) != (rows, cols):
                block = (slice(r0, r0 + block_rows), slice(c0, c0 + block_cols))
                for name, values in layers.items():
                    layers[name] = np.full((rows, cols), -1 if name == "grain_id" else 0, values.dtype)
                    layers[name][block] = values
        layers["edge_gap"] = self._edge_gap(xs, ys, math.sqrt(dx * dy))
        layers.setdefault("grain_id", np.full((rows, cols), -1, np.int64))
        for name in ("orientation", "boundary", "pore", "pore_rim", "pore_depth_um", "second_phase", "scratch"):
            layers.setdefault(name, np.zeros((rows, cols), np.float32))

        relief = (
            layers["pore_depth_um"]
            + GROOVE_DEPTH_UM * layers["boundary"]
            + SCRATCH_MAX_DEPTH_UM * layers["scratch"]
            + SECOND_PHASE_RELIEF_UM * layers["second_phase"]
        )
        out = Fields(x_um=xs, y_um=ys, region=region, surface_um=surface, height_um=surface - relief, **layers)
        for f in dataclass_fields(out):
            getattr(out, f.name).setflags(write=False)
        return out

    def _edge_gap(self, xs: np.ndarray, ys: np.ndarray, foot: float) -> np.ndarray:
        """Coverage of each sample by the gap just outside the specimen's outline."""
        beyond_x = np.abs(xs) - self.specimen_size_um[0] / 2
        beyond_y = np.abs(ys) - self.specimen_size_um[1] / 2
        reach = EDGE_GAP_UM + foot
        # The gap is in view only when one side of the outline is, which is rare.
        side_x = (np.abs(beyond_x) < reach).any() and (beyond_y < reach).any()
        side_y = (np.abs(beyond_y) < reach).any() and (beyond_x < reach).any()
        if not (side_x or side_y):
            return np.zeros((len(ys), len(xs)), np.float32)
        # How far a sample is outside the rectangle, measured square to the
        # nearer side. Negative inside.
        beyond = np.maximum(beyond_x[None, :], beyond_y[:, None])
        return (_overlap(beyond - foot / 2, beyond + foot / 2, 0.0, EDGE_GAP_UM) / foot).astype(np.float32)

    def _microstructure(self, grid: _Grid) -> dict[str, np.ndarray]:
        x_hi = grid.x0 + grid.dx * (grid.cols - 1)
        y_hi = grid.y0 + grid.dy * (grid.rows - 1)
        reach = _SAFE * self._pitch + grid.footprint
        found = self._gather(grid.x0 - reach, x_hi + reach, grid.y0 - reach, y_hi + reach)

        def near(lo_x, hi_x, lo_y, hi_y, margin):
            """Whether a box comes within `margin` of the grid."""
            across = (hi_x >= grid.x0 - margin) & (lo_x <= x_hi + margin)
            return across & (hi_y >= grid.y0 - margin) & (lo_y <= y_hi + margin)

        e = found.edges
        lo_x, hi_x = np.minimum(e.ax, e.bx), np.maximum(e.ax, e.bx)
        edges = _take(e, near(lo_x, hi_x, np.minimum(e.ay, e.by), np.maximum(e.ay, e.by), reach))
        labels = _label_image(found.cells, edges, grid)
        # A grain's angle is a hash of its number. Hash each number in the window
        # once and look the samples up, rather than hashing every sample.
        lowest = int(labels.min())
        angles = np.pi * _unit(self._key, _CH_ANGLE, np.arange(lowest, int(labels.max()) + 1))
        out = {"grain_id": labels, "orientation": angles.astype(np.float32)[labels - lowest]}

        if self.etched:
            half = np.full(len(edges.ax), GROOVE_WIDTH_UM / 2)
            # How well the etch revealed a boundary is a hash of its two grains.
            low, high = np.minimum(edges.left, edges.right), np.maximum(edges.left, edges.right)
            draw = _unit(self._key, _CH_GROOVE, low, high)
            rest = (draw - _FAINT_BOUNDARIES) / (1 - _FAINT_BOUNDARIES)
            strength = np.where(draw < _FAINT_BOUNDARIES, 0.15 * draw / _FAINT_BOUNDARIES, 0.25 + 0.75 * rest)
            out["boundary"] = _splat_segments(edges.ax, edges.ay, edges.bx, edges.by, half, strength, grid)

        f = found.features
        out.update(_splat_features(_take(f, near(f.x, f.x, f.y, f.y, f.radius + grid.footprint)), grid))

        if self.scratches:
            ax, ay, bx, by, width, depth = (
                np.array([getattr(item, name) for item in self.scratches])
                for name in ("x0_um", "y0_um", "x1_um", "y1_um", "width_um", "depth_um")
            )
            out["scratch"] = _splat_segments(ax, ay, bx, by, width / 2, depth / SCRATCH_MAX_DEPTH_UM, grid)
        # A pore is a hole and a pocket is another material: neither carries the
        # grooves of the grains around it, and a pore carries no scratch either.
        for name, cleared_by in (("boundary", ("pore", "second_phase")), ("scratch", ("pore",))):
            if name in out:
                for other in cleared_by:
                    out[name] *= 1.0 - out[other]
        return out

    def truth(self, x_um: float, y_um: float, width_um: float, height_um: float) -> Truth:
        """The true answers for a window centred on (x_um, y_um).

        Worked out from the geometry itself, not from pixels: pore and pocket
        areas are added up (cut at the window's edge where needed), grain areas
        and perimeters come from the grain polygons.
        """
        half_x, half_y = self.specimen_size_um[0] / 2, self.specimen_size_um[1] / 2
        x_lo, x_hi = max(x_um - width_um / 2, -half_x), min(x_um + width_um / 2, half_x)
        y_lo, y_hi = max(y_um - height_um / 2, -half_y), min(y_um + height_um / 2, half_y)
        if x_lo >= x_hi or y_lo >= y_hi:
            nothing = np.zeros((0, 3))
            return Truth(x_um, y_um, width_um, height_um, 0.0, 0.0, nothing, 0, 0.0, 0.0, 0.0, ())
        specimen_area = (x_hi - x_lo) * (y_hi - y_lo)
        # A tile holds the grains of its cells, whose points may lie a little
        # outside it, and the pores centred in it, which reach a little outside it.
        reach = max(_OVERHANG, _POCKET_RADIUS_PER_PITCH[1], _PORE_RADIUS_PER_PITCH[1]) * self._pitch
        found = self._gather(x_lo - reach, x_hi + reach, y_lo - reach, y_hi + reach)

        def inside(x, y):
            return (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)

        cells = _take(found.cells, inside(found.cells.x, found.cells.y))
        count = len(cells.id)
        features = found.features
        area = _area_inside(features, x_lo, x_hi, y_lo, y_hi)
        listed = features.is_pore & inside(features.x, features.y)
        scratches = []
        for s in self.scratches:
            part = _clip_segment(s.x0_um, s.y0_um, s.x1_um, s.y1_um, x_lo, x_hi, y_lo, y_hi)
            if part:
                scratches.append(Scratch(s.id, *part, s.width_um, s.depth_um))
        return Truth(
            x_um=x_um,
            y_um=y_um,
            width_um=width_um,
            height_um=height_um,
            specimen_fraction=specimen_area / (width_um * height_um),
            porosity=float(area[features.is_pore].sum() / specimen_area),
            pores_um=np.column_stack([features.x, features.y, 2 * features.radius])[listed],
            grain_count=count,
            mean_grain_diameter_um=float(np.sqrt(4 * cells.area / np.pi).mean()) if count else 0.0,
            mean_intercept_um=float(np.pi * cells.area.sum() / cells.perimeter.sum()) if count else 0.0,
            second_phase_fraction=float(area[~features.is_pore].sum() / specimen_area),
            scratches=tuple(scratches),
        )
