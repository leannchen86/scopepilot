"""From a specimen and the microscope's settings to a camera frame.

A reflected-light (metallurgical) microscope looking down at a polished, opaque
surface. Nothing here is calibrated against an instrument: the objective
figures are typical published values for standard brightfield metallurgical
objectives, and everything about brightness is a plausible choice made so that
a person's instincts about the real thing carry over (darkfield needs far more
exposure than brightfield, a 100x frame is dimmer than a 5x one, a 50x
objective loses focus within a micrometre).

What is modelled
    A reflectance picture per contrast method, built from the specimen's
    fields; blur from the objective's resolution; extra blur that grows with
    the distance between the focal plane and the surface under each pixel;
    illumination that falls off smoothly towards the corners of the frame;
    signal proportional to lamp x exposure, clipped at full scale; shot noise
    and read noise.

What is not
    Colour, dust, glare, diffraction rings, blooming, camera black level (a
    dark frame is zeros plus noise, clipped at zero), the darkfield collar's
    own aperture, and anything a rotating stage would show between crossed
    polarisers (the polarisers are fixed). Defocus looks the same above and
    below focus. Surfaces at different heights are blurred separately and
    added, so an out-of-focus specimen spills a little light over the resin
    beside it; only the stage plate, 15 mm below, is kept from spilling onto
    the mount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from scopepilot.sim.sample import Fields, Material, Region, Sample

WAVELENGTH_UM = 0.55
"""Green light, the usual reference wavelength for resolution figures."""

CONTRASTS = ("Brightfield", "Darkfield", "Polarised")


@dataclass(frozen=True)
class Objective:
    name: str
    magnification: float
    numerical_aperture: float
    working_distance_um: float
    """Clear distance from the front lens to the focal plane."""
    front_radius_um: float
    """Radius of the nose around the front lens: how far from the optical axis
    the objective can touch something."""

    @property
    def resolution_um(self) -> float:
        """Rayleigh resolution, 0.61 * wavelength / NA, at 550 nm."""
        return 0.61 * WAVELENGTH_UM / self.numerical_aperture

    @property
    def depth_of_field_um(self) -> float:
        """Wave-optical depth of field in air, wavelength / NA^2, at 550 nm.

        This is the first term of the usual total depth of field; the second,
        which depends on the camera's pixel size, is left out. Some catalogues
        quote half this figure.
        """
        return WAVELENGTH_UM / self.numerical_aperture**2

    @property
    def brightness(self) -> float:
        """Image brightness relative to the 5x objective, for the same lamp and exposure.

        Light gathered goes as NA^2 and is spread over an image area that goes
        as magnification^2.
        """
        return (self.numerical_aperture / self.magnification) ** 2 / (0.15 / 5) ** 2


# Numerical aperture and working distance are those commonly published for
# standard (not long-working-distance) brightfield metallurgical objectives.
# No catalogue gives the nose radius; these are guesses for a body that tapers
# to a nose 7 to 12 mm across.
METALLURGICAL_OBJECTIVES: tuple[Objective, ...] = (
    Objective("5x", 5, 0.15, 20000.0, 6000.0),
    Objective("10x", 10, 0.30, 11000.0, 5000.0),
    Objective("20x", 20, 0.45, 3100.0, 4000.0),
    Objective("50x", 50, 0.80, 1000.0, 3500.0),
    Objective("100x", 100, 0.90, 1000.0, 3500.0),
)

# --- illustrative constants ---------------------------------------------------

MIRROR_SATURATION_MS = 1.0
"""Exposure at which a perfect mirror just saturates the camera, in brightfield
with the 5x objective and the lamp at full. With it, a ceramic that reflects 12%
sits at 60% of full scale at 10 ms and half lamp."""
FULL_WELL_ELECTRONS = 10000.0
READ_NOISE_ELECTRONS = 3.0

ILLUMINATION_FALLOFF = 0.08
"""How much dimmer the corners of the frame are than its centre. Fixed to the
camera, not the specimen, as on an instrument with no flat-field correction."""
STAGE_RETURN = 0.002
"""Fraction of the lamp's light a matt black stage plate sends back into the
objective: it scatters what little it reflects in all directions. About 1% of
full scale when the ceramic beside it is well exposed."""
PORE_DARKNESS = 0.12
"""A pore returns this fraction of what the matrix around it returns."""
GROOVE_DARKENING = 0.3
"""How much of the light a sample loses when the most strongly etched groove
covers it, in brightfield. Chosen so that a groove in sharp focus at 100x stays
above 0.7 of the matrix level, clear of the pores: at 0.55 the grooves were as
dark as pore edges and no threshold could measure porosity on an etched
specimen at 50x or 100x."""
SCRATCH_DARKENING = 0.25
"""The same for the deepest scratch."""
EDGE_GAP_DARKENING = 0.7
"""And for the gap between the resin and the specimen."""
MOTTLE = 0.015
"""Slow brightness variation of a translucent ceramic, as a fraction of its level."""

# Darkfield and crossed polarisers, as fractions of the matrix's brightfield
# level. Flat surface returns almost nothing; edges and slopes scatter light in.
# With these, a pore rim in darkfield needs about 11 times the brightfield
# exposure to reach the same level. That ratio is a plausible choice: accounts
# of real instruments range from ten to a hundred.
DARKFIELD = {
    "flat": 0.004,
    "second_phase": 0.008,
    "boundary": 0.025,
    "scratch": 0.1,
    "pore_rim": 0.09,
    "pore": 0.012,
    "edge_gap": 0.09,
}
DARKFIELD_RESIN = 0.02
"""Translucent resin glows a little in darkfield."""
DARKFIELD_STAGE = 0.002
POLARISED = {"flat": 0.012, "pore": 0.08, "scratch": 0.03}
"""Leak through crossed polarisers, and the depolarised light from pores (and
the gap round the specimen) and from scratches. Pores are the brightest thing
in a polarised frame of a ceramic."""
POLARISED_MOTTLE = 0.25
"""The glow from below the surface of a translucent ceramic is uneven by about
this fraction of itself."""
POLARISED_RESIN = 0.02
POLARISED_STAGE = 0.004


def field_of_view_um(
    objective: Objective, sensor_shape: tuple[int, int] = (768, 1024), pixel_pitch_um: float = 3.45
) -> tuple[float, float]:
    """Width (along x) and height (along y) of the specimen seen by the camera."""
    pixel = pixel_pitch_um / objective.magnification
    return sensor_shape[1] * pixel, sensor_shape[0] * pixel


def blur_sigma_um(objective: Objective, defocus_um) -> np.ndarray:
    """Standard deviation of the blur, on the specimen, at a given distance from focus.

    In focus it is a Gaussian whose full width at half maximum is the Rayleigh
    resolution. Out of focus it widens as sqrt(1 + (defocus / half the depth of
    field)^2): 41% wider at the edge of the depth of field, and far from focus
    the same as the cone of light geometry gives for a small aperture.

    For a large aperture that far-field rate is too slow: the true cone widens
    as NA / sqrt(1 - NA^2), which is 1.6 times faster at 50x (NA 0.8) and 2.2
    times at 100x (NA 0.9). So several micrometres out of focus, those two
    objectives still show smudges of pores that a real one would have lost.
    Within a micrometre or two of focus, where focusing is decided, the
    difference is small.
    """
    in_focus = objective.resolution_um / 2.3548
    depths_of_field = np.asarray(defocus_um, float) / objective.depth_of_field_um
    return in_focus * np.sqrt(1.0 + (2.0 * depths_of_field) ** 2)


def exposure_for(
    sample: Sample,
    objective: Objective,
    contrast: str = "Brightfield",
    *,
    lamp_fraction: float = 1.0,
    level: float = 0.65,
) -> float:
    """Exposure in ms that puts the contrast method's reference tone at `level`
    of full scale at the centre of the frame.

    The reference tone is the dense matrix in brightfield, a fully lit pore rim
    in darkfield, and between crossed polarisers the brightest of a pore and
    the best-placed grain. In the last two most of the frame is far darker
    than that, as it should be.
    """
    material = sample.material
    tone = {
        "Brightfield": 1.0,
        "Darkfield": DARKFIELD["pore_rim"],
        "Polarised": max(POLARISED["pore"], POLARISED["flat"] + material.subsurface_glow + material.anisotropy),
    }[_checked(contrast)]
    if lamp_fraction <= 0:
        raise ValueError("no exposure is long enough with the lamp off")
    returned = lamp_fraction * objective.brightness * sample.material.reflectance * tone
    return level * MIRROR_SATURATION_MS / returned


def render(
    sample: Sample,
    *,
    x_um: float,
    y_um: float,
    z_um: float,
    objective: Objective,
    contrast: str,
    lamp_fraction: float,
    exposure_ms: float,
    shutter_open: bool,
    sensor_shape: tuple[int, int] = (768, 1024),
    pixel_pitch_um: float = 3.45,
    bit_depth: int = 12,
    noise_seed: int | None = None,
) -> np.ndarray:
    """One camera frame, as uint16 counts from 0 to 2^bit_depth - 1.

    (x_um, y_um) is the point of the specimen on the optical axis, at the
    centre of the frame; z_um is the height of the focal plane. Columns run
    with +x and rows with +y. With `noise_seed` None the frame is noiseless;
    with a seed it gets shot and read noise that the same seed reproduces.
    """
    _checked(contrast)
    if not 0.0 <= lamp_fraction <= 1.0:
        raise ValueError(f"lamp_fraction must be between 0 and 1, not {lamp_fraction}")
    if exposure_ms < 0:
        raise ValueError(f"exposure_ms cannot be negative, not {exposure_ms}")
    light = lamp_fraction * exposure_ms * objective.brightness / MIRROR_SATURATION_MS
    if not shutter_open or light == 0:
        signal = np.zeros(sensor_shape, np.float32)
    else:
        width, height = field_of_view_um(objective, sensor_shape, pixel_pitch_um)
        fields = sample.fields(x_um, y_um, width, height, sensor_shape)
        picture = response(fields, sample.material, contrast, seed=sample.seed)
        pixel_um = pixel_pitch_um / objective.magnification
        signal = light * _defocused(picture, fields, z_um, objective, pixel_um) * illumination(sensor_shape)

    # `signal` is in units of full scale and may exceed it; the clip comes last,
    # after the noise, the way a full well or a converter clips.
    if noise_seed is not None:
        # Shot noise as a Gaussian of variance equal to the electron count:
        # wrong in shape only where there are a handful of electrons, and there
        # the read noise dominates anyway.
        electrons = signal * FULL_WELL_ELECTRONS
        spread = np.sqrt(electrons + READ_NOISE_ELECTRONS**2)
        draw = np.random.default_rng(noise_seed).standard_normal(signal.shape, dtype=np.float32)
        signal = (electrons + spread * draw) / FULL_WELL_ELECTRONS
    return np.rint(np.clip(signal, 0.0, 1.0) * (2**bit_depth - 1)).astype(np.uint16)


def illumination(sensor_shape: tuple[int, int] = (768, 1024)) -> np.ndarray:
    """How evenly the lamp lights the frame: 1 at the centre, falling with the
    square of the distance from it to 1 - ILLUMINATION_FALLOFF at the corners.

    Dividing a frame by this is a perfect flat-field correction, which a real
    instrument only gets from a picture of a mirror.
    """
    rows, cols = sensor_shape
    half_diagonal_sq = (rows**2 + cols**2) / 4.0
    along_rows = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2) ** 2 / half_diagonal_sq
    along_cols = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2) ** 2 / half_diagonal_sq
    return (1.0 - ILLUMINATION_FALLOFF * along_rows)[:, None] - (ILLUMINATION_FALLOFF * along_cols)[None, :]


def response(fields: Fields, material: Material, contrast: str, *, seed: int = 0) -> np.ndarray:
    """The fraction of the lamp's light each sample sends back to the camera,
    before any blur. This is where the three contrast methods differ."""
    _checked(contrast)
    matrix = material.reflectance
    specimen = fields.region == Region.SPECIMEN
    if contrast == "Brightfield":
        mottle = 1.0 + MOTTLE * _mottle(fields, seed)
        picture = matrix + (material.second_phase_reflectance - matrix) * fields.second_phase
        picture = picture * mottle
        picture *= (1.0 - GROOVE_DARKENING * fields.boundary) * (1.0 - SCRATCH_DARKENING * fields.scratch)
        # A pore replaces whatever was there, grooves included.
        picture += (PORE_DARKNESS * matrix - picture) * fields.pore
        resin = material.resin_reflectance * mottle * (1.0 - EDGE_GAP_DARKENING * fields.edge_gap)
        stage = STAGE_RETURN
    elif contrast == "Darkfield":
        d = DARKFIELD
        picture = d["flat"] + (d["second_phase"] - d["flat"]) * fields.second_phase
        picture = picture + d["boundary"] * fields.boundary + d["scratch"] * fields.scratch
        # Inside a pore only the pore itself scatters: a dim floor and a bright wall.
        picture = picture + (d["pore"] - picture) * fields.pore + d["pore_rim"] * fields.pore_rim
        picture = matrix * picture
        # The step down from the specimen to the resin scatters like a pore's wall.
        resin = matrix * (DARKFIELD_RESIN + d["edge_gap"] * fields.edge_gap)
        stage = matrix * DARKFIELD_STAGE
    else:
        # What crossed polarisers pass: a leak, the glow from below the surface
        # of a translucent matrix, and the little a grain's own birefringence
        # adds, most with its axis at 45 degrees to the polarisers and nothing
        # along them. The second phase is taken as cubic (no response of its
        # own); the yttrium aluminates of a real aluminium nitride are not.
        p = POLARISED
        glow = material.subsurface_glow * (1.0 + POLARISED_MOTTLE * _mottle(fields, seed))
        grains = material.anisotropy * np.sin(2.0 * fields.orientation) ** 2
        picture = p["flat"] + glow + grains * (1.0 - fields.second_phase) + p["scratch"] * fields.scratch
        picture = matrix * (picture + (p["pore"] - picture) * fields.pore)
        resin = matrix * (POLARISED_RESIN + (p["pore"] - POLARISED_RESIN) * fields.edge_gap)
        stage = matrix * POLARISED_STAGE
    if not specimen.all():
        picture = np.where(specimen, picture, np.where(fields.region == Region.RESIN, resin, stage))
    return picture.astype(np.float32)


def _checked(contrast: str) -> str:
    if contrast not in CONTRASTS:
        raise ValueError(f"unknown contrast method {contrast!r}; choose from {', '.join(CONTRASTS)}")
    return contrast


def _mottle(fields: Fields, seed: int) -> np.ndarray:
    """A smooth pattern between about -1 and 1, fixed to the specimen, 20 to 80 um across."""
    rng = np.random.default_rng([seed & 0xFFFFFFFF, 0x6D6F74])
    count = 6
    wavelength = 20.0 * 4.0 ** rng.random(count)
    heading = 2 * np.pi * rng.random(count)
    phase = 2 * np.pi * rng.random(count)
    kx = (2 * np.pi * np.cos(heading) / wavelength)[:, None, None]
    ky = (2 * np.pi * np.sin(heading) / wavelength)[:, None, None]
    along_x = kx * fields.x_um[None, None, :] + phase[:, None, None]
    along_y = ky * fields.y_um[None, :, None]
    # sin(a + b) as sin a cos b + cos a sin b: a sine per axis, not per pixel.
    in_x = np.concatenate([np.sin(along_x), np.cos(along_x)])
    in_y = np.concatenate([np.cos(along_y), np.sin(along_y)])
    return (np.einsum("k...,k...->...", in_x, in_y) * (1.4 / count)).astype(np.float32)


def _defocused(
    picture: np.ndarray, fields: Fields, z_um: float, objective: Objective, pixel_um: float
) -> np.ndarray:
    """Blur each part of the picture by as much as its own distance from focus calls for.

    The frame is split by what is under it (specimen, resin, stage plate) and,
    where the surface height varies by more than the blur can ignore, into
    bands of height. Each part is blurred on its own and the parts are added,
    which is how light from surfaces at different depths adds up in a camera.
    """
    widest = max(picture.shape) / 4.0  # beyond this a frame is featureless anyway

    def sigma(defocus_um: float) -> float:
        return min(float(blur_sigma_um(objective, defocus_um)) / pixel_um, widest)

    defocus = z_um - fields.surface_um
    out = np.zeros(picture.shape, np.float32)
    for code in Region:
        part = fields.region == code
        if not part.any():
            continue
        whole = bool(part.all())
        here = defocus if whole else defocus[part]
        layer = picture if whole else np.where(part, picture, np.float32(0.0))
        if code == Region.STAGE:
            # The plate is featureless, so blurring it changes nothing except
            # at the rim of the mount, where it would spread the plate's light
            # over the resin as if the mount were glass.
            out += layer
            continue
        lo, hi = float(here.min()), float(here.max())
        sharpest = sigma(0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi)))
        softest = sigma(max(abs(lo), abs(hi)))
        if softest <= 1.2 * sharpest + 0.1:
            out += _blur(layer, sigma(float(here.mean())))
            continue
        bands = min(9, math.ceil((hi - lo) / (objective.depth_of_field_um / 2)) + 1)
        step = (hi - lo) / (bands - 1)
        for centre in np.linspace(lo, hi, bands):
            weight = np.clip(1.0 - np.abs(defocus - centre) / step, 0.0, 1.0).astype(np.float32)
            out += _blur(layer * weight, sigma(float(centre)))
    return out


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation `sigma` pixels.

    A wide blur is done on a shrunken copy and stretched back, which costs the
    same whatever the width. The shrinking and stretching blur a little
    themselves, and the Gaussian is narrowed to make up for it.
    """
    if sigma < 0.05:
        return image
    if sigma <= 4.0:
        return ndimage.gaussian_filter(image, sigma, mode="nearest", truncate=3.5)
    factor = int(sigma / 2.0)
    rows, cols = image.shape
    # One block of margin all round, so the frame's own edge is interpolated
    # like any other pixel, plus whatever makes the size a whole number of blocks.
    margins = ((factor, factor + -rows % factor), (factor, factor + -cols % factor))
    padded = np.pad(image, margins, mode="edge")
    blocks = (padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    small = padded.reshape(blocks).mean(axis=(1, 3))
    # Averaging blocks of `factor` adds factor^2 / 12 of variance and linear
    # stretching about factor^2 / 6: together 0.25 in shrunken pixels squared.
    remaining = math.sqrt(sigma**2 / factor**2 - 0.25)
    small = ndimage.gaussian_filter(small, remaining, mode="nearest", truncate=3.5)

    def stretch(values, first, size, axis):
        at = (first + np.arange(size) + 0.5) / factor - 0.5
        lower = np.floor(at).astype(int)
        t = (at - lower).astype(np.float32)
        t = t[:, None] if axis == 0 else t[None, :]
        return np.take(values, lower, axis) * (1 - t) + np.take(values, lower + 1, axis) * t

    return stretch(stretch(small, factor, rows, 0), factor, cols, 1)
