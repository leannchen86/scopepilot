"""The pretend microscope's devices, as Micro-Manager sees them.

Six devices written in Python for pymmcore-plus's experimental UniMMCore: a
camera, an XY stage, a focus drive, an objective turret, a contrast selector and
a lamp. A caller talks to them through the ordinary core calls and cannot tell
them from device adapters, except where the notes below say so.

They share one `World`, because that is the only way they can know about each
other: the camera needs the stage position to make a picture, and the stage
needs the objective to know whether it has just hit something. UniMMCore gives
a device no way to ask the core about another device.

Two behaviours are the point of this module:

Finite travel
    A move outside the stage's or the focus drive's range is refused with a
    RuntimeError and nothing moves. RuntimeError is what a C++ adapter's error
    turns into in Python. The one adapter this was compared with is
    Micro-Manager's own demo focus drive, which refuses the same way
    ('Error in device "Z": Position out of range'); the demo XY stage has no
    limits at all, and real stages differ (some stop at the limit instead), so
    for XY "refuse" is a choice, not a measured match.

Contact
    A real objective that touches a specimen raises no error, so nothing here
    does either. The move completes and an `Incident` is appended to
    `World.incidents`.

The travel ranges and the camera are plausible choices, not a real instrument.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from pymmcore_plus.core import FocusDirection
from pymmcore_plus.experimental.unicore import (
    ShutterDevice,
    SimpleCameraDevice,
    StageDevice,
    StateDevice,
    XYStageDevice,
    pymm_property,
)

from scopepilot.sim.optics import CONTRASTS, METALLURGICAL_OBJECTIVES, Objective, render
from scopepilot.sim.sample import Sample

# --- illustrative constants ---------------------------------------------------

XY_TRAVEL_UM = (-25000.0, 25000.0)
"""Travel of each stage axis: 50 mm, enough to drive the whole 30 mm mount out
from under the largest objective."""
Z_TRAVEL_UM = (0.0, 25000.0)
"""Range of the focal plane's height above the stage plate."""
SENSOR_SHAPE = (768, 1024)
"""Rows, columns."""
PIXEL_PITCH_UM = 3.45
BIT_DEPTH = 12


@dataclass(frozen=True)
class Incident:
    """One move that brought the objective's front lens into something."""

    kind: str
    """What moved: "z", "xy" or "objective"."""
    objective: str
    """Name of the objective that touched."""
    x_um: float
    y_um: float
    """Stage position where contact began. For "z" and "objective" that is
    where the stage was; for "xy" it is the first point along the path at which
    the nose touched anything, found to within a micrometre or so."""
    z_um: float
    """Focus drive position (the focal plane, not the front lens)."""
    penetration_um: float
    """The most the front lens was below the highest thing under the nose, at
    any point of the move. When the nose runs sideways into the mount this is
    the height of the step it ran into, not a distance travelled into it."""
    hit: str
    """The worst thing touched during the whole move, not what was under the
    nose at (x_um, y_um): "specimen" when any of the specimen was above the
    front lens at any point of the move, otherwise "resin" (the mount)."""
    ongoing: bool = False
    """True when the objective was already touching before this move began, so
    the move carried an earlier crash on rather than starting a new one. Count
    the incidents with `ongoing` False to count crashes."""


@dataclass(frozen=True)
class Refusal:
    """A move that was asked for and refused because it is outside the travel."""

    device: str
    requested_um: tuple[float, ...]


@dataclass
class World:
    """The state of the pretend instrument, shared by its devices."""

    sample: Sample
    objectives: tuple[Objective, ...] = METALLURGICAL_OBJECTIVES
    noise: bool = True
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = Z_TRAVEL_UM[1]
    objective_index: int = 0
    contrast_index: int = 0
    lamp_open: bool = False
    lamp_percent: float = 50.0
    exposure_ms: float = 10.0
    frames: int = 0
    """Frames taken so far. It seeds the camera noise, so the same sequence of
    calls gives the same frames."""
    incidents: list[Incident] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Nothing on the stage is higher than this. A front lens above it cannot
        # touch anything, which spares the common case a search of the surface.
        self._ceiling_um = self.sample.max_height_um(0.0, 0.0, self.sample.mount_diameter_um / 2) + 1.0

    @property
    def objective(self) -> Objective:
        return self.objectives[self.objective_index]

    @property
    def contrast(self) -> str:
        return CONTRASTS[self.contrast_index]

    def contact(self, x_um: float, y_um: float, z_um: float, objective: Objective) -> tuple[float, bool]:
        """How far the front lens is below the highest thing under the nose (0
        when clear), and whether any of the specimen is above the front lens."""
        front_lens = z_um + objective.working_distance_um
        if front_lens >= self._ceiling_um:
            return 0.0, False
        depth = self.sample.max_height_um(x_um, y_um, objective.front_radius_um) - front_lens
        if depth <= 0:
            return 0.0, False
        ring = objective.front_radius_um * np.sqrt(np.arange(33) / 32.0)[:, None]
        angle = 2 * np.pi * np.arange(96)[None, :] / 96
        x, y = x_um + ring * np.cos(angle), y_um + ring * np.sin(angle)
        width, length = self.sample.specimen_size_um
        on_specimen = (np.abs(x) < width / 2) & (np.abs(y) < length / 2)
        return depth, bool((on_specimen & (self.sample.height_um(x, y) > front_lens)).any())

    def check_z_move(self, target_um: float) -> None:
        """Record an incident if driving the focus down to `target_um` ends in contact.

        Moving up never makes contact, and moving up while still touching is the
        way out of one, so only downward moves are recorded.
        """
        if target_um < self.z_um:
            depth, specimen = self.contact(self.x_um, self.y_um, target_um, self.objective)
            ongoing = depth > 0 and self.contact(self.x_um, self.y_um, self.z_um, self.objective)[0] > 0
            self._record("z", self.x_um, self.y_um, target_um, self.objective, depth, specimen, ongoing)

    def check_xy_move(self, x_um: float, y_um: float) -> None:
        """Record an incident if the straight path to (x_um, y_um) touches anything.

        The path is checked every half nose radius and at its closest approach
        to the centre of the mount, so a move that only clips the mount's rim is
        caught, and so is one that crosses the mount and ends clear of it. A
        move that starts in contact counts, marked `ongoing`: the nose is being
        dragged.
        """
        dx, dy = x_um - self.x_um, y_um - self.y_um
        distance = math.hypot(dx, dy)
        objective = self.objective
        if distance == 0 or self.z_um + objective.working_distance_um >= self._ceiling_um:
            return
        steps = min(48, math.ceil(distance / (objective.front_radius_um / 2)))
        nearest = min(1.0, max(0.0, -(self.x_um * dx + self.y_um * dy) / distance**2))
        along = sorted({*np.linspace(0.0, 1.0, steps + 1).tolist(), nearest})

        def depth_at(t: float) -> tuple[float, bool]:
            return self.contact(self.x_um + t * dx, self.y_um + t * dy, self.z_um, objective)

        found = [depth_at(t) for t in along]
        touching = [index for index, (depth, _) in enumerate(found) if depth > 0]
        if not touching:
            return
        # Where contact began: halve the step before the first touching sample
        # until it is a few thousandths of it long.
        first = along[touching[0]]
        if touching[0] > 0:
            clear = along[touching[0] - 1]
            for _ in range(12):
                middle = (clear + first) / 2
                clear, first = (clear, middle) if depth_at(middle)[0] > 0 else (middle, first)
        deepest = max(depth for depth, _ in found)
        specimen = any(touched for _, touched in found)
        began = (self.x_um + first * dx, self.y_um + first * dy)
        self._record("xy", *began, self.z_um, objective, deepest, specimen, touching[0] == 0)

    def check_objective_change(self, index: int) -> None:
        """Record an incident if the objective at `index` is too long for where the focus is.

        Objectives are parfocal: the focal plane stays where it is, so one with
        a shorter working distance puts its front lens lower.
        """
        if index != self.objective_index:
            depth, specimen = self.contact(self.x_um, self.y_um, self.z_um, self.objectives[index])
            ongoing = depth > 0 and self.contact(self.x_um, self.y_um, self.z_um, self.objective)[0] > 0
            self._record(
                "objective", self.x_um, self.y_um, self.z_um, self.objectives[index], depth, specimen, ongoing
            )

    def _record(
        self,
        kind: str,
        x_um: float,
        y_um: float,
        z_um: float,
        objective: Objective,
        depth: float,
        specimen: bool,
        ongoing: bool,
    ) -> None:
        if depth > 0:
            hit = "specimen" if specimen else "resin"
            self.incidents.append(Incident(kind, objective.name, x_um, y_um, z_um, depth, hit, ongoing))


class _Shared:
    """What every device here needs: the world, and a lock that can be re-entered.

    UniMMCore holds a plain lock on a device for the length of every call to
    it. A listener that asks the core about the same device from inside one of
    its events (a position readout updating itself, say) then waits on itself
    for ever. `_lock` is a private attribute of pymmcore-plus 0.18.
    """

    def _join(self, world: World) -> None:
        self.world = world
        self._lock = threading.RLock()


def property_text(value: object) -> str:
    """A property value as Micro-Manager hands it over: text, a float with four decimals."""
    if isinstance(value, bool):
        return str(int(value))
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _refuse(device: str, world: World, requested: tuple[float, ...], travel: tuple[float, float]) -> None:
    # The MDA engine logs a failed stage move and carries on, so the exception
    # alone would leave no trace of the request there.
    world.refusals.append(Refusal(device, requested))
    target = ", ".join(f"{value:.1f}" for value in requested)
    raise RuntimeError(
        f'Error in device "{device}": position {target} um is outside the travel range '
        f"{travel[0]:.0f} to {travel[1]:.0f} um; nothing moved"
    )


class Camera(_Shared, SimpleCameraDevice):
    """A monochrome camera whose frames come from `optics.render`."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self._join(world)

    def get_exposure(self) -> float:
        return self.world.exposure_ms

    def set_exposure(self, exposure: float) -> None:
        try:  # it arrives as text when set as the "Exposure" property
            exposure = float(exposure)
        except (TypeError, ValueError):
            exposure = float("nan")
        if not exposure >= 0:
            raise RuntimeError(f'Error in device "{self.get_label()}": exposure cannot be {exposure} ms')
        self.world.exposure_ms = exposure
        self.core.events.exposureChanged.emit(self.get_label(), self.world.exposure_ms)

    def sensor_shape(self) -> tuple[int, int]:
        return SENSOR_SHAPE

    def dtype(self) -> type:
        return np.uint16

    @pymm_property(is_read_only=True, name="BitDepth")
    def bit_depth(self) -> int:
        """Counts run from 0 to 2^BitDepth - 1, in 16-bit pixels."""
        return BIT_DEPTH

    def snap(self, buffer: np.ndarray) -> Mapping:
        world = self.world
        seed = [world.sample.seed & 0xFFFFFFFF, world.frames] if world.noise else None
        world.frames += 1
        buffer[:] = render(
            world.sample,
            x_um=world.x_um,
            y_um=world.y_um,
            z_um=world.z_um,
            objective=world.objective,
            contrast=world.contrast,
            lamp_fraction=world.lamp_percent / 100.0,
            exposure_ms=world.exposure_ms,
            shutter_open=world.lamp_open,
            sensor_shape=SENSOR_SHAPE,
            pixel_pitch_um=PIXEL_PITCH_UM,
            bit_depth=BIT_DEPTH,
            noise_seed=seed,
        )
        return {}


class XYStage(_Shared, XYStageDevice):
    """The stage. Its position is the specimen point on the optical axis."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self._join(world)

    def set_position_um(self, x: float, y: float) -> None:
        x, y = float(x), float(y)
        low, high = XY_TRAVEL_UM
        if not (low <= x <= high and low <= y <= high):
            _refuse(self.get_label(), self.world, (x, y), XY_TRAVEL_UM)
        self.world.check_xy_move(x, y)
        self.world.x_um, self.world.y_um = x, y
        # The core does not announce moves of a Python stage; the device must.
        self.core.events.XYStagePositionChanged.emit(self.get_label(), x, y)

    def get_position_um(self) -> tuple[float, float]:
        return (self.world.x_um, self.world.y_um)

    def home(self) -> None:
        """Go to the centre of the travel, which is the centre of the specimen."""
        self.set_position_um(0.0, 0.0)

    def stop(self) -> None:
        """Moves are instant, so there is never one to stop."""

    def set_origin_x(self) -> None:
        raise RuntimeError(f'Error in device "{self.get_label()}": the origin is fixed to the specimen')

    def set_origin_y(self) -> None:
        self.set_origin_x()


class FocusDrive(_Shared, StageDevice):
    """The focus drive. Its position is the height of the focal plane above the
    stage plate, so a smaller number brings the objective closer to the specimen."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self._join(world)
        self._direction = FocusDirection.AwayFromSample

    def set_position_um(self, val: float) -> None:
        val = float(val)
        low, high = Z_TRAVEL_UM
        if not low <= val <= high:
            _refuse(self.get_label(), self.world, (val,), Z_TRAVEL_UM)
        self.world.check_z_move(val)
        self.world.z_um = val
        self.core.events.stagePositionChanged.emit(self.get_label(), val)

    def get_position_um(self) -> float:
        return self.world.z_um

    def get_focus_direction(self) -> FocusDirection:
        return self._direction

    def set_focus_direction(self, sign: int) -> None:
        """Micro-Manager lets a configuration relabel the direction. It is a
        label only: the drive moves the same way whatever it says."""
        self._direction = FocusDirection(int(sign))

    def home(self) -> None:
        """Go to the top of the travel, as far from the specimen as the drive reaches."""
        self.set_position_um(Z_TRAVEL_UM[1])

    def stop(self) -> None:
        """Moves are instant, so there is never one to stop."""

    def set_origin(self) -> None:
        raise RuntimeError(f'Error in device "{self.get_label()}": the origin is fixed to the stage plate')


class Turret(_Shared, StateDevice):
    """The objective nosepiece, one state per objective."""

    def __init__(self, world: World) -> None:
        super().__init__({index: objective.name for index, objective in enumerate(world.objectives)})
        self._join(world)

    def get_state(self) -> int:
        return self.world.objective_index

    def set_state(self, position: int) -> None:
        self.world.check_objective_change(int(position))
        self.world.objective_index = int(position)


class ContrastSelector(_Shared, StateDevice):
    """Brightfield, darkfield or crossed polarisers."""

    def __init__(self, world: World) -> None:
        super().__init__(dict(enumerate(CONTRASTS)))
        self._join(world)

    def get_state(self) -> int:
        return self.world.contrast_index

    def set_state(self, position: int) -> None:
        self.world.contrast_index = int(position)


class Lamp(_Shared, ShutterDevice):
    """The illuminator: a shutter, and an intensity in percent."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self._join(world)

    def get_open(self) -> bool:
        return self.world.lamp_open

    def set_open(self, open: bool) -> None:
        self.world.lamp_open = bool(open)

    # The core announces a "State" property whenever the shutter changes, and a
    # listener that reads it back needs it to exist.
    @pymm_property(allowed_values=[0, 1], name="State")
    def state(self) -> int:
        return int(self.world.lamp_open)

    @state.setter
    def _set_state(self, value: int) -> None:
        self.world.lamp_open = bool(int(value))

    @pymm_property(limits=(0.0, 100.0), name="Intensity")
    def intensity(self) -> float:
        return self.world.lamp_percent

    @intensity.setter
    def _set_intensity(self, value: float) -> None:
        self.world.lamp_percent = float(value)
        self.core.events.propertyChanged.emit(self.get_label(), "Intensity", property_text(float(value)))
