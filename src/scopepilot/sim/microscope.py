"""The pretend microscope, assembled: a Micro-Manager core with the devices loaded.

`build_microscope()` returns a `SimMicroscope`. Its `.core` is driven with the
same calls as a core that has real hardware behind it. The rest of it is what
real hardware cannot give: the true answers about the field under the camera,
and a list of the times the objective touched the specimen.

pymmcore-plus's UniMMCore is experimental, and release 0.18 leaves several
things undone, or done differently from a core with compiled device adapters,
for devices written in Python. `SimCore` puts them right, with private parts of
that release, so the dependency is pinned below 0.19 and each has a test:

- the pixel size never follows the objective (it stays 0);
- auto-shutter does not open a Python shutter, for a snap or for a sequence;
- the image bit depth is the 16 of the pixel type, not the camera's 12;
- property values come back as Python numbers, and text values (which is what
  a configuration file and most Micro-Manager code pass) are refused;
- a mistake raises KeyError or ValueError where Micro-Manager raises RuntimeError;
- the system-state cache goes stale, so image metadata can name the wrong objective;
- the core's own Camera, Shutter, Focus and XYStage properties are empty;
- nothing announces a change of pixel size or of a config group's preset.

What it does not put right is listed in docs/simulated-microscope.md.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymmcore
from pymmcore_plus.core import Configuration, Keyword, PropertyType
from pymmcore_plus.experimental.unicore import UniMMCore

from scopepilot.sim import devices
from scopepilot.sim.devices import Incident, Refusal, World, property_text
from scopepilot.sim.optics import CONTRASTS, Objective
from scopepilot.sim.sample import Sample, Truth

START_ABOVE_SURFACE_UM = 300.0
"""Where the focal plane starts, above the centre of the specimen: far enough
that the picture is plainly out of focus, with every objective still clear."""
SEQUENCE_BUFFER_MB = 64
"""Room for about 40 frames. UniMMCore reserves 1 GB by default."""

_ROLES = {
    Keyword.CoreCamera: "CameraDevice",
    Keyword.CoreShutter: "ShutterDevice",
    Keyword.CoreFocus: "FocusDevice",
    Keyword.CoreXYStage: "XYStageDevice",
}
"""The core's own properties that name the device in each role, and the pair of
calls (get... and set...) that each stands for."""


@contextmanager
def _as_device_error(label: object) -> Iterator[None]:
    """Report a mistake the way Micro-Manager does: as RuntimeError.

    UniMMCore lets a Python device's KeyError (no such property) and ValueError
    (value not allowed, wrong kind of device) through as they are, so error
    handling written against this would not catch the same mistake on real
    hardware, nor the other way round.
    """
    try:
        yield
    except (KeyError, ValueError) as error:
        reason = error.args[0] if error.args else error
        raise RuntimeError(f'Error in device "{label}": {reason}') from error


def _reporting_as_a_device(name: str):
    inherited = getattr(UniMMCore, name)

    @functools.wraps(inherited)
    def method(self, *args: Any, **kwargs: Any) -> Any:
        with _as_device_error(args[0] if args and isinstance(args[0], str) else ""):
            return inherited(self, *args, **kwargs)

    return method


class SimCore(UniMMCore):
    """UniMMCore with what release 0.18 leaves undone for Python devices."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Once a Qt application exists in the process, pymmcore-plus switches to
        # Qt signals and imports qtpy, which is not among its requirements nor
        # ours. Without qtpy, say so first, unless the caller has chosen.
        if importlib.util.find_spec("qtpy") is None:
            os.environ.setdefault("PYMM_SIGNALS_BACKEND", "psygnal")
        super().__init__(*args, **kwargs)
        self._light_lock = threading.Lock()
        self._sequence = 0
        self._lit: Any = None

    # --- properties ---------------------------------------------------------------

    def getProperty(self, label: str, propName: str) -> Any:
        """Text, as every Micro-Manager adapter returns: "1", "10.0000", "5x"."""
        if label == Keyword.CoreDevice and propName in _ROLES:
            return getattr(self, f"get{_ROLES[propName]}")()
        with _as_device_error(label):
            value = super().getProperty(label, propName)
        return property_text(value) if label in self._pydevices else value

    def getPropertyFromCache(self, deviceLabel: str, propName: str) -> Any:
        """The live value. Python devices answer at once, so nothing is gained
        by a cache, and UniMMCore's is not kept up when a device changes."""
        if deviceLabel in self._pydevices or deviceLabel == Keyword.CoreDevice:
            return self.getProperty(deviceLabel, propName)
        return super().getPropertyFromCache(deviceLabel, propName)

    def setProperty(self, label: str, propName: str, propValue: bool | float | int | str) -> None:
        """Accepts text for a numeric property, as Micro-Manager does."""
        if label == Keyword.CoreDevice and propName in _ROLES:
            return getattr(self, f"set{_ROLES[propName]}")(str(propValue))
        if label not in self._pydevices:
            return super().setProperty(label, propName, propValue)
        with self._announcing(), _as_device_error(label):
            if isinstance(propValue, str):
                kind = self.getPropertyType(label, propName)
                if kind == PropertyType.Integer:
                    propValue = int(float(propValue))
                elif kind == PropertyType.Float:
                    propValue = float(propValue)
            super().setProperty(label, propName, propValue)

    def setState(self, stateDeviceLabel: str, state: int) -> None:
        with self._announcing(), _as_device_error(stateDeviceLabel):
            super().setState(stateDeviceLabel, state)

    def setStateLabel(self, stateDeviceLabel: str, stateLabel: str) -> None:
        with self._announcing(), _as_device_error(stateDeviceLabel):
            super().setStateLabel(stateDeviceLabel, stateLabel)

    @contextmanager
    def _announcing(self) -> Iterator[None]:
        """Tell listeners what a change did to the config groups and the pixel
        size, as a real core does. UniMMCore announces only the property."""
        groups = {group: self.getCurrentConfig(group) for group in self.getAvailableConfigGroups()}
        pixel_um = self.getPixelSizeUm()
        yield
        for group, was in groups.items():
            if (now := self.getCurrentConfig(group)) != was:
                self.events.configGroupChanged.emit(group, now)
        if (now_um := self.getPixelSizeUm()) != pixel_um:
            self.events.pixelSizeChanged.emit(now_um)
            self.events.pixelSizeAffineChanged.emit(*self.getPixelSizeAffine())

    def _getSystemStateCache(self, cache: bool, native: bool = False) -> Configuration | pymmcore.Configuration:
        """The system state, with Python devices read live and the roles filled in.

        This is what image metadata is made from. UniMMCore falls back to the
        live value for anything missing from its cache, so emptying the cache
        first is enough to keep a stale objective out of the metadata.
        """
        self._state_cache.clear()
        state = super()._getSystemStateCache(cache, native=True)
        for role, device in _ROLES.items():
            current = getattr(self, f"get{device}")()
            state.addSetting(pymmcore.PropertySetting(Keyword.CoreDevice, role, current))
        return state if native else Configuration.from_configuration(state)

    def setShutterOpen(self, *args: Any) -> None:
        """UniMMCore announces a change of the current shutter under an empty label."""
        if len(args) == 1:
            args = (self.getShutterDevice(), *args)
        super().setShutterOpen(*args)

    # --- pixel size and bit depth ---------------------------------------------------

    def getCurrentPixelSizeConfig(self, cached: bool = False) -> str:
        """The pixel-size preset whose settings match the devices as they are now.

        UniMMCore keeps these presets in the C++ core, which cannot see Python
        devices and so never finds a match. `cached` is ignored: UniMMCore's own
        cache is not updated when a state device changes.
        """
        for preset in self.getAvailablePixelSizeConfigs():
            wanted = self.getPixelSizeConfigData(preset, native=True)
            try:
                if all(
                    str(self.getProperty(setting.getDeviceLabel(), setting.getPropertyName()))
                    == setting.getPropertyValue()
                    for setting in (wanted.getSetting(i) for i in range(wanted.size()))
                ):
                    return preset
            except Exception:  # a preset naming a device that is not loaded matches nothing
                continue
        return ""

    def getPixelSizeUm(self, cached: bool = False) -> float:
        preset = self.getCurrentPixelSizeConfig(cached)
        return self.getPixelSizeUmByID(preset) * self._binning_over_magnification() if preset else 0.0

    def getPixelSizeAffine(self, cached: bool = False) -> tuple[float, ...]:
        preset = self.getCurrentPixelSizeConfig(cached)
        if not preset:
            return (0.0,) * 6
        factor = self._binning_over_magnification()
        return tuple(value * factor for value in self.getPixelSizeAffineByID(preset))

    def _binning_over_magnification(self) -> float:
        camera = self._py_camera()
        binning = float(camera.get_property_value(Keyword.Binning)) if camera is not None else 1.0
        return binning / self.getMagnificationFactor()

    def getImageBitDepth(self) -> int:
        """The camera's own bit depth where it states one, as a C++ adapter reports it."""
        camera = self._py_camera()
        if camera is not None and camera.has_property("BitDepth"):
            return int(camera.get_property_value("BitDepth"))
        return super().getImageBitDepth()

    # --- auto-shutter ------------------------------------------------------------------

    def _auto_shutter(self) -> Any:
        """The Python shutter to open for an exposure, or None when auto-shutter is off."""
        return self._py_shutter() if self.getAutoShutter() and self.getShutterDevice() else None

    def snapImage(self) -> None:
        if self.isSequenceRunning():  # UniMMCore would quietly snap alongside the sequence
            raise RuntimeError("This operation can not be executed while sequence acquisition is running.")
        super().snapImage()

    def _do_snap_image(self) -> None:
        """Open a Python shutter around the exposure when auto-shutter is on.

        Afterwards it is closed, even if it was open before, which is what the
        C++ core does.
        """
        shutter = self._auto_shutter()
        if shutter is None:
            return super()._do_snap_image()
        with shutter:
            shutter.set_open(True)
        try:
            super()._do_snap_image()
        finally:
            with shutter:
                shutter.set_open(False)

    def _do_start_sequence_acquisition(
        self, cameraLabel: str, numImages: int, intervalMs: float, stopOnOverflow: bool
    ) -> None:
        sequence = self._light()
        super()._do_start_sequence_acquisition(cameraLabel, numImages, intervalMs, stopOnOverflow)
        # A counted sequence ends by itself, and the shutter closes when it does.
        if (acquiring := self._acquisition_thread) is not None:
            threading.Thread(target=lambda: (acquiring.join(), self._darken(sequence)), daemon=True).start()

    def _do_start_continuous_sequence_acquisition(self, intervalMs: float = 0) -> None:
        self._light()
        super()._do_start_continuous_sequence_acquisition(intervalMs)

    def _do_stop_sequence_acquisition(self, cameraLabel: str) -> None:
        super()._do_stop_sequence_acquisition(cameraLabel)
        self._darken()

    def _light(self) -> int:
        """Open the shutter for a sequence if auto-shutter is on; return the sequence's number."""
        with self._light_lock:
            self._sequence += 1
            self._lit = self._auto_shutter()
            if self._lit is not None:
                with self._lit:
                    self._lit.set_open(True)
            return self._sequence

    def _darken(self, sequence: int | None = None) -> None:
        """Close the shutter a sequence opened, unless a later sequence has started since."""
        with self._light_lock:
            if sequence in (None, self._sequence) and self._lit is not None:
                with self._lit:
                    self._lit.set_open(False)
                self._lit = None

    # --- the same mistakes, reported the same way ---------------------------------------

    getState = _reporting_as_a_device("getState")
    getStateLabel = _reporting_as_a_device("getStateLabel")
    getPosition = _reporting_as_a_device("getPosition")
    setPosition = _reporting_as_a_device("setPosition")
    setRelativePosition = _reporting_as_a_device("setRelativePosition")
    getXYPosition = _reporting_as_a_device("getXYPosition")
    getXPosition = _reporting_as_a_device("getXPosition")
    getYPosition = _reporting_as_a_device("getYPosition")
    setXYPosition = _reporting_as_a_device("setXYPosition")
    setRelativeXYPosition = _reporting_as_a_device("setRelativeXYPosition")
    getShutterOpen = _reporting_as_a_device("getShutterOpen")
    getExposure = _reporting_as_a_device("getExposure")
    setExposure = _reporting_as_a_device("setExposure")


@dataclass(frozen=True, eq=False)
class SimMicroscope:
    """A core to drive, and the truth about what it is looking at."""

    core: SimCore
    world: World

    @property
    def sample(self) -> Sample:
        return self.world.sample

    @property
    def objectives(self) -> tuple[Objective, ...]:
        return self.world.objectives

    @property
    def incidents(self) -> list[Incident]:
        """Every time the objective touched something, oldest first."""
        return self.world.incidents

    @property
    def refusals(self) -> list[Refusal]:
        """Every move refused for being outside the travel, oldest first."""
        return self.world.refusals

    def truth_in_view(self) -> Truth:
        """Ground truth for the field the camera sees now, region of interest included."""
        world = self.world
        pixel_um = devices.PIXEL_PITCH_UM / world.objective.magnification
        rows, cols = devices.SENSOR_SHAPE
        left, top, width, height = self.core.getROI()
        return world.sample.truth(
            world.x_um + (left + width / 2 - cols / 2) * pixel_um,
            world.y_um + (top + height / 2 - rows / 2) * pixel_um,
            width * pixel_um,
            height * pixel_um,
        )


def build_microscope(
    sample: Sample | None = None, *, seed: int = 0, core: SimCore | None = None, noise: bool = True
) -> SimMicroscope:
    """A microscope looking at `sample`, or at `Sample.polished_ceramic(seed)`.

    It starts the way a careful person leaves one: lowest magnification, focal
    plane a few hundred micrometres above the specimen, lamp shutter closed.
    Auto-shutter is on, as in Micro-Manager, so a snap still gets light.

    With `noise` off, frames are noiseless. With it on, the noise is seeded by
    the frame count, so the same calls in the same order give the same frames.

    Pass `core` to load the devices into a core of your own, which must be a
    `SimCore`: a plain UniMMCore would report a pixel size of 0.

    The stage and focus drive have fixed travel, so a specimen that does not
    fit under them is refused with ValueError.
    """
    if sample is None:
        sample = Sample.polished_ceramic(seed)
    start_um = sample.height_um(0.0, 0.0) + START_ABOVE_SURFACE_UM
    if start_um > devices.Z_TRAVEL_UM[1]:
        raise ValueError(
            f"the specimen's surface is {sample.height_um(0.0, 0.0):.0f} um above the stage plate; the focus "
            f"drive tops out at {devices.Z_TRAVEL_UM[1]:.0f} um and has to start {START_ABOVE_SURFACE_UM:.0f} um "
            "above the surface"
        )
    if sample.mount_diameter_um / 2 > min(-devices.XY_TRAVEL_UM[0], devices.XY_TRAVEL_UM[1]):
        raise ValueError(
            f"a mount {sample.mount_diameter_um:.0f} um across reaches beyond the stage's travel of "
            f"{devices.XY_TRAVEL_UM[0]:.0f} to {devices.XY_TRAVEL_UM[1]:.0f} um each way from its centre"
        )
    if core is None:
        # Any folder that exists stops the core logging an error about finding
        # no Micro-Manager installation, which this does not need.
        core = SimCore(mm_path=str(Path(__file__).parent))
    elif not isinstance(core, SimCore):
        raise TypeError("core must be a scopepilot.sim.microscope.SimCore")

    world = World(sample, noise=noise, z_um=start_um)
    world.objective_index = min(range(len(world.objectives)), key=lambda i: world.objectives[i].magnification)

    core.loadPyDevice("Camera", devices.Camera(world))
    core.loadPyDevice("XY", devices.XYStage(world))
    core.loadPyDevice("Z", devices.FocusDrive(world))
    core.loadPyDevice("Objective", devices.Turret(world))
    core.loadPyDevice("Contrast", devices.ContrastSelector(world))
    core.loadPyDevice("Lamp", devices.Lamp(world))
    core.initializeAllDevices()
    core.setCameraDevice("Camera")
    # Only once a Python camera is current does this reach the Python buffer;
    # before, UniMMCore hands it to the C++ core and keeps its own 1 GB.
    core.setCircularBufferMemoryFootprint(SEQUENCE_BUFFER_MB)
    core.setXYStageDevice("XY")
    core.setFocusDevice("Z")
    core.setShutterDevice("Lamp")

    for objective in world.objectives:
        name, pixel_um = objective.name, devices.PIXEL_PITCH_UM / objective.magnification
        core.defineConfig("Objective", name, "Objective", "Label", name)
        core.definePixelSizeConfig(f"Res{name}", "Objective", "Label", name)
        core.setPixelSizeUm(f"Res{name}", pixel_um)
        # Without this the affine is the identity, whatever the pixel size.
        core.setPixelSizeAffine(f"Res{name}", (pixel_um, 0.0, 0.0, 0.0, pixel_um, 0.0))
    for contrast in CONTRASTS:
        core.defineConfig("Contrast", contrast, "Contrast", "Label", contrast)
    # The contrast method is the nearest thing here to a fluorescence channel.
    core.setChannelGroup("Contrast")
    # As a real core does at the end of loading a configuration.
    core.updateSystemStateCache()
    return SimMicroscope(core, world)
