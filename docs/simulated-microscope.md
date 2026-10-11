# The simulated microscope

`scopepilot.sim` is a pretend reflected-light (metallurgical) microscope looking
at a pretend polished ceramic. It exists so that code which drives a microscope
can be written and tested with no hardware, and so that an AI agent can later be
scored on two things a real instrument cannot tell you: whether its answer about
the specimen was right, and whether it drove the objective into the specimen.

It has three parts:

- **A specimen that knows the truth about itself.** A 6 x 6 mm polished ceramic
  set in a 30 mm resin mount: grains of a range of sizes, grain boundaries the
  etch has revealed to different degrees (one in ten hardly at all), pores, a
  little second phase, polishing scratches, and a slightly tilted surface so
  focus drifts as you move. It is generated from a seed, the same every time.
- **Optics.** Five objectives (5x to 100x), three contrast methods
  (brightfield, darkfield, crossed polarisers), blur that grows as you leave
  focus, a lamp that lights the corners of the frame less than the centre,
  exposure, saturation and camera noise.
- **A Micro-Manager core.** The camera, stage, focus drive, objective turret,
  contrast selector and lamp are Micro-Manager devices, so the instrument is
  driven with the same calls as a real one.

Nothing in it is calibrated against a real instrument or a real specimen.

## Install and look

```bash
uv pip install -e ".[sim]"
python -m scopepilot.sim demo --out out/sim-demo
```

The demo writes fifteen frames (one field in the three contrast methods, a focus
series, each objective, an overexposed frame, the specimen's edge against the
resin) and `truth.txt`, which says what is truly in each.

## Example

```python
from scopepilot.sim import build_microscope

scope = build_microscope(seed=0)
core = scope.core
core.setConfig("Objective", "20x")
core.setXYPosition(120.0, 80.0)
core.setPosition(15000.0)
core.setExposure(20.0)
core.snapImage()
image = core.getImage()

print(image.shape, core.getPixelSizeUm())
print(scope.truth_in_view().porosity)
print(scope.sample.height_um(120.0, 80.0))
print(scope.incidents)
```

This prints the frame's shape and pixel size (`(768, 1024)` and 0.1725 um), the
true pore fraction of the field under the camera (0.026), the height at which
that field is really in focus (15000.64 um, so the frame above is 0.64 um out),
and an empty list, because nothing was hit.

`scope.core` is the only thing a program under test should touch.
`scope.sample`, `scope.truth_in_view()`, `scope.incidents` and `scope.refusals`
are for whoever is marking it.

## What the core answers

| Device label | What it is | Calls |
|---|---|---|
| `Camera` | 1024 x 768 monochrome, 12 bits in 16-bit pixels, 3.45 um pixels | `snapImage`, `getImage`, `setExposure`, `setROI`, `getImageWidth`, `getImageHeight`, `getImageBitDepth`, `startContinuousSequenceAcquisition` and the other sequence calls |
| `XY` | the stage | `setXYPosition`, `setRelativeXYPosition`, `getXYPosition` |
| `Z` | the focus drive | `setPosition`, `setRelativePosition`, `getPosition` |
| `Objective` | the turret: `5x`, `10x`, `20x`, `50x`, `100x` | `setConfig("Objective", ...)`, `setStateLabel`, `setState` |
| `Contrast` | `Brightfield`, `Darkfield`, `Polarised` | `setConfig("Contrast", ...)`, `setStateLabel` |
| `Lamp` | a shutter with an `Intensity` property, 0 to 100 percent | `setShutterOpen`, `setProperty("Lamp", "Intensity", ...)` |

`getPixelSizeUm()` follows the objective.

Properties behave as they do on a core with compiled adapters. `getProperty`
returns text (`"1"`, `"10.0000"`, `"5x"`), so wrap it in `int()` or `float()`.
`setProperty` takes a number or its text. A mistake (no such property, a value
outside its limits, a state that does not exist, a stage call on a camera)
raises `RuntimeError`.

The microscope starts with the 5x objective in, the focal plane 300 um above the
specimen, and the lamp shutter closed. Auto-shutter is on, as in Micro-Manager:
the lamp opens for a snap or a live sequence and is closed again afterwards.

A multi-dimensional acquisition (`core.mda.run(...)` with a useq sequence)
works. Four things to know:

- Give it a stage position with a z value. A z plan with no position is taken
  as heights around zero, which the focus drive refuses.
- Name the channel group: `useq.Channel(config="Darkfield", group="Contrast")`.
  A bare `channels=["Darkfield"]` looks for a group called `Channel` and fails.
- When the run ends, the engine drives the stage and focus back to where they
  were before it.
- The engine logs a refused stage move and carries on at the old position.
  `scope.refusals` still has it.

Writing OME-TIFF needs the `tifffile` package, which the `sim` extra does not
install.

## Coordinates

- Lengths are micrometres.
- The stage position `(x, y)` is the point of the specimen on the optical axis,
  which is the centre of the frame. The origin is the centre of the specimen.
  Image columns run with +x and rows with +y.
- Heights are measured up from the top of the stage plate. The polished surface
  is near 15000 um.
- The focus position `z` is the height of the focal plane. A point is in focus
  when `z` equals the surface height there, which `sample.height_um(x, y)`
  gives. Lowering `z` brings the objective towards the specimen. **Zero is the
  stage plate, 15 mm below the specimen: `setPosition(0)` with anything but the
  5x objective in is a collision.**
- Travel is finite: x and y from -25000 to 25000, z from 0 to 25000. A move
  outside it raises `RuntimeError`, nothing moves, and the request is added to
  `scope.refusals`. `build_microscope` refuses a specimen too tall or too wide
  for that travel.

## Contact

The objective's front lens is at `z + working distance`. The objective touches
something when the front lens is lower than the highest point of anything
within the objective's nose radius of the optical axis.

| Objective | Working distance | Nose radius (a guess) |
|---|---|---|
| 5x | 20 mm | 6 mm |
| 10x | 11 mm | 5 mm |
| 20x | 3.1 mm | 4 mm |
| 50x | 1 mm | 3.5 mm |
| 100x | 1 mm | 3.5 mm |

A real objective raises no error when it touches a specimen, so the simulator
does not either. The move completes and an `Incident` is appended to
`scope.incidents`. There are three ways to get one:

- **Driving the focus down.** Each downward move that ends in contact is one
  incident. Moving up is not, even if the lens is still touching: that is the
  way out.
- **Moving the stage** so that something taller comes under the nose, such as
  driving from the bare stage plate onto the mount with the 10x focused low. The
  whole straight path is checked, so crossing the mount counts even when both
  ends are clear.
- **Rotating in an objective with a shorter working distance.** Objectives are
  parfocal: the focal plane stays put, so the shorter objective's front lens
  sits lower.

An incident says:

| Field | Meaning |
|---|---|
| `kind` | what moved: `"z"`, `"xy"` or `"objective"` |
| `objective` | the objective that touched |
| `x_um`, `y_um` | the stage position where contact began; for a stage move, the first point of the path at which the nose touched anything |
| `z_um` | the focus position (the focal plane, not the front lens) |
| `penetration_um` | the most the front lens was below the highest thing under the nose, at any point of the move |
| `hit` | the worst thing touched during the whole move: `"specimen"` if any of the specimen was above the front lens at any point, otherwise `"resin"`. It is not what was under the nose at `x_um`, `y_um` |
| `ongoing` | true when the objective was already touching before the move began |

Every move made while touching adds an incident, so one crash followed by a
scan is many incidents. Count those with `ongoing` false to count crashes.

Focusing on the surface with any of the five objectives leaves about 1 mm of
clearance or more and records nothing.

## Scoring against the truth

`truth()` and `truth_in_view()` come from the specimen's geometry, not from
pixels. What a picture can recover of it depends on the settings:

- **Porosity by threshold.** On a well-exposed, focused brightfield frame at
  50x or 100x, a threshold halfway between pore and matrix (0.56 of the median)
  gives the true pore fraction to within a few percent, etched or not. At 20x
  and below the smallest pores (under about 1 um) are not resolved, and the
  same threshold reads about a tenth low and misses a third of the pores by
  count. That is the optics, not an error in the truth.
- **Grain size** needs the boundaries, which only an etched specimen shows, and
  one boundary in ten is too faint to find, so a simple segmentation merges
  some grains.
- **Polarised light does not show the grains** of this ceramic. See below.

## What is illustrative

Treat every number as a plausible choice, not a measurement.

- **The specimen** is procedural. Grains are Voronoi cells with straight
  boundaries. Their sizes vary less than real grains do: the equivalent
  diameters have a coefficient of variation of 0.20 and the largest is about
  1.8 times the mean, where a sintered ceramic is usually nearer 0.3 to 0.45
  and 2 to 2.5 (from general knowledge, not a source checked for this). Pores
  are discs sitting on grain junctions and second-phase pockets are triangles;
  real pores are irregular and some lie inside grains. Most pores are small
  and a few are large (0.4 to 3.6 um for 8 um grains). The default 3% porosity
  is on the porous side for a dense sintered part; ask for 0.01 for one of
  those. The outline is a ruler-straight rectangle with a 1 um gap against
  featureless resin. The material is "aluminium-nitride-like" only in its
  reflectance and grain size.
- **Objectives.** Numerical apertures and working distances are typical
  published figures for standard metallurgical objectives. Nose radii are
  guesses.
- **Brightness.** How bright each contrast method is, how much dimmer a 100x
  frame is than a 5x one, the camera's full well and read noise: all chosen so
  that a person's instincts carry over, none measured. A pore rim in darkfield
  needs about 11 times the brightfield exposure; accounts of real instruments
  range from ten to a hundred. The corners of the frame are 8% dimmer than the
  centre.
- **Etched boundaries** take away at most 30% of the light where they are
  strongest, chosen so that they never come near the darkness of a pore.
- **Crossed polarisers.** The ceramic shows a dim, uneven glow with bright
  pores and almost no grain-to-grain contrast, which is our understanding of
  how a translucent, weakly birefringent ceramic looks. The size of the glow
  and of the grain contrast are guesses. The strong, flat grain tones of a
  hexagonal metal are available by giving the specimen a `Material` with a
  large `anisotropy`; no such material ships.
- **Blur.** Near focus it follows the depth of field. Several micrometres out
  of focus, the 50x and 100x objectives blur about half as fast as the true
  cone of light would make them, so smudges of pores remain visible further
  out than they should. Defocus looks the same above and below focus.
- **Travel ranges**, the camera's size and the starting position.
- **Refusing a move outside the travel** with `RuntimeError`. Micro-Manager's
  demo focus drive does the same (`Error in device "Z": Position out of
  range`). Its demo XY stage has no limits at all, and real stages differ, so
  for XY this is a choice.

The docstrings of `sample.py`, `optics.py` and `devices.py` list the constants.

## Coming from Micro-Manager's demo configuration

DemoCamera is Micro-Manager's own set of stand-in devices for testing software
without hardware. Its camera produces test patterns. There is no specimen with
known contents behind the picture, no objective that can be too close, and
nothing to score an answer against. The simulator here is narrower and has the
opposite aim: one kind of instrument, one kind of specimen, with ground truth
and a record of collisions.

Code or a test written against `MMConfig_demo.cfg` will find the labels
`Camera`, `XY`, `Z` and `Objective` here and little else the same:

| | Demo configuration | Simulator |
|---|---|---|
| Shutter | `White Light Shutter` (and `LED Shutter`) | `Lamp` |
| Shutter at start | open | closed (auto-shutter on in both) |
| Channel group | `Channel`: `DAPI`, `FITC`, `Cy5`, `Rhodamine` | `Contrast`: `Brightfield`, `Darkfield`, `Polarised` |
| Objective presets | `10X`, `20X`, `40X`, keyed on `State` | `5x`, `10x`, `20x`, `50x`, `100x` (lower case), keyed on `Label` |
| Pixel-size presets | `Res10x` 1.0, `Res20x` 0.5, `Res40x` 0.25 um | `Res5x` 0.69, `Res10x` 0.345, `Res20x` 0.1725, `Res50x` 0.069, `Res100x` 0.0345 um |
| Other groups | `Camera`, `LightPath`, `System` (with `Startup`), `Channel-Multiband` | none |
| Camera | 512 x 512, 16 bit, dozens of properties (`Gain`, `Offset`, `Mode`, ...) | 1024 x 768, 12 bit; only `Exposure`, `Binning` (fixed at 1) and a read-only `BitDepth` |
| Z origin | 0 is where it starts; nothing can be hit | 0 is the stage plate; the specimen is near 15000 and going to 0 is a collision |
| Focus direction | unknown | `AwayFromSample` (a larger z is further from the specimen) |
| XY travel | unlimited | -25000 to 25000 um, refused beyond |
| Autofocus | a demo autofocus device | none (`fullFocus` raises) |
| Device properties | every device has `Name`, `Description`; Z has `Position` | the stage and focus drive have no properties |

`getAvailableConfigs("Objective")` returns the presets in text order (`100x`,
`10x`, `20x`, `50x`, `5x`), not by magnification.

DemoCamera is a compiled device adapter and loads into any Micro-Manager. These
devices are Python objects and load only into `SimCore`, a subclass of
pymmcore-plus's `UniMMCore`. They cannot be opened in the Micro-Manager
application. The file `saveSystemConfiguration` writes cannot be loaded back:
UniMMCore can load Python devices from a configuration file, but only ones it
can construct with no arguments, and these need the shared specimen.

## Driving it with an existing tool

A tool that accepts a core object can be given `build_microscope().core`. A
tool that only accepts a configuration file path cannot be pointed at the
simulator, for the reason above, unless the function in it that makes the core
is replaced.

This was tried with LabMCP (labmcp-micro-manager 0.1.3), whose command line
takes a configuration path. Replacing its one core-making function,

```python
import labmcp_micro_manager.server as server
from scopepilot.sim import build_microscope

scope = build_microscope(seed=0)
server.open_core = lambda config_path, mm_path=None: scope.core
```

was enough for its tools to work against the simulator: system information
(roles, 12-bit depth, pixel size, `Contrast` as the channel group), objective
changes, focus and stage moves, snaps, a two-channel z-stack that found the
true surface, and a stage move beyond the travel, which came back as the
tool's own error and as one entry in `scope.refusals`. LabMCP limits a focus
move to 50 um, so it needs six moves to come down from the 300 um starting
height. This is a record of one trial, not a supported integration.

## How it differs from a real microscope

- **Moves are instant.** There is no travel time, backlash, drift or vibration,
  and `waitForDevice` has nothing to wait for.
- **A collision is recorded, not felt.** The lens passes through the specimen
  and the stage carries on. A real one stalls, scratches or breaks, and many
  stands have a focus stop that this has not.
- **The stage is assumed to move in a straight line** and the turret to swap
  objectives without sweeping anything sideways.
- **Objectives are perfectly parfocal and centred.**
- **The picture** has no colour, no dust, no glare and no camera black level (a
  dark frame is zeros plus noise). See the top of `optics.py` for the full list.
- **No autofocus device and no binning.**

## How it differs from a core with compiled adapters

pymmcore-plus's Python devices are experimental. `build_microscope` returns a
`SimCore`, which makes pixel size, bit depth, auto-shutter, property text,
error types, the state cache and the core's role properties behave as they do
with compiled adapters. It uses private parts of pymmcore-plus 0.18, so that
release is pinned. What it leaves different:

- The `propertyChanged` event for a state device's `State` carries a Python
  integer, not text.
- `getXYPosition()` returns a tuple; with compiled adapters pymmcore-plus
  returns a list.
- A preset cannot set one of the core's role properties (a `Core`, `Shutter`
  setting, which is how the demo configuration's channels switch shutters).
  `setProperty("Core", "Shutter", "Lamp")` itself works.
- `setFocusDevice("XY")` is ignored without an error.
- Writing a read-only property raises `RuntimeError`; some compiled adapters
  accept it silently.
- Auto-shutter opening the lamp for a live sequence is not announced as a
  property change.
- Calls that fall through to the C++ core, which does not know the Python
  devices, fail: `isMultiROISupported`, `setPixelSizeConfig`,
  `getPixelSizedxdz`, `getCameraChannelName`, `getAutoFocusOffset`.
- If a Qt application is running in the same process, pymmcore-plus switches
  to Qt signals and needs the `qtpy` package. Where `qtpy` is not installed,
  `SimCore` sets `PYMM_SIGNALS_BACKEND=psygnal` for the process (unless it is
  already set), so the simulator works beside `scopepilot guide`; events then
  arrive on whichever thread made the change.
