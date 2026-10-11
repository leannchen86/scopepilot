"""`python -m scopepilot.sim demo --out DIR`: a small gallery for a person to look at.

The frames are taken through the Micro-Manager core, the way any other caller
would take them. Two things here a real session could not do: the focus is set
from the specimen's known surface height instead of being found, and each
exposure comes from the model's own arithmetic instead of from a histogram.

Counts are written to 8-bit PNG without stretching, so a dim frame looks dim
and an overexposed one looks white.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from scopepilot.sim.devices import BIT_DEPTH
from scopepilot.sim.microscope import SimMicroscope, build_microscope
from scopepilot.sim.optics import CONTRASTS, exposure_for

FIELD_UM = (120.0, 80.0)
"""An ordinary place on the specimen, away from its edges."""
EDGE_UM = (2970.0, 500.0)
"""Just inside the specimen's +x edge, so the frame is half specimen, half resin."""


def shoot(
    scope: SimMicroscope,
    path: Path,
    *,
    at: tuple[float, float] = FIELD_UM,
    objective: str = "50x",
    contrast: str = "Brightfield",
    defocus_um: float = 0.0,
    overexpose: float = 1.0,
) -> str:
    """Take one frame, save it, and return what is truly in it as text."""
    core, sample = scope.core, scope.sample
    surface = sample.height_um(*at)
    core.setConfig("Objective", objective)
    core.setConfig("Contrast", contrast)
    core.setXYPosition(*at)
    core.setPosition(surface + defocus_um)
    lamp = float(core.getProperty("Lamp", "Intensity"))
    lens = next(o for o in scope.objectives if o.name == objective)
    core.setExposure(overexpose * exposure_for(sample, lens, contrast, lamp_fraction=lamp / 100))
    core.snapImage()
    frame = core.getImage()
    Image.fromarray((frame >> (BIT_DEPTH - 8)).astype("uint8")).save(path)

    truth = scope.truth_in_view()
    pores = truth.pores_um
    largest = f", the largest {pores[:, 2].max():.1f} um across" if len(pores) else ""
    saturated = (frame == 2**BIT_DEPTH - 1).mean()
    return "\n".join(
        [
            path.name,
            f"  {objective}, {contrast}, exposure {core.getExposure():.2f} ms, lamp {lamp:.0f} %,"
            f" pixel {core.getPixelSizeUm():.4f} um, {saturated:.0%} of pixels saturated",
            f"  stage x {at[0]:.1f} um, y {at[1]:.1f} um; focal plane at {core.getPosition():.2f} um,"
            f" surface at {surface:.2f} um",
            f"  field {truth.width_um:.1f} x {truth.height_um:.1f} um, {truth.specimen_fraction:.0%} of it specimen",
            f"  porosity {truth.porosity:.2%}; {len(pores)} pores centred in the field{largest}",
            f"  {truth.grain_count} grains, mean diameter {truth.mean_grain_diameter_um:.2f} um,"
            f" mean intercept {truth.mean_intercept_um:.2f} um",
            f"  second phase {truth.second_phase_fraction:.2%}; {len(truth.scratches)} scratches cross the field",
        ]
    )


def demo(out: Path, seed: int = 0) -> list[Path]:
    """Write the gallery and its ground truth to `out`; return the files written."""
    out.mkdir(parents=True, exist_ok=True)
    scope = build_microscope(seed=seed)
    shots: list[tuple[str, dict]] = []
    for contrast in CONTRASTS:
        shots.append((f"contrast-{contrast.lower()}.png", {"contrast": contrast}))
    for step, defocus in enumerate((-3.0, -1.0, 0.0, 1.0, 3.0), start=1):
        where = f"{abs(defocus):.0f}um-{'below' if defocus < 0 else 'above'}" if defocus else "in-focus"
        shots.append((f"focus-{step}-{where}.png", {"defocus_um": defocus}))
    for objective in scope.objectives:
        shots.append((f"objective-{objective.name}.png", {"objective": objective.name}))
    shots.append(("overexposed.png", {"objective": "20x", "overexpose": 3.0}))
    shots.append(("edge.png", {"objective": "10x", "at": EDGE_UM}))

    paths = [out / name for name, _ in shots]
    notes = [shoot(scope, path, **settings) for path, (_, settings) in zip(paths, shots)]
    header = (
        f"Ground truth for the frames in this folder: Sample.polished_ceramic(seed={seed}).\n"
        "Procedural and uncalibrated; see docs/simulated-microscope.md.\n"
        f"Times the objective touched anything while these were taken: {len(scope.incidents)}.\n"
    )
    truth_file = out / "truth.txt"
    truth_file.write_text("\n\n".join([header.rstrip(), *notes]) + "\n")
    return [*paths, truth_file]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scopepilot.sim", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    gallery = commands.add_parser("demo", help="write a gallery of frames and their ground truth")
    gallery.add_argument("--out", type=Path, required=True, help="folder to write to")
    gallery.add_argument("--seed", type=int, default=0, help="which specimen to make")
    args = parser.parse_args(argv)
    written = demo(args.out, args.seed)
    print(f"wrote {len(written) - 1} frames and {written[-1].name} to {args.out}")


if __name__ == "__main__":
    main()
