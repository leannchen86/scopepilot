"""A pretend reflected-light materials microscope, for developing with no hardware.

`sample` is the specimen and knows the true answers about it; `optics` turns a
stage position, a focus position and a set of light settings into a camera
frame; `devices` and `microscope` put those behind a Micro-Manager core, so the
pretend instrument is driven with the same calls as a real one. Nothing here is
calibrated against a real instrument. See docs/simulated-microscope.md.

Needs the optional `sim` extra (numpy, scipy, pymmcore-plus). Importing
`scopepilot` itself does not. The names below are looked up on first use, so
`scopepilot.sim.sample` and `scopepilot.sim.optics` work without pymmcore-plus.
"""

from __future__ import annotations

from importlib import import_module

_HOME = {
    "Sample": "sample",
    "Incident": "devices",
    "SimMicroscope": "microscope",
    "build_microscope": "microscope",
}

__all__ = sorted(_HOME)


def __getattr__(name: str):
    if name not in _HOME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
