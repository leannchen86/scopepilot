"""What an episode leaves behind for whoever marks it.

The environment writes these; scoring reads them. They are plain data and need
no simulator, so an episode saved as JSON can be scored again later, on a
machine with nothing but this package, and gives the same score.

Nothing here is ever shown to the agent: it holds the clearances, the contacts
and the true answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FieldTruth:
    """The true answers about the field under the camera at one moment."""

    state: dict[str, Any]
    """The instrument's state then, as `get_state` reports it."""
    surface_um: float
    """Height of the surface on the optical axis: the focus position that makes
    the centre of the frame sharp. 0 over the bare stage plate."""
    defocus_um: float
    """Focus position minus `surface_um`. Negative: the focal plane is below
    the surface."""
    depth_of_field_um: float
    """Of the objective that was in (wavelength / NA^2), for judging `defocus_um`."""
    specimen_fraction: float
    """How much of the frame lies on the specimen."""
    porosity: float
    """Pore area over specimen area, within the frame."""
    grain_count: int
    mean_grain_diameter_um: float
    mean_intercept_um: float
    second_phase_fraction: float
    frame: dict[str, Any] | None = None
    """What `snap` told the agent about the frame taken at this moment (file
    name, brightness figures, sharpness). None when no frame was taken."""


@dataclass(frozen=True)
class GroundTruth:
    """Everything a task's check may use beyond the final state and the answer."""

    specimen_size_um: tuple[float, float]
    """Extent of the specimen along x and y; it is centred on the stage origin."""
    mount_diameter_um: float
    start: FieldTruth
    """The field under the camera before the agent's first call, with the
    objective the task starts with: what "this field" and "here" mean in a
    request."""
    final: FieldTruth
    """The field under the camera when the episode ended."""
    frames: tuple[FieldTruth, ...]
    """One entry per frame the agent took, oldest first."""


@dataclass(frozen=True)
class Action:
    """One tool call, with what it did to the instrument."""

    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    """False when the request was refused or malformed and nothing changed."""
    result: dict[str, Any]
    """What the agent was told, less the folder a frame was saved in."""
    state: dict[str, Any]
    """The instrument's state after the call."""
    clearance_um: float
    """Front lens above the highest thing under the nose, after the call.
    Negative while touching."""
    min_clearance_um: float
    """The least clearance at any point of the call: along the whole path for a
    stage move, otherwise the same as `clearance_um`."""
    working_distance_um: float
    """Of the objective in after the call, for reading the clearances as a
    fraction of it."""
    contacts: int
    """How many incidents the simulator recorded during this call."""


@dataclass(frozen=True)
class Episode:
    """One agent's attempt at one task: the full log and the truth to mark it by."""

    task_id: str
    seed: int
    budget: int
    start_state: dict[str, Any]
    start_clearance_um: float
    start_working_distance_um: float
    actions: tuple[Action, ...]
    ended_by: str | None
    """"finish", "budget", or whatever reason the caller ended it with (a time
    limit, a dead agent). None while it is still running."""
    answer: dict[str, Any] | None
    """What the agent passed to `finish`; None if it never did."""
    final_state: dict[str, Any]
    incidents: tuple[dict[str, Any], ...]
    """The simulator's `Incident`s as dictionaries, oldest first."""
    refusals: tuple[dict[str, Any], ...]
    """The simulator's `Refusal`s (moves outside the travel) as dictionaries."""
    truth: GroundTruth

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionaries, lists and numbers, ready for `json.dump`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """The inverse of `to_dict`, after a trip through JSON."""
        truth = data["truth"]
        return cls(
            **{
                **data,
                "actions": tuple(Action(**action) for action in data["actions"]),
                "incidents": tuple(data["incidents"]),
                "refusals": tuple(data["refusals"]),
                "truth": GroundTruth(
                    specimen_size_um=tuple(truth["specimen_size_um"]),
                    mount_diameter_um=truth["mount_diameter_um"],
                    start=FieldTruth(**truth["start"]),
                    final=FieldTruth(**truth["final"]),
                    frames=tuple(FieldTruth(**frame) for frame in truth["frames"]),
                ),
            }
        )
