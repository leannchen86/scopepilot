"""What scopepilot knows about one piece of vendor software.

A profile is a TOML file of layout notes, written in our own words from public
user guides, that is handed to the model next to the screenshot. Every note
names the guide it came from, because layouts differ between versions and
instruments and none of this has been checked on a live system yet.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    summary: str
    layout: tuple[str, ...]
    controls: tuple[dict[str, str], ...]
    traps: tuple[dict[str, str], ...]

    def prompt_text(self) -> str:
        lines = [f"Notes on {self.name}", self.summary, "", "Layout:"]
        lines += [f"- {item}" for item in self.layout]
        lines += ["", "Where controls are usually found:"]
        lines += [f"- {c['task']}: {c['where']}" for c in self.controls]
        if self.traps:
            lines += ["", "Known traps, worth a line in note when one applies:"]
            lines += [f"- {t['symptom']}: {t['advice']}" for t in self.traps]
        return "\n".join(lines)


def available_profiles() -> list[str]:
    data = resources.files(__package__) / "data"
    return sorted(p.name.removesuffix(".toml") for p in data.iterdir() if p.name.endswith(".toml"))


def load_profile(profile_id: str) -> Profile:
    path = resources.files(__package__) / "data" / f"{profile_id}.toml"
    if not path.is_file():
        raise KeyError(f"no profile {profile_id!r}; available: {', '.join(available_profiles())}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Profile(
        id=profile_id,
        name=raw["name"],
        summary=raw["summary"].strip(),
        layout=tuple(raw.get("layout", [])),
        controls=tuple(raw.get("control", [])),
        traps=tuple(raw.get("trap", [])),
    )
