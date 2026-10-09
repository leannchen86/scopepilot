"""Measure how often the pointer lands on the right control.

A case is one screenshot, one question and a labelled box around the control
that answers it. Labels start as drafts and count as verified only once a
person has checked them. A case is a hit when the centre of the box scopepilot
returns falls inside the labelled box: that is where a user following the
pointer would click.
"""

from __future__ import annotations

import html
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from scopepilot.corpus import Source
from scopepilot.imaging import annotate
from scopepilot.locate import locate
from scopepilot.model import Backend, ModelError
from scopepilot.profiles import load_profile
from scopepilot.types import Box, Guidance


@dataclass(frozen=True)
class Case:
    id: str
    source: str
    page: int
    image: int
    question: str
    box: Box | None = None
    """The labelled control, in pixels of the extracted image. None when `expect` is absent."""
    control: str = ""
    """What the labelled control is, for the person checking the label."""
    status: str = "draft"
    """draft until a person has checked the label, then verified."""
    expect: str = "present"
    """present: the control is on this screenshot and `box` marks it.
    absent: it is not on this screenshot, and the right answer is to say so."""
    kind: str = ""
    """What makes the case hard, for the person reading the results."""


@dataclass(frozen=True)
class Outcome:
    case: Case
    guidance: Guidance | None
    error: str = ""

    @property
    def hit(self) -> bool:
        if self.guidance is None:
            return False
        if self.case.box is None:
            return self.guidance.box is None
        if self.guidance.box is None:
            return False
        return self.case.box.contains(*self.guidance.box.center)


def load_cases(path: Path) -> list[Case]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cases = []
    for entry in raw.get("case", []):
        expect = entry.get("expect", "present")
        if expect not in ("present", "absent"):
            raise ValueError(f"case {entry.get('id')}: expect must be present or absent")
        if (expect == "present") != ("box" in entry):
            raise ValueError(f"case {entry.get('id')}: a present case needs a box, an absent one must not have one")
        box = Box(*entry["box"]) if "box" in entry else None
        cases.append(Case(**{**entry, "box": box}))
    return cases


def run(
    cases: list[Case],
    sources: dict[str, Source],
    backend: Backend,
    *,
    refine: bool = True,
    jobs: int = 1,
) -> list[Outcome]:
    def one(case: Case) -> Outcome:
        source = sources[case.source]
        screenshot = Image.open(source.image_path(case.page, case.image))
        try:
            guidance = locate(
                screenshot, case.question, load_profile(source.profile), backend, refine=refine
            )
        except ModelError as error:
            return Outcome(case, None, str(error))
        return Outcome(case, guidance)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cases))


def summarize(outcomes: list[Outcome]) -> str:
    lines = []
    for outcome in outcomes:
        verdict = "HIT " if outcome.hit else "MISS"
        if outcome.error or outcome.guidance is None:
            verdict = f"ERROR {outcome.error}"
        elif outcome.guidance.box is None:
            verdict += "  said the control is not on this screen"
        else:
            verdict += f"  pointed at {outcome.guidance.label!r}"
            if outcome.case.box is None:
                verdict += ", but the control is not on this screen"
        lines.append(f"{outcome.case.id:<28} {verdict}")
    hits = sum(outcome.hit for outcome in outcomes)
    verified = [outcome for outcome in outcomes if outcome.case.status == "verified"]
    lines.append("")
    lines.append(f"{hits} of {len(outcomes)} answers were right.")
    absent = [outcome for outcome in outcomes if outcome.case.box is None]
    if absent:
        present = len(outcomes) - len(absent)
        lines.append(
            f"  control on screen: {hits - sum(o.hit for o in absent)} of {present} pointed at it; "
            f"control not on screen: {sum(o.hit for o in absent)} of {len(absent)} said so."
        )
    if len(verified) < len(outcomes):
        lines.append(
            f"Only {len(verified)} of {len(outcomes)} labels have been checked by a person; "
            f"treat the score as provisional."
        )
    return "\n".join(lines)


def write_review(
    cases: list[Case], sources: dict[str, Source], out_dir: Path, outcomes: list[Outcome] | None
) -> Path:
    """Write a local page showing each label (and each answer, if given) on its screenshot.

    The page embeds publishers' screenshots, so it is for private checking only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {outcome.case.id: outcome for outcome in outcomes or []}
    rows = []
    for case in cases:
        screenshot = Image.open(sources[case.source].image_path(case.page, case.image))
        picture = annotate(screenshot, case.box) if case.box else screenshot.convert("RGB")
        outcome = by_id.get(case.id)
        caption = f"label status: {case.status}"
        if case.box is None:
            caption += " · expected: not on this screen"
        if outcome is not None and outcome.guidance is not None and outcome.guidance.box:
            picture = annotate(picture, outcome.guidance.box, pad=0, color=(0, 122, 255))
            caption += f" · answer: {'hit' if outcome.hit else 'miss'} ({outcome.guidance.label})"
        file = out_dir / f"{case.id}.png"
        picture.save(file)
        rows.append(
            f"<figure><figcaption><b>{html.escape(case.id)}</b> — {html.escape(case.question)}"
            f"<br>{html.escape(caption)}</figcaption><img src='{file.name}'></figure>"
        )
    page = out_dir / "index.html"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><title>scopepilot label review</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%;border:1px solid #ccc}figure{margin:0 0 2.5rem}</style>"
        "<h1>Label review</h1><p>Red: the labelled control. Blue: scopepilot's answer.</p>"
        + "".join(rows),
        encoding="utf-8",
    )
    return page
