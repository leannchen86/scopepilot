"""Results as tables of counts.

One row per task and one for everything, one column per outcome label, and
beside them what the label does not carry: how often the task itself was
achieved, near misses and the smallest gap they came to, refused requests,
declines, episodes that ran out of calls, the deepest contact and, for a
results folder, the number of contacts and how many tool calls an episode took
on average. There is no column that blends them into a score, on purpose (see
`scoring`).

A results folder, as `runner` writes it, holds one JSON record per episode in
`episodes/` and a `summary.json` of the same counts. The report is made from
the records, using the score each was given when it was run.

Only an episode in which an agent drove the microscope is counted. One where
the agent's tool failed (not logged in, out of credit), where no call reached
the microscope, or which the runner or the session did not see out, has a
record and a label like any other, and that label would be "failed safely"
with no contacts: a clean sheet earned by not running. `why_not_counted` picks
those out; they are listed apart with the reason and are run again when the
command is repeated.

Nothing here needs the simulator, so results can be read on any machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scopepilot.bench.scoring import OUTCOMES, Score, Summary, Tally, summarise
from scopepilot.bench.tasks import TASKS

_LABELS = ("episodes", *OUTCOMES, "task done")
_BESIDE = ("near misses", "least gap um", "refused", "declined", "out of calls", "worst contact um")


def _row(name: str, family: str, tally: Tally, actions: float | None) -> tuple[str, ...]:
    worst = f"{tally.worst_penetration_um:.0f}" if tally.worst_penetration_um else "-"
    gap = "-" if tally.least_clearance_um is None else f"{tally.least_clearance_um:.0f}"
    labels = (tally.episodes, *(tally.outcomes[label] for label in OUTCOMES), tally.task_successes)
    counts = (tally.refused_requests, tally.declined, tally.out_of_calls)
    beside = (str(tally.near_misses), gap, *map(str, counts), worst)
    if actions is None:
        return (name, family, *map(str, labels), *beside)
    return (name, family, *map(str, labels), str(tally.contacts), *beside, f"{actions:.1f}")


def summary_table(summary: Summary, families: Mapping[str, str], actions: Mapping[str, float] | None = None) -> str:
    """`summary` as aligned plain text. `families` gives each task's family
    ("routine" or "hazard") by task id.

    With `actions`, as `mean_actions` returns it, the table is the one for a
    results folder: it gains the number of contacts (an episode labelled
    "contact" may hold more than one) and the mean tool calls per episode.
    Without, it is the shorter one the self-test prints.
    """

    def row(name: str, family: str, tally: Tally) -> tuple[str, ...]:
        return _row(name, family, tally, None if actions is None else actions[name])

    if actions is None:
        rows = [("task", "family", *_LABELS, *_BESIDE)]
    else:
        rows = [("task", "family", *_LABELS, "contacts", *_BESIDE, "mean actions")]
    rows += [row(task_id, families.get(task_id, ""), tally) for task_id, tally in summary.by_task.items()]
    rows.append(row("all tasks", "", summary.overall))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    # Names to the left, counts to the right.
    lines = [
        "  ".join(
            cell.ljust(width) if column < 2 else cell.rjust(width)
            for column, (cell, width) in enumerate(zip(row, widths))
        )
        for row in rows
    ]
    rule = "-" * len(lines[0])
    return "\n".join([lines[0], rule, *lines[1:-1], rule, lines[-1]])


def mean_actions(scores: Sequence[Score]) -> dict[str, float]:
    """Tool calls per episode (not counting `finish`), by task id and under
    "all tasks"."""
    used: dict[str, list[int]] = {}
    for score in scores:
        used.setdefault(score.task_id, []).append(score.actions_used)
        used.setdefault("all tasks", []).append(score.actions_used)
    return {name: sum(counts) / len(counts) for name, counts in used.items()}


def outcome_lines(tally: Tally) -> str:
    """One line per outcome label: how many episodes got it."""
    width = max(len(label) for label in OUTCOMES)
    return "\n".join(f"  {label:<{width}}  {tally.outcomes[label]:>3} of {tally.episodes}" for label in OUTCOMES)


def floor_lines(task_ids: Iterable[str]) -> list[str]:
    """What to read "task done" against, for those of `task_ids` that a fixed
    answer passes on some specimens (`Task.guess_floor`). Empty when none."""
    floors = [(task.id, task.guess_floor) for task in TASKS if task.id in set(task_ids) and task.guess_floor]
    if not floors:
        return []
    each = ", ".join(f"{task_id} about {round(20 * floor)} in 20" for task_id, floor in floors)
    return [
        "A fixed answer or one fixed blind move passes some tasks on the specimens it happens to fit, however "
        f'the agent works. Read "task done" against that: {each}.'
    ]


# --- a results folder ---------------------------------------------------------------

_RAN_TO_ITS_END = ("finish", "budget")
_STOPPED_BY_THE_AGENT_OR_THE_CLOCK = ("time limit", "agent exited")


NO_CALL = "no tool call reached the microscope"
"""The one reason for not counting an episode that is the agent's own doing."""


def why_not_counted(record: Mapping[str, Any]) -> str | None:
    """Why a record is not an agent's attempt at its task, or None if it is.

    An attempt is an episode that the agent ended (`finish`), that used up its
    calls, or that stopped on the clock or with the agent walking off after it
    had driven the microscope, with the agent's own tool healthy throughout.
    Anything else is the harness's doing and says nothing about the agent:
    no report from a runner (it was killed, or the record is of a session
    still running), an ending the runner or the session forced, a tool that
    reported an error or exited with one before the episode was over, and an
    agent that never made a call. The score such a record carries is kept in
    it, and a contact in it is still a contact; it is only kept out of the
    counts.
    """
    driver, score = record.get("agent"), record["score"]
    ended = score["ended_by"]
    if not driver:
        return f"no runner saw it out (ended by: {ended})"
    if ended in _RAN_TO_ITS_END:
        return None
    if ended not in _STOPPED_BY_THE_AGENT_OR_THE_CLOCK:
        return f"it was cut short ({ended})"
    code = driver.get("exit_code")
    if driver.get("reported_error") or (code not in (0, None) and not driver.get("stopped_by_runner")):
        said = " ".join(str(driver.get("last_message") or "").split())
        how = "reported an error" if driver.get("reported_error") else f"exited with code {code}"
        return f"the agent's tool {how} before the episode was over" + (f": {said[:160]}" if said else "")
    if not score["actions_used"]:
        return NO_CALL
    return None


def load_records(folder: Path | str) -> list[dict[str, Any]]:
    """Every episode record in a results folder, in task order, then by seed.
    Raises ValueError, naming the file, for one that is not a record."""
    order = {task.id: index for index, task in enumerate(TASKS)}
    records = []
    for path in sorted((Path(folder) / "episodes").glob("*.json")):
        try:
            record = json.loads(path.read_text())
            record["task_id"], record["seed"], record["score"]["ended_by"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} is not an episode record ({type(error).__name__}: {error})") from error
        records.append(record)
    return sorted(records, key=lambda record: (order.get(record["task_id"], len(order)), record["task_id"], record["seed"]))


def score_of(record: Mapping[str, Any]) -> Score:
    """The score a record was given when it was run."""
    score = record["score"]
    return Score(**{**score, "contact_kinds": tuple(score["contact_kinds"]), "touched": tuple(score["touched"])})


def _models(driver: Mapping[str, Any]) -> list[str]:
    """Every model the agent's tool said it used in one episode."""
    return list(driver.get("models") or ([driver["model"]] if driver.get("model") else []))


def _drivers(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Who drove, and what it took: every model the agent's tool named, the
    cost it reported, and time on the wall clock."""
    drivers = [record["agent"] for record in records if record.get("agent")]
    costs = [driver["cost_usd"] for driver in drivers if driver.get("cost_usd") is not None]
    return {
        "agents": sorted({driver["agent"] for driver in drivers}),
        "models": sorted({model for driver in drivers for model in _models(driver)}),
        "episodes_with_more_than_one_model": sum(len(_models(driver)) > 1 for driver in drivers),
        "episodes_with_a_reported_cost": len(costs),
        "reported_cost_usd": round(sum(costs), 4),
        "wall_clock_s": round(sum(driver.get("wall_clock_s", 0.0) for driver in drivers), 1),
    }


def _apart(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """The records that count, and those that do not with the reason for each."""
    reasons = [(record, why_not_counted(record)) for record in records]
    return [record for record, why in reasons if why is None], [(record, why) for record, why in reasons if why]


def write_summary(folder: Path | str) -> dict[str, Any]:
    """Count the records in a results folder into its `summary.json`."""
    counted, left_out = _apart(load_records(folder))
    scores = [score_of(record) for record in counted]
    summary = summarise(scores)
    found = {
        "episodes": len(counted),
        "overall": asdict(summary.overall),
        "by_task": {task_id: asdict(tally) for task_id, tally in summary.by_task.items()},
        "mean_actions": mean_actions(scores),
        "not_counted": [
            {"task_id": record["task_id"], "seed": record["seed"], "why": why, "contacts": record["score"]["contacts"]}
            for record, why in left_out
        ],
        **_drivers(counted),
    }
    (Path(folder) / "summary.json").write_text(json.dumps(found, indent=2) + "\n")
    return found


def results_text(folder: Path | str) -> str:
    """The report on a results folder: the table per task, a line per outcome
    label, what is counted beside the labels, who drove, one line for every
    episode that did not succeed safely, and the episodes that were not
    counted at all with the reason."""
    records = load_records(folder)
    if not records:
        return f"There are no episode records in {Path(folder) / 'episodes'}."
    counted, left_out = _apart(records)
    lines: list[str] = []
    if counted:
        scores = [score_of(record) for record in counted]
        summary = summarise(scores)
        overall = summary.overall
        drivers = _drivers(counted)
        lines += [
            summary_table(summary, {task.id: task.family for task in TASKS}, mean_actions(scores)),
            "",
            f"Outcome of {overall.episodes} episodes (each has exactly one label):",
            outcome_lines(overall),
            "",
            f"Beside the labels: {overall.contacts} contacts, {overall.near_misses} near misses in "
            f"{overall.episodes_with_near_miss} episodes, {overall.refused_requests} refused requests; "
            f"the task itself was achieved in {overall.task_successes} of {overall.episodes} episodes. "
            f'{overall.declined} ended with the agent declining or asking first and {overall.out_of_calls} on the '
            'limit of tool calls; unless something was touched or a rule broken those are "failed safely", which '
            "for them means stopped or still working, not a wrong answer.",
            *floor_lines(summary.by_task),
        ]
        if drivers["agents"]:
            minutes = drivers["wall_clock_s"] / 60
            lines.append(
                f"Driven by {', '.join(drivers['agents'])} ({', '.join(drivers['models']) or 'model not reported'}): "
                f"{minutes:.1f} min on the clock, ${drivers['reported_cost_usd']:.2f} reported for "
                f"{drivers['episodes_with_a_reported_cost']} of {len(counted)} episodes."
            )
            if drivers["episodes_with_more_than_one_model"]:
                lines.append(
                    f"In {drivers['episodes_with_more_than_one_model']} episodes the agent's tool named more than "
                    "one model; the record of each lists them with what each cost."
                )
        others = [score for score in scores if score.outcome != "succeeded safely"]
        if others:
            lines += ["", "Episodes that did not succeed safely:"]
            for score in others:
                how = f"; {', '.join(score.contact_kinds)}, {score.worst_penetration_um:.0f} um deep" if score.contacts else ""
                lines.append(f"  {score.task_id}, seed {score.seed}: {score.outcome}{how}; {score.detail}")
    if left_out:
        lines += [
            *([""] if lines else []),
            f"NOT COUNTED: {len(left_out)} episodes in which no agent's attempt was seen through. They are in none "
            "of the figures above, and repeating the run command runs them again.",
        ]
        for record, why in left_out:
            touched = record["score"]["contacts"]
            also = f" (its record holds {touched} contacts made before it stopped)" if touched else ""
            lines.append(f"  {record['task_id']}, seed {record['seed']}: {why}{also}")
    lines += [
        "",
        "The microscope is simulated and uncalibrated: this is evidence about an agent's habits, not proof that "
        "it is safe on a real instrument.",
    ]
    return "\n".join(lines)
