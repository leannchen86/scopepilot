"""Command line for the safety test.

    python -m scopepilot.bench list
        The tasks: id, family, action budget, title, and with --prompts what
        the agent is asked. With --requests, only the id, the budget and the
        request: the family and the title name the hazard, and somebody about
        to try a task should not see them.

    python -m scopepilot.bench check [--seeds 0 1 2 3 4] [--tasks ID ...]
        Run the two scripted operators over the tasks and print what the
        scoring made of them. A self-test that needs no model: the careful
        operator must come out "succeeded safely" on every task, and the
        reckless one must never do so on a hazard task. Exits 1 otherwise.

    python -m scopepilot.bench run --agent claude-code --out DIR
            [--tasks ID ...] [--seeds 0 1 2] [--model NAME] [--time-limit SECONDS]
        Put a real agent in front of the microscope: one episode per task and
        seed, each a fresh agent process that can do nothing but call the
        tools. Writes one record per episode and a summary under DIR and
        prints the report. Episodes already on file in DIR are not run again.
        Every episode uses the account the agent's tool is logged in to. Stops
        with exit code 1 at the first episode in which the agent's tool failed
        or made no call: that is not an attempt and is not counted as one.

    python -m scopepilot.bench session --task ID --session DIR --record FILE
            [--seed N] [--time-limit SECONDS]
        Hold one episode for somebody to drive by hand from another terminal,
        with `python3 DIR/act.py <tool> '<json>'`. Prints the request and
        where things are, waits until the episode is over, and says how it
        came out. See docs/safety-test-by-hand.md.

    python -m scopepilot.bench report DIR
        The table for a results folder: per task, how many episodes got each
        outcome label, with contacts, near misses, refused requests, declines
        and mean actions beside them, and one line per label overall. No
        blended score. Episodes that were not an agent's attempt are listed
        apart and counted nowhere.

The microscope is simulated and uncalibrated, so what this prints is evidence
about the test and about an operator's habits, not about any real instrument.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from scopepilot.bench.policies import POLICIES, run_policy
from scopepilot.bench.report import floor_lines, results_text, summary_table
from scopepilot.bench.runner import AGENTS, RunStopped, run
from scopepilot.bench.scoring import Score, score_episode, summarise
from scopepilot.bench.session import DEFAULT_TIME_LIMIT_S, hold
from scopepilot.bench.tasks import TASKS, Task, get_task

CHECK_SEEDS = (0, 1, 2, 3, 4)
RUN_SEEDS = (0, 1, 2)
"""Three specimens per task by default. Every episode with a real agent takes
minutes and is paid for, so the default is the least that shows whether an
outcome repeats; it is too few to put a rate on anything."""


def as_expected(policy: str, task: Task, score: Score) -> bool:
    """Whether a scripted operator came out as the self-test requires. What
    the reckless one does on a routine task is hurried, not unsafe, and is not
    held to anything."""
    if policy == "careful":
        return score.outcome == "succeeded safely"
    return task.family != "hazard" or score.outcome != "succeeded safely"


def _list(tasks: Sequence[Task], prompts: bool, requests: bool) -> int:
    width = max(len(task.id) for task in tasks)
    for task in tasks:
        if requests:
            print(f"{task.id:<{width}}  {task.budget:>3} calls  {task.prompt}\n")
            continue
        print(f"{task.id:<{width}}  {task.family:<7}  {task.budget:>3} calls  {task.title}")
        if prompts:
            print(f"{'':<{width}}  {task.prompt}\n")
    return 0


def _check(tasks: Sequence[Task], seeds: Sequence[int]) -> int:
    families = {task.id: task.family for task in tasks}
    surprises = []
    with tempfile.TemporaryDirectory(prefix="scopepilot-bench-") as folder:
        for name, policy in POLICIES.items():
            scores = []
            for task in tasks:
                for seed in seeds:
                    frames = Path(folder) / name / f"{task.id}-{seed}"
                    score = score_episode(task, run_policy(policy, task, seed, frames_dir=frames))
                    scores.append(score)
                    if not as_expected(name, task, score):
                        surprises.append(f"{name} on {task.id}, seed {seed}: {score.outcome} ({score.detail})")
                print(f"{name}: {task.id} done", file=sys.stderr)
            print(f"\nThe {name} operator, seeds {' '.join(map(str, seeds))}\n")
            print(summary_table(summarise(scores), families))
    print()
    for line in floor_lines(families):
        print(line + "\n")
    if surprises:
        print("NOT AS EXPECTED:\n  " + "\n  ".join(surprises))
        return 1
    print(
        'As expected: the careful operator "succeeded safely" on every task, and the reckless one never did '
        "on a hazard task."
    )
    return 0


def _run(agent: str, model: str | None, tasks: Sequence[Task], seeds: Sequence[int], out: Path, limit: float) -> int:
    # Stopped from outside, leave by the ordinary way out, so that the agent
    # and the session of the episode in hand are stopped too.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        run(AGENTS[agent](model=model), tasks, seeds, out, time_limit_s=limit)
    except RunStopped as stopped:
        print(f"\n{results_text(out)}\n", flush=True)
        print(f"scopepilot.bench: {stopped}", file=sys.stderr)
        return 1
    print()
    print(results_text(out))
    return 0


def _seconds(text: str) -> float:
    """A time limit: a finite number of seconds above zero. "nan" compares
    false with everything, so a limit of it would never pass."""
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds above zero")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scopepilot.bench",
        description="A test that scores an agent driving the simulated microscope on task success and on safety.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="show the tasks")
    listing.add_argument("--prompts", action="store_true", help="also show what the agent is asked")
    listing.add_argument(
        "--requests", action="store_true", help="show only id, budget and request, with nothing that names a hazard"
    )
    check = commands.add_parser("check", help="self-test with the scripted careful and reckless operators")
    check.add_argument("--seeds", type=int, nargs="+", default=list(CHECK_SEEDS), help="specimens to run on")
    check.add_argument("--tasks", nargs="+", metavar="ID", help="only these tasks (default: all)")
    running = commands.add_parser("run", help="put a real agent in front of the microscope and record what it does")
    running.add_argument("--agent", required=True, choices=sorted(AGENTS), help="the agent harness to drive")
    running.add_argument("--out", required=True, type=Path, metavar="DIR", help="results folder")
    running.add_argument("--tasks", nargs="+", metavar="ID", help="only these tasks (default: all)")
    running.add_argument("--seeds", type=int, nargs="+", default=list(RUN_SEEDS), help="specimens to run on")
    running.add_argument("--model", help="model for the agent (default: its own default)")
    running.add_argument(
        "--time-limit",
        type=_seconds,
        default=DEFAULT_TIME_LIMIT_S,
        metavar="SECONDS",
        help="wall-clock limit per episode",
    )
    holding = commands.add_parser("session", help="hold one episode for somebody to drive by hand")
    holding.add_argument("--task", required=True, metavar="ID", help="the task to hold")
    holding.add_argument("--seed", type=int, default=0, help="the specimen")
    holding.add_argument(
        "--session", required=True, type=Path, metavar="DIR", help="folder for what the operator may see"
    )
    holding.add_argument(
        "--record", required=True, type=Path, metavar="FILE", help="file for the record; not inside the session folder"
    )
    holding.add_argument(
        "--time-limit", type=_seconds, default=DEFAULT_TIME_LIMIT_S, metavar="SECONDS", help="wall-clock limit"
    )
    report = commands.add_parser("report", help="print the table for a results folder")
    report.add_argument("out", type=Path, metavar="DIR", help="results folder written by run")
    args = parser.parse_args(argv)

    if args.command == "list":
        return _list(TASKS, args.prompts, args.requests)
    try:
        named = [args.task] if args.command == "session" else getattr(args, "tasks", None)
        tasks = [get_task(task_id) for task_id in named] if named else list(TASKS)
    except KeyError as error:
        parser.error(error.args[0])
    # A mistake on the command line or a file that is not what it should be
    # gets one line, not a traceback; so does being interrupted.
    try:
        if args.command == "report":
            print(results_text(args.out))
            return 0
        if args.command == "session":
            hold(tasks[0], args.seed, session_dir=args.session, record_path=args.record, time_limit_s=args.time_limit)
            return 0
        if args.command == "run":
            return _run(args.agent, args.model, tasks, args.seeds, args.out, args.time_limit)
        return _check(tasks, args.seeds)
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and args.command == "check":
            raise  # nothing a person typed: a fault in the self-test should show where it is
        print(f"scopepilot.bench: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("scopepilot.bench: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
