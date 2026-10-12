"""Putting a real agent in front of the simulated microscope.

For each task and seed the runner starts a session process (`session`), starts
the agent as a separate process with the request and the tool description,
waits until the episode is over, and files one record under the results
folder:

    OUT/episodes/<task>-seed<n>.json   the score, who was driving, the full log and the truth
    OUT/frames/<task>-seed<n>/         copies of the frames the agent took
    OUT/logs/                          what the agent's process printed
    OUT/summary.json                   counts per outcome label (see `report`)

The agent's working folder is a temporary one outside the results folder, and
holds only what it may see. It is removed when the episode ends.

What the agent is told is `request.txt` as the session wrote it: the request,
how to call the tools, and to end with `finish`. Nothing about hazards, about
how it is marked, or that it is being tested on anything (see
`session.agent_prompt`).

An episode in which the agent's own tool failed (not logged in, out of
credit), or no call reached the microscope, is the harness's failure and not
the agent's. Its record would read "failed safely" with no contacts. `run`
therefore stops there with the reason, the report keeps such records out of
every count (`report.why_not_counted`), and repeating the command runs them
again.

The first and so far only agent is the Claude Code command-line tool. It is
run with `--restricted --tools Bash,Read` and a permission rule that allows
the thin client's command line and nothing else that was not already harmless,
so that it cannot import the simulator and read the answers. What that was
seen to refuse, and what it lets through, is in docs/safety-test.md; the check
there is manual, because it needs a login.

POSIX only (macOS, Linux): it stops an agent by its process group.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scopepilot.bench.record import Episode
from scopepilot.bench.report import NO_CALL, why_not_counted, write_summary
from scopepilot.bench.session import (
    ACT_FILE,
    DEFAULT_TIME_LIMIT_S,
    FRAMES_DIR,
    REQUEST_FILE,
    SessionProcess,
    act_command,
    start_session,
    write_record,
)
from scopepilot.bench.tasks import Task

CLEAN_ENVIRONMENT = ("HOME", "PATH", "USER", "TERM")
"""The only variables an agent's process is started with.

Claude Code reads a great many environment variables: credentials, endpoints,
and markers it sets for its own child processes. When the runner is itself
started from inside a Claude Code session (an agent running the test on
another agent), those inherited markers break the child's login. Starting
every agent from the same four variables avoids that, and also keeps one
person's shell settings from changing what is measured."""

GRACE_S = 60.0
"""How long an agent is given to say its last words after the episode is over
before it is stopped. Its closing message is where the model and the cost are
reported, so it is worth waiting for."""


def clean_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment an agent is started in: `CLEAN_ENVIRONMENT` and nothing
    else, taken from this process. TERM is "xterm" where this process has
    none, so that the four are always there."""
    environ = os.environ if environ is None else environ
    return {"TERM": "xterm", **{name: environ[name] for name in CLEAN_ENVIRONMENT if name in environ}}


class RunStopped(RuntimeError):
    """A run was stopped because an episode could not be counted. Carrying on
    would spend every remaining episode on the same failure."""


@dataclass(frozen=True)
class ClaudeCode:
    """The Claude Code command-line tool, one fresh `claude --print` per episode."""

    model: str | None = None
    """A model alias or full name for `--model`. None: whatever the tool defaults to."""
    executable: str = "claude"
    name: str = "claude-code"

    def command(self, session_dir: Path) -> list[str]:
        """The command to run in `session_dir`, with the prompt on standard input.

        --print --output-format json
            One prompt, no interaction, and a closing JSON object naming the
            model and the cost.
        --restricted --tools Bash,Read
            Only these two built-in tools exist. --restricted ignores the
            user's, the project's and the local settings files, and confines
            Read to the working folder, which is the session folder.
        --allowedTools "Bash(python3 <session>/act.py *)" "Bash(python3 act.py *)" Read, --permission-mode dontAsk
            The thin client's command line is allowed, with the full path the
            prompt gives or the bare file name, since the working folder is the
            session folder; whatever would need asking is refused, since
            nobody is there to ask. The bare form is there because a smaller
            model wrote it that way on its first call, was refused, and
            stopped: a refusal about command spelling, not about microscopy.
        --safe-mode --strict-mcp-config --disable-slash-commands
            None of the user's memory files, hooks, plugins, skills or MCP
            servers come along.
        --no-session-persistence
            The conversation is not kept in the user's history.

        Refuses a session folder with a space in its path. The thin client's
        command line would then need quoting, and whether the permission rule
        still matches a quoted command has not been tried against the tool; a
        rule that matched nothing would leave the agent unable to make a call.
        """
        if any(character.isspace() for character in str(session_dir)):
            raise ValueError(
                f"the session folder {str(session_dir)!r} has a space in its path, which the permission rule for "
                "the thin client has not been checked with; set TMPDIR to a folder without one"
            )
        command = [
            self.executable,
            "--print",
            "--output-format", "json",
            "--restricted",
            "--safe-mode",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--tools", "Bash,Read",
            "--allowedTools", f"Bash({act_command(session_dir)} *)", f"Bash(python3 {ACT_FILE} *)", "Read",
            "--permission-mode", "dontAsk",
        ]  # fmt: skip
        return command + (["--model", self.model] if self.model else [])

    @staticmethod
    def parse(stdout: str) -> dict[str, Any]:
        """What the tool's closing JSON says about the run: the models that
        answered, the cost it reports, and the commands it was refused.

        Everything is None or empty when there is no such JSON, as when the
        process was stopped before it finished. It never raises, whatever the
        text: by the time it is read the episode has been run and paid for.
        """
        try:
            closing = json.loads(stdout)
        except (ValueError, RecursionError):
            closing = {}
        if not isinstance(closing, dict):
            closing = {}
        usage = closing.get("modelUsage")
        costs = {}
        for name, use in (usage.items() if isinstance(usage, dict) else ()):
            cost = use.get("costUSD") if isinstance(use, dict) else None
            costs[str(name)] = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0
        by_cost = sorted(costs, key=costs.get, reverse=True)
        denials = closing.get("permission_denials")
        total = closing.get("total_cost_usd")
        return {
            # The tool can name several models for one run: a small one that
            # does housekeeping, and sometimes two full-size ones. All that
            # cost anything are kept, dearest first; `model` is the first.
            "model": by_cost[0] if by_cost else None,
            "models": [name for name in by_cost if costs[name] > 0] or by_cost[:1],
            "cost_usd_by_model": costs,
            "cost_usd": total if isinstance(total, (int, float)) and not isinstance(total, bool) else None,
            "turns": closing.get("num_turns"),
            "reported_error": bool(closing.get("is_error", False)),
            "refused_commands": [
                denial.get("tool_input") if isinstance(denial, dict) else denial
                for denial in (denials if isinstance(denials, list) else ())
            ],
            "last_message": closing.get("result"),
        }


AGENTS: dict[str, Callable[..., ClaudeCode]] = {"claude-code": ClaudeCode}


def episode_name(task_id: str, seed: int) -> str:
    return f"{task_id}-seed{seed}"


def run_episode(
    agent: ClaudeCode,
    task: Task,
    seed: int,
    out_dir: Path | str,
    *,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    grace_s: float = GRACE_S,
) -> dict[str, Any]:
    """One episode of `task` with `agent` driving. Returns the record, which
    is also saved under `out_dir`.

    Neither process outlives the call, however it ends. The time limit is the
    session's: it counts from when the microscope is ready, not from when the
    agent's process starts. Frames left under `out_dir` by an earlier attempt
    at the same episode are removed first, so the folder matches the record.
    """
    out_dir = Path(out_dir).resolve()
    name = episode_name(task.id, seed)
    record_path = out_dir / "episodes" / f"{name}.json"
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    printed, complained = logs / f"{name}.agent.json", logs / f"{name}.agent.err"
    kept_frames = out_dir / "frames" / name
    shutil.rmtree(kept_frames, ignore_errors=True)

    with tempfile.TemporaryDirectory(prefix="scopepilot-session-") as folder:
        session_dir = Path(folder).resolve()
        session = start_session(task.id, seed, session_dir=session_dir, record_path=record_path, time_limit_s=time_limit_s)
        process, stopped, ending = None, None, "runner interrupted"
        began = time.monotonic()
        try:
            with printed.open("w") as out, complained.open("w") as err:
                process = subprocess.Popen(
                    agent.command(session_dir),
                    cwd=session_dir,
                    env=clean_environment(),
                    stdin=subprocess.PIPE,
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=True,  # its own process group, so that it can be stopped whole
                )
            session.tell(agent_pgid=process.pid)
            try:
                process.stdin.write((session_dir / REQUEST_FILE).read_text())
                process.stdin.close()
            except OSError:
                pass  # it died at once; the wait below finds that out
            stopped = _wait(process, session, grace_s, give_up_at=began + time_limit_s + 2 * grace_s)
            ending = "agent exited" if stopped is None else "agent stopped by the runner"
        finally:
            if process is not None:
                _stop_group(process)
            session.close(ending)  # only matters if the episode is still running
            frames = session_dir / FRAMES_DIR
            if frames.is_dir():
                shutil.copytree(frames, kept_frames, dirs_exist_ok=True)
        wall_clock_s = time.monotonic() - began

    record = json.loads(record_path.read_text())
    record["agent"] = {
        "agent": agent.name,
        "requested_model": agent.model,
        **agent.parse(printed.read_text(errors="replace")),
        "exit_code": process.returncode,
        "stopped_by_runner": stopped,
        "wall_clock_s": round(wall_clock_s, 1),
    }
    write_record(record_path, task, Episode.from_dict(record["episode"]), record["agent"])
    if not complained.stat().st_size:
        complained.unlink()
    return record


def _wait(process: subprocess.Popen[str], session: SessionProcess, grace_s: float, *, give_up_at: float) -> str | None:
    """Wait for the agent's process to end by itself. Returns None if it did,
    or why it had to be stopped."""
    over_since = None
    while process.poll() is None:
        now = time.monotonic()
        if over_since is None and session.over:
            over_since = now
        if over_since is not None and now - over_since > grace_s:
            return f"still running {grace_s:.0f} s after the episode was over"
        if now > give_up_at:
            return "the session did not end at its time limit"
        time.sleep(0.1)
    return None


def _stop_group(process: subprocess.Popen[str]) -> None:
    """Stop the agent and anything it started: ask, then insist."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError):
            break
        try:
            process.wait(5.0)
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def _on_file(path: Path) -> dict[str, Any] | None:
    """The record at `path`, or None if there is none that can be read."""
    try:
        record = json.loads(path.read_text())
        record["score"]["ended_by"]
    except (OSError, ValueError, KeyError, TypeError):
        return None  # missing, or a file cut off mid-write by something other than us
    return record


def run(
    agent: ClaudeCode,
    tasks: Sequence[Task],
    seeds: Sequence[int],
    out_dir: Path | str,
    *,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    say: Callable[[str], None] = lambda line: print(line, flush=True),
) -> list[dict[str, Any]]:
    """Every task on every seed, one episode after another, into `out_dir`.

    An episode already on file there that counts as an agent's attempt
    (`report.why_not_counted`) is not run again, so a run that was interrupted
    can be picked up by repeating the command; one on file that does not count
    is run again. Delete a record to have it redone. The summary is rewritten
    after each episode, and `say` is given one line about it.

    Raises ValueError before anything is run if the agent's executable is not
    there, or if the folder holds episodes driven by another agent or model
    than the one asked for: a report that mixed them would say nothing clear.
    Raises RunStopped at the first episode that cannot be counted.
    """
    out_dir = Path(out_dir)
    episodes = out_dir / "episodes"
    wanted = [(task, seed, episodes / f"{episode_name(task.id, seed)}.json") for task in tasks for seed in seeds]
    kept = {}
    for task, seed, path in wanted:
        record = _on_file(path)
        if record is not None and why_not_counted(record) is None:
            kept[path] = record
            driver = record["agent"]
            if (driver.get("agent"), driver.get("requested_model")) != (agent.name, agent.model):
                raise ValueError(
                    f"{path} was driven by {driver.get('agent')} (model asked for: {driver.get('requested_model')}), "
                    f"not by {agent.name} (model asked for: {agent.model}); use another results folder"
                )
    if len(kept) < len(wanted) and shutil.which(agent.executable) is None:
        raise ValueError(f"there is no {agent.executable!r} to run: it is not on PATH and not a file that can be run")

    records = []
    for task, seed, path in wanted:
        name = episode_name(task.id, seed)
        if path in kept:
            record = kept[path]
            say(f"{name}: already on file, not run again")
        else:
            record = run_episode(agent, task, seed, out_dir, time_limit_s=time_limit_s)
            write_summary(out_dir)
            score, driver = record["score"], record["agent"]
            cost = "?" if driver["cost_usd"] is None else f"{driver['cost_usd']:.2f}"
            took = f"{' + '.join(driver['models']) or 'no model named'}, {driver['wall_clock_s']:.0f} s, ${cost}"
            if (why := why_not_counted(record)) is not None:
                say(f"{name}: NOT COUNTED: {why} ({took})")
                if why == NO_CALL:
                    # The tool was healthy and the agent chose not to act: that
                    # will not mend itself on a repeat, so the run goes on.
                    records.append(record)
                    continue
                raise RunStopped(
                    f"stopped at {name}, which cannot be counted: {why}. Nothing after it was run. "
                    "Repeat the command to run it again once the cause is dealt with."
                )
            say(
                f"{name}: {score['outcome']} ({score['contacts']} contacts, {score['actions_used']} actions, "
                f"ended by {score['ended_by']}; {took})"
            )
        records.append(record)
    return records
