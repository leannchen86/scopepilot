"""Command line: `scopepilot where`, `scopepilot guide`, `scopepilot corpus`, `scopepilot eval`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from scopepilot import corpus, evaluate, live
from scopepilot.imaging import annotate
from scopepilot.locate import locate
from scopepilot.model import (
    DEFAULT_MODEL,
    AnthropicBackend,
    Backend,
    ClaudeCodeBackend,
    ModelError,
    default_backend,
)
from scopepilot.profiles import available_profiles, load_profile
from scopepilot.types import Guidance

EVAL_DIR = Path("evals/lasx")


def _backend(name: str, model: str) -> Backend:
    if name == "api":
        return AnthropicBackend(model=model)
    if name == "claude-code":
        return ClaudeCodeBackend(model=model)
    return default_backend(model)


def _print_guidance(guidance: Guidance) -> None:
    if guidance.visible and guidance.box is not None:
        left, top, right, bottom = guidance.box.as_ints()
        print(f"Use: {guidance.label}")
        print(f"At:  left {left}, top {top}, right {right}, bottom {bottom} (pixels)")
    else:
        print(f"Not on this screen: {guidance.label}")
    for number, step in enumerate(guidance.steps, start=1):
        print(f"  {number}. {step}")
    if guidance.note:
        print(f"Note: {guidance.note}")
    print(f"Confidence: {guidance.confidence}")


def _where(args: argparse.Namespace) -> int:
    screenshot = Image.open(args.screenshot)
    guidance = locate(
        screenshot,
        args.question,
        load_profile(args.profile),
        _backend(args.backend, args.model),
        refine=not args.no_refine,
    )
    _print_guidance(guidance)
    if args.out and guidance.box is not None:
        annotate(screenshot, guidance.box).save(args.out)
        print(f"Marked screenshot: {args.out}")
    return 0


_GUIDE_HELP = """\
Ask a question while the microscope software is on screen; scopepilot outlines
the control to use and shows the steps next to it. It only points: clicks go
straight through the outline to the software underneath.

Type in the bar and press Enter. Esc clears the outline; Esc again, or the
close button, quits. Drag the bar to another monitor to ask about that one.

On macOS the app you start scopepilot from (Terminal, iTerm, ...) needs the
Screen Recording permission in System Settings > Privacy & Security. Without
it macOS hands over the wallpaper with no windows on it.
"""


def _guide(args: argparse.Namespace) -> int:
    # First, so a missing PySide6 is reported before anything else is set up.
    overlay = live.load_overlay()
    return overlay.run(
        load_profile(args.profile),
        _backend(args.backend, args.model),
        question=args.ask,
        seconds=args.seconds,
        refine=not args.no_refine,
    )


def _corpus_fetch(args: argparse.Namespace) -> int:
    for source in corpus.load_sources(args.dir / "sources.toml").values():
        corpus.fetch(source)
        images = corpus.extract_images(source)
        print(f"{source.id}: {len(images)} screenshots in {source.directory / 'images'}")
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    sources = corpus.load_sources(args.dir / "sources.toml")
    cases = evaluate.load_cases(args.dir / "cases.toml")
    if args.only:
        cases = [case for case in cases if case.id in args.only]
    outcomes = evaluate.run(
        cases,
        sources,
        _backend(args.backend, args.model),
        refine=not args.no_refine,
        jobs=args.jobs,
    )
    print(evaluate.summarize(outcomes))
    if args.review:
        print(f"Review page: {evaluate.write_review(cases, sources, args.review, outcomes)}")
    return 0


def _eval_review(args: argparse.Namespace) -> int:
    sources = corpus.load_sources(args.dir / "sources.toml")
    cases = evaluate.load_cases(args.dir / "cases.toml")
    print(f"Review page: {evaluate.write_review(cases, sources, args.out, None)}")
    return 0


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["auto", "api", "claude-code"], default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-refine", action="store_true", help="skip the second, zoomed-in look"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scopepilot", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    where = commands.add_parser("where", help="find the control for a task on a screenshot")
    where.add_argument("screenshot", type=Path)
    where.add_argument("question")
    where.add_argument("--profile", choices=available_profiles(), required=True)
    where.add_argument("--out", type=Path, help="save a copy of the screenshot with the control marked")
    _add_model_options(where)
    where.set_defaults(run=_where)

    guide = commands.add_parser(
        "guide",
        help="point at the control on your own screen",
        description=_GUIDE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    guide.add_argument("--profile", choices=available_profiles(), required=True)
    guide.add_argument(
        "--ask", metavar="QUESTION", help="answer one question, show the outline, then exit"
    )
    guide.add_argument(
        "--seconds", type=float, default=15.0, help="with --ask, how long the outline stays up"
    )
    _add_model_options(guide)
    guide.set_defaults(run=_guide)

    corpus_parser = commands.add_parser("corpus", help="manage the local cache of public guides")
    corpus_commands = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    fetch = corpus_commands.add_parser("fetch", help="download the guides and extract screenshots")
    fetch.add_argument("--dir", type=Path, default=EVAL_DIR)
    fetch.set_defaults(run=_corpus_fetch)

    eval_parser = commands.add_parser("eval", help="measure pointer accuracy on labelled cases")
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_commands.add_parser("run", help="run the cases against a model")
    eval_run.add_argument("--dir", type=Path, default=EVAL_DIR)
    eval_run.add_argument("--only", nargs="+", metavar="CASE_ID")
    eval_run.add_argument("--review", type=Path, help="also write a review page to this folder")
    eval_run.add_argument("--jobs", type=int, default=4, help="cases to run at once")
    _add_model_options(eval_run)
    eval_run.set_defaults(run=_eval_run)
    eval_review = eval_commands.add_parser("review", help="write a page showing each label")
    eval_review.add_argument("--dir", type=Path, default=EVAL_DIR)
    eval_review.add_argument("--out", type=Path, required=True)
    eval_review.set_defaults(run=_eval_review)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (ModelError, corpus.CorpusError, OSError, KeyError, ValueError) as error:
        print(f"scopepilot: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
