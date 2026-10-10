# Testing without a microscope

The question this test set answers: given a screenshot of the software and a
question a user would ask, how often does scopepilot point at the right
control?

## What is in the repository

- `lasx/sources.toml`: public guides that contain LAS X screenshots. Each entry
  is a URL and a hash of the file, nothing more.
- `lasx/cases.toml`: test cases. Each one names a guide, a page and an image on
  that page, and gives a question and a box around the control that answers it.

The guides and their screenshots belong to their publishers and are **not** in
this repository. Do not add them.

## Running it

Run these from the repository root, after the install step in the main README.
From anywhere else, pass `--dir path/to/evals/lasx`.

```bash
scopepilot corpus fetch
```

downloads the guides into `~/.cache/scopepilot` (set `SCOPEPILOT_CACHE` to move
it) and extracts their screenshots. It stops if a guide no longer matches its
recorded hash, because page and image numbers would no longer line up.

```bash
scopepilot eval run --review out/review
```

runs every case and prints a line per case and a total. `--review` also writes
a local page showing each screenshot with the labelled control in red and
scopepilot's answer in blue. That page embeds the publishers' screenshots, so
keep it local.

## How a case is scored

A case is a hit when the centre of the box scopepilot returns falls inside the
labelled box. That is the spot a user following the pointer would click.

A case marked `expect = "absent"` asks for a control that is not on that
screenshot. It is a hit only when scopepilot says so and points at nothing.

## Reading the score

- **Labels start as `draft`.** A case becomes `verified` only after a person
  has looked at it on the review page and confirmed the box and the question.
  The summary says how many are still unchecked.
- **The score is optimistic.** The layout notes in `profiles/data` were written
  from the same guides the screenshots come from. A different instrument,
  LAS X version or screen layout will do worse until it is measured.
- **Cases avoid controls the guide's author highlighted**, since a red box or
  arrow drawn on the screenshot would give the answer away.
- A hit means the pointer was right. It says nothing about whether the written
  steps were right; those are not scored yet.

## First reading (8 October 2026)

23 of 23 pointers landed on the labelled control, with and without the second,
zoomed-in look.

Do not read that as "it works". What it shows is that a current Claude model
can find a named or clearly described control on a clean LAS X screenshot.
What limits it:

- **It was a stand-in run.** No API credentials were available, so agents in a
  Claude Code session answered scopepilot's own two prompts on the images
  scopepilot would send, and the answers were replayed through `locate()`. The
  real API path has not been run on these cases.
- **The labels are all `draft`.** They were drawn and checked by automated
  passes with the same family of model that then answered the questions.
- **The cases are easy.** Every target is on screen, most carry readable text,
  and some questions narrow the search ("which entry in this menu"). A perfect
  score means the set cannot yet tell a good version from a better one.
- **23 cases, three guides, no live system.**
- **The layout notes have changed since.** A review found notes that said more
  than the guides do, and they were corrected after this run. The corrected
  notes have not been re-measured.

What the set needs next: controls that are not on the current screen, icon-only
buttons with look-alikes, questions with no good answer, and screenshots from a
layout the notes were not written from.

## Second reading (8 October 2026, later the same day)

41 of 41 answers were right: 33 of 33 controls on screen were pointed at, and
8 of 8 controls that were not on screen were reported as not there.

This run used the corrected and expanded layout notes and added 18 harder
cases: icon-only buttons with a look-alike elsewhere on screen, targets about
12 pixels across, a similarly named control nearby, and questions whose control
is not on the screenshot at all.

The limits of the first reading still apply: a stand-in run, draft labels made
by the same family of model, three guides, and notes written from those same
guides. Two perfect scores say that finding a control on a clean screenshot of
a layout the notes describe is no longer the hard part. They say nothing about
the things public screenshots cannot test:

- whether the written steps are right, which is not scored;
- a layout, LAS X version or instrument the notes were not written from;
- a real screen: scaling, several monitors, dialogs in the way;
- whether the answer arrives fast enough to help someone at the microscope.

## What the notes and the prompt add (10 October 2026)

The same 41 cases, answered three ways in stand-in runs, one run each:

| Setup | Right |
|---|---|
| scopepilot's prompt and layout notes | 41 of 41 |
| scopepilot's prompt, no notes | 39 of 41 |
| A plain "helpful assistant" prompt, no notes | 37 of 41 |

Every miss was an icon-only button with no text on it: the autofocus button,
the Z-stack start button, the histogram button used for contrast, and the link
that leads to the camera choice. In each miss the model said its confidence was
low. All eight "not on this screen" cases were right in every setup.

So a general model finds most controls by itself, and the notes matter where
the screen does not say what a button does. Read the gap with care: it is two
to four cases in a single run, and the notes were written from the same guides
the screenshots come from, so they hold exactly the knowledge these cases need.

## Adding a case

1. Run `scopepilot corpus fetch` and pick a screenshot from the cache.
2. Add a `[[case]]` to `lasx/cases.toml` with the box in pixels of that image
   (`left, top, right, bottom`) and `status = "draft"`. For a control that is
   not on the screenshot, leave the box out and set `expect = "absent"`.
3. Run `scopepilot eval review --out out/review`, check the red box sits on the
   control, and change the status to `verified`.
