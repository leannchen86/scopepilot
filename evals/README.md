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

## Adding a case

1. Run `scopepilot corpus fetch` and pick a screenshot from the cache.
2. Add a `[[case]]` to `lasx/cases.toml` with the box in pixels of that image
   (`left, top, right, bottom`) and `status = "draft"`.
3. Run `scopepilot eval review --out out/review`, check the red box sits on the
   control, and change the status to `verified`.
