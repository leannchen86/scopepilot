# Driving the safety test by hand

This page is for a person (or an agent under somebody's eye) who is about to do
the tasks of `scopepilot.bench` themselves, one tool call at a time, to see
whether the requests are fair and the tools are enough.

It says nothing about what the tasks are looking for, on purpose. **If you are
going to drive, do not read [safety-test.md](safety-test.md) first, and do not
run `python -m scopepilot.bench list` without `--requests`:** both name, task
by task, what can go wrong. Somebody else can start the sessions for you, or
you can start them yourself as below, which shows only the request.

The microscope is simulated and uncalibrated. What you learn here is about the
test, not about any real instrument.

## What you need

```bash
uv pip install -e ".[bench]"
```

## The requests

```bash
python -m scopepilot.bench list --requests
```

prints, for each task, its id, how many tool calls it allows, and the request
as a lab user would word it. Nothing else.

## One session

Two terminals. In the first, hold the session:

```bash
python -m scopepilot.bench session --task focus-10x --seed 0 \
    --session /tmp/hand/focus-10x --record /tmp/hand-records/focus-10x-seed0.json
```

- `--session` is the folder you work in. It holds `request.txt` (the request
  and how to make a call, word for word what an agent is given), `tools.txt`
  (the tools), `act.py` (the command that makes a call) and `frames/` (the
  pictures `snap` saves).
- `--record` is where the full record goes: every call, and the true answers.
  It must be outside the session folder, and you should not open it until the
  session is over.
- `--seed` picks the specimen. Each seed is a different one.
- `--time-limit SECONDS` changes the wall-clock limit, which is 1800 (half an
  hour) unless you say otherwise. A person reading every frame may want
  `--time-limit 10800`.

It prints the request and the folder, then waits. It ends by itself when you
call `finish`, when the calls or the time run out, or when you press Ctrl-C in
that terminal; then it says how the session was marked. Nothing is left
running, and it needs no input held open.

In the second terminal, make calls:

```bash
python3 /tmp/hand/focus-10x/act.py describe_microscope
python3 /tmp/hand/focus-10x/act.py get_state
python3 /tmp/hand/focus-10x/act.py move_stage '{"x_um": 100, "y_um": 0, "relative": true}'
python3 /tmp/hand/focus-10x/act.py snap
```

Each prints one JSON object. `ok` is true or false; when false, `error` says
why and nothing has changed. `actions_left` is how many calls remain.

To be in the position an agent is in:

- One call per command. No loops, no scripts, no other program on the frames
  or the numbers. An agent run by this test can do nothing but call `act.py`
  and look at the PNG files.
- Every call uses one action, including a refused one, `describe_microscope`
  and `get_state`. `finish` uses none and is accepted after the last action.
- End with `finish` and the answer the request asks for, as a JSON object.
  This request asks for an empty one; one that asks for an image would end
  `finish '{"frame": "frame_0003.png"}'`:

```bash
python3 /tmp/hand/focus-10x/act.py finish '{}'
```

  If you think the right thing is not to do what was asked, or to ask a
  question first, finish with `{"declined": true, "reason": "..."}`.

## Afterwards

The first terminal prints one line: how the session ended, how many calls it
took, and the label it was given. The record file holds the rest, under
`score` (what was measured) and `episode` (the log, with the gap between lens
and specimen after every call, and the truth). Now read
[safety-test.md](safety-test.md) for what the labels mean and what each task
was for.

Worth writing down while it is fresh, for each task: whether the request left
you unsure what was wanted, whether a tool surprised you, how many calls you
needed, and whether you think the label is right.
