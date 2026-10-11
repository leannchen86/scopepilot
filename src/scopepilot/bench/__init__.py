"""A test that scores an AI agent driving the simulated microscope.

It marks two things at once: did the agent do what was asked, and did it drive
safely. `env` is the microscope as the agent sees it (a handful of tools, units
in every name); `record` is what an episode leaves behind; `scoring` reads that
record and gives the episode one label; `tasks` is what the agent is asked;
`policies` holds a careful and a reckless scripted operator, which show that
the tasks can be done and that the scoring catches what it should
(`python -m scopepilot.bench check` runs them); `session` holds the microscope
in a process of its own, which an agent in another process reaches through the
thin client `act.py` (`python -m scopepilot.bench session` holds one for a
person to drive by hand); `runner` puts a real agent in front of it
(`python -m scopepilot.bench run`); `report` prints the counts, and keeps out
of them any episode in which the agent's own tool failed or no call was made.

Scoring is ordinary code reading the simulator's own records. No model is ever
asked to judge a result.

The microscope is simulated and uncalibrated (see docs/simulated-microscope.md),
so a good result here is evidence about an agent's habits, not proof that it is
safe on a real instrument.

Needs the optional `bench` extra. Nothing is imported here, so `tasks`,
`record`, `scoring`, `policies`, `report`, `session` and `runner` load without
the simulator; a session process imports it when it starts.

See docs/safety-test.md.
"""
