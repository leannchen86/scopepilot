"""Make one tool call on the microscope and print the result as JSON.

    python3 act.py <tool> ['<arguments as a JSON object>']

    python3 act.py get_state
    python3 act.py move_stage '{"x_um": 100, "y_um": 0, "relative": true}'

This file sits in a session folder beside `address.json`, which says where the
process holding the microscope is listening. It passes the call on and prints
the answer; it knows nothing about the microscope itself. It uses only the
standard library, so it starts in a few hundredths of a second with any
Python 3, where importing the simulator takes several seconds.

It exits 0 whenever the microscope answered, including when it refused the
request ({"ok": false, "error": ...}), and 2 when no tool was named.
"""

import json
import socket
import sys
from pathlib import Path

ADDRESS_FILE = "address.json"
TIMEOUT_S = 60.0
OVER = {"ok": False, "error": "the session is over; no further calls are accepted"}


def call(folder, tool, arguments_text):
    """Send one call to the session that `folder` belongs to; return its answer.

    The arguments travel as the text that was typed. The session parses them,
    so a call with unreadable arguments is refused and written down there like
    any other, not dropped here unseen.
    """
    try:
        address = json.loads((Path(folder) / ADDRESS_FILE).read_text())
        with socket.create_connection((address["host"], address["port"]), timeout=TIMEOUT_S) as connection:
            request = json.dumps({"tool": tool, "arguments_text": arguments_text})
            connection.sendall(request.encode() + b"\n")
            with connection.makefile("rb") as answer:
                line = answer.readline()
        return json.loads(line)
    except (OSError, ValueError, KeyError):
        # Nobody is listening, or the session went away mid-call: it has ended.
        return dict(OVER)


def main(argv):
    if len(argv) < 2:
        usage = "usage: python3 act.py <tool> ['<arguments as a JSON object>']"
        print(json.dumps({"ok": False, "error": usage}))
        return 2
    print(json.dumps(call(Path(__file__).resolve().parent, argv[1], " ".join(argv[2:]) or "{}")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
