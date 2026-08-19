"""Caregiver side of the live room: send one command, wait for its
reply, print it. Usage:
  python3 scripts/room_cli.py <life_path> <cmd> [payload...]
"""

import os
import sys
import time


def main():
    life, cmd = sys.argv[1], sys.argv[2]
    payload = " ".join(sys.argv[3:])
    inbox, outbox = life + ".inbox", life + ".outbox"
    with open(inbox) as f:
        n = sum(1 for _ in f) + 1
    with open(inbox, "a") as f:
        f.write(f"{n}|{cmd}|{payload}\n")
    deadline = time.time() + 120
    while time.time() < deadline:
        if os.path.exists(outbox):
            with open(outbox) as f:
                for line in f:
                    if line.startswith(f"{n}|"):
                        _, kind, text = (line.rstrip("\n")
                                         .split("|", 2) + [""])[:3]
                        print(f"{kind}: {text}")
                        return
        time.sleep(0.2)
    print("TIMEOUT")
    sys.exit(1)


if __name__ == "__main__":
    main()
