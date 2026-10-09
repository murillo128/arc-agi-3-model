#!/usr/bin/python3 -I
"""Sandboxed client for the local Devin supervisor's authenticated host broker."""

import json
from pathlib import Path
import socket
import sys


def main(argv):
    if len(argv) < 2 or argv[0] not in ("git", "gh"):
        print("Use host client git|gh ARGS", file=sys.stderr)
        return 64
    job = json.loads(Path(__file__).with_name("job.json").read_text())
    payload = json.dumps({"argv": argv, "cwd": str(Path.cwd().resolve())}).encode("utf-8")
    if len(payload) > 200_000:
        print("Host request is too large", file=sys.stderr)
        return 64
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(185)
            connection.connect(job["host_socket"])
            connection.sendall(payload)
            connection.shutdown(socket.SHUT_WR)
            response = bytearray()
            while True:
                part = connection.recv(65536)
                if not part:
                    break
                response.extend(part)
                if len(response) > 2_100_000:
                    raise ValueError("Host response is too large")
        result = json.loads(response)
        sys.stdout.write(result["stdout"])
        sys.stderr.write(result["stderr"])
        return int(result["exit_code"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Host broker unavailable: {error}", file=sys.stderr)
        return 69


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
