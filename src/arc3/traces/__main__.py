"""Small offline conformance entry point: python -m arc3.traces FILE.arc3."""

import argparse
import json
import sys

from . import Arc3Error, MAX_UNCOMPRESSED_BYTES, read_attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one complete .arc3 v1 attempt offline")
    parser.add_argument("path")
    parser.add_argument("--max-uncompressed-bytes", type=int, default=MAX_UNCOMPRESSED_BYTES)
    args = parser.parse_args(argv)
    try:
        attempt = read_attempt(args.path, max_uncompressed_bytes=args.max_uncompressed_bytes)
    except (Arc3Error, OSError) as exc:
        print(f"Invalid .arc3: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "valid": True,
        "game_id": attempt["metadata"]["game_id"],
        "attempt_index": attempt["metadata"]["attempt_index"],
        "steps": len(attempt["steps"]),
        "termination": attempt["termination"]["reason"],
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
