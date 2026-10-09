#!/usr/bin/env python3
"""Safely remove persistent issue and review worktrees after an issue closes."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time


class CleanupError(RuntimeError):
    pass


def git(root, *args, check=True):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    }
    result = subprocess.run(
        ["git", "-C", str(root), *map(str, args)],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    if check and result.returncode:
        detail = result.stderr.strip().replace("\n", " ")[:500]
        raise CleanupError(
            f"git {' '.join(map(str, args))} failed ({result.returncode})"
            + (f": {detail}" if detail else "")
        )
    return result


def resolved(path):
    return Path(path).expanduser().resolve()


def verify_repo(root, repository):
    root = resolved(root)
    top = resolved(git(root, "rev-parse", "--show-toplevel").stdout.strip())
    if top != root:
        raise CleanupError("SKILLFORGE_REPO_ROOT is not the repository root")

    origin = git(root, "remote", "get-url", "origin").stdout.strip()
    allowed = {
        f"{prefix}{repository}{suffix}"
        for prefix in ("https://github.com/", "git@github.com:", "ssh://git@github.com/")
        for suffix in ("", ".git")
    }
    if origin not in allowed:
        raise CleanupError("Persistent clone has an unexpected origin")
    return root


def worktrees(root):
    records = []
    current = None
    raw = git(root, "worktree", "list", "--porcelain", "-z").stdout
    for field in raw.split("\0"):
        if not field:
            continue
        if field.startswith("worktree "):
            if current is not None:
                records.append(current)
            current = {"path": resolved(field[9:]), "branch": None, "detached": False}
        elif current is not None and field.startswith("branch "):
            current["branch"] = field[7:]
        elif current is not None and field == "detached":
            current["detached"] = True
    if current is not None:
        records.append(current)
    return records


def within(path, root):
    path = resolved(path)
    root = resolved(root)
    return path == root or root in path.parents


def lock_is_held(path):
    path = Path(path)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def wait_for_reviews(run_root, repo_key, pr_numbers, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    while True:
        active = [
            pr
            for pr in pr_numbers
            if lock_is_held(resolved(run_root) / repo_key / f"pr-review-{pr}" / "active.lock")
        ]
        if not active:
            return
        if time.monotonic() >= deadline:
            joined = ", ".join(f"#{number}" for number in active)
            raise CleanupError(f"Timed out waiting for active PR audit(s): {joined}")
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))


def cleanup(repo_root, worktree_root, run_root, repository, issue_number, pr_numbers, wait_seconds=900):
    if not re.fullmatch(r"[1-9][0-9]*", str(issue_number)):
        raise CleanupError("issue number must be a positive decimal integer")
    normalized_prs = sorted({int(number) for number in pr_numbers})
    if any(number < 1 for number in normalized_prs):
        raise CleanupError("PR numbers must be positive integers")

    repo_root = verify_repo(repo_root, repository)
    repo_key = repository.replace("/", "-")
    controlled_root = resolved(worktree_root) / repo_key
    issue_path = controlled_root / f"issue-{issue_number}"
    issue_branch = f"refs/heads/codex/issue-{issue_number}"
    reviews_root = controlled_root / "reviews"

    wait_for_reviews(run_root, repo_key, normalized_prs, wait_seconds)

    records = worktrees(repo_root)
    targets = []
    for record in records:
        path = record["path"]
        if record["branch"] == issue_branch:
            if not within(path, controlled_root):
                raise CleanupError(f"Refusing to remove issue worktree outside managed root: {path}")
            targets.append(path)
            continue

        if path == issue_path and record["branch"] != issue_branch:
            raise CleanupError(f"Canonical issue worktree has unexpected branch: {record['branch']}")

        if within(path, reviews_root):
            match = re.fullmatch(r"pr-([1-9][0-9]*)-.+", path.name)
            if match and int(match.group(1)) in normalized_prs:
                if not record["detached"]:
                    raise CleanupError(f"Refusing to remove non-detached review worktree: {path}")
                targets.append(path)

    removed = []
    for path in dict.fromkeys(targets):
        # Never discard unfinished tracked, staged or untracked work.
        git(repo_root, "worktree", "remove", str(path))
        removed.append(path)

    git(repo_root, "worktree", "prune")

    leftovers = []
    if issue_path.exists():
        leftovers.append(issue_path)
    for pr in normalized_prs:
        if reviews_root.exists():
            leftovers.extend(sorted(reviews_root.glob(f"pr-{pr}-*")))

    return removed, leftovers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--run-root", default="~/.skillforge/run")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--pr-number", action="append", default=[])
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args()

    if args.wait_seconds < 0:
        raise SystemExit("--wait-seconds must be >= 0")

    try:
        removed, leftovers = cleanup(
            args.repo_root,
            args.worktree_root,
            args.run_root,
            args.repository,
            args.issue_number,
            args.pr_number,
            args.wait_seconds,
        )
    except CleanupError as exc:
        raise SystemExit(str(exc)) from None

    for path in removed:
        print(f"removed worktree: {path}")
    for path in leftovers:
        print(f"warning: unregistered managed path left in place: {path}")
    if not removed and not leftovers:
        print("no matching worktrees remain")


if __name__ == "__main__":
    main()
