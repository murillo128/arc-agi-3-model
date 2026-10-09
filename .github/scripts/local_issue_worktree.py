"""Persistent local issue worktrees; never use the Actions checkout as state."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

from executor_control import ControlError

CONTEXT_MARKER = "<!-- codex-execution-context:v1 -->"


def run(args, *, cwd=None, env=None, check=True):
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                            text=True, capture_output=True, timeout=180)
    if check and result.returncode:
        # Git output may contain remote URLs or authentication details.
        # tmux receives only fixed control arguments and private local paths;
        # preserve its bounded diagnostic so a failed spawn is actionable.
        detail = ": " + result.stderr.strip()[:500].replace("\n", " ") if Path(args[0]).name == "tmux" else ""
        raise ControlError(f"{Path(args[0]).name} command failed (exit {result.returncode}){detail}")
    return result


def git(root, *args, check=True):
    env = {k: v for k, v in os.environ.items()
           if k not in {"GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"}
           and not k.startswith(("ACTIONS_", "GIT_CONFIG_"))}
    return run(["git", "-C", str(root), *args], env=env, check=check)


def durable_path(value):
    path = Path(value).expanduser().resolve()
    if "_work" in path.parts or any(c in str(path) for c in "\r\n\0"):
        raise ControlError("Persistent paths must be outside Actions _work and contain no control characters")
    return path


def verify_repo(root, repo):
    root = durable_path(root)
    if Path(git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve() != root:
        raise ControlError("Repository root is not a Git worktree root")
    origin = git(root, "remote", "get-url", "origin").stdout.strip()
    allowed = {f"{prefix}{repo}{suffix}" for prefix in
               ("https://github.com/", "git@github.com:", "ssh://git@github.com/")
               for suffix in ("", ".git")}
    if origin not in allowed:
        raise ControlError("Persistent clone has an unexpected origin")
    return root


@contextmanager
def lock(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield fd
    finally:
        os.close(fd)


def context(comments, number):
    """Accept the documented canonical three-scalar YAML form, fail otherwise.

    No YAML object constructors, aliases, merges or implicit typing. JSON-style
    double quotes and YAML single quotes are accepted for these scalar fields.
    """
    matches = [c.get("body") or "" for c in comments
               if CONTEXT_MARKER in (c.get("body") or "")]
    if not matches:
        return None
    if len(matches) != 1:
        raise ControlError("Duplicate canonical execution context")
    fences = re.findall(r"(?m)^```ya?ml\s*\n([\s\S]*?)^```[ \t]*$", matches[0])
    if len(fences) != 1:
        raise ControlError("Expected one canonical execution-context YAML fence")
    fields = {}
    for line in fences[0].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([a-z_]+):[ \t]*(.*?)[ \t]*", line)
        if not match or match[1] in fields:
            raise ControlError("Malformed or duplicate execution-context scalar")
        key, value = match.groups()
        if value.startswith('"'):
            try:
                value, end = json.JSONDecoder().raw_decode(value)
            except ValueError:
                raise ControlError("Malformed quoted execution-context scalar") from None
            suffix = match[2][end:].strip()
            if suffix and not suffix.startswith("#"):
                raise ControlError("Unexpected execution-context scalar suffix")
        elif value.startswith("'"):
            quoted = re.fullmatch(r"'([^']*)'(?:[ \t]+#.*)?", value)
            if not quoted:
                raise ControlError("Malformed quoted execution-context scalar")
            value = quoted[1]
        else:
            value = re.split(r"[ \t]+#", value, maxsplit=1)[0].strip()
        fields[key] = value
    if (set(fields) != {"epic_issue", "integration_branch", "base_sha"}
            or not all(isinstance(v, str) for v in fields.values())
            or not re.fullmatch(r"[1-9][0-9]*", fields["epic_issue"])
            or fields["epic_issue"] == str(number)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", fields["base_sha"])):
        raise ControlError("Invalid canonical execution context")
    return fields


def verify_worktree(root, path, repo, branch):
    path = verify_repo(path, repo)
    if path == root:
        raise ControlError("Issue work must not use the durable coordination clone")
    if git(path, "symbolic-ref", "--short", "HEAD").stdout.strip() != branch:
        raise ControlError("Issue worktree has the wrong branch or detached HEAD")
    common = lambda p: Path(git(p, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()).resolve()
    if common(path) != common(root):
        raise ControlError("Issue worktree belongs to a different clone")
    return path


def prepare(root, base_dir, repo, number, default_branch, activation=None):
    root = verify_repo(root, repo)
    branch = f"codex/issue-{number}"
    if not re.fullmatch(r"[1-9][0-9]*", str(number)):
        raise ControlError("Invalid issue number")
    target = activation["integration_branch"] if activation else default_branch
    git(root, "check-ref-format", f"refs/heads/{target}")
    if target.startswith("-"):
        raise ControlError("Invalid base branch")
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip())
    with lock(common / "skillforge-worktrees.lock"):
        git(root, "fetch", "--prune", "origin")
        tip = git(root, "rev-parse", "--verify", f"refs/remotes/origin/{target}^{{commit}}").stdout.strip()
        base = activation["base_sha"] if activation else tip
        if activation:
            git(root, "cat-file", "-e", base + "^{commit}")
            if git(root, "merge-base", "--is-ancestor", base, tip, check=False).returncode:
                raise ControlError("Canonical activation pin is not in the integration branch")
        registered = []
        current = None
        for item in git(root, "worktree", "list", "--porcelain", "-z").stdout.split("\0"):
            if item.startswith("worktree "):
                current = item[9:]
            elif item == f"branch refs/heads/{branch}":
                registered.append(current)
        if len(registered) > 1:
            raise ControlError("Issue branch registered in multiple worktrees")
        if registered:
            return verify_worktree(root, registered[0], repo, branch)
        path = durable_path(base_dir) / repo.replace("/", "-") / f"issue-{number}"
        if path.exists():
            raise ControlError("Unregistered issue path already exists; preserve and reconcile it")
        path.parent.mkdir(parents=True, exist_ok=True)
        if git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0:
            git(root, "worktree", "add", str(path), branch)
        elif git(root, "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}", check=False).returncode == 0:
            git(root, "worktree", "add", "--track", "-b", branch, str(path), f"origin/{branch}")
        else:
            git(root, "worktree", "add", "-b", branch, str(path), base)
        return verify_worktree(root, path, repo, branch)
