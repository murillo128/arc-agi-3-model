"""Constrained host Git/GitHub broker for a sandboxed local Devin turn.

The client and its shell stay in the OS sandbox. This supervisor-owned broker
validates argument vectors and repository identity before using host auth.
"""

import json
import os
from pathlib import Path
import pwd
import re
import socket
import struct
import subprocess
import threading


class Rejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Rejected(message)


def decimal(value):
    require(re.fullmatch(r"[1-9][0-9]*", value) is not None, "Expected an issue or PR number")


def local_file(value, worktree):
    path = Path(value).resolve()
    require(path.is_file() and (path.is_relative_to(worktree) or path.is_relative_to(Path("/tmp"))),
            "Body file must be a regular file in the issue worktree or /tmp")
    with path.open("r", encoding="utf-8") as source:
        content = source.read(100_001)
        require(len(content.encode("utf-8")) <= 100_000, "GitHub body file is too large")
        require(Path(f"/proc/self/fd/{source.fileno()}").resolve() == path,
                "GitHub body file changed during validation")
    return content


def workspace_path(value, worktree):
    require(not value.startswith("-"), "Expected a worktree path")
    path = (worktree / value).resolve()
    require(path == worktree or path.is_relative_to(worktree), "Git path escapes the issue worktree")


def parse(args, *, switches=(), values=None):
    """Accept only named options; return positional args without invoking a shell."""
    values = values or {}
    positionals = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in switches:
            pass
        elif arg in values:
            i += 1
            require(i < len(args), f"Missing value for {arg}")
            values[arg](args[i])
        elif arg.startswith("--") and "=" in arg and arg.split("=", 1)[0] in values:
            name, value = arg.split("=", 1)
            values[name](value)
        elif arg.startswith("-"):
            raise Rejected(f"Unsupported option: {arg}")
        else:
            positionals.append(arg)
        i += 1
    return positionals


def git_args(args, job, worktree):
    require(bool(args), "Missing Git subcommand")
    command, rest = args[0], args[1:]
    branch = job["branch"]
    issue = job["number"]

    if command == "status":
        require(not parse(rest, switches=("--short", "--branch", "--porcelain", "--porcelain=v1",
                                         "--porcelain=v2", "--untracked-files=all")),
                "Git status takes no path here")
    elif command == "rev-parse":
        positionals = parse(rest, switches=("--show-toplevel", "--git-common-dir", "--verify",
                                            "--abbrev-ref", "--is-inside-work-tree"))
        require(all(re.fullmatch(r"[A-Za-z0-9_./^~{}:-]+", x) for x in positionals),
                "Invalid Git revision")
    elif command in ("diff", "log", "show"):
        suffix = []
        options = rest
        if "--" in rest:
            separator = rest.index("--")
            options, suffix = rest[:separator], rest[separator + 1:]
            require(command == "diff" and bool(suffix), "Only Git diff accepts worktree paths")
            for path in suffix:
                workspace_path(path, worktree)
        switches = ("--stat", "--name-only", "--oneline", "--no-ext-diff", "--check",
                    "--cached", "--staged", "--decorate", "--quiet")
        positionals = parse(options, switches=switches, values={
            "-n": decimal, "--max-count": decimal,
            "--format": lambda value: require(bool(value), "Missing Git format"),
        })
        require(all(re.fullmatch(r"[A-Za-z0-9_./^~{}:-]+", x) for x in positionals),
                "Invalid Git revision or path")
    elif command == "ls-remote":
        positionals = parse(rest, switches=("--heads", "--tags", "--exit-code"))
        require(positionals and positionals[0] == "origin" and
                all(re.fullmatch(r"[A-Za-z0-9_./*^-]+", x) for x in positionals[1:]),
                "Git remote must be origin")
    elif command == "ls-files":
        require(not parse(rest, switches=("-m", "-o", "--modified", "--others",
                                         "--cached", "--exclude-standard", "--stage")),
                "Git ls-files takes no path here")
    elif command == "fetch":
        positionals = parse(rest, switches=("--no-tags", "--prune"))
        require(positionals and positionals[0] == "origin" and
                all(re.fullmatch(r"[A-Za-z0-9_./*^-]+", x) and not x.startswith("-")
                    for x in positionals[1:]), "Git fetch must use origin and refs")
    elif command == "add":
        require(rest, "Git add needs an explicit path or -A")
        if rest == ["-A"] or rest == ["--all"]:
            pass
        else:
            paths = rest[1:] if rest[0] == "--" else rest
            require(bool(paths), "Git add needs a path")
            for path in paths:
                workspace_path(path, worktree)
    elif command == "commit":
        messages = []
        positionals = parse(rest, switches=("-a", "--all"), values={
            "-m": lambda value: messages.append(value),
            "--message": lambda value: messages.append(value),
        })
        require(not positionals and messages, "Git commit needs a message and no path/options")
    elif command == "push":
        lease = []
        positionals = parse(rest, switches=("-u", "--set-upstream"), values={
            "--force-with-lease": lambda value: lease.append(value),
        })
        require(len(positionals) == 2 and positionals[0] == "origin",
                "Git push must target origin and one ref")
        refspec = positionals[1]
        destinations = {branch, f"refs/heads/{branch}"}
        if branch == f"codex/issue-{issue}":
            destinations.add(f"refs/heads/codex/epic-issue-{issue}")
        source, separator, destination = refspec.partition(":")
        require(source in ("HEAD", branch) or re.fullmatch(r"[0-9a-f]{40}", source),
                "Git push needs the owned branch, HEAD, or a full commit SHA")
        require((not separator and source == branch) or
                (separator and destination in destinations),
                "Git push destination is outside the owned issue branches")
        if lease:
            require(len(lease) == 1 and
                    re.fullmatch(rf"refs/heads/{re.escape(branch)}:[0-9a-f]{{40}}", lease[0]) and
                    (not separator or destination == f"refs/heads/{branch}"),
                    "Force lease must pin the owned issue branch and prior SHA")
    elif command == "merge-base":
        positionals = parse(rest, switches=("--is-ancestor",))
        require(len(positionals) == 2 and
                all(re.fullmatch(r"[A-Za-z0-9_./^~{}:-]+", x) for x in positionals),
                "Git merge-base needs two revisions")
    elif command == "remote":
        require(rest == ["get-url", "origin"], "Only the origin URL may be read")
    elif command == "branch":
        require(rest in ([], ["--show-current"], ["--list"], ["-vv"]),
                "Git branch is read-only here")
    elif command == "merge":
        positionals = parse(rest, switches=("--no-edit", "--no-ff"))
        require(len(positionals) == 1 and re.fullmatch(r"[A-Za-z0-9_./^-]+", positionals[0]),
                "Git merge needs one revision")
    elif command == "rebase":
        require(rest in (["--continue"], ["--abort"]) or
                (len(rest) == 1 and re.fullmatch(r"[A-Za-z0-9_./^-]+", rest[0])),
                "Git rebase needs one revision or continue/abort")
    else:
        raise Rejected(f"Unsupported host Git subcommand: {command}")

    # Disable issue-controlled hooks. Git configuration and executable overrides
    # are never accepted from the agent's argv or environment.
    return ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", command, *rest]


def gh_args(args, job, worktree):
    require(bool(args), "Missing GitHub subcommand")
    repo = job["repo"]
    def exact_repo(value):
        require(value == repo, "GitHub repository differs from the issue owner")
    def text_value(value):
        require(bool(value) and "\x00" not in value, "Missing text value")
    def head_ref(value):
        require(value == job["branch"], "PR head must be the owned issue branch")
    def base_ref(value):
        require(value == job.get("default_branch", "main") or
                re.fullmatch(r"codex/epic-issue-[1-9][0-9]*", value),
                "PR base must be the default or an epic integration branch")
    def body_file(value):
        body_files.append(local_file(value, worktree))
    body_files = []
    common = {"--repo": exact_repo, "-R": exact_repo}
    json_options = {**common, "--json": text_value, "--jq": text_value, "-q": text_value}
    group = args[0]

    if group == "api":
        methods = []
        positionals = parse(args[1:], switches=("--paginate", "--slurp"), values={
            "--jq": text_value, "-q": text_value,
            "--method": lambda value: methods.append(value),
            "-X": lambda value: methods.append(value),
        })
        require(len(positionals) == 1 and positionals[0].startswith(f"repos/{repo}/") and
                "%" not in positionals[0].split("?", 1)[0] and
                ".." not in positionals[0].split("?", 1)[0].split("/") and
                all(method == "GET" for method in methods),
                "GitHub API access is limited to GET on the owned repository")
    elif group == "auth":
        require(args[1:] in (["status"], ["status", "--hostname", "github.com"]),
                "Only read-only GitHub auth status is allowed")
    elif group == "repo":
        require(len(args) >= 2 and args[1] == "view", "Only GitHub repo view is allowed")
        positionals = parse(args[2:], values=json_options)
        require(positionals in ([], [repo]), "GitHub repo view must use the owned repository")
    elif group in ("issue", "pr", "run"):
        require(len(args) >= 2, "Missing GitHub action")
        action = args[1]
        rest = args[2:]
        if group == "issue" and action == "view":
            positionals = parse(rest, switches=("--comments",), values=json_options)
        elif group == "issue" and action == "comment":
            positionals = parse(rest, values={**common, "--body": text_value,
                                             "--body-file": body_file})
            require(any(x == "--body" or x.startswith("--body=") or x == "--body-file" or
                        x.startswith("--body-file=") for x in rest), "Issue comment needs a body")
        elif group == "issue" and action == "edit":
            positionals = parse(rest, values={**common, "--add-label": text_value,
                                             "--remove-label": text_value,
                                             "--title": text_value, "--body": text_value,
                                             "--body-file": body_file})
            require(any(x.startswith(("--add-label", "--remove-label", "--title", "--body"))
                        for x in rest), "Issue edit needs an explicit change")
        elif group == "pr" and action == "list":
            positionals = parse(rest, values={
                **json_options, "--head": head_ref, "--base": base_ref,
                "--state": lambda value: require(value in ("open", "closed", "merged", "all"),
                                                  "Unsupported PR state"),
                "--limit": decimal,
            })
            require(not positionals and any(value == "--head" or value.startswith("--head=")
                                            for value in rest),
                    "PR list must filter the owned head branch")
        elif group == "pr" and action in ("view", "checks", "diff"):
            positionals = parse(rest, values=json_options)
        elif group == "pr" and action == "create":
            positionals = parse(rest, switches=("--draft",), values={
                **common, "--base": base_ref, "--head": head_ref,
                "--title": text_value, "--body": text_value, "--body-file": body_file,
            })
            require(not positionals and job["branch"] in rest and "--base" in rest,
                    "PR creation needs the owned head branch and an explicit base")
        elif group == "pr" and action == "edit":
            positionals = parse(rest, values={**common, "--base": base_ref,
                                             "--title": text_value, "--body": text_value,
                                             "--body-file": body_file})
        elif group == "pr" and action == "patch":
            fields = {}
            def set_field(name, validator):
                def accept(value):
                    validator(value)
                    require(name not in fields, f"Duplicate PR field: {name}")
                    fields[name] = value
                return accept
            def patch_body_file(value):
                require("body" not in fields, "Duplicate PR body")
                fields["body"] = local_file(value, worktree)
            positionals = parse(rest, values={
                **common, "--base": set_field("base", base_ref),
                "--title": set_field("title", text_value),
                "--body": set_field("body", text_value),
                "--body-file": patch_body_file,
            })
            require(bool(fields), "PR patch needs an explicit change")
        elif group == "pr" and action == "ready":
            positionals = parse(rest, values=common)
        elif group == "run" and action in ("view", "list"):
            positionals = parse(rest, switches=("--log", "--log-failed"), values={
                **json_options, "--limit": decimal, "--workflow": text_value,
            })
        else:
            raise Rejected(f"Unsupported host GitHub action: {group} {action}")
        if (group == "run" or group == "pr") and action == "list":
            require(not positionals, "GitHub run list takes no ID")
        elif group == "pr" and action == "create":
            pass
        else:
            require(len(positionals) == 1, "GitHub action needs one issue, PR or run number")
            decimal(positionals[0])
    else:
        raise Rejected(f"Unsupported host GitHub subcommand: {group}")

    if group == "pr" and action == "patch":
        command = ["/usr/bin/gh", "api", "-X", "PATCH",
                   f"repos/{repo}/pulls/{positionals[0]}"]
        for key, value in fields.items():
            command.extend(("-f", f"{key}={value}"))
        return [*command, "--jq", "{number,html_url}"]
    if body_files:
        require(len(body_files) == 1 and not any(x == "--body" or x.startswith("--body=")
                                                 for x in args), "Use one GitHub body source")
        rewritten = []
        i = 0
        while i < len(args):
            if args[i] == "--body-file":
                i += 2
                rewritten.extend(("--body", body_files[0]))
            elif args[i].startswith("--body-file="):
                i += 1
                rewritten.extend(("--body", body_files[0]))
            else:
                rewritten.append(args[i])
                i += 1
        args = rewritten
    return ["/usr/bin/gh", *args]


def host_environment(job):
    user = pwd.getpwuid(os.getuid())
    environment = {"HOME": user.pw_dir, "PATH": "/usr/local/bin:/usr/bin:/bin",
                   "LANG": "C.UTF-8", "GH_REPO": job["repo"],
                   "GH_PROMPT_DISABLED": "1", "GIT_TERMINAL_PROMPT": "0",
                   "GIT_PAGER": "cat", "GH_PAGER": "cat", "PAGER": "cat"}
    if job.get("ssh_auth_sock"):
        environment["SSH_AUTH_SOCK"] = job["ssh_auth_sock"]
    return environment


def verify_identity(job, environment):
    root = Path(job["root"]).resolve(strict=True)
    worktree = Path(job["worktree"]).resolve(strict=True)
    require(root != worktree, "Host commands require a separate issue worktree")
    repo = job["repo"]
    origins = {f"https://github.com/{repo}", f"https://github.com/{repo}.git",
               f"git@github.com:{repo}", f"git@github.com:{repo}.git",
               f"ssh://git@github.com/{repo}", f"ssh://git@github.com/{repo}.git"}

    def git_at(path, *args):
        result = subprocess.run(["/usr/bin/git", "-C", str(path), *args], env=environment,
                                text=True, capture_output=True, timeout=20, check=False)
        require(result.returncode == 0, "Registered Git worktree verification failed")
        return result.stdout.strip()

    for path in (root, worktree):
        require(git_at(path, "remote", "get-url", "origin") in origins,
                "Git origin differs from the owned repository")
    require(git_at(worktree, "symbolic-ref", "--short", "HEAD") == job["branch"],
            "Issue worktree branch changed")

    def common(path):
        return Path(git_at(path, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()

    require(common(root) == common(worktree), "Issue worktree belongs to another clone")
    return worktree


def execute_request(job, argv, cwd):
    try:
        require(isinstance(argv, list) and len(argv) >= 2 and
                all(isinstance(value, str) and len(value) <= 100_000 for value in argv) and
                argv[0] in ("git", "gh"), "Use host client git|gh ARGS")
        require(isinstance(cwd, str) and Path(cwd).resolve() == Path(job["worktree"]).resolve(),
                "Host client must run from the owned issue worktree")
        environment = host_environment(job)
        worktree = verify_identity(job, environment)
        command = git_args(argv[1:], job, worktree) if argv[0] == "git" else gh_args(argv[1:], job, worktree)
        if argv[:3] in (["gh", "pr", "edit"], ["gh", "pr", "patch"],
                        ["gh", "pr", "ready"]):
            number = next((value for value in argv[3:] if re.fullmatch(r"[1-9][0-9]*", value)), None)
            require(number is not None, "PR mutation needs an explicit PR number")
            observed = subprocess.run(["/usr/bin/gh", "api", f"repos/{job['repo']}/pulls/{number}",
                                       "--jq", ".head.ref"], cwd=worktree, env=environment,
                                      capture_output=True, text=True, timeout=30, check=False)
            require(observed.returncode == 0 and observed.stdout.strip() == job["branch"],
                    "PR mutation requires the owned issue head branch")
        completed = subprocess.run(command, cwd=worktree, env=environment, text=True,
                                   errors="replace", capture_output=True, timeout=180, check=False)
        return {"exit_code": completed.returncode, "stdout": completed.stdout[:1_000_000],
                "stderr": completed.stderr[:1_000_000]}
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "stdout": "", "stderr": "Host command timed out\n"}
    except (Rejected, OSError, ValueError, KeyError, RuntimeError) as error:
        return {"exit_code": 64, "stdout": "", "stderr": f"Host command rejected: {error}\n"}


class HostBroker:
    def __init__(self, job):
        self.job = job
        self.path = Path(job["host_socket"])
        self.stop = threading.Event()
        self.listener = None
        self.thread = None

    def start(self):
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.path.chmod(0o600)
        self.listener.listen(4)
        self.listener.settimeout(0.2)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while not self.stop.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    _, uid, _ = struct.unpack("3i", connection.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                    require(uid == os.getuid(), "Host client user differs from the runner")
                    data = bytearray()
                    while True:
                        part = connection.recv(65536)
                        if not part:
                            break
                        data.extend(part)
                        require(len(data) <= 200_000, "Host request is too large")
                    request = json.loads(data)
                    require(isinstance(request, dict), "Host request must be an object")
                    result = execute_request(self.job, request.get("argv"), request.get("cwd"))
                except (Rejected, OSError, ValueError, TypeError) as error:
                    result = {"exit_code": 64, "stdout": "", "stderr": f"Host command rejected: {error}\n"}
                except Exception:
                    result = {"exit_code": 70, "stdout": "", "stderr": "Host broker request failed\n"}
                try:
                    connection.sendall(json.dumps(result).encode("utf-8"))
                except OSError:
                    pass

    def close(self):
        self.stop.set()
        if self.listener is not None:
            self.listener.close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)
