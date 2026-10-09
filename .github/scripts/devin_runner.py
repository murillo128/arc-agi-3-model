"""Local Devin CLI turns in isolated tmux sessions, not Devin Cloud.

Actions acknowledges a detached local supervisor. Durable receipts, explicit
session IDs and real Git worktrees survive the job; no Actions token does.
"""

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from executor_control import (ControlError, GitHub, SESSION_MARKER, devin_settings,
                              eligible, executor, read_record, require_lease)
from devin_host_broker import HostBroker
from local_issue_worktree import (context, durable_path, git, lock, prepare, run,
                                 verify_repo, verify_worktree)

LOCAL_MARKER = "<!-- skillforge-devin-local:v1 -->"
TERMINAL = {"finished", "failed"}
PERMISSION_REJECTION = b"rejected a tool call that requires confirmation"
TRUST_REJECTION = b"refusing to run in an untrusted workspace"
ATIF_SCHEMA = re.compile(r"ATIF-v1\.[0-9]+")
STRUCTURED_REJECTION = "tool execution was rejected by the user"
APPROVAL_DENIAL_PREFIX = "permission to run the command `"
APPROVAL_DENIAL_SUFFIX = "` was denied. the user needs to approve command execution."
HOST_REJECTION = re.compile(
    r"^output from command in shell [^\n]+:\n(?:host command rejected|host broker unavailable):"
    r".*\nexit code: (?:64|69|70)$", re.DOTALL)


class ExportError(ControlError):
    def __init__(self, message, saved_id=None):
        super().__init__(message)
        self.saved_id = saved_id


def clean_env(source):
    denied = {"GH_TOKEN", "GITHUB_TOKEN", "CI", "GITHUB_ACTIONS", "TMUX", "TMUX_PANE",
              "GIT_ASKPASS", "SSH_ASKPASS", "DEVIN_API_KEY", "DEVIN_ORG_ID", "DEVIN_MAX_ACU_LIMIT"}
    result = {k: v for k, v in source.items() if k not in denied and
              k not in {"DEVIN_PERMISSION_MODE", "DEVIN_SANDBOX", "DEVIN_MODEL"} and
              not k.startswith(("ACTIONS_", "RUNNER_", "CODEX_", "SKILLFORGE_CODEX_", "GIT_CONFIG_"))}
    result["RUNNER_TRACKING_ID"] = ""
    return result


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def read_json(path, *, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise ControlError(f"Unreadable local receipt: {path.name}; reconcile before retrying") from None
    if not isinstance(value, dict):
        raise ControlError("Local receipt must be an object")
    return value


def session_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", value):
        raise ControlError("Missing or invalid exported Devin session ID; never resume the latest session")
    return value


def read_export(path, expected=None):
    # CLI --export is ATIF; unsupported/missing schemas require reconciliation,
    # not a guessed private database location or a 'latest session' fallback.
    try:
        value = read_json(path)
    except ControlError:
        raise ExportError("Missing or malformed Devin ATIF export; reconcile before retrying") from None
    try:
        sid = session_id(value.get("session_id"))
    except ControlError:
        raise ExportError("Missing or invalid exported Devin session ID; reconcile before retrying") from None
    if expected is not None and sid != expected:
        raise ExportError("Devin exported a different session than the requested resume")
    steps = value.get("steps")
    if (not isinstance(value.get("schema_version"), str)
            or not ATIF_SCHEMA.fullmatch(value["schema_version"])
            or not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps)):
        raise ExportError("Missing or malformed Devin ATIF export; reconcile before retrying", sid)
    return sid, steps


def turn_evidence(steps, previous_steps):
    if not isinstance(previous_steps, int) or previous_steps < 0 or previous_steps > len(steps):
        raise ControlError("Devin export lost prior session steps; reconcile before retrying")
    executed = False
    rejected = False
    host_rejected = False
    for step in steps[previous_steps:]:
        if step.get("source") != "agent":
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict) or not isinstance(observation.get("results"), list):
            continue
        call_ids = {call.get("tool_call_id") for call in (step.get("tool_calls") or [])
                    if isinstance(call, dict) and isinstance(call.get("tool_call_id"), str)}
        for result in observation["results"]:
            if not isinstance(result, dict) or result.get("source_call_id") not in call_ids:
                continue
            content = result.get("content")
            if not isinstance(content, str):
                continue
            lowered = content.lower()
            normalized = lowered.strip()
            if (normalized == STRUCTURED_REJECTION or
                    (normalized.startswith(APPROVAL_DENIAL_PREFIX) and
                     normalized.endswith(APPROVAL_DENIAL_SUFFIX))):
                rejected = True
            elif HOST_REJECTION.fullmatch(normalized):
                host_rejected = True
            elif not lowered.startswith("tool call canceled because another tool call"):
                executed = True
    return executed, rejected, host_rejected


def agent_reported_block(steps, previous_steps):
    """Recognize an explicit blocked final answer, even after earlier tools ran."""
    if len(steps) <= previous_steps:
        return False
    final = steps[-1]
    if (final.get("source") != "agent" or final.get("tool_calls") or
            final.get("observation") or not isinstance(final.get("message"), str)):
        return False
    return re.match(r"\s*(?:blocked\b|execution is blocked\b)",
                    final["message"], re.IGNORECASE) is not None


def permission_config():
    # Autonomous mode keeps every shell command inside Devin's OS sandbox.
    # Authenticated host operations use the supervisor's checked socket broker.
    credential_rules = [
        "Read(~/.ssh/**)", "Read(~/.gnupg/**)", "Read(~/.config/gh/**)",
        "Read(~/.config/devin/**)", "Read(~/.local/share/devin/credentials.toml)",
        "Read(~/.git-credentials)", "Read(~/.netrc)", "Read(~/.aws/**)",
        "Write(~/.ssh/**)", "Write(~/.gnupg/**)", "Write(~/.config/gh/**)",
        "Write(~/.config/devin/**)", "Write(~/.local/share/devin/credentials.toml)",
        "Write(~/.git-credentials)", "Write(~/.netrc)", "Write(~/.aws/**)",
        "Write(.env*)", "Write(**/.env*)",
    ]
    command_denials = [
        "Exec(sudo)", "Exec(su)", "Exec(doas)", "Exec(pkexec)",
        "Exec(devin auth login)", "Exec(devin auth logout)",
        "Exec(gh auth login)", "Exec(gh auth logout)",
        "Exec(gh auth refresh)", "Exec(gh auth setup-git)",
        "Exec(gh auth switch)", "Exec(gh auth token)",
        "Exec(gh auth status --show-token)", "Exec(gh auth status -t)",
        "Exec(gh auth status --hostname github.com --show-token)",
        "Exec(gh auth status --hostname github.com -t)",
        "Exec(gh config)",
        "Exec(git credential)", "Exec(git config)",
    ]
    return {
        "permissions": {
            "allow": ["Read(**)", "Write(**)", "Write(/tmp/**)"],
            "deny": credential_rules + command_denials,
        },
    }


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, ValueError, IndexError, TypeError):
        return None


def tmux_args(socket, *args):
    return ["tmux", "-L", socket, "-f", "/dev/null", *args]


def active(state_dir, record, env):
    try:
        with lock(state_dir / "turn.lock", blocking=False):
            pass
    except BlockingIOError:
        return True
    if not record:
        return False
    for name in ("worker", "child"):
        pid, start = record.get(name + "_pid"), record.get(name + "_start")
        if pid and start and process_identity(pid) == start:
            return True
    panes = run(tmux_args(record["socket"], "list-panes", "-t", "=" + record["session"] + ":0",
                          "-F", "#{pane_dead}"), env=env, check=False)
    return panes.returncode == 0 and any(line == "0" for line in panes.stdout.splitlines())


def cli_command(binary, job):
    command = [binary, "--print", "--prompt-file", job["prompt"], "--export", job["export"],
               "--config", job["config"], "--sandbox", "--permission-mode", "autonomous",
               "--respect-workspace-trust", "false"]
    if job.get("session_id"):
        command += ["--resume", session_id(job["session_id"])]
    if "model" in job["settings"]:
        command += ["--model", job["settings"]["model"]]
    return command


def prompt(repo, number, host_client):
    return f"""Execute https://github.com/{repo}/issues/{number} using local Devin CLI.
You are in the prepared persistent issue worktree on this host, not Devin Cloud.
Read AGENTS.md and skills/execution-runner-selection/SKILL.md, then the live
controlling issue and its relevant top-level comments using persistent host auth.
Respect SKILLFORGE_LOCAL_RUNNER=1, SKILLFORGE_ISSUE_WORKTREE and
SKILLFORGE_ISSUE_BRANCH. Do not create another worktree or change to the durable
coordination clone. Preserve unfinished changes on resume. Re-read current state
and exclusive ownership before editing. Dispatcher-owned executor/local-host and
session receipts are read-only to you. Never modify, delete or duplicate them.
If execution_mode is epic-dag use skills/codex-epic-scheduler/SKILL.md; otherwise
use skills/spec-driven-codex-loop/SKILL.md. Verify canonical
codex-execution-context:v1, pinned base and PR target before implementation edits;
reconcile a reused branch according to that skill, never reset valid issue work.
Only the scheduler activates queued children; preserve dependencies, holds and
max_parallel_workers. Parent executor/model settings do not propagate to children.
Use only local tools and persistent host Git/gh credentials. Do not use /handoff,
--cloud, a cloud VM, the Codex App Server, or another issue's session or worktree.
Use the shell exec tool for file edits; direct edit/write tools require interactive
approval in the CLI sandbox. Keep shell work inside this issue worktree or /tmp.
For every Git and GitHub command, including read-only status, diff, ls-files,
fetch, PR list, and issue view, use the launcher-owned
sandboxed client: {host_client} git ARGS or {host_client} gh ARGS. Run it from the
issue worktree. It sends argument vectors to the local supervisor, which checks
repository identity and operations before using host credentials. No shell exec
command leaves the sandbox. The broker rejects Git global options,
config/credential commands, GitHub auth changes, unrelated repositories, and
unsupported operations. Do not call git or gh directly: the sandbox cannot
write to the shared Git/LFS directory, and direct gh lacks host authentication.
Use origin/codex/epic-issue-N after fetching an epic branch; the local branch
name may not exist. Use git remote get-url origin instead of git config for the
origin check. Make separate calls for separate GitHub requests.
Push this issue branch with one exec command:
{host_client} git push origin HEAD:refs/heads/codex/issue-{number}
The broker requires an explicit owned ref.
If gh pr edit fails due to GraphQL project-card access, use the host client's
gh pr patch NUMBER --body-file PATH or --base BRANCH for the owned PR.
On resume, retry safe commands rejected by an earlier permission policy; this
turn's sandbox runs shell exec unattended. Report any new rejection you observe.
Never recover Actions tokens, change host authentication or bypass approvals.
Finish at a ready PR and review-ready handoff to the independent Codex audit.
Never merge, enable auto-merge, close issues, mark completed, or mutate GitHub
after review-ready. A successful CLI exit is not task completion or test evidence.
Report concrete missing tools/auth/model/test infrastructure; do not switch executor.
"""


def worker(job_path):
    job = read_json(job_path)
    state_path = Path(job["state"])
    env = clean_env(os.environ)
    with lock(state_path.parent / "turn.lock", blocking=False):
        record = read_json(state_path)
        if record.get("run_id") != job["run_id"] or record.get("phase") != "pending":
            raise ControlError("Turn receipt is not the expected pending launch")
        child = None
        broker = None
        permission_rejected = False
        trust_rejected = False
        try:
            verify_worktree(Path(job["root"]), job["worktree"], job["repo"], job["branch"])
            record.update(worker_pid=os.getpid(), worker_start=process_identity(os.getpid()))
            # Receipt is already pending before Popen; no retry after uncertain launch.
            atomic_json(state_path, record)
            broker = HostBroker(job)
            broker.start()
            env.update(SKILLFORGE_LOCAL_RUNNER="1", SKILLFORGE_EXECUTOR="devin",
                       SKILLFORGE_ISSUE_WORKTREE=job["worktree"],
                       SKILLFORGE_ISSUE_BRANCH=job["branch"],
                       SKILLFORGE_REPO_ROOT_RESOLVED=job["root"],
                       GITHUB_REPOSITORY=job["repo"], ISSUE_NUMBER=job["number"])
            with open(job["log"], "ab", buffering=0) as output, open(job["log"], "rb") as mirror:
                mirror.seek(0, os.SEEK_END)
                permission_probe_tail = b""
                # A log file, not a pipe/PTY, prevents background command children
                # from holding the supervisor's output stream open after CLI exit.
                child = subprocess.Popen(cli_command(job["binary"], job), cwd=job["worktree"],
                                         env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT)
                def interrupt(signum, frame):
                    if child.poll() is None:
                        child.terminate()
                    raise ControlError("Local supervisor interrupted; reconcile the session before retrying")
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    signal.signal(sig, interrupt)
                record.update(phase="running", child_pid=child.pid,
                              child_start=process_identity(child.pid))
                atomic_json(state_path, record)
                while True:
                    data = mirror.read(65536)
                    if data:
                        probe = permission_probe_tail + data
                        lowered = probe.lower()
                        if PERMISSION_REJECTION in lowered:
                            permission_rejected = True
                        if TRUST_REJECTION in lowered:
                            trust_rejected = True
                        permission_probe_tail = probe[-256:]
                        # tmux scrollback is convenience; the local log is durable.
                        with suppress(BrokenPipeError, OSError):
                            os.write(sys.stdout.fileno(), data)
                    elif child.poll() is not None:
                        break
                    else:
                        time.sleep(0.1)
                rc = child.wait()
                record["exit_code"] = rc
                record["session_id"], steps = read_export(job["export"], job.get("session_id"))
                executed, structured_rejection, host_rejection = turn_evidence(
                    steps, job["previous_steps"])
                if trust_rejected:
                    record["error"] = "workspace-trust-rejection"
                    record["phase"] = "failed"
                elif permission_rejected or structured_rejection:
                    # Devin may exit 0 after rejecting non-interactive approvals.
                    # Treat that as a failed turn so an authorized wake can resume
                    # the exact exported session after launcher policy is corrected.
                    record["error"] = "permission-rejection"
                    record["phase"] = "failed"
                elif host_rejection:
                    record["error"] = "host-command-rejection"
                    record["phase"] = "failed"
                elif rc != 0:
                    record["error"] = "cli-failure"
                    record["phase"] = "failed"
                elif agent_reported_block(steps, job["previous_steps"]):
                    record["error"] = "agent-blocked"
                    record["phase"] = "failed"
                elif not executed:
                    # A resumed agent may only repeat an old approval request.
                    record["error"] = "no-tool-progress"
                    record["phase"] = "failed"
                else:
                    record["phase"] = "finished"
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if trust_rejected:
                error = "workspace-trust-rejection"
            elif permission_rejected:
                error = "permission-rejection"
            elif isinstance(exc, ExportError) or (isinstance(exc, ControlError) and
                                                  "export" in str(exc).lower()):
                error = "invalid-export"
            else:
                error = type(exc).__name__
            if isinstance(exc, ExportError) and exc.saved_id:
                record["session_id"] = exc.saved_id
            record.update(phase="needs-reconciliation", error=error)
            atomic_json(state_path, record)
            raise
        finally:
            if broker is not None:
                broker.close()
        atomic_json(state_path, record)
        if record["phase"] != "finished" and record["exit_code"] == 0:
            return 75
        return record["exit_code"]


def launch(gh, number, root, worktree_base, state_dir, log_dir, default_branch, run_id,
           *, wait=False, sleep=time.sleep, clock=time.monotonic, wait_seconds=900):
    if not re.fullmatch(r"[1-9][0-9]*", str(number)) or not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ControlError("Invalid issue number or Actions run ID")
    number = str(number)
    root = verify_repo(root, gh.repo)
    env = clean_env(os.environ)
    binary = shutil.which("devin", path=env.get("PATH"))
    if not binary or not shutil.which("tmux", path=env.get("PATH")):
        raise ControlError("Install/authenticate local Devin CLI and tmux for the runner user first")
    help_text = run([binary, "--help"], env=env).stdout
    for flag in ("--print", "--prompt-file", "--export", "--resume", "--config",
                 "--sandbox", "--permission-mode", "--respect-workspace-trust"):
        if flag not in help_text:
            raise ControlError(f"Installed Devin CLI lacks required capability {flag}")
    if not shutil.which("bwrap", path=env.get("PATH")) or not shutil.which("socat", path=env.get("PATH")):
        raise ControlError("Devin's unattended OS sandbox requires bwrap and socat on the runner host")
    run([binary, "auth", "status"], env=env)
    # Persistently configured gh must work without the short-lived Actions token.
    run(["gh", "auth", "status", "--hostname", "github.com"], env=env)
    state_dir, log_dir = durable_path(state_dir), durable_path(log_dir)
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    machine = Path("/etc/machine-id").read_text().strip()
    host = hashlib.sha256(f"{machine}:{os.getuid()}:{root}".encode()).hexdigest()
    binding = {"transport": "cli-tmux", "host": host}
    state_path = state_dir / "state.json"

    def current():
        issue = gh.issue(number)
        if not eligible(issue, wait):
            return None
        if executor(issue["body"]) != "devin":
            raise ControlError("Executor selection changed before local launch")
        require_lease(gh, number, "devin")
        comments = gh.comments(number)
        if any(SESSION_MARKER in (c.get("body") or "") for c in comments):
            raise ControlError("Legacy cloud session receipt exists; explicit migration required")
        _, owner = read_record(comments, LOCAL_MARKER)
        if owner is not None and owner != binding:
            raise ControlError("Issue is bound to another local host/user/clone; do not start a replacement")
        return devin_settings(issue["body"]), context(comments, number), owner

    with lock(state_dir / "launch.lock", blocking=False):
        selected = current()
        if selected is None:
            return {"result": "skipped-current-state"}
        record = read_json(state_path, optional=True)
        if record and (record.get("repo") != gh.repo or record.get("number") != number
                       or record.get("host") != host):
            raise ControlError("Local receipt belongs to a different issue/host")
        deadline = clock() + wait_seconds
        while active(state_dir, record, env):
            if not wait:
                return {"result": "already-active"}
            if clock() >= deadline:
                raise ControlError("Active Devin turn exceeded bounded scheduler wait; retry later")
            sleep(10)
            selected = current()
            if selected is None:
                return {"result": "skipped-current-state"}
            record = read_json(state_path, optional=True)
        if record and record.get("phase") not in TERMINAL:
            raise ControlError("Incomplete local receipt; reconcile logs/session before retrying")
        if record and record.get("run_id") == run_id:
            if record["phase"] == "failed":
                raise ControlError("This Actions run already ended with CLI failure; inspect local logs")
            return {"result": "already-acknowledged"}
        if selected[2] is not None and not record:
            raise ControlError("Local session receipt is missing on the bound host; do not create another session")
        # Never adopt a pre-existing Codex session as local Devin work.
        codex_state = state_dir.parent
        if any((codex_state / name).exists() for name in
               ("app-server-thread-id", "app-server-client.pid", "codex.pid")):
            raise ControlError("Codex session state exists; explicit idle ownership transfer required")
        settings, activation, owner = selected
        if record is None and git(root, "show-ref", "--verify", "--quiet",
                                  f"refs/heads/codex/issue-{number}", check=False).returncode == 0:
            raise ControlError("Unowned local issue branch exists; explicitly reconcile legacy work")
        if activation:
            gh.issue(activation["epic_issue"])
        worktree = prepare(root, worktree_base, gh.repo, number, default_branch, activation)
        if record and record.get("worktree") != str(worktree):
            raise ControlError("Recorded session worktree moved; explicit idle migration required")
        refreshed = current()
        if refreshed is None:
            return {"result": "skipped-current-state"}
        if refreshed != selected:
            raise ControlError("Issue settings/context/ownership changed during worktree preparation")
        previous_steps = 0
        if record:
            prior_session = session_id(record.get("session_id"))
            previous_run = record.get("run_id")
            if not isinstance(previous_run, str) or not re.fullmatch(r"[1-9][0-9]*", previous_run):
                raise ControlError("Invalid previous Actions run ID in local receipt")
            _, prior_steps = read_export(
                state_dir / f"run-{previous_run}" / "conversation.json", prior_session)
            previous_steps = len(prior_steps)
        turn_dir = state_dir / f"run-{run_id}"
        # Exclusive mkdir refuses replay of even a partially written launch bundle.
        turn_dir.mkdir(mode=0o700)
        for name in ("devin_runner.py", "executor_control.py", "local_issue_worktree.py",
                     "devin_host_broker.py", "devin_host_client.py"):
            shutil.copyfile(Path(__file__).with_name(name), turn_dir / name)
        host_client = turn_dir / "devin_host_client.py"
        host_client.chmod(0o700)
        host_socket = Path("/tmp") / ("sf-devin-host-" + hashlib.sha256(
            f"{gh.repo}:{number}:{run_id}".encode()).hexdigest()[:24] + ".sock")
        (turn_dir / "prompt.txt").write_text(prompt(gh.repo, number, host_client))
        permission_config_path = turn_dir / "devin-config.json"
        atomic_json(permission_config_path, permission_config())
        socket = "sf-devin-" + hashlib.sha256(f"{gh.repo}:{number}".encode()).hexdigest()[:20] + "-" + run_id
        session = f"issue-{number}"
        if record:
            # Only our verified inactive pane, never a user's tmux server.
            run(tmux_args(record["socket"], "kill-session", "-t", "=" + record["session"]), env=env, check=False)
        pending = {"phase": "pending", "run_id": run_id, "repo": gh.repo, "number": number,
                   "host": host, "socket": socket, "session": session,
                   "worktree": str(worktree), "settings": settings}
        if record:
            pending["session_id"] = record["session_id"]
        pending["previous_steps"] = previous_steps
        job = {**pending, "state": str(state_path), "root": str(root),
               "default_branch": default_branch,
               "branch": f"codex/issue-{number}", "binary": binary,
               "host_client": str(host_client), "host_socket": str(host_socket),
               "ssh_auth_sock": env.get("SSH_AUTH_SOCK"),
               "prompt": str(turn_dir / "prompt.txt"), "export": str(turn_dir / "conversation.json"),
               "config": str(permission_config_path),
               "log": str(log_dir / f"issue-{number}-{run_id}-devin.log")}
        atomic_json(turn_dir / "job.json", job)
        atomic_json(state_path, pending)
        if owner is None:
            gh.save_record(number, LOCAL_MARKER, binding)
        # The dedicated tmux server and all descendants receive sanitized env.
        # No shell interpolation of issue prose; prompt is a private local file.
        command = shlex.join([sys.executable, str(turn_dir / "devin_runner.py"),
                              "worker", str(turn_dir / "job.json")])
        run(tmux_args(socket, "new-session", "-d", "-s", session, "-c", str(worktree),
                      command, ";", "set-option", "-w", "-t", "=" + session + ":0",
                      "remain-on-exit", "on"), env=env)
        for _ in range(100):
            observed = read_json(state_path)
            if observed.get("phase") in {"running", "finished"}:
                return {"result": "launched-local-cli", "socket": socket, "session": session,
                        "log": job["log"], "state": str(state_path)}
            if observed.get("phase") in {"failed", "needs-reconciliation"}:
                raise ControlError("Devin CLI failed to start/finish; inspect local receipt and log")
            sleep(0.2)
        raise ControlError("Local launch acknowledgement timed out; reconcile before retrying")


def main():
    os.umask(0o077)
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        return worker(sys.argv[2])
    if sys.argv[1:] != ["launch"]:
        raise ControlError("usage: devin_runner.py launch | worker JOB_JSON")
    if os.environ.get("GITHUB_REF") != "refs/heads/" + os.environ["DEFAULT_BRANCH"]:
        raise ControlError("Local Devin launches require the trusted default-branch workflow")
    repo, number = os.environ["GITHUB_REPOSITORY"], os.environ["ISSUE_NUMBER"]
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ControlError("Invalid controlling issue number")
    gh = GitHub(repo, os.environ.get("GITHUB_TOKEN"))
    repo_key = repo.replace("/", "-")
    home = Path.home()
    result = launch(gh, number, os.environ["SKILLFORGE_REPO_ROOT"],
                    os.environ.get("SKILLFORGE_WORKTREE_ROOT", str(home / ".skillforge/worktrees")),
                    home / ".skillforge/run" / repo_key / f"issue-{number}" / "devin",
                    Path(os.environ.get("SKILLFORGE_LOG_ROOT", str(home / ".skillforge/logs"))) / repo_key,
                    os.environ["DEFAULT_BRANCH"], os.environ["GITHUB_RUN_ID"],
                    wait=os.environ.get("WAIT_FOR_EXISTING_TURN") == "true")
    print(json.dumps(result, sort_keys=True))
    # No private host paths/log contents in public Actions summaries.
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
        output.write(f"Local Devin CLI: **{result['result']}**. Launch is not task completion.\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControlError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Local Devin runner stopped: {type(exc).__name__}: {exc}") from None
