"""Build a fresh audit client from the trusted, published App Server transport.

The implementation launcher's host-local generated client and state are never
inputs. Keep its WebSocket protocol, but replace the whole resume/start decision
with an audit-only thread/start. Fail closed when the source contract changes.
"""

import ast
from pathlib import Path
import sys
import textwrap


FRESH_THREAD = '''    if os.path.exists(THREAD_FILE):
        raise RuntimeError("An audit attempt must never resume an existing thread")
    started = client.request(
        "thread/start",
        {
            "cwd": WORKTREE,
            "approvalPolicy": "on-request",
            "approvalsReviewer": "auto_review",
            "sandbox": "workspace-write",
            "serviceName": "skillforge",
            **policy.thread_options(),
        },
    )
    live_thread = started.get("thread") or {}
    thread_id = live_thread.get("id")
    if not thread_id:
        raise RuntimeError("thread/start returned no thread id")
    write_atomic(THREAD_FILE, thread_id)
    policy.confirm(started)
    display_name = (
        f"Review PR #{os.environ['PR_NUMBER']} / issue #{ISSUE_NUMBER}"
        f" / {os.environ['REVIEW_HEAD_SHA'][:12]}"
    )
    client.request("thread/name/set", {"threadId": thread_id, "name": display_name})
    log("thread_started", threadId=thread_id, name=display_name, cwd=WORKTREE)

'''


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Published App Server client contract changed: " + old[:80])
    return source.replace(old, new, 1)


def build_audit_client(workflow: str) -> str:
    """Reuse only published protocol code; return a fresh-session audit client."""
    begin = '          cat > "$client" <<\'PY\'\n'
    end = '\n          PY\n'
    if workflow.count(begin) != 1:
        raise ValueError("Expected exactly one published App Server client heredoc")
    body = workflow.split(begin, 1)[1]
    if end not in body:
        raise ValueError("Unterminated App Server client heredoc")
    source = textwrap.dedent(body.split(end, 1)[0]) + "\n"

    start_marker = '    thread_id = None\n    if os.path.exists(THREAD_FILE):\n'
    end_marker = '    policy.log_confirmation(log, thread_id=thread_id, cwd=WORKTREE, resumed=was_resumed)\n'
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("Published thread lifecycle contract changed")
    start = source.index(start_marker)
    stop = source.index(end_marker, start)
    source = source[:start] + FRESH_THREAD + source[stop:]
    source = replace_once(
        source,
        'WORKTREE = os.environ["SKILLFORGE_ISSUE_WORKTREE"]',
        'WORKTREE = os.environ["SKILLFORGE_REVIEW_WORKTREE"]',
    )
    source = replace_once(
        source,
        '    prompt = f"Execute GitHub issue #{ISSUE_NUMBER} according to AGENTS.md."',
        '    prompt = os.environ["SKILLFORGE_TASK_PROMPT"]',
    )
    source = replace_once(
        source,
        'f"skillforge:{os.environ.get(\'GITHUB_REPOSITORY\', \'repo\')}:{ISSUE_NUMBER}:{RUN_ID}"',
        'f"skillforge-review:{os.environ[\'GITHUB_REPOSITORY\']}:{os.environ[\'PR_NUMBER\']}:{RUN_ID}:{os.environ[\'GITHUB_RUN_ATTEMPT\']}"',
    )
    tree = ast.parse(source)
    forbidden = {"thread/resume", "thread/read", "thread/fork", "SKILLFORGE_ISSUE_WORKTREE"}
    if any(isinstance(node, ast.Constant) and isinstance(node.value, str)
           and node.value in forbidden for node in ast.walk(tree)):
        raise ValueError("Audit client still contains an executor-session entry point")
    return source


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_pr_audit.py <published-workflow> <new-client>")
    client = build_audit_client(Path(sys.argv[1]).read_text(encoding="utf-8"))
    compile(client, sys.argv[2], "exec")
    with Path(sys.argv[2]).open("x", encoding="utf-8") as handle:
        handle.write(client)
    Path(sys.argv[2]).chmod(0o700)


if __name__ == "__main__":
    main()
