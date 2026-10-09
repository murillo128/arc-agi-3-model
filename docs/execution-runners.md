# Execution runners

This document owns executor selection, persistent local execution, prerequisites
and recovery. [Codex operating policy](codex-operations.md) owns Codex model
selection. Configuration changes are not authorization to activate an issue,
provision a host, change credentials or migrate a session.

## Independent executor and model selection

The default executor is Codex through the existing shared Codex App Server.
Select installed local Devin CLI explicitly in the controlling issue body:

```execution
executor = "devin"
```

The only executor values are `codex` and `devin`. An optional separate `devin`
TOML fence accepts only `model`; omission preserves the installed CLI's native
selection. Use the account's installed CLI model catalog to verify an exact ID.
Do not derive an executor from a model name. `devin_mode`, `max_acu_limit`, custom
commands, paths, credentials, permission settings and arbitrary CLI arguments
are rejected; there is no Cloud API fallback.

A separate optional `codex` fence accepts `model`, `model_reasoning_effort` and
`profile`. On a Devin issue it configures only the fresh independent Codex audit,
not Devin. Explicit model settings are validated rather than silently replaced.
Quoted examples, comments, prose and a parent's settings do not configure a child.
Duplicate, malformed, oversized and unsupported selections fail closed.

Every epic child and parent selects independently. A parent chooses its scheduler;
changing a whole epic intentionally requires changing every requested child.
Mixed executors preserve the same canonical DAG, dependency/mutex rules,
integration base and `max_parallel_workers`. Only the scheduler activates queued
children. Preserve user activation holds; a completed specification or selection
change is not permission to release a hold.

## Routing, ownership and audit

`.github/workflows/codex-issue-state.yml` is the only issue-label listener. It
routes `execution-ready` to the selected reusable executor and `review-ready` to
Codex audit. Completed/queued child events wake only the unique active canonical
parent. Delayed events re-read current state and do not release manual holds.

The physical runner label remains `[self-hosted, codex]` for both executors.
Do not register redundant runners or rename labels to choose Devin. GitHub
Actions owns `skillforge-executor:v1` and `skillforge-devin-local:v1` receipts;
executors and schedulers may read them, not edit/delete/duplicate them.

Implementation and scheduling follow the same existing skills regardless of
executor. A successful executor handoff is a ready PR and `review-ready`, never
self-approval, merge, issue closure or `completed`. Final audit always uses a
fresh Codex thread and detached review worktree pinned to the exact PR head.
Historical `codex/issue-N` branches and `codex-*` workflow markers remain shared
identifiers, not a requirement that implementation use Codex.

## Persistent worktrees and terminal cleanup

`SKILLFORGE_REPO_ROOT` points to a durable clone outside Actions `_work`.
Implementation uses its registered persistent issue worktree and
`codex/issue-N` branch. Defaults for worktrees/logs are
`~/.skillforge/worktrees` and `~/.skillforge/logs`; existing host-specific overrides
remain valid. New epic children use the canonical pinned integration base.
Retries verify/adopt the registered worktree and preserve unfinished files.

An issue-close event removes only that issue's registered implementation worktree
and matching detached `reviews/pr-*` worktrees for its associated PRs. Cleanup
verifies repository origin and managed paths, waits for active PR-audit locks,
retains branch refs and reports unregistered leftovers rather than deleting them.
This terminal operation uses force removal for those verified closed-issue
worktrees, so it is not a general-purpose cleanup command for unfinished work.
Open-issue work is outside its scope. Cleanup does not authorize a model turn.

Runner provisioning is never performed against the canonical
`murillo128/skillforge` template repository. Its issue-close cleanup job is
therefore skipped; repositories created from the template retain the cleanup
behavior once their local runner is explicitly configured.

## Local Devin transport and permissions

The trusted default-branch launcher snapshots its Python helpers at `GITHUB_SHA`,
prepares/verifies the worktree, writes a private per-turn prompt/config/job bundle,
and starts a detached supervisor in an isolated tmux server. The supervisor uses
a lifetime lock and atomic durable receipts. Actions launch acknowledgement is
neither CLI completion nor task success.

The detached environment removes Actions tokens, tracking and provider-selection
variables. Git/GitHub authentication is the user's existing persistent host
authentication; a workflow token must never be recovered or persisted for the
model. The implementation shell remains in Devin's OS sandbox with
`--sandbox --permission-mode autonomous`. The launcher disables the interactive
workspace-trust prompt only after repository/worktree/branch verification.
Organization policy remains authoritative. No dangerous/bypass mode or
issue-provided permissions are supported.

File edits use shell execution in the worktree or `/tmp`; direct edit/write tools
can still require approval. Every Git/GitHub operation, including status, diff,
ls-files and issue/PR reads, uses the launcher-owned sandbox client. It sends
argument vectors to the supervisor's Unix-socket host broker, which verifies
repository/worktree identity and permits only bounded operations before using
host credentials. Direct Git/GitHub commands may fail on authentication or the
shared Git/LFS directory outside the sandbox.

The broker disables Git hooks, rejects Git global/config/credential overrides,
other repositories, unsupported operations and authentication changes. Pushes
require explicit owned branch refs; PR mutations require the owned head. It
supports an owned-PR REST `pr patch` fallback for GraphQL project-card failures.
Shell chains, substitutions and redirections do not execute on the host through
the broker. Credential files and privilege escalation remain excluded by the
launcher policy. This is not a claim of stronger isolation than the installed
CLI, OS sandbox and host configuration actually provide.

## Completion evidence and recovery

A valid supported ATIF export must identify the exact session and contain a valid
step sequence. Resume always names that saved session; never use `--continue`, a
latest-session guess or a replacement cloud session. A previous export/session
mismatch or ambiguous pending receipt requires reconciliation, not a fresh launch.

Known permission/trust diagnostics, structured ATIF approval denials, host-broker
rejections, a nonzero CLI exit, an explicit blocked final answer, and an exit-zero
turn with no new tool result are not successful turns. Only new results since the
previous export count toward progress. Preserve valid exported session identity
for a separately authorized resume after correcting the concrete prerequisite.
Malformed/missing exports and uncertain launches require explicit reconciliation.

Inspect `~/.skillforge/run/<owner>-<repo>/issue-N/devin/state.json`, its recorded
local log, and the exact tmux socket/session. A dead pane may intentionally remain
for diagnosis. Attach/detach for observation without killing it or starting a
second process. Never overwrite active ownership or delete receipts to make a
retry succeed. A host restart cannot preserve its tmux process.

Legacy cloud receipts, another host/user/clone binding, existing Codex thread
state, missing local receipts or moved worktrees block automatic adoption. Any
migration requires explicit idle reconciliation while preserving branch,
uncommitted changes, session exports and prior evidence. Selection edits do not
steer active turns. No automatic provider migration or heartbeat is introduced.

## Host prerequisites and offline validation

Verify the intended non-root Linux service user's Git, Python 3.11+, tmux,
installed Devin CLI, `bwrap`, `socat`, durable clone and persistent authentication.
The CLI must support print/prompt-file/export/resume/config/sandbox/permission-mode
and workspace-trust flags. Login is performed by the user. Do not copy another
repository's paths, CLI version, model availability or live-session history as
facts about this repository. Host provisioning and real model smoke tests require
separate authorization.

Run the full offline suite with Python 3.11+ and tmux:

```sh
REQUIRE_TMUX_TEST=1 python3 -m unittest discover -s .github/scripts -p 'test_*.py' -v
```

The suite uses real temporary Git worktrees and tmux with fake model clients;
it makes no paid model calls. It tests routing, policy, lifecycle failure/resume,
worktree cleanup and deterministic epic discovery/waves. Passing it does not
prove live-host readiness, installed-CLI sandbox enforcement or model availability.
Use `skills/devin-local-runner/SKILL.md` and
`skills/codex-local-runner/SKILL.md` for explicitly authorized host work.
