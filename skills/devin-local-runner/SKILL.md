---
name: devin-local-runner
description: Verify or explicitly provision the local Devin CLI and tmux adapter on the existing self-hosted machine without changing Codex sessions, issue ownership or authentication.
---

# Devin Local Runner

## Responsibility and authority

Use for local Devin host prerequisites, inspection and explicitly requested repair.
`docs/execution-runners.md` owns the execution/session/recovery contract; executor
selection belongs to `skills/execution-runner-selection/SKILL.md`.

This adapter replaces the former cloud launcher. Never configure a Cloud API
key, organization VM, `/handoff`, `--cloud`, or a second local Codex App Server.
Do not provision anything on the canonical `murillo128/skillforge` template repo.
Provisioning is opt-in and does not authorize real issue activation, paid test
turns, credential replacement, blanket permission bypass or live-session resets.

## Preconditions

Establish the exact repository/default branch, intended non-root runner user,
Linux host, persistent clone and existing `[self-hosted, codex]` registration.
The label identifies the physical runner and is shared by the two executors.
Do not rename it, register a redundant runner or interrupt a live Codex turn.

Verify `SKILLFORGE_REPO_ROOT` origin and location outside Actions `_work`, optional
worktree/log roots, Git, Python 3.11+, tmux and the installed `devin` on the service
user's PATH. Devin's help must expose `--print`, `--prompt-file`, `--export`,
`--resume`, `--config`, `--permission-mode` and `--respect-workspace-trust`.
Unattended execution also needs the CLI's `--sandbox` flag plus host `bwrap` and
`socat` executables.
Verify `devin auth status` and persistent `gh auth status --hostname github.com`
without Actions tokens. Login belongs to the user; never copy Codex authentication
or persist a workflow token.

For real unattended issue turns, the trusted launcher supplies a private per-turn
permission config, starts Devin in `--sandbox --permission-mode autonomous` and
disables the interactive workspace-trust prompt only after it has verified the
exact repository/worktree/branch identity. Sandbox shell writes are scoped to the
worktree and `/tmp`. No shell command is excluded from the CLI sandbox. For
every Git/GitHub operation, the launcher supplies a sandboxed client
that sends argument vectors to a Unix-socket broker in the supervisor. The broker
verifies the registered worktree/repository and allows only bounded operations
before using host credentials. Shell redirections, chains and substitutions
remain sandboxed. Direct `git`/`gh` calls can fail on host auth or shared Git/LFS
directory access and should not be used. The generated
policy denies privilege escalation, authentication/configuration mutation and
direct credential access while allowing read-only GitHub auth status through the
broker. The prompt
directs file edits through shell exec because direct edit/write
tools still prompt in autonomous mode. Organization policy still wins. Never use
`dangerous`/bypass mode or issue-provided permission settings. A structured ATIF
tool or broker rejection, known approval/trust diagnostic, or exit-0 turn with no
new tool result is not a successful turn. An explicit blocked final agent answer also
fails even after successful earlier tools. Preserve a valid exported session and
resume only after an authorized wake.

## Inspect launch infrastructure

Confirm `codex-issue-state.yml` is the only label listener; execution selection
routes Devin to `devin-execute-ready.yml`; the reusable workflow targets the local
host, checks the default branch and snapshots trusted scripts by `GITHUB_SHA`.
The worktree helper must prepare/adopt the actual registered branch and preserve
unfinished files, rather than delegate all Git setup to the model.

Confirm the supervisor runs local CLI inside an isolated tmux server, uses a
durable prompt/export/log/state directory, holds its lifetime lock, removes
Actions credentials/tracking and records exit/session identity and new ATIF tool
evidence. Acknowledgement
is not task completion. Resume must use the saved explicit ID, never `--continue`.
The final audit remains the existing fresh Codex audit workflow.

## Safe inspection and repair

Inspect `~/.skillforge/run/<owner>-<repo>/issue-N/devin/state.json`, the recorded
local log and exact tmux socket/session. Attach for monitoring and detach without
terminating the pane. Do not run a second Devin process against an active session.
A dead pane may intentionally remain for inspection after CLI completion.

Check the remote executor lease and local-host binding before adoption. Existing
cloud receipts, other-host bindings, Codex session state, missing/ambiguous local
receipts, mismatched exports and moved worktrees require explicit idle recovery.
Preserve the branch, uncommitted files, session export and prior run evidence.
Do not delete records or fall back to a new session just to make dispatch succeed.

A host restart cannot keep a tmux process alive; inspect the interrupted receipt
and session before an authorized retry. No heartbeat/cron or automatic provider
migration is supplied. Follow the detailed recovery contract in the operations doc.

## Verification and completion report

Prefer harmless version/help/auth/identity checks and the offline regression CI.
Never use a dummy issue, label transition or paid model request as a provisioning
smoke test without separate authorization. Do not claim installed-CLI compatibility
from a test double or that a started process completed the issue.

Report the verified host/user/repository, runner availability, CLI/tmux capability,
worktree/log/state identities, authentication/trust/permission status, actual changes
and any remaining prerequisite. Never print tokens, raw credentials or private logs.
