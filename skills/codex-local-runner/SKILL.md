---
name: codex-local-runner
description: Install, repair, and verify the optional repository-scoped GitHub Actions bridge that launches Skillforge execution and audit turns through the Codex App Server shared with Desktop Remote Control.
---

# Codex Local Runner

## Responsibility

Use this skill to provision or repair Skillforge's optional host-side bridge. The public label entrypoint is `.github/workflows/codex-issue-state.yml`; it is the **only** workflow that reacts to `issues:labeled`. It routes:

- `execution-ready` to `.github/workflows/codex-execute-ready.yml` for the default/explicit `codex` executor, or `.github/workflows/devin-execute-ready.yml` for explicit local `devin`;
- `review-ready` to `.github/workflows/codex-review-ready.yml`;
- `completed` / `queued` child events back to the unique active epic parent when its canonical DAG contains that child.

The executor and audit workflows are reusable `workflow_call` workflows. They do not interpret labels themselves. The dispatcher may use a hosted runner for control-plane discovery; both local execution workflows and Codex audit use the existing repository-scoped self-hosted runner carrying `self-hosted` and `codex`. The latter is a physical runner label, not a forced executor selection.

The Codex executor never uses the Actions `_work` checkout as project state. It receives a durable clone through `SKILLFORGE_REPO_ROOT`, creates/reuses one persistent worktree and `codex/issue-N` branch, and connects to the **existing Codex App Server control socket used by Desktop Remote Control**. The audit workflow creates a fresh detached review worktree and fresh App Server thread for the exact PR head. When the controlling issue is closed, the dispatcher launches a local cleanup job for the registered `issue-N` worktree and detached `reviews/pr-*` worktrees belonging to that issue's PRs. It keeps Git branches intact and waits for an active PR-audit lock to release before deletion. Cleanup never force-removes a worktree: tracked, staged or untracked unfinished changes make cleanup fail closed and require explicit reconciliation.

GitHub Actions is authorization/routing/launch infrastructure; the actual Codex model turn is owned by the shared App Server and continues after the Actions job exits.

This skill owns Codex runner installation, registration, service configuration, durable-repository environment wiring, App Server prerequisites, repair, optional scaling, and verification. It does not implement issues or modify repository files.

Never execute this skill against the canonical `murillo128/skillforge` template repository itself.

## Executor selection boundary

Use `skills/execution-runner-selection/SKILL.md` and `docs/execution-runners.md`
to choose an executor independently of its model. Devin uses the installed local
CLI inside tmux on the same host, with actual persistent issue worktrees; it does
not use a cloud API, remote VM or the Codex App Server. Its host prerequisites
and inspection belong to `skills/devin-local-runner/SKILL.md`.

Keep the Codex App Server/runner for existing work and independent final audit.
Do not rename runner labels, copy credentials, migrate active sessions, edit
executor/session lease comments or relabel issues as a provisioning test.

## Authority and safety

Runner provisioning is opt-in. Explicit invocation grants only host/GitHub control-plane authority needed to install/register/repair/start/verify the target runner. It does not authorize repository edits, OpenAI API key creation, replacement of existing Codex auth, public listeners/ports, root service execution, removal of unrelated runners/services, or triggering a real issue merely as a test.

A self-hosted runner executes repository workflow code on the machine. Prefer repository-scoped registration, run it as the intended non-root user who owns the durable clone and Codex remote state, and never persist registration/removal tokens. Before enabling on a public repository, fail closed if untrusted fork/PR-controlled code can target the `codex` runner.

## Execution topology

Treat the Actions runner `_work` tree as disposable infrastructure. Each repository runner exposes:

`SKILLFORGE_REPO_ROOT=<absolute durable local clone>`

Optional overrides:

`SKILLFORGE_WORKTREE_ROOT=<persistent worktree parent>`
`SKILLFORGE_LOG_ROOT=<persistent event-log parent>`
`SKILLFORGE_CODEX_APP_SERVER_SOCKET=<shared App Server Unix socket>`

Defaults are `$HOME/.skillforge/worktrees`, `$HOME/.skillforge/logs`, and `${CODEX_HOME:-$HOME/.codex}/app-server-control/app-server-control.sock`.

For `<owner>/<repo>` issue `N`, execution uses persistent `codex/issue-N`. A retry validates/adopts the registered worktree rather than discarding unfinished state. The durable clone is coordination state; implementation occurs in issue worktrees.

### Shared App Server

For an SSH/Desktop project, Skillforge joins the same remote App Server through its Unix control socket. Do not start a second `codex app-server`, `codex exec`, or TUI as fallback. If the socket is absent/unreachable, restore/reconnect the intended Desktop SSH App Server and retry.

Codex executor semantics:

1. prepare/adopt the issue worktree;
2. create or resume the issue's durable thread;
3. start the turn with thread/turn cwd anchored to the issue worktree;
4. use on-request approvals, workspace-write and network access;
5. keep a detached WebSocket subscriber connected until matching `turn/completed`;
6. let Actions exit after `threadId`/`turnId` launch acknowledgement.

A scheduler wake may arrive while the parent turn is still active. The reusable executor accepts `wait_for_existing_turn=true` only for this routed follow-up case and serializes the next turn after the active client ends. Ordinary `execution-ready` launches refuse duplicate active turns.

Audit semantics are deliberately different: every audit attempt gets a fresh detached review worktree pinned to current exact PR head and a fresh App Server thread; it never resumes the executor conversation. The audit launcher removes executor worktree/branch variables and ephemeral Actions tokens before starting the review client.

The background subscriber is transport, not another model actor. Preserve empty `RUNNER_TRACKING_ID` detachment so it survives Actions cleanup.

Desktop project grouping is cosmetic: a thread may appear under Recents rather than the saved project. Repository/worktree/thread identity, not Desktop grouping, is operational authority.

## Preconditions

Before changing the host establish:

1. exact repository/default branch and confirm it is not canonical `murillo128/skillforge`;
2. `.github/workflows/codex-issue-state.yml` exists and is the only `issues:labeled` state router;
3. `.github/workflows/codex-execute-ready.yml` and `.github/workflows/codex-review-ready.yml` are reusable `workflow_call` workflows targeting `[self-hosted, codex]` and do not directly interpret label events;
4. executor path requires `SKILLFORGE_REPO_ROOT`, uses persistent worktrees, and does not use a runner `_work` checkout as implementation state;
5. durable clone absolute path, exact matching origin, and location outside `_work`;
6. OS/architecture/hostname/current user/service manager/disk state;
7. Git, Python 3, `setsid`, `flock` (for audit), and persistent GitHub transport as required;
8. shared Codex Remote Control App Server socket under the intended user;
9. installed App Server accepts current thread/turn protocol and environment overrides;
10. persistent Git/GitHub authentication usable by detached Codex turns;
11. current runner installations/services/registrations.

The hosted dispatcher may use `actions/checkout` only to read its routing/selection helpers. That does **not** relax the executor rule: self-hosted model execution must never treat Actions `_work` as project state.

## Idempotent provisioning

### 1. Identify durable repo and runner

Canonicalize `SKILLFORGE_REPO_ROOT`, verify origin and that it is outside `_work`. Use a stable runner name/install directory outside the repository. Reuse healthy installations.

### 2. Reuse before replacing

Inspect local service and GitHub registration first. Repair a correct runner rather than creating duplicates. Remove stale registrations only when ownership is unambiguous.

### 3. Register official runner securely

Resolve the current official Actions runner for detected OS/architecture; do not hard-code a version. Obtain a temporary repository registration token only at registration time and never print/persist it. Configure exact repo, normal self-hosted labels plus `codex`, persistent/non-ephemeral operation, and normal internal `_work` directory.

### 4. Configure persistent environment

Expose `SKILLFORGE_REPO_ROOT` and optional worktree/log/socket overrides via supported runner `.env` or narrow service environment. Preserve unrelated entries and restart after changes.

### 5. Install service

Use official runner service mechanism. Run as intended non-root user, active now and enabled at boot.

### 6. Verify without model execution

Verify service identity/active/boot enablement; GitHub runner online for exact repo with `self-hosted`/`codex`; environment paths/socket; exact repo origin and harmless fetch; required host tools; shared socket is reachable by intended user; dispatcher has the sole label trigger; reusable executor/audit target the codex runner; executor uses persistent issue worktree and shared App Server; audit uses fresh detached review worktree/session.

Do not create a dummy issue, add/remove workflow labels, or launch a model request merely to test provisioning. The first real transition is the end-to-end test.

## Repair behavior

Prefer repair. Wrong `_work` execution means fix `SKILLFORGE_REPO_ROOT`; missing socket means restore Desktop/shared App Server; protocol rejection means repair version compatibility; Recents grouping alone is not a topology defect; active thread without matching launcher fails closed; detached client dying after Actions means verify `RUNNER_TRACKING_ID`; missing GitHub mutations means repair persistent auth rather than persisting Actions token; offline runner means inspect service/network/environment.

Never interrupt a live model turn merely to repair the runner service without explicit authorization.

## Bootstrap integration

`repository-bootstrap` may offer runner setup only as an optional **post-bootstrap handoff**. Bootstrap validity never depends on runner presence and its file-write boundary is not widened. Runner failure after successful bootstrap is a separate retryable infrastructure failure.

## Completion report

Report target repository/durable root, runner name/install directory, service identity/state/boot enablement, worktree/log/socket paths, GitHub online/labels state, shared App Server reachability, whether setup reused/repaired/created/scaled, and any real auth/permission/workflow/protocol/host blocker. Never report tokens or credential contents.
