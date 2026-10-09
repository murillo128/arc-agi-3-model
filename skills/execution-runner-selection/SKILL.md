---
name: execution-runner-selection
description: Select local Codex or local Devin CLI independently of model settings, preserve issue/worktree/session ownership and the independent audit boundary.
---

# Execution Runner Selection

## Responsibility

Use when designing or changing an issue's executor or inspecting its launch
configuration. `docs/execution-runners.md` owns configuration, prerequisites and
recovery. This skill does not grant activation, provisioning, authentication
changes, active ownership transfer or merge authority.

## Selection procedure

1. Read the exact controlling issue and its single workflow-state label.
2. Retain Codex when unspecified, or choose the user's explicit local executor.
   Do not infer an executor from a model name or prose.
3. Use one top-level `execution` TOML fence in the issue body with
   `executor = "codex"` or `executor = "devin"`.
4. Keep optional Codex model/effort/profile in `codex`. Local Devin accepts only
   optional `model` in its separate `devin` fence. Omission keeps native selection;
   do not translate Codex roles/efforts or silently substitute models.
5. Reject old cloud `devin_mode`/`max_acu_limit` fields rather than treating them
   as CLI settings. There is no cloud fallback or issue-supplied command, path,
   credential, permission mode, or arbitrary launcher argument.
6. Inspect existing executor/host/session records and local ownership before
   changing a started issue. A body edit does not migrate a session or worktree.
7. Verify the selected local environment can satisfy validation. Missing tools,
   trust, permissions, authentication or resources are concrete prerequisites,
   not reasons to switch executor or weaken validation.
8. Preserve labels/holds unless activation is separately authorized. Report actual
   selection and unverified installed-CLI/host capabilities honestly.

## Local execution contract

Codex uses its existing App Server. Devin uses the installed CLI in a dedicated
tmux session on the same local runner, never its Cloud API or `/handoff`. Both
retain `SKILLFORGE_REPO_ROOT`, persistent issue worktree/branch identity, and
`SKILLFORGE_LOCAL_RUNNER=1`; do not leave the assigned worktree or manufacture a
second implementation branch. Existing dirty work is preserved.

Devin's launcher prepares and verifies the actual Git worktree, snapshots a
supervisor, removes ephemeral Actions credentials, records CLI completion, and
resumes only its confirmed explicit session ID. tmux keeps the turn independent
of the Actions job. The launcher verifies repository/worktree trust before
disabling the interactive trust prompt, then uses Devin's OS sandbox for
unattended shell execution under a launcher-owned permission policy.
Codex-only App Server/profile/delegation settings do not configure Devin helpers.

Use `skills/devin-local-runner/SKILL.md` for Devin host prerequisites/inspection;
Codex host provisioning stays in `skills/codex-local-runner/SKILL.md`. No new
`devin` Actions runner label is required: the existing `codex` label identifies
the installed physical host. Do not rename it or restart live sessions.

## Epics and audit

Each child is an independent controlling issue. The parent selects only its
scheduler. For a whole-epic request, intentionally update every requested child;
never imply inheritance. Mixed-provider children retain the canonical DAG,
integration base, dependency/mutex constraints and `max_parallel_workers`.
Only the scheduler activates `queued` children.

Historical `codex/issue-N`, `codex-epic-dag:v1`, `codex-execution-context:v1` and
skill names remain shared workflow identifiers. Both local executors follow the
ordinary execution or epic-scheduler skill. Final `review-ready` audit remains a
fresh independent Codex session; a Devin issue's `codex` block is for that audit.
Implementation sessions never acquire merge/completion or self-review authority.

## Infrastructure ownership

The Actions dispatcher owns `skillforge-executor:v1` and
`skillforge-devin-local:v1` comments. Executors and schedulers may read them, not
edit/delete/duplicate them. A legacy cloud `skillforge-devin-session:v1` receipt
blocks local launch until explicit reconciliation. A pending local receipt may
represent a started process: inspect it before any retry or migration.

Do not launch paid sessions, relabel real issues, copy credentials, alter native
permissions, or restart existing turns merely to test configuration. Offline
CLI doubles and real tmux tests do not prove the user's installed CLI works.
