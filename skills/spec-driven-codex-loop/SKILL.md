---
name: spec-driven-codex-loop
description: Execute an approved controlling issue through bounded implementation, repository-native validation, publication, integration stabilization, and a review-ready handoff.
---

# Spec-Driven Codex Loop

## Responsibility

Use this skill for non-trivial implementation under an approved controlling issue. The issue is the task-specific contract; repository documents define durable architecture and project-wide constraints; branches/PRs preserve implementation; tests and evidence preserve observed behavior.

The executor owns implementation, validation, commits, publication, explicitly issue-declared intermediate review checkpoints, and handoff. Delegate GitHub mechanics to `codex-github-operations` and intermediate independent review to `codex-independent-review`. The executor may not independently review its own work.

The terminal delivery state is a **ready-for-review PR** and issue state `review-ready`, not merged. Do not invoke a duplicate final independent review merely to reach that state: `codex-pr-audit` owns the final independent review and verdict-derived completion path.

This skill does not execute an epic parent. If the controlling issue declares `execution_mode: epic-dag`, route to `codex-epic-scheduler` instead.

## Context and authority

Load `AGENTS.md`, the controlling issue and only top-level comments needed to resolve execution context, exact authoritative design/source/test/config/evidence needed for the current outcome, and only the skill that owns the current action. Do not weaken the issue or silently promote exploratory material into requirements. Missing material design returns to design authority.

On resume, verify branch, `HEAD`, worktree, single workflow-state label, canonical execution context when present, and new material issue/PR discussion. Reuse already inspected facts while their source identity is unchanged.

## Canonical execution context

Search top-level controlling-issue comments for exact marker:

```text
<!-- codex-execution-context:v1 -->
```

With zero matches, treat the issue as standalone: PR target is repository default branch and do not infer an epic parent. With exactly one match, parse exactly one usable `epic_issue`, `integration_branch`, and full 40-hex `base_sha`. The context is authoritative only for execution base and PR target; child body remains authoritative for technical scope/acceptance. More than one or malformed required fields fails closed.

Validate that `integration_branch` exists, contains `base_sha` in history, and `epic_issue` exists and differs from the child. Ordinary forward branch movement is valid; rewritten history that drops `base_sha` is a control-plane defect.

### Prepare issue branch from pinned base

Implementation remains on executor-owned `codex/issue-N`. Before first implementation edit of an activation:

1. fetch integration branch and exact pin;
2. require a clean worktree before automatic reconciliation; inspect/preserve any existing issue work;
3. fast-forward an untouched issue branch to `base_sha` when possible;
4. when valid issue commits already exist and the pin is not their ancestor, merge exact `base_sha` rather than resetting/discarding those commits;
5. resolve only conflicts within existing issue authority; semantic conflict needing a new decision returns to `design-required`;
6. verify `base_sha` is an ancestor of resulting issue `HEAD`.

This is the activation rule. Final integration freshness may later rebase the executor-owned branch through the exact-lease procedure owned by `codex-github-operations` while the issue is still `in-progress`.

## Local-runner execution lease

When `SKILLFORGE_LOCAL_RUNNER=1`, require current directory to equal `SKILLFORGE_ISSUE_WORKTREE`, current branch to equal `SKILLFORGE_ISSUE_BRANCH`, repository identity to match, and the worktree/branch to belong to the controlling issue. A retry may contain unfinished prior issue state; inspect/adopt it rather than replacing it.

Do not create another worktree, switch to the durable coordination clone/default branch, invent a second implementation branch, or reset valid issue work. The Actions job only launches the long-lived local turn: a shared App Server turn for Codex or a tmux-supervised CLI turn for Devin. Launch success is not issue success. Detached execution intentionally has no Actions token and must use persistent transports defined by `codex-github-operations`.

Local Devin follows this same lease and workflow without invoking Codex or a cloud handoff. `skills/execution-runner-selection/SKILL.md` and `docs/execution-runners.md` own provider settings, native permissions and session recovery. Do not modify dispatcher-owned executor/host records or start a second process against an active session.

## Entry gate and state

Before editing require exactly one workflow-state label, `execution-ready` or `in-progress`; safe branch/worktree; absent or uniquely valid canonical execution context with required pinned-base preparation; clear scope/invariants/acceptance/inputs; and no competing ownership.

If already `review-ready`, implementation has been handed to audit. Do not resume unless audit or explicit authority returns the issue to executable state.

Before first implementation edit, replace `execution-ready` with `in-progress` without a state-only comment. Runtime returns are `design-required` for missing material design, `investigation-required` for required evidence before design, and `blocked` only for a genuinely unavailable external capability. Executor never sets `completed` or closes the issue.

## Execution loop

### 1. Establish bounded outcome

Confirm intended behavior, permitted subsystem/files, invariants, validation/evidence, execution target, and any explicit intermediate checkpoint. Do not combine unrelated work or invent project-wide machinery.

### 2. Implement smallest coherent delta

Follow the issue and accepted architecture, preserve behavior outside scope, add tests/evaluation coverage when required, use repository-native integration, avoid unrelated cleanup, and stop when evidence invalidates the contract.

Before test edits, use [test-quality](../test-quality/SKILL.md) to plan independent assertions and inspect existing coverage. Reuse adequate tests, demonstrate bug regressions failing before the fix and passing after it, and require evidence before removing coverage. Never conform expectations to a known defect or disable a failing test merely to obtain green validation.

### 3. Handle dependencies deliberately

Preserve identities/provenance/licensing of submodules, vendored code, external repositories/packages/datasets/artifacts. Update only at coherent boundaries and never present unavailable/ambiguous dependency state as a review target.

### 4. Validate honestly

Prefer repository-native build/test/lint/type/evaluation/benchmark paths. Run required and risk-appropriate focused checks; record material deviations/environment limits; never claim an unrun check passed. Local implementation failures are corrected in scope rather than labeled external blockers.

Any final integration rebase/conflict resolution creates a new candidate head. Rerun affected focused validation and obtain fresh exact-head CI/check evidence; do not reuse pre-rebase CI as final evidence.

### 5. Retain proportional evidence

Keep enough technical evidence for the claim without committing bulky generated outputs/caches/binaries/datasets/traces unless explicitly required and distributable. Keep GitHub bookkeeping out of technical artifacts unless it is itself a test input.

### 6. Publish intentionally

Use `codex-github-operations`. When canonical context exists, PR target must equal `integration_branch`; retarget an executor-owned reusable PR if necessary before trusting diff/CI/review. Update durable docs only when durable knowledge changes.

### 7. Stabilize against integration branch

This gate applies only with one valid canonical `integration_branch`, after candidate PR head publication and before PR ready/`review-ready`.

1. Fetch current remote integration branch and remote issue branch. Capture integration tip and candidate/remote issue heads.
2. Require issue still open with sole state `in-progress`, one matching PR targeting intended integration base, clean worktree, and remote issue head equal local candidate.
3. If integration tip is already ancestor of candidate, no rewrite is needed. Otherwise require canonical `base_sha` still ancestor of integration tip.
4. Ask `codex-github-operations` to rebase issue-owned commits onto exact integration tip and publish with an **exact old-head lease**; never force integration branch.
5. Resolve conflicts only within issue authority; otherwise return to design.
6. After rebase rerun affected validation and obtain fresh exact-head CI/checks while PR remains draft.
7. Fetch integration branch again after required checks. If its tip changed, repeat.
8. Gate is stable only when required evidence belongs to the current candidate and the integration tip is still the same tip used for stabilization.

Do not busy-poll CI/branch state. Use normal blocking waits and recheck at material completion boundaries. Standalone issues skip this integration loop.

## Intermediate review checkpoints

Executor-side independent review exists only for a **material intermediate checkpoint explicitly required by the issue**. Publish the exact target and evidence, invoke one fresh read-only review, and continue on `PASS`/non-blocking `PASS_WITH_NOTES`. `FAIL` causes bounded correction/design/investigation return; `BLOCKED` is only genuine unavailable review/evidence capability.

Do not create a final checkpoint merely because implementation finished; final review belongs to `codex-pr-audit` after handoff.

After two consecutive failures in substantially the same validation/attestation/parser/documentation-sync/bookkeeping mechanism, return to design before a third cycle unless the defect is materially different.

## Handoff

When complete final diff has required validation, all declared intermediate checkpoints are satisfied, and integration freshness is stable when applicable:

1. update PR description with final technical state;
2. mark PR ready for review;
3. replace issue `in-progress` with `review-ready` as the executor's **final GitHub mutation**;
4. verify handoff and stop.

Do not merge, enable auto-merge, close the issue, set `completed`, or continue mutating GitHub after `review-ready`. The immediate next action is `codex-pr-audit`.
