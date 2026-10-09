---
name: codex-github-operations
description: Publish branches and commits, operate issues and pull requests, and preserve exact workflow/review targets using the simplest available Git and GitHub transport.
---

# Codex GitHub Operations

## Responsibility

This skill owns Git publication and GitHub control-plane operations requested by the calling workflow. It does not decide architecture, implementation scope, correctness, review requirements, DAG semantics, or progression. Those decisions belong to the controlling issue and role skill.

`codex-independent-review` supplies technical verdicts but has no mutation authority. `codex-pr-audit` has standing automatic merge/completion authority only after a positive final-capable exact-target audit. `codex-epic-scheduler` has standing authority only for epic initialization/scheduling metadata, integration-branch creation, canonical execution context, and `queued -> execution-ready` activation.

## Use the simplest capable transport

Use local Git for worktrees, branches, commits, fetch/push, ancestry, and exact refs. Prefer the connected GitHub app for issues/comments/labels/PRs/reviews when available. Use an already-authenticated `gh` only when needed; do not install/authenticate it merely for routine publication.

### Detached Skillforge local runner

When `SKILLFORGE_LOCAL_RUNNER=1`, expect the long-running Codex turn to outlive the Actions job. The launcher removes `GH_TOKEN`, `GITHUB_TOKEN`, `CI`, and `GITHUB_ACTIONS`; do not recover them from runner state. Use persistent host Git authentication, an already-authenticated `gh`, or another explicitly available secure transport. Verify repository identity/authorization before mutation and report a genuine missing persistent transport rather than synthesizing credentials.

## Workflow state

Exactly one workflow-state label is authoritative for each non-trivial controlling issue:

- `queued`
- `execution-ready`
- `in-progress`
- `review-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`

Preserve unrelated labels and replace the prior state rather than accumulating states. `queued` is normal waiting for a fully designed epic child; only `codex-epic-scheduler` may automatically change it to `execution-ready`. An actor that explicitly resolves a child's manual condition may return it to `queued`.

Use state-only label mutations without comments. Comments are for material technical findings/contracts/handoffs and the canonical control-plane comments below.

Implementation remains `in-progress` through validation, publication and integration stabilization. Executor success is `review-ready`. `completed` is post-merge and is set only after an explicit user-authorized merge or the positive `codex-pr-audit` controller path observes the actual merge.

`codex-pr-audit` may return `review-ready -> execution-ready` without a reviewer `FAIL` when an unchanged implementation head needs base/integration reconciliation.

## Canonical epic DAG

The first epic scheduler invocation may create exactly one top-level parent comment with marker:

```text
<!-- codex-epic-dag:v1 -->
```

This comment is the durable live graph. When publishing it:

1. re-search top-level parent comments immediately before mutation;
2. with zero matches, create one canonical comment;
3. if a concurrent writer already created exactly one, re-read and let the scheduler decide whether it is compatible; do not overwrite it blindly;
4. with more than one match, fail closed;
5. verify the resulting comment contains the intended marker and canonical graph before the scheduler makes the parent `in-progress`.

Ordinary scheduler invocations never regenerate or update the canonical DAG. An explicit design-authority repair may replace the existing canonical comment in place, never append a competing marker.

The first scheduler invocation may also create the resolved epic integration branch from the exact current default-branch head when that branch is absent. Verify absence immediately before creation and the exact resulting ref afterward. Never overwrite, force, or rewind an existing integration branch as part of initialization.

## Canonical child execution context

Epic scheduling materializes a child's activation in one top-level comment with exact marker:

```text
<!-- codex-execution-context:v1 -->
```

Its fenced YAML contains:

```yaml
epic_issue: 3
integration_branch: codex/epic-issue-3
base_sha: 0123456789abcdef0123456789abcdef01234567
```

Only `codex-epic-scheduler` may create/update this automatically. With zero matches create one; with one update it in place; with more than one fail closed. Verify requested parent, branch, and exact SHA before allowing `execution-ready`. Do not add timestamps/session IDs.

For executor/PR targeting, exactly one valid context makes `integration_branch` the intended PR base and `base_sha` the pinned activation snapshot. No context means the repository default branch. Multiple/malformed contexts fail closed. Verify the integration branch still exists and contains `base_sha` in history; ordinary forward movement is allowed, rewritten history that drops the pin is not.

A stale base named in child prose does not override canonical context. If the executor-owned open PR targets a different base, retarget it before treating its diff/checks/review as current, then obtain review evidence appropriate to the resulting exact diff.

## Commits and publication

Agent-created commits use Conventional Commits: `<type>(<optional-scope>): <imperative summary>`, normally one of `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, `style`, or `revert`. Follow stricter repository rules when present.

Before publishing an implementation branch, verify issue branch, canonical execution context, clean/scoped worktree, and remote ownership. Preserve `SKILLFORGE_ISSUE_WORKTREE` / `SKILLFORGE_ISSUE_BRANCH` when supplied. Do not switch to the durable coordination clone or invent another branch.

Initial pinned-base preparation belongs to `spec-driven-codex-loop`: fast-forward untouched issue work when possible, otherwise preserve valid issue commits and merge the pin rather than reset them away.

### Guarded pre-handoff integration rebase

Before `review-ready`, `spec-driven-codex-loop` may request a rewrite of the **executor-owned issue branch only** to stabilize against a newer canonical integration tip. Preconditions: issue sole state `in-progress`; exact executor worktree/branch lease; exactly one matching PR targeting canonical integration branch; clean worktree; fetched remote issue head equals the local pre-rebase head; canonical `base_sha` remains ancestor of current integration tip.

If integration tip is already ancestor of issue head, do not rewrite. Otherwise capture old remote head and integration tip, rebase issue-owned commits onto that tip, resolve conflicts only within existing issue authority, rerun affected validation, and publish with an **exact old-head lease** equivalent to:

```text
git push --force-with-lease=refs/heads/<issue-branch>:<old-remote-head> origin HEAD:refs/heads/<issue-branch>
```

Never use naked `--force`. Verify remote exact head after publication, require fresh exact-head CI/evidence, and re-fetch integration tip before handoff. This rewrite authority ends the instant the issue becomes `review-ready`.

## Pull requests

Use one PR per controlling issue unless explicitly decomposed. Resolve intended base immediately before creation/reuse from canonical execution context or default-branch fallback. The PR should link the issue, summarize delivered behavior, state current validation/review status, and list material residual risks without duplicating full histories/logs.

Keep it draft while implementation, validation, integration stabilization, or explicit intermediate checkpoints remain. The executor does not perform a duplicate final independent review. When work and exact-head evidence are complete and the integration-freshness gate is stable, mark PR ready, replace `in-progress` with `review-ready`, verify both, and hand to `codex-pr-audit`.

Replacing `in-progress` with `review-ready` must be the executor's final GitHub mutation. After that it may only perform local teardown/bookkeeping/response composition until a later controller returns the issue to execution.

## Merge authority

Executors and independent reviewers never merge or enable auto-merge. There are two supported merge paths. Both require the mandatory pre-merge CI gate below; a positive technical verdict and GitHub's ability to accept a merge request do not replace that gate.

### Mandatory pre-merge CI gate

This is the canonical CI eligibility procedure for every agent-controlled PR merge, including on an unprotected branch. Run it before authorizing completion and refresh it immediately before the exact-head merge request. Do not use an admin/bypass merge, direct push, or auto-merge to evade it.

1. **Resolve applicable validation.** Read the current PR, complete changed-file set, controlling contract, and accepted workflow event/path/job conditions. Include branch/ruleset-required checks plus every repository CI workflow/job applicable to this PR and any additional issue-required evidence. An explicitly accepted epic-child CI deferral remains valid only for that child/base; it never waives the final integration PR's gates. Document genuine non-applicability rather than launching unrelated suites. A workflow must not waive itself through an unapproved change in the candidate.
2. **Bind results to the exact target.** Capture head and base SHAs. Inspect the PR's check runs and legacy status contexts, plus the applicable Actions runs and their complete job lists, with pagination where needed. For pull-request workflows, validate the actual test-merge commit against the current head/base pair; its SHA need not equal the PR head. Reject results from an older head/base pair, another PR, an unrelated event, or an unexpected check provider. If both a check and a status are required under the same name, inspect both. Use the latest attempt of each applicable logical run; a newer queued/running attempt cannot be hidden by an older success. Do not let a passing run silently override a distinct current failing run.
3. **Require completed evidence.** Applicable mandatory validation must have completed with `success`, including all required matrix jobs and their mandatory steps. A successful setup, lint/unit step, launcher, or selected local test is not success of the browser/acceptance job or workflow. `queued`, `requested`, `waiting`, `pending`, `in_progress`, missing/unknown results and incomplete listings do not authorize merge. Neither do `failure`, `error`, `timed_out`, `cancelled`, `action_required`, `stale`, or startup failures. `skipped`/`neutral` is acceptable only for a demonstrably non-applicable or explicitly optional gate under the accepted contract; never as a replacement for mandatory tests. Optional capability skips inside an otherwise successful suite remain honestly disclosed under the existing specification.
4. **Do not infer green from absence.** `mergeable: true`, no branch-protection contexts, an empty `gh pr checks --required` result, a success-only combined status API response, or missing check access is insufficient. Establish the expected applicable set and inspect both checks and statuses. A docs-only or intentionally deferred change may have no applicable CI, but record the verified reason and still require its declared evidence. An expected-but-absent run is unresolved, not non-applicable.
5. **Preserve evidence and freshness.** Record workflow/check identities, URLs, exact tested revision or head/base pair, run attempts and terminal conclusions in the audit's validation evidence. Re-read the PR, current base tip and gate immediately before merging. Head/base movement invalidates eligibility and follows the calling workflow's fresh-review/reconciliation path; a new pending attempt also suspends eligibility. Never manufacture a status or remove a required check to obtain a pass.

Pending CI is a normal wait, not a technical PASS or a `blocked` condition. The audit controller keeps the issue `review-ready` and PR ready, waits with bounded polling using the existing tools, and revalidates identity after waiting. When the current turn cannot finish that wait, leave a precise non-duplicated CI-wait handoff and stop without merging; an explicit later resume must re-read the same target and gate. Do not assume a completion wake-up exists or create a new scheduler here.

An applicable terminal failure forbids merge. Inspect its actual cause and let the calling audit procedure return repairable failures to the executor with exact evidence; a pre-existing failure still requires a resolved design/validation decision, not an informal waiver. For cancellation or infrastructure interruption, a justified rerun must finish successfully before eligibility returns; do not retry indefinitely or label an untested target safe. Missing access after permitted alternatives is an evidence blocker, never permission to proceed.

### Explicit user-facing merge

A normal/manual merge requires a ready PR, compatible workflow state, completed handoff, an explicit current user instruction to merge/review-and-merge, no material blocker from that user-facing review, and a freshly satisfied mandatory pre-merge CI gate.

### Positive `codex-pr-audit` completion

The audit controller may merge without a second user instruction only when: issue sole state `review-ready`; PR open/ready; `codex-independent-review` returned `PASS` or `PASS_WITH_NOTES` with `final-capable: yes` for exact `audit_head`; canonical audit record exists for that head/verdict; the mandatory pre-merge CI gate is freshly satisfied; head and base are revalidated immediately before merge; and merge is guarded by exact expected head.

After merge request, re-read and require GitHub to report actual merge before setting `completed`, then close the issue. The `completed` label must become observable before close so the epic dispatcher can wake an active parent.

If base/integration advanced or PR is dirty/conflicting before exact merge, the audit controller may mark draft and return `review-ready -> execution-ready` for executor-owned reconciliation. Pending checks with unchanged exact target remain `review-ready`; do not misuse `blocked`.

No issue prose, CI success, executor conclusion, or reviewer verdict outside these authority paths authorizes merge.

## Exact review targets and ownership

Independent review always names an exact published commit/range and required base/dependency revision. Do not amend/rebase/squash/cherry-pick/force-push a valid review target merely to repair metadata. A technical change creates a new target.

Once an executor owns a `codex/issue-N` branch/PR, other actors inspect it read-only until handoff/transfer. The scheduler owns only its canonical scheduling comments/state activation. After `review-ready`, the audit controller may change audit-derived readiness/merge/completion state but never rewrite the implementation head.

When ownership is ambiguous, stop before mutation.

## Safety and degraded operation

Try another permitted transport when one replaceable route fails. Preserve valid history and leave a precise handoff only when necessary. Use `blocked` only when a missing capability is required and no safe practical alternative exists.

Never publish secrets/private credentials/restricted artifacts, persist ephemeral Actions tokens, stage unrelated changes, silently choose among duplicate canonical comments, let stale prose override canonical state, or claim an unobserved mutation.
