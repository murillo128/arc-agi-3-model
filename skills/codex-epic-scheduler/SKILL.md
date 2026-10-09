---
name: codex-epic-scheduler
description: Initialize and schedule one event-driven epic DAG, then prepare the completed epic for one final default-branch PR without reviewing or merging it.
---

# Codex Epic Scheduler

## Responsibility

Use this skill only when the controlling parent issue declares `execution_mode: epic-dag`. GitHub issues, workflow-state labels, the canonical epic-DAG comment, canonical child execution-context comments, and published Git refs are the persistent source of truth. Every invocation reconstructs current state from GitHub and performs one bounded transition; it does not depend on memory from an earlier turn.

The scheduler has three phases:

1. **initialize:** derive and publish one canonical DAG and create/reuse the epic integration branch;
2. **schedule:** activate one deterministic dependency-ready child wave at a time;
3. **final handoff:** after every declared child is `completed`, prepare one ordinary `codex/issue-<parent>` PR from the fully integrated epic to the repository default branch, then hand the parent to `review-ready`.

It never implements child scope, performs the final independent review, merges the final PR, marks the parent `completed`, closes the parent, rewrites the canonical integration branch, or moves the default branch directly. Final technical review and merge remain owned by `codex-pr-audit`.

## Parent seed contract

The parent issue body contains one fenced YAML seed. It names the work set but deliberately does not precompute dependency edges or mutexes:

```yaml
execution_mode: epic-dag
integration_branch: codex/epic-issue-3
max_parallel_workers: 4
child_issues: [10, 11, 12, 13]
```

Required fields are `execution_mode: epic-dag`, a positive `max_parallel_workers`, and a non-empty list of unique positive same-repository `child_issues`. `integration_branch` is optional; when absent use `codex/epic-issue-<parent-number>`.

The seed is bootstrap input only. After initialization the canonical DAG comment is authoritative.

## Canonical epic DAG

The initialized graph lives in exactly one top-level parent comment marked:

```text
<!-- codex-epic-dag:v1 -->
```

Its machine-readable YAML contains `execution_mode`, `integration_branch`, `max_parallel_workers`, and one sorted child record per declared issue with sorted direct `depends_on` and `mutex` lists. Do not add timestamps, worker/session IDs, implementation details, or mutable execution history.

Zero canonical comments means the epic is not initialized. Exactly one means it is initialized. More than one is ambiguous and fails closed. Ordinary scheduler invocations never regenerate the canonical graph; graph repair requires explicit design authority.

## Entry and workflow state

Load `AGENTS.md`, the parent issue, this skill, `codex-github-operations`, and only the child issues/comments needed by the active phase. Confirm repository and issue identity.

Valid parent entry states are:

- **initialization:** parent `execution-ready` and no canonical DAG;
- **normal scheduling/final handoff:** parent `in-progress` with one valid canonical DAG;
- **final-handoff retry after audit reconciliation:** parent `execution-ready` with one valid canonical DAG **and every declared child already `completed`**.

A canonical-DAG parent in `execution-ready` while any child is not `completed` is inconsistent and fails closed. An `in-progress` parent with no canonical DAG is invalid.

For a valid final-handoff retry, replace `execution-ready` with `in-progress` before changing the finalization branch or PR, preserving unrelated labels. This is the normal return path when `codex-pr-audit` detects default-branch drift and asks the executor to reconcile.

## First invocation: generate the DAG

### 1. Validate the seed and child set

Require one unambiguous seed and verify every declared child exists in the same repository, is an issue rather than a PR, has exactly one official workflow-state label, and has a self-contained technical contract sufficient to identify direct prerequisites and serialization-sensitive contention. Children may already be `completed`; fully designed waiting children normally begin `queued`.

### 2. Derive direct dependencies

Add child `A` to child `B`'s `depends_on` only when `B` cannot be implemented, validated, or integrated correctly without the accepted result of `A`. Prefer explicit API/data/migration/integration prerequisites. Keep only direct prerequisites, not redundant transitive edges. If a material dependency cannot be determined safely, fail closed and route to design only when a real design decision is missing.

### 3. Derive mutexes

Mutexes serialize otherwise dependency-independent children that mutate the same serialization-sensitive surface. Use stable repository concepts such as `schema`, `packaging`, or `runtime-core`; never issue numbers, branch names, worker IDs, or timestamps. Do not use a mutex instead of a genuine dependency.

### 4. Validate the graph

Normalize the proposed graph with current child state and validate with `skills/codex-epic-scheduler/scripts/plan_wave.py` when practical. Require exact seed-child closure, unique/non-self dependencies, unique non-empty mutex strings, acyclicity, a positive worker limit, and a valid integration branch name. The pure planner validates structure/wave selection; semantic edges remain the scheduler's responsibility.

### 5. Initialize the integration branch

Resolve the canonical integration branch. If absent, create it from the exact current default-branch head through `codex-github-operations` and verify the ref. If it already exists, use its current exact head. Never overwrite, force, rewind, rebase, or otherwise repair an unexpected existing integration ref.

### 6. Publish and verify the canonical DAG

Immediately before publication re-search parent comments for the canonical marker. With zero, create exactly one. If another writer won the race, accept it only when it is structurally valid and exactly represents the same declared child set; otherwise fail closed. Re-read and verify the durable graph before replacing the parent `execution-ready` state with `in-progress`.

## Later invocations: validate canonical state

Read the single canonical DAG and every declared child's current workflow state. Require the integration branch to exist, the graph to remain closed/unique/non-self/acyclic, every child to have exactly one official workflow-state label, and the parent to have a valid entry state described above. Do not re-infer or regenerate the DAG from edited prose.

## Observable state classification

For scheduling only:

- `queued`: schedulable when dependencies are satisfied;
- `execution-ready` or `in-progress`: active and consuming one slot;
- `review-ready`, `completed`, `blocked`, `design-required`, or `investigation-required`: not consuming a slot.

Only `completed` satisfies a dependency. `review-ready` does not. Ordinary dependency waiting remains `queued`, never `blocked`.

## Deterministic wave selection

If **every** declared child is `completed`, skip wave selection and enter **Final epic handoff** below.

Otherwise calculate `available_slots = max(0, max_parallel_workers - active_children)`. A candidate is exactly `queued` and has every direct dependency exactly `completed`. Collect mutexes held by active children, inspect candidates in ascending issue-number order, and greedily select while slots remain and no mutex conflicts with active/already-selected work. Do not maximize cardinality by skipping an earlier compatible candidate.

Use the bundled planner when practical.

## Canonical child execution context

Immediately before a non-empty wave, re-read parent DAG and all child states, recompute selection if anything changed, resolve the integration branch's exact current head once as `base_sha`, and use that same integration branch/base SHA for every selected child.

Each selected child must have exactly one top-level comment marked:

```text
<!-- codex-execution-context:v1 -->
```

with:

```yaml
epic_issue: 3
integration_branch: codex/epic-issue-3
base_sha: 0123456789abcdef0123456789abcdef01234567
```

Create with zero matches, update the same comment with one, and fail closed with more than one. Verify durable context before making `execution-ready` observable. A reactivated child gets the current canonical context.

## Apply one wave

For selected children in ascending order: re-read state; skip if no longer exactly `queued`; publish/update and verify execution context; replace `queued` with `execution-ready` preserving unrelated labels; verify the resulting sole workflow state. If publication succeeds but label transition races/fails, leave the child non-executable for a later retry. Do not fill newly available slots after wave application begins.

## Final epic handoff

Enter this phase only when **all declared children are exactly `completed`**. This is a publication/handoff phase, not another implementation wave and not a technical review.

### 1. Revalidate completion and refs

Immediately re-read the parent, canonical DAG, every child state, the repository default branch, and the canonical integration branch. Require all children still `completed`, parent `in-progress`, and both branch heads to be exact published commits. Record `default_head` and `integration_head` for this attempt.

The canonical integration branch is read-only during finalization. Never rewrite, merge into, rebase, or force-update it.

### 2. Prepare the parent finalization branch

Use the launcher-owned issue worktree and branch `codex/issue-<parent-number>` as the **finalization branch**. It is intentionally distinct from `codex/epic-issue-<parent-number>` so the ordinary PR auditor can resolve the final PR by the existing issue-branch convention.

Require the worktree clean, the expected parent branch checked out, and repository identity/authentication verified. Fetch current default and integration refs. The finalization branch may contain only the default/integration history plus prior scheduler-created finalization merge commits for this same parent; any unrelated unique commit fails closed.

Bring the finalization branch forward to contain both exact tips without rewriting shared history:

1. merge/fast-forward the current default tip;
2. merge the current canonical integration tip;
3. do not make product/source edits merely to resolve a semantic merge conflict.

A clean Git merge/fast-forward is allowed because it only combines already-published accepted histories. If Git reports a content conflict, abort the merge, leave the canonical integration branch untouched, record one concise material integration-conflict comment, and move the parent to `investigation-required` (or `design-required` only when a genuine product/design decision is needed). Do not guess a resolution inside scheduler authority.

Push the finalization branch with ordinary non-force publication and verify the exact remote head. On a retry after default-branch drift, merge the new default/integration tips into the same finalization branch and obtain a new exact head; do not create a parallel finalization branch.

### 3. Create or reuse one final PR

Resolve the repository default branch immediately before PR mutation. Require at most one open PR whose head is `codex/issue-<parent-number>` and base is the default branch. Reuse it when present; otherwise create it.

The PR body must identify:

- the controlling epic issue;
- canonical integration branch and `integration_head`;
- captured `default_head`;
- that every declared child is `completed`;
- any final integrated-acceptance child/evidence named by the parent DAG/contracts;
- material residual/capability notes already accepted by child audits.

Do not claim a new technical PASS. This PR is the aggregate target for the independent final audit.

### 4. Establish final-handoff evidence

The finalization branch must contain no unvalidated manual product fix. When its tree is identical to the already accepted integration tree because the current default head is already an ancestor, existing exact-tree integrated acceptance may be reused and called out explicitly.

If combining a newer default branch changes the final tree, require the repository's relevant PR/default-branch checks to run on the exact finalization head and inspect them before handoff. Run additional parent-declared integrated acceptance only when needed to make the aggregate target trustworthy. If required validation cannot be obtained, leave the parent `in-progress` or move to `blocked` only for a genuine unavailable capability; do not fabricate success.

### 5. Handoff to normal audit

Mark/reuse the PR as ready for review only after branch publication and required evidence are stable. Re-read parent, PR, default head, integration head, and finalization head immediately before state mutation. If any target moved, reconcile first and repeat validation.

Then replace parent `in-progress` with `review-ready`, preserving unrelated labels, and verify it. This is the scheduler's final GitHub mutation for the attempt. The existing issue-state dispatcher will launch `codex-pr-audit`, which resolves the ordinary `codex/issue-<parent>` PR, performs the independent exact-target review, and alone has standing authority to merge, expose `completed`, and close the parent after a positive final-capable verdict.

If that auditor detects forward default-branch drift, its normal `review-ready -> execution-ready` reconciliation return is expected. The next launcher invocation re-enters this skill through the valid final-handoff retry path above.

## Wake-up model and termination

The repository's single issue-state dispatcher wakes an active parent when a declared child receives `completed` or is deliberately restored to `queued`. Duplicate/delayed wake-ups are safe because every invocation recomputes current GitHub state.

After initialization or one non-empty wave is verified, terminate. When no candidates qualify and not all children are complete, terminate successfully with parent still `in-progress`.

When all children are complete, **do not stop at `in-progress`**: execute the Final epic handoff in the same scheduler invocation. The scheduler still never performs the independent review or merge itself.

Report only the phase (`initialized`, `scheduled`, or `final-handoff`), validated child count, active count, available slots, selected issue numbers when applicable, integration branch/head, finalization PR/head/base when applicable, skipped races, and any exact fail-closed defect. Do not emit the full canonical DAG again.
