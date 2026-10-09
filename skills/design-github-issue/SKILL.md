---
name: design-github-issue
description: Define a self-contained GitHub execution contract for a standalone issue, epic child, or epic parent seed without leaking design-session reasoning into implementation.
---

# Design a GitHub Execution Issue

## Responsibility

Use this skill before non-trivial implementation starts, or when execution returns because a material design or validation decision is unresolved. The design authority owns the observable outcome, material architectural and validation decisions, bounded scope, invariants, exclusions, failure semantics, acceptance criteria, and initial workflow state. It does not implement code, operate implementation branches, perform independent review, or authorize an executor to merge.

Design for a fresh executor with no access to hidden chat reasoning. The issue must contain every task-specific fact, decision, constraint, and acceptance rule needed for correct execution. Links supplement the contract; they do not replace material instructions.

Keep user-facing teaching outside the issue. Technical flow belongs in the issue only when an executor needs it to implement correctly.

## Load only material design context

Start with `AGENTS.md` when present and the user request, roadmap item, or existing controlling issue. Then inspect only the accepted plans/decisions, relevant implementation seams/tests/configuration, baseline evidence, dependencies/artifacts, and plausibly overlapping active work required to settle this task.

Do not promote exploratory notes, hypotheses, derived wiki text, or provisional chat conclusions into requirements unless an authoritative source explicitly adopts them.

## Explicit activation holds

An explicit user instruction not to activate work overrides all initial-state
and publication defaults below. Preserve existing held labels when updating an
issue. For a newly published, fully specified standalone issue or epic parent
that must remain inactive, use `queued` and state that it awaits manual activation;
this is an activation hold, not missing technical design. Keep its children
`queued` too. Never add `execution-ready`, release a hold, initialize a DAG or
launch a session merely because specification or executor selection is complete.
Only a separate explicit authorization may activate that held parent/standalone
issue. Do not place a manually held child under an active parent without an
explicit non-executable workflow state: an active scheduler can activate `queued`
children. The normal defaults below apply only when no explicit hold exists.

## Workflow states

Use exactly one current workflow-state label:

- `queued`
- `execution-ready`
- `in-progress`
- `review-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`

At publication, set exactly one through `codex-github-operations`. A fully designed standalone issue starts at `execution-ready`. A fully designed child that belongs to an epic starts at `queued` so only `codex-epic-scheduler` activates it. Unresolved design, evidence, or external capability uses `design-required`, `investigation-required`, or `blocked` rather than `queued`.

A normal issue must not be published initially as `in-progress`, `review-ready`, or `completed`.

## The issue is the execution contract

Depending on the task, record the current limitation and observable goal; baseline/default behavior that must remain unchanged; relevant dependencies/artifacts/data; accepted APIs/data/control flow; ownership/lifetime/concurrency/failure semantics; permitted implementation scope and explicit exclusions; validation targets and objective pass/fail criteria; material negative evidence; and intermediate checkpoints only when they reduce a distinct technical risk.

Use precise names, paths, values, examples, and equations when they remove ambiguity. Do not copy generic Git, publication, label, audit, merge, or reporting procedure already owned by skills.

The executor's normal successful delivery is a ready-for-review PR plus `review-ready`. Final independent review and verdict-derived completion belong to `codex-pr-audit`; do not make every issue define a duplicate final review checkpoint.

## Executor and model selection

Use `skills/execution-runner-selection/SKILL.md` when the user chooses local
Codex or local Devin CLI. Record executor selection in a dedicated top-level
`execution` TOML block. Keep optional Codex model/effort/profile in `codex`; local
Devin accepts optional `model` in `devin`, not cloud mode/ACU fields or Codex effort.
Without an executor override, preserve Codex and its native model settings.
`docs/execution-runners.md` owns supported keys, local prerequisites and recovery.

Each child and parent selects independently; there is no implicit epic inheritance.
For a whole-epic request, intentionally set each requested child as well as the
parent. Verify the local host, CLI authentication, worktree trust/permissions and
model/GPU/test requirements. Devin runs locally in tmux, not through Cloud API.
Changing selection is not activation or an active-session ownership transfer.
Final independent audit remains Codex regardless of implementation executor.

## Epic child design

A child of an epic remains a normal self-contained controlling issue. It must not rely on the parent to define its technical scope or acceptance criteria. The parent may define membership and scheduling, but each child must be executable by a fresh worker once activated.

When fully designed, publish the child as `queued`. Do not precompute or duplicate its eventual `depends_on`, mutex, execution base, or PR target in the child body. The first epic scheduler invocation derives the canonical DAG from all child contracts, and each activation receives canonical execution context separately.

If a child has a real unresolved blocker, use the corresponding manual state rather than `queued`; it can be returned to `queued` after that condition is explicitly resolved.

## Epic parent seed design

An epic parent is a scheduling/integration contract, not a giant technical issue and not a pre-written DAG. Its first execution is owned by `codex-epic-scheduler`.

Before publishing the parent:

1. design every intended child sufficiently that dependency relationships can be inferred from the child contracts without hidden context;
2. ensure the declared child set is complete for the intended epic scope and contains no accidental duplicates/superseded issues;
3. choose a conscious parallelism limit;
4. optionally choose an integration branch name, otherwise accept the scheduler default `codex/epic-issue-<parent-number>`;
5. put fully designed waiting children in `queued` and leave genuinely unresolved children in their correct manual state;
6. publish the parent as `execution-ready`.

The parent body contains one compact fenced YAML seed:

```yaml
execution_mode: epic-dag
integration_branch: codex/epic-issue-123  # optional
max_parallel_workers: 4
child_issues: [124, 125, 126, 127]
```

Do **not** put generated `depends_on` or mutex data in the seed. On the first scheduler invocation, `codex-epic-scheduler` reads the child contracts, derives a minimal direct dependency graph and mutexes, validates it, initializes the integration branch when necessary, publishes exactly one canonical `codex-epic-dag:v1` parent comment, changes the parent to `in-progress`, and selects the first wave.

After canonical DAG publication, the seed is bootstrap history rather than live graph state. A later graph change is an explicit design repair of the canonical DAG; ordinary scheduler invocations never regenerate it from edited prose.

## Design method

### 1. Define the observable outcome

State what must become true, why it matters, the current limitation, and the boundary of the requested change.

### 2. Resolve material unknowns

Resolve questions that can change behavior, compatibility, architecture, data handling, correctness, failure handling, validation, licensing, security, performance, or deployment. Keep `OPEN`/`SPECULATIVE` material non-contractual until intentionally resolved.

Do not invent project-wide roadmaps, schemas, frameworks, ontologies, or process machinery merely because they might be useful later.

### 3. Bound implementation

Define the smallest coherent outcome, permitted subsystem/files, explicit exclusions, and invariants. Name exact seams where an executor could otherwise modify the wrong layer.

### 4. Define validation

Specify repository-native build/test/lint/type-check/evaluation/benchmark targets and the correctness, failure, data-integrity, concurrency, security, or performance cases that materially prove the outcome. Use exact commands when invocation details are part of the evidence; otherwise name the target/result without freezing replaceable syntax.

Never require evidence that the expected environment cannot practically produce unless the task explicitly establishes that capability as a prerequisite.

Use [test-quality](../test-quality/SKILL.md) for test planning: name behavioral contracts, independent expected-result sources, test owners and existing coverage gaps. Define pre-fix failure/post-fix success evidence for bug regressions and required consumer/integration checks. Keep the plan proportional rather than requiring duplicate tests or numeric coverage targets; design remains non-implementing.

### 5. Add only material intermediate checkpoints

Use independent executor-side checkpoints only when work should not safely continue past a distinct architecture, ownership/lifetime, data integrity, numerical, concurrency, security, backend, or broad-refactor boundary without review. Do not add a final checkpoint merely because implementation finished; `codex-pr-audit` provides the final independent review after `review-ready`.

### 6. Define restart semantics

Distinguish local implementation defects from design defects, evidence gaps, replaceable transport failures, and genuine external blockers. Two consecutive review failures for substantially the same validation/bookkeeping mechanism should return to design before a third cycle unless the defect is materially different.

### 7. Check overlap

Inspect only plausibly overlapping open issues/PRs/branches. Link superseded work and summarize the material constraint instead of copying its history.

## Publication readiness

Before assigning `execution-ready` or `queued`, confirm a fresh executor can act without design-session reasoning; terminology and outcome are unambiguous; material decisions and inputs are present; scope/invariants/failure behavior/acceptance are clear; validation is feasible; exploratory material was not silently promoted; no unnecessary project-wide machinery was invented; and no issue text grants the implementation executor merge authority.

For an epic parent also confirm every `child_issues` member exists, is intended, and has enough contract detail for first-run DAG generation; the parent itself contains no manually generated DAG; and the only parent workflow state is `execution-ready`.

## Standalone/child issue structure

```markdown
# <Outcome-oriented title>

## Readiness
**Initial state:** execution-ready | queued | design-required | investigation-required | blocked

## Goal and current limitation
<Observable outcome and current behavior.>

## Baseline and inputs
<Material facts, dependencies, artifacts, defaults.>

## Resolved technical contract
<APIs, data/control flow, ownership, failure semantics, bounds, concrete seams.>

## Scope
### In scope
### Out of scope
### Invariants

## Validation and evidence
<Required targets, cases, environment, artifacts, objective gates.>

## Intermediate checkpoints
<Only distinct material-risk checkpoints, if needed.>

## Delivery
<PR shape, publication boundaries, and ready-for-review handoff.>
```

For an epic parent, keep the body smaller: goal/scope, scheduling/integration policy that is genuinely specific to the epic, and the seed contract above. Child technical contracts stay in the child issues.
