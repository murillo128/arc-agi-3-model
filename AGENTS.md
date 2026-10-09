# AGENTS.md

Repository-wide instructions for ChatGPT, Codex, and other development agents.

## Mission and scope

The project's durable mission and product/domain scope belong in `README.md` and the repository documentation that explicitly owns them. Do not broaden the project, invent adjacent goals, or promote exploratory discussion into settled scope without an explicit repository or issue-level decision.

This file owns repository-wide agent invariants and routes work to reusable skills. Skills define reusable procedure, issues define bounded task contracts, and repository documents define durable project knowledge.

## One-time template bootstrap

This section exists only in the canonical Skillforge template and repositories not yet initialized.

- Never execute `repository-bootstrap` inside canonical `murillo128/skillforge` itself.
- In a repository created from this template, presence of `skills/repository-bootstrap/SKILL.md` means initialization is incomplete.
- Before normal non-trivial project work, run that skill once with actual mission/scope/established constraints.
- Successful bootstrap creates/verifies required labels, makes README project-specific, adapts project invariants, deletes bootstrap skill, and removes this section/routing entry.
- Local Codex runner provisioning is optional and is a separate post-bootstrap capability.

## Load context progressively

For non-trivial work start with `AGENTS.md` and the controlling issue. Then load only accepted decisions/spec sections, source/tests/config/evidence, and the one workflow skill needed by the current role/action. Do not preload every document, skill, issue/PR history, result directory, or derived wiki.

On resume, verify branch, `HEAD`, worktree, controlling issue's current single workflow-state label, canonical execution context when applicable, and new material issue/PR discussion. Reuse unchanged inspected context rather than replaying history.

## Source-of-truth hierarchy

Unless project-specific documentation defines a stricter hierarchy:

1. Tests, formal checks, evaluation outputs, and captured evidence establish observed behavior.
2. Accepted specifications, decisions, architecture docs, `docs/**`, and other explicitly normative documents establish durable intended behavior.
3. The controlling issue establishes the bounded execution contract.
4. PRs, checks, reviews, commits, and Git history preserve implementation and reproducible evidence.
5. Roadmaps/epics/planning establish planning/dependency status only within their declared authority.
6. Exploratory notes/research/drafts are provisional unless explicitly adopted.
7. `wiki/**` is agent-generated derived non-normative knowledge and never overrides stronger sources.
8. Chat is provisional until intentionally recorded in an authoritative repository/GitHub source.

Do not promote `OPEN`, `SPECULATIVE`, exploratory, or wiki-derived statements into requirements without explicit adoption. Surface material conflicts rather than silently choosing.

## Skill-driven workflow

Load skills lazily by role:

- one-time initialization: `skills/repository-bootstrap/SKILL.md`;
- optional local runner provisioning/repair: `skills/codex-local-runner/SKILL.md`;
- design authority: `skills/design-github-issue/SKILL.md`;
- ordinary issue executor: `skills/spec-driven-codex-loop/SKILL.md`;
- final PR audit controller: `skills/codex-pr-audit/SKILL.md`;
- Git/GitHub mutations/publication: `skills/codex-github-operations/SKILL.md`;
- independent technical review: `skills/codex-independent-review/SKILL.md`;
- test planning, authoring and audit: `skills/test-quality/SKILL.md`;
- event-driven epic initialization/scheduling: `skills/codex-epic-scheduler/SKILL.md`;
- explicit manual multi-issue orchestration: `skills/codex-issue-orchestrator/SKILL.md`;
- derived wiki curation: `skills/repository-wiki-curation/SKILL.md`.

The generic launcher for an execution target is the same. After launch, read the controlling issue: when it declares `execution_mode: epic-dag`, route to `codex-epic-scheduler`; otherwise route to `spec-driven-codex-loop`. Keep this decision in agent instructions, not duplicated in the launcher workflow.

Before planning, writing, changing, reviewing or removing tests, load `test-quality` alongside the current role skill. Follow its repository-testing reference for native guidance without replacing accepted configuration, scope or workflow authority.

## Workflow state

Every non-trivial controlling issue uses exactly one current workflow-state label:

- `queued`
- `execution-ready`
- `in-progress`
- `review-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`

The label is authoritative. State-only transitions normally produce no comment.

`queued` is normal waiting for a fully designed epic child; it is not a blocker. Only `codex-epic-scheduler` may automatically promote `queued -> execution-ready`. An actor that explicitly resolves `blocked`, `design-required`, or `investigation-required` may return an epic child to `queued`.

`review-ready` is the executor's successful terminal handoff: implementation/validation and issue-declared intermediate checkpoints are complete, the PR is ready, and integration freshness is stable when applicable. It hands control to `codex-pr-audit`; the executor does not perform a duplicate final independent review.

`completed` is post-merge. Executors and independent reviewers never merge, enable auto-merge, close issues, or set completed. After `codex-pr-audit` obtains `PASS` or `PASS_WITH_NOTES` with `final-capable: yes` on its current exact PR head, the audit controller has standing authority to merge that exact head, make `completed` observable, and close the issue. The reviewer verdict alone has no mutation authority.

## Epic workflow

An epic parent starts with a compact seed contract (`execution_mode: epic-dag`, child issue set, parallelism limit, optional integration branch), **not** a manually generated DAG. Its initial state is `execution-ready`; fully designed children normally start `queued`.

The first `codex-epic-scheduler` turn inspects all child contracts, derives direct dependency edges and serialization mutexes, validates acyclicity/closure, creates the integration branch from the current default branch when necessary, and persists exactly one parent comment marked `<!-- codex-epic-dag:v1 -->`. Only after that durable graph exists does it make the parent `in-progress` and activate the first wave.

Later scheduler turns consume the canonical graph; they do not re-infer it from edited prose. Every child activation receives exactly one canonical `<!-- codex-execution-context:v1 -->` comment with parent, integration branch, and exact base SHA before `execution-ready` becomes observable.

The single `.github/workflows/codex-issue-state.yml` dispatcher is the only `issues:labeled` workflow. It routes `execution-ready` to execution, `review-ready` to audit, and `completed`/`queued` child events to the unique active parent discovered from canonical DAG state. Duplicate/delayed scheduler wake-ups are expected and must remain idempotent.

## Design and execution discipline

For non-trivial changes, design a self-contained controlling issue; implement the smallest coherent outcome; preserve behavior outside scope; validate proportionally with repository-native checks; retain enough evidence for claims; and use independent review only at issue-declared intermediate risk boundaries plus the final audit controller.

Do not invent project-wide roadmaps, schemas, frameworks, ontologies, or process machinery merely because they might be useful. Do not mechanically implement every reviewer suggestion; judge findings against contract/materiality/invariants.

## Durable knowledge and wiki

Keep deliberate normative docs and generated memory distinct. `docs/**` and other explicitly normative sources use normal design/execution workflow. `wiki/**` is generated, derived, non-normative memory maintained by `repository-wiki-curation`. GitHub issues own actionable discrepancies, unresolved decisions, suspected defects/drift, and bounded work contracts.

The curator has standing write authority only for `wiki/**`, and only after its adversarial review and hard write-boundary gate. It never modifies non-wiki paths. New curator-created issues use `curator-detected`.

## Evidence and artifacts

Keep evidence proportional. Commit source, tests, configuration, small deterministic fixtures, concise reports, and compact reproducibility evidence. Keep large generated outputs, binaries, model weights, datasets, caches, traces, or bulky logs outside Git unless explicitly required and distributable. Never publish secrets/private data/restricted artifacts.

## Git and GitHub behavior

- Keep changes scoped and use explicit paths.
- Agent-created commits follow `codex-github-operations` conventions.
- Do not rewrite shared valid history without explicit authority. The standing exception is the executor-owned `codex/issue-N` branch, before `review-ready`, using the exact-old-head `--force-with-lease` integration-rebase protocol.
- Direct commits to default branch require explicit user instruction except the narrow wiki-curator authority and one-time bootstrap authority.
- An implementation workflow ends with a ready PR and `review-ready` handoff to audit.
- Before handoff, an epic child executor performs the final integration-freshness gate: if integration advances, reconcile/revalidate/republish and require fresh exact-head CI until stable.
- Replacing `in-progress` with `review-ready` is the executor's final GitHub mutation. Afterward only local teardown/bookkeeping/response composition is allowed until another controller returns it to execution.
- Outside the positive `codex-pr-audit` path, merge requires a later explicit user-facing instruction after review finds no material blocker.

## Project-specific additions

Repositories created from the template should add only genuine domain-specific repository-wide invariants: language/runtime/coding constraints, ownership/lifetime/concurrency/security invariants, required validation paths, dependency/licensing constraints, hardware requirements, correctness/performance hard failures, or explicit planning authority. Keep reusable procedure in skills and task detail in issues.
## Executor selection and terminal worktree cleanup

Use `skills/execution-runner-selection/SKILL.md` for executor selection and `docs/execution-runners.md` for its contract. Native Codex remains the default; optional local Devin prerequisites belong to `skills/devin-local-runner/SKILL.md`. `docs/codex-operations.md` owns Codex model/effort/profile selection. Each parent/child selects independently. Configuration edits do not activate issues, release holds or migrate active sessions; final audit remains fresh independent Codex.

The existing dispatcher also handles issue closure with narrowly scoped cleanup of that issue's registered implementation and detached PR-review worktrees after active audit locks release. It retains branch refs and preserves unrelated/open-issue work. Canonical SkillForge has no local runner and skips this cleanup; initialized repositories retain it. Cleanup grants no authority to launch a new model turn.

## Efficient shell execution

Prioritize the shortest total time to a correct result, not the fewest characters or the smallest output at any cost.

- For routine shell operations, choose a direct command and execute it. Do not prepare a separate plan or compare equivalent alternatives.
- Reuse existing project commands and available tools. Do not write helper scripts for simple operations.
- Scope searches and reads to the necessary paths and data. Do not repeat queries unless the information may have changed.
- Batch related checks when no intermediate decision is needed. Avoid both tiny separate calls and giant command chains.
- Use Python when it is clearer and more reliable than a complicated pipeline. Do not force everything into a one-liner.
- Avoid unnecessary `bash -c` layers, wrappers, temporary files, retries, and frequent polling.
- Preserve errors, exit codes, and required checks. Simplifying must not hide failures or skip validation.

These rules do not waive repository workflow, authorization, validation, or review requirements.
