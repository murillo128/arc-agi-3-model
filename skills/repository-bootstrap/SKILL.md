---
name: repository-bootstrap
description: Initialize a repository created from the Skillforge template exactly once, establish required GitHub workflow labels and project-specific top-level instructions, then remove this bootstrap skill and its template-only references; optionally hand off to local Codex runner provisioning.
---

# Repository Bootstrap

## Responsibility

Use exactly once when initializing a repository created from the Skillforge template. Never execute it in canonical `murillo128/skillforge` itself.

Successful terminal state is: project-specific `README.md`; `AGENTS.md` contains only established repository-specific invariants plus generic Skillforge workflow; all required workflow labels exist; `skills/repository-bootstrap/` is gone; and `README.md`/`AGENTS.md` no longer route to bootstrap.

Local Codex runner setup is optional and separately owned by `codex-local-runner`; it is not a bootstrap acceptance criterion.

## Narrow authority

Explicit invocation permits direct default-branch bootstrap publication only for:

- create/replace `README.md`;
- update `AGENTS.md`;
- delete `skills/repository-bootstrap/**` as final self-removal;
- create/normalize required repository labels.

It may not modify code, tests, workflows, other skills, docs/wiki, build/configuration, fixtures, data, issues/PRs/releases/projects, or unrelated repository settings. Optional runner opt-in is only a post-publication handoff and does not widen bootstrap authority.

## Required labels

Verify these nine workflow-state/provenance labels exist:

- `queued`
- `execution-ready`
- `in-progress`
- `review-ready`
- `design-required`
- `investigation-required`
- `blocked`
- `completed`
- `curator-detected`

Names are normative; descriptions/colors are not. Create missing labels without deleting or repurposing unrelated labels. If all required labels cannot be verified, bootstrap remains incomplete and must not self-remove.

`queued` is required because fully designed epic children wait there until `codex-epic-scheduler` activates them.

## Inherited workflow capability

Repositories inherit:

- `.github/workflows/codex-issue-state.yml`, the single `issues:labeled` state dispatcher;
- reusable `.github/workflows/codex-execute-ready.yml` and `.github/workflows/codex-review-ready.yml`;
- `codex-epic-scheduler`, which generates/persists an epic DAG on the parent's first scheduler execution and schedules later waves;
- optional `codex-local-runner` host provisioning.

The workflows may exist with no self-hosted infrastructure. Absence of a `codex` runner does not make bootstrap incomplete. Only explicit user opt-in may trigger post-bootstrap runner provisioning.

## Project-specific README and AGENTS

Rewrite template README into the actual project's durable entry point using only authoritative supplied/repository context. Do not invent architecture, commands, roadmap, features, or maturity merely to fill a template. If project mission/scope is insufficiently known, stop before self-removal.

Adapt `AGENTS.md` with only established repository-wide invariants such as runtime/coding constraints, ownership/concurrency/security invariants, required validation paths, dependency/licensing constraints, hardware/runtime requirements, hard correctness/performance failures, or explicit planning authority. Keep reusable procedure in skills and task-specific detail in issues.

Preserve generic Skillforge routing including optional local-runner, issue execution/audit, epic scheduler, GitHub operations, review, and wiki curation. Remove only bootstrap-specific routing/sections.

## Idempotent procedure

1. Verify exact repository identity and stop with no mutation if it is `murillo128/skillforge`.
2. Load this skill, current README/AGENTS, repository metadata, labels, and only project context needed for truthful top-level docs. Record explicit runner opt-in if supplied; unspecified means no runner setup.
3. Create missing required labels and verify all nine.
4. Prepare project-specific README.
5. Adapt AGENTS with established project invariants while preserving generic workflow.
6. Prepare mandatory self-removal: delete this skill and remove all bootstrap references from README/AGENTS.

## Publication hard gate

Before final commit, complete candidate path set must be a subset of exactly:

```text
README.md
AGENTS.md
skills/repository-bootstrap/SKILL.md
```

The skill path may only be deleted. Abort publication if any candidate path escapes this set.

Also verify all nine labels exist; README is truthful/project-specific; AGENTS contains no bootstrap routing/section; AGENTS still routes persistent local-runner and other generic workflow skills; resulting tree lacks bootstrap skill; and no other content changes.

Publish the completed bootstrap as **one coherent commit directly to default branch**, normally `chore: bootstrap repository`. If available transport cannot publish atomically, stop rather than leave partial bootstrap.

## Post-publication verification and optional runner handoff

Re-read default-branch state and verify exact changed path boundary, project README, AGENTS cleanup/routing, bootstrap absence, and all labels. Only then report bootstrap complete.

After successful verification, invoke `codex-local-runner` only if user explicitly opted in. Runner failure is a separate infrastructure failure and does not invalidate completed bootstrap.
