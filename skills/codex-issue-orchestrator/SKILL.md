---
name: codex-issue-orchestrator
description: Orchestrate an explicit set of controlling GitHub issues through fresh Codex workers in parallel or sequential order while preserving each issue's normal spec-driven execution workflow.
---

# Codex Issue Orchestrator

## Responsibility

Use when one parent Codex session must execute a known explicit set of ordinary controlling issues through separate worker agents. This is manual/batch orchestration, distinct from `codex-epic-scheduler`: it does not generate or own an epic DAG and does not react to workflow labels.

It owns issue ordering/dispatch, parallel vs sequential scheduling, independent vs dependent semantics, base selection, worker-failure propagation, progress observability, and final batch summary. It never implements, redesigns, audits, or merges a child issue itself.

Each worker executes exactly one controlling issue with `spec-driven-codex-loop`. Its successful terminal outcome is a ready PR plus `review-ready`; the separate issue-state dispatcher/audit workflow may later audit/merge/complete it.

## Inputs

Resolve explicit ordered issue list, `schedule` (`parallel` or `sequential`), `dependency` (`independent` or `dependent`), and initial base ref. Valid combinations are parallel+independent, sequential+independent, and sequential+dependent. Parallel+dependent is invalid.

Do not infer extra work from neighboring issues, labels, milestones, or an epic body after the issue list is resolved. Do not promote provisional conclusions from one worker into another contract.

## Worker contract

For each issue start a fresh worker, give it exactly one controlling issue and selected base, require normal `AGENTS.md` + `spec-driven-codex-loop`, optionally request coarse progress observability, and let it own implementation, validation, evidence, commits, issue-declared intermediate review, PR publication, and `review-ready` handoff. Wait/record only what orchestration needs.

Do not transfer temporary diagnostics, hidden reasoning, uncommitted state, or ad-hoc prompt conclusions between workers. Each issue keeps its own PR/branch unless its contract explicitly says otherwise.

## Progress observability

Workers may add concise issue comments at useful phase boundaries such as start/context established, completion of a material setup/preflight phase, start/end of a long-running evaluation/build/migration, coarse cheaply available progress, or a material retry/fallback/blocker. Avoid per-item/log chatter.

Progress comments are not review checkpoints, do not alter acceptance/scope, and do not replace normal material findings/handoff comments.

## Independent issues

Independent workers must not inherit unmerged sibling work. They branch from the designated common base/ref and keep outcomes isolated.

Sequential independent execution waits for each worker before the next but continues after `review-ready`, `blocked`, `design-required`, `investigation-required`, implementation/validation failure, or worker/transport failure unless caller explicitly requested otherwise. Record failure; do not repair it in orchestrator.

Parallel independent execution launches one fresh worker per issue; one failure does not cancel siblings. Wait for all to terminate then summarize.

## Sequential dependent issues

Use only when each issue intentionally builds on the previous issue's unmerged implementation. First starts from initial base; each later worker starts from exact published head of immediately preceding successful `review-ready` issue and its PR targets predecessor branch so its diff is incremental. Never auto-merge predecessors merely to advance chain.

A downstream worker starts only after predecessor provides a valid `review-ready` published branch/head. Any other predecessor outcome stops the chain and marks later issues not executed due to dependency failure. If branch topology changes externally and intended next base becomes ambiguous, stop rather than guessing.

## Existing work and resume

`completed` may be recorded as already complete unless re-execution is explicit. `review-ready` may be reused as successful executor outcome when no more execution was requested. An `in-progress` issue with existing branch/PR is resumed through normal executor workflow rather than duplicated. Ambiguous ownership blocks a second executor.

## Completion

After all permitted workers terminate, produce one compact table with issue, outcome, PR/branch, and result/blocker. Useful outcomes: `review-ready`, `already-completed`, `blocked`, `design-required`, `investigation-required`, `failed`, and `not-run-dependency-failure`.

Orchestration completion means workers reached terminal execution outcomes; it does not imply PR audit/merge/completion.
