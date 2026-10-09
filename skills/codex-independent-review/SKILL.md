---
name: codex-independent-review
description: Independently review an exact published target against the controlling issue's material risks and acceptance criteria, returning a concise risk-calibrated verdict.
---

# Codex Independent Review

## Responsibility

Use this skill for an explicitly required intermediate checkpoint or for the final technical review invoked by `codex-pr-audit`.

The reviewer owns independent exact-target inspection, proportional validation, materiality, and the technical verdict. It does not implement fixes, redesign the issue, mutate workflow state, publish commits, continue execution, or authorize/perform merge while acting as reviewer.

A `PASS` or `PASS_WITH_NOTES` means the reviewed target is technically safe to progress according to the calling workflow. The verdict itself has no mutation authority. When invoked by `codex-pr-audit`, a positive final-capable verdict may be consumed by that controller after the reviewer role has ended.

## Trust the issue as the technical contract

Judge implementation against the controlling issue's explicit contract and authoritative repository invariants. Do not replace settled decisions with a new design merely because another approach is possible. Exploratory notes, derived wiki content, and chat history are context, not acceptance criteria, unless explicitly adopted.

## Minimal review packet

A fresh reviewer needs `AGENTS.md` when present, this skill, the controlling issue or checkpoint contract, the exact published target/range, and technical evidence required for that target. Load prior comments/reviews only when an unresolved material finding or circuit breaker depends on them.

## Independence

Use a fresh context that does not inherit the executor's hidden reasoning, inspect exactly the requested target, remain read-only, judge evidence rather than intent, and do not implement corrections or advance later work. Independence does not mean maximal hostility: test declared risks and plausible normal use rather than every imaginable malformed variant outside the contract.

## Materiality

Return `FAIL` only when a finding violates an explicit invariant/acceptance criterion, exposes a plausible normal-path defect, makes required evidence materially false/incomplete/misleading, introduces unapproved material scope/architecture/dependency/behavior, or makes progression unsafe.

Use `PASS_WITH_NOTES` for editorial wording, bookkeeping, optional hardening, stale non-authoritative prose, or robustness outside the declared boundary when the technical outcome remains trustworthy.

For every `FAIL`, state the exact criterion violated, material consequence, and smallest corrective delta or why design must reopen. If those cannot be stated concretely, do not fail the target.

## Review procedure

### 1. Establish risk and authority

Identify scope, invariants, acceptance criteria, exact target, evidence, and explicit failure boundary.

### 2. Inspect the exact target

Check complete diff/scope compliance, implementation/integration, credible required evidence, plausible correctness/ownership/numerical/data/concurrency/security/backend/performance risks covered by the contract, unexpected dependencies/secrets/restricted artifacts, and safety to progress. Do not replay accepted earlier ranges without a concrete unresolved risk.

### 3. Test proportionally

Prefer checks capable of falsifying the claimed outcome. Exact-head green CI, retained executor evidence, and deterministic artifacts are valid when target/environment are clear. Inspect and reuse them rather than mechanically rerunning a broad suite. Run focused or additional checks when they materially increase independence, target a plausible risk, close an evidence gap, or are explicitly required. Distinguish checks personally run from evidence inspected and never claim an unrun check passed.

Use [test-quality](../test-quality/SKILL.md) read-only when assessing coverage: verify independent expected results, credible regression failure/pass evidence, and retained contracts after deletion or relocation. Challenge mocks that supply the asserted behavior and changes that weaken assertions to hide defects. Apply the existing materiality bar; this adds neither implementation authority nor another final review stage.

### 4. Determine final capability

For an **intermediate checkpoint**, call it final-capable only when the controlling issue explicitly says that checkpoint may serve as final review and the exact target already contains the complete final diff, final dependencies, required evidence, and all remaining acceptance criteria.

For a **final review invoked by `codex-pr-audit`**, the controller itself establishes final-review intent. Return `final-capable: yes` only if the exact `audit_base..audit_head` actually contains the complete final PR diff, final dependency revisions, required final evidence, and all remaining acceptance criteria/findings. Return a non-positive verdict or `final-capable: no` when that condition is not met; do not require a redundant issue declaration saying the audit is final.

A later technical change invalidates the verdict for the changed target. Changes only to issue/PR prose, labels, merge metadata, or other workflow state do not change the reviewed technical target.

Final technical review returns control to the caller. It never allows the reviewer role itself to merge or mutate GitHub.

### 5. Report briefly

Record exact target, verdict, safety to progress, final-capable yes/no, material findings, validation run/evidence inspected, and smallest required delta or non-blocking notes. Do not imply the reviewer performed or authorized a merge.

## Reviewer transport

Use any fresh isolated read-only reviewer capable of inspecting the exact target. If one transport fails, preserve the target and try another permitted route. Do not amend commits, guess a different target, or fall back to executor self-review. Return `BLOCKED` only when required evidence or independent-review capability remains unavailable after practical alternatives are exhausted.

## Repeated-review circuit breaker

After two consecutive reviews fail for substantially the same validation, attestation, parser, documentation-sync, or bookkeeping mechanism, stop open-ended representational searches. Use `PASS_WITH_NOTES` when progression is technically safe and the concern is non-material; request design review when the validation strategy itself prevents a trustworthy decision. This never waives a continuing material defect.

## Verdicts

Return exactly one of `PASS`, `PASS_WITH_NOTES`, `FAIL`, or `BLOCKED`. Transport failure is an attempt result, not a verdict.
