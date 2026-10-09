---
name: codex-pr-audit
description: Audit one pull request at an exact head through the independent-review procedure, record the verdict, and apply the verdict-derived merge or correction workflow.
---

# Codex Pull Request Audit

## Responsibility

Use this skill when one concrete pull request must be audited after the executor has handed it off with the controlling issue in `review-ready`.

This skill is the audit controller. It owns unambiguous controlling-issue resolution, exact base/head capture, the final technical review under `codex-independent-review`, one canonical audit result, and the GitHub mutations derived from that result. `codex-independent-review` remains read-only and has no mutation authority.

When the audit launcher already created a fresh App Server thread isolated from the executor and a dedicated detached worktree pinned to the exact PR head, the current Codex session may apply `codex-independent-review` directly in that same session. Do not create a nested reviewer merely to manufacture independence the launcher already established. During the review phase remain read-only; after the verdict is fixed, leave the reviewer role and resume this controller procedure.

If the current context is not demonstrably fresh and isolated from the executor, invoke exactly one fresh isolated reviewer instead. An executor context may never self-certify its own implementation.

Use `codex-github-operations` for Git/GitHub mechanics after the verdict. This skill grants standing workflow authority for the positive exact-target audit path; no separate user-facing merge instruction is required. The audit controller never implements a technical fix. A failed review or stale integration target returns work to the normal executor.

## Preconditions

Before technical review:

1. Load `AGENTS.md`, this skill, `codex-independent-review`, `codex-github-operations`, the PR, and the controlling issue.
2. Resolve exactly one controlling issue from authoritative evidence such as explicit caller input, the `codex/issue-N` head convention, or an explicit structured PR-to-issue relationship. Contradictory authoritative sources fail closed.
3. Require an open PR and an open controlling issue whose sole workflow-state label is `review-ready` for a fresh audit. A matching already-recorded audit may be replayed only to finish interrupted compatible controller mutations, never to skip the current CI gate.
4. Capture PR number, head ref, `audit_head`, base ref, and `audit_base`. Both SHAs must be exact published 40-character commits.
5. Require the PR head branch to be owned by the controlling issue and the PR to be ready rather than draft.
6. Establish review mode: direct only when this session is fresh relative to the executor and uses a dedicated detached worktree pinned to `audit_head`; otherwise use one fresh isolated reviewer.
7. Resolve the applicable validation set and its current state through the mandatory pre-merge CI gate in `codex-github-operations`. Source inspection may proceed while CI runs, but completion authority remains suspended.

Fail closed on ambiguous ownership, contradictory issue references, wrong workflow state, missing published commits, or a moving target.

## Exact target invariants

A fresh audit targets the PR's current head at audit start. `audit_head` and `audit_base` are snapshots used to detect races, not caller-selectable historical targets. The review covers the complete PR diff `audit_base..audit_head`.

Immediately before recording a verdict and again before verdict-derived mutation, re-read the PR and require the current head SHA to equal `audit_head`, the base ref to remain expected, and repository/issue ownership to remain unchanged.

If the head moved, the verdict is stale: record at most one concise stale-attempt note, perform no verdict-derived mutation, and require a fresh audit of the new head.

If the head is unchanged but the base branch tip moved from `audit_base`, distinguish normal forward integration drift from rewritten/ambiguous base history. Forward drift is executable reconciliation work, not technical `FAIL` and not `BLOCKED`: end the reviewer role, mark the PR draft if ready, replace `review-ready` with `execution-ready`, leave the issue open, and stop. The dispatcher relaunches the executor so it can reconcile, republish, and obtain fresh CI. Rewritten or ambiguous base history fails closed with a concise control-plane finding.

## Review phase

Apply `codex-independent-review` exactly as written. Supply `AGENTS.md`, the controlling issue, PR number and exact `audit_base..audit_head`, required evidence, the applicable CI set and current results, and unresolved material findings.

This invocation is the workflow's final technical review, so ask the reviewer to determine whether the exact target is **final-capable**: it must contain the complete final PR diff, final dependency revisions, required evidence, and all remaining acceptance criteria. The review returns exactly `PASS`, `PASS_WITH_NOTES`, `FAIL`, or `BLOCKED`, plus exact head, final-capable yes/no, evidence inspected/run, and findings.

A positive result is actionable only when the reviewed head equals `audit_head`, `final-capable: yes`, and the mandatory pre-merge CI gate is satisfied. Do not mutate implementation or GitHub state while acting in the reviewer role.

### CI completion is a separate mandatory condition

Use the canonical gate in `skills/codex-github-operations/SKILL.md#mandatory-pre-merge-ci-gate`; do not substitute GitHub mergeability, branch-protection configuration, or a passing subset of tests. Required final evidence includes the completed applicable CI jobs, not merely their successful setup/check steps. In particular, a green `npm run check` does not establish success of a still-running browser or application-acceptance job.

Do not issue a positive canonical audit with `final-capable: yes` while applicable mandatory CI is pending, missing, failed, cancelled, timed out, or otherwise unresolved. Pending CI is not a non-blocking note and must not be waived because focused review tests passed. A material technical FAIL may still be returned promptly without waiting for unrelated remaining CI.

When the inspected code has no material finding but CI is still running, retain the technical assessment, keep the PR ready and issue `review-ready`, and wait with bounded polling as specified by the gate. If the turn ends first, leave at most one concise CI-wait handoff identifying the exact target and outstanding run URLs; do not fabricate a final verdict or mark the issue `blocked` merely for normal CI latency. A subsequent explicit audit resume must revalidate head/base and inspect the completed CI evidence; reuse unchanged independent review evidence rather than rerunning all tests. Do not assume a CI-completion dispatcher exists.

For a terminal applicable test failure, inspect and record the exact failing gate and smallest corrective scope, then return the PR to draft and issue to execution through the normal FAIL path. Failures outside the current contract must be routed to a resolved design/dependency decision, never silently accepted as pre-existing. Infrastructure cancellation or timeout is not a successful test outcome: obtain an explicitly justified successful rerun or leave a precise unresolved-evidence handoff. Do not rerun a reproducible assertion failure to search for a lucky pass.

Every replay of a previous positive audit must refresh the gate. A newer pending/failing attempt or stale head/base suspends the prior completion authority even if the old comment says `final-capable: yes`.

## Canonical audit record

Before changing PR or issue state, write one concise PR comment with this exact marker:

```markdown
<!-- codex-pr-audit:v1 head=<full-head-sha> verdict=<VERDICT> -->
## Codex PR audit

**Controlling issue:** #<issue>
**Reviewed head:** `<full-head-sha>`
**Observed base:** `<full-base-sha>`
**Verdict:** `<VERDICT>`
**Final-capable:** `<yes-or-no>`
**Validation/evidence:** <concise summary, including applicable CI run/check URLs, tested head/base or merge revision, attempts, terminal conclusions and justified non-applicability>
**Material findings:** <none or concise findings>
**Non-blocking notes:** <none or concise notes; unresolved mandatory CI is never non-blocking>
```

Do not submit a formal GitHub `APPROVE` review. If the exact marker already exists for the same head and verdict, do not duplicate it; use it only to reconcile missing compatible controller mutations after refreshing the gate. A different head requires a new audit. Integration drift detected before a valid verdict does not invent a verdict or canonical record.

## Verdict mapping

Re-fetch the exact PR and controlling issue before applying any result. Positive rows below are conditional on a freshly satisfied mandatory pre-merge CI gate.

| Verdict | PR action | Issue action | Terminal result |
| --- | --- | --- | --- |
| `PASS` | Merge exact audited head only after CI eligibility | After merge is observed, `review-ready` -> `completed`, then close | Integrated and complete |
| `PASS_WITH_NOTES` | Same gated merge; preserve notes | Same completion | Complete with non-blocking notes |
| `FAIL` | Mark draft if ready | `review-ready` -> `execution-ready`; leave open | Returned to executor |
| `BLOCKED` | Keep readiness | `review-ready` -> `blocked` only for genuine unavailable capability/evidence | Blocked with exact cause |

Integration-base drift or a `dirty`/conflicting PR caused by base reconciliation is a control-plane return, not an extra reviewer verdict. Mark draft, return to `execution-ready`, and let the normal executor reconcile it.

### Positive automatic completion

For `PASS` or `PASS_WITH_NOTES`:

1. require `final-capable: yes` and exact head/base equality;
2. require the PR still open and ready and the controlling issue solely `review-ready`;
3. refresh the complete mandatory pre-merge CI gate, including current attempts and the actual PR test-merge revision where applicable; stop without merge unless it is satisfied;
4. revalidate the current head and base tip, then merge exactly `audit_head` through `codex-github-operations` with an expected-head guard; never rewrite the reviewed head first or bypass pending/failed checks;
5. re-read the PR and require GitHub to report it actually merged;
6. if the controlling issue is still open with sole state `review-ready`, replace that state with `completed` preserving unrelated labels;
7. verify `completed` is observable — this event intentionally wakes an active epic scheduler when the issue is a DAG child;
8. close the issue with completed resolution and verify it remains labeled `completed`;
9. stop.

If exact-head merge cannot complete, classify why. Integration drift/conflict returns to executor. Pending checks/mergeability with unchanged exact head/base leave the issue `review-ready` for later controller reconciliation. Use `BLOCKED` only for a genuine unavailable capability with no safe alternative. A server accepting an early merge is not evidence that this controller's CI gate passed.

Never mark `completed` until GitHub reports the exact PR actually merged.
