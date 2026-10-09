---
name: test-quality
description: Plan, write, change, review or audit tests against independent behavioral contracts, prove regressions, and reduce redundant validation without weakening required evidence.
---

# Test Quality

## Responsibility and authority

Use alongside the current design, execution or review skill whenever test coverage
is planned, added, changed, reviewed or removed. Authoring mode evaluates proposed
tests; audit mode evaluates an existing bounded test surface. Optimize defect
detection and maintainability per unit of validation cost, not test count,
coverage percentage or deletion volume.

This skill does not authorize product changes, test deletion, issue activation,
CI exemptions, runner provisioning or publication. The controlling contract and
existing workflow retain those decisions. Analysis-only requests remain read-only;
independent reviewers remain read-only. Do not add a second final review stage.

## Load focused context

Read root and applicable scoped `AGENTS.md`, the controlling contract, and the
relevant accepted specification before choosing expected behavior. Then inspect
the complete candidate tests, production entry point and owner, relevant callers
and dependencies, nearby coverage, and applicable CI routing. Inspect dependency
source or types when a claim depends on their behavior. Search before declaring a
coverage gap; do not preload unrelated subsystems.

Read [repository testing guidance](references/repository-testing.md) for native
commands and boundaries. Confirm commands and options against the current checked
out scripts/configuration. Keep the reusable procedure here and repository-specific
instructions in that reference; a template's infrastructure tests are not proof
of an initialized application's behavior.

## Authoring mode: plan before writing

Record a compact plan in the existing task/PR evidence, not a new mandatory
manifest. Each case or parameterized family needs: the observable contract and
independent expected-result source; a plausible defect it detects; the responsible
boundary and proposed test location; existing coverage and the remaining gap;
representative inputs, action and assertions; and a focused validation target.
Reuse a prior plan while its source/contract remains applicable, explaining only
material additions or removals. If existing tests already provide the required
proof, say so rather than manufacturing more cases.

Choose the cheapest boundary that faithfully exercises the responsibility being
claimed. Give each behavior a primary test owner; another layer needs a distinct
risk, such as serialization, lifecycle, deployment or real rendering. A unit test
cannot replace integration proof for a boundary it mocks. Conversely, do not
repeat the entire domain input matrix through every transport and browser layer.
Avoid exposing internals or adding production flags/wrappers solely to reach a
test; first exercise the real interface. A genuine architectural seam still needs
its own approved production justification.

Partition cases by behavior and independently changeable conditions, not by method
count or a mechanical target for every branch. Include relevant positive, failure
and nearest valid/invalid boundary cases, keeping other constraints valid so each
negative control reaches the intended guard. The same error code does not make
two independent conditions duplicates. Shapes, empty inputs, Unicode, nullability,
ordering and concurrency are relevant when the contract or credible failure mode
requires them, even without an explicit source branch. Do not invent unsupported
requirements or variations inside an unchanged equivalence class.

## Authoring mode: independent assertions and execution

Keep setup, action and expected outcomes understandable together and follow native
naming/fixture conventions. Prefer extending parameterized cases or a small shared
fixture over duplicating scenarios; do not hide the essential cause in a distant
setup framework. Multiple assertions may jointly describe one observable result.

Expected values must not be calculated by the implementation under test or by a
mock that already implements the claimed behavior. Use explicit small examples,
accepted contract fixtures, independently derived references or justified
properties. Simple loops, property-based cases and independent numerical oracles
are allowed when they improve the proof; do not blindly duplicate the production
algorithm. Explain tolerances and seeds where relevant. Assert observable state,
output or side effects, not incidental private calls. Mock unrelated external
costs/failures only; a fake transport or model proves only the boundary exercised.

Compile/type-check when applicable, run the focused tests and required sibling or
integration checks, and inspect actual assertions and outcomes. For a bug fix,
demonstrate the regression test failing on the pre-fix revision for the intended
reason and passing with the repair. Missing imports, setup failures or unrelated
guards are not a valid red result. Use an isolated snapshot or the repository's
permitted worktree procedure; never reset shared work or edit a checkout during
its checks. If baseline execution is genuinely unavailable, state the evidence
gap and obtain a contract-appropriate validation decision instead of claiming the
regression is proven.

When a test fails, distinguish a bad fixture/assertion from a product defect using
the accepted contract. Correct a wrong test with an independent justification;
fix a product defect only within authorized scope. Otherwise retain the reproducer
and report the discrepancy through the existing investigation/design workflow.
Characterization tests may record current behavior only when that is the explicit
task, clearly separate from required behavior. Never change expectations to match
a known defect, refresh a snapshot blindly, widen a tolerance, remove an assertion
or add retries just to make the build green.

Do not automatically delete, skip, disable or mark an expected failure after a
fixed number of attempts. Any accepted quarantine needs an explicit reason,
tracked owner and revisit condition, and must remain visible as missing coverage;
it cannot satisfy the failing acceptance criterion. Existing workflow circuit
breakers still apply and never turn unresolved failures into success.

## Audit mode: evidence before deletion

Start with read-only discovery for one coherent subsystem or test family. Read
relevant history before judging why a test or support seam exists. Look for
assertion-free execution, self-comparisons, expected values generated by the
subject, mocks that supply the asserted outcome, repeated contracts with no new
risk, private-call choreography, source-text restatements and test-only production
scaffolding. These are investigation signals, not automatic deletion rules.

For each proposed removal or relocation, record the exact test/path, defect it can
actually detect, relevant history, non-test consumers of any affected seam,
remaining independent proof (or evidence that the contract is obsolete), proposed
support/production cleanup, risk and focused validation. Missing evidence means
retain the candidate or continue investigating, not delete it speculatively.
Obtain edit authority before applying a discovery-only recommendation.

Retain independent API, protocol, storage, security, numerical, packaging,
generated-binding, configuration, platform and architecture contracts. Ordering
is worth asserting when observable. A source check can be a valid economical
contract guard if it tests a durable public property rather than identifier
spelling. Slowness, static inspection or sensitivity to refactoring alone does
not establish redundancy. A baseline failure may expose a real product defect.

Apply only the authorized coherent batch. Consolidate duplicate setup, keep
necessary regressions at their responsible boundary, and remove genuinely unused
support code only after checking consumers. Validate retained proof against the
claimed risk; for migrated bug regressions preserve red/green evidence. When a
static check is replaced, execute the real contract or its supported dry-run.
Do not trade a difficult test for a cheaper mock that cannot detect the same bug.

## Proportional CI and optional experiments

Map changed production, test, fixture, generator and configuration paths to the
accepted validation owners. Inspect workflow event/base/path/job conditions;
important contract-only changes must not silently lose their consumers' proof.
Report a routing gap and address it only with appropriate change authority. Run
focused checks while iterating and preserve every applicable final CI gate.
A documented epic-child deferral does not waive local scoped validation or the
aggregate integration gate. This skill creates no new deferral and does not
change final audit or merge eligibility.

Reuse evidence only when the exact revision, relevant base/dependencies and
environment match the claim. A rebase or changed dependency invalidates affected
proof. Reviewers should inspect credible exact-target evidence instead of
mechanically repeating broad suites, adding focused checks for distinct risks.
Cache dependencies and reproducible inputs within existing policy, not stale
pass/fail results presented as evidence for another revision. Keep concurrent
checks in isolated snapshots with separate ports/artifacts and resource budgets.

Measure setup, build and test time separately before claiming an optimization.
Compare equivalent scope/environment and record retries, flakes and retained
contracts; fewer tests or higher line/branch coverage alone is not success.
Never remove real boundary or platform proof merely because it is expensive.

Mutation testing is an optional, explicitly scoped experiment, not a default PR
requirement or permission to install tools. Choose a small critical module, an
agreed runtime budget and isolated state; first require the unmodified baseline
to pass, then verify that meaningful deliberately incorrect variants fail for the
intended reason. Classify equivalent, invalid, timed-out and surviving variants
honestly. Report tool/configuration, scope and detected defects; do not promise a
global mutation percentage or start a heavy scheduled suite without approval.

## Handoff and review

Use the existing task/PR report: contracts and test owners, cases added/reused or
removed with reasons, expected-result provenance, regression failure/pass evidence,
exact revisions and commands, checks personally run versus inspected, skips and
remaining risks. Include measured cost only when comparable evidence exists.
Missing optional formatting is not itself a product failure; judge material proof
and the calling workflow's acceptance criteria. Compilation or a green badge is
not enough to establish that assertions can detect the claimed defect.

## Sources and deliberate adaptation

This is a locally authored procedure informed by
[OpenClaw test-audit](https://github.com/openclaw/openclaw/blob/a15f3f72a6aabe4d51ce53426d717b4c65a4589e/.agents/skills/test-audit/SKILL.md)
and [Mavka generate-tests](https://github.com/mavka-ai/unit-tests-skills/blob/66d5aa34b51f3db39431dac2b7515c7d761e773c/skills/generate-tests/SKILL.md).
The source revisions are pinned for traceability, not automatically installed.
We adopt behavioral value review and planning before generation, not their
repository-specific commands, Java-only conventions, extra review controllers or
Mavka's rule to conform tests to current production behavior and disable them
after repeated failures. No benchmark result is assumed transferable here.
