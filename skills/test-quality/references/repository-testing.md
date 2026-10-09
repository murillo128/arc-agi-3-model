# Skillforge testing boundaries

This reference covers the reusable template infrastructure, not a generated
application. In an initialized repository, inspect its accepted specification,
package/build configuration and native tests before selecting language-specific
checks. Keep product guidance in that repository; do not introduce Java/JUnit,
pytest, Vitest or mutation tools merely because an upstream skill uses them.

## Native checks and ownership

The existing [executor CI](../../../.github/workflows/executor-routing-ci.yml)
is the owner of the hosted offline regression gate. From the repository root,
select the relevant command while iterating:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_executor_routing.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_worktree_cleanup.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_epic_scheduler.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_canonical_template.py' -v
```

`test_executor_routing.py` covers selection, local supervision, the host broker,
real temporary Git worktrees and a fake Devin CLI. When claiming transport proof,
run it with `REQUIRE_TMUX_TEST=1` and a real installed `tmux`; a skipped transport
test is not that proof. `test_codex_profile.py` owns native model/profile and
App Server lifecycle behavior. Cleanup and epic/discovery suites own their
separate filesystem and scheduling contracts. The canonical-template test guards
the template's no-host-execution boundary, not initialized product behavior.

Read complete tests and owners before extending them. Representative risks are
cross-issue ownership, delayed events releasing a hold, wrong-session resume,
unfinished work preservation and an apparent successful exit hiding rejected
tools. Prefer real temporary filesystem/Git behavior for those claims, while
keeping paid models and authenticated live infrastructure out of offline tests.
No offline double proves installed CLI compatibility, real model execution or
host sandbox isolation. Never activate an issue or provision the canonical
Skillforge runner as a test.

## Evidence and documentation-only changes

Use isolated temporary state and the existing scripts rather than a new harness.
Read CI filters at the candidate revision to determine the final gate; do not
expand application or paid execution in response to a skill-only edit. For a
procedural Markdown change, verify relative links, frontmatter, routing from
`AGENTS.md` and the role skills, and preservation of existing authority boundaries.
Review behavioral examples against the procedure, but do not present that review
as an executed agent benchmark. Source-text keyword assertions alone would not
prove that an agent follows the skill.

Measure actual offline job/test durations before optimizing them. Broader audits,
CI restructuring and mutation tooling require their own bounded authorization.
