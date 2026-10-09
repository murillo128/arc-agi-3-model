# Codex operating policy

Executor selection belongs to [Execution runners](execution-runners.md). This
page owns model selection for this repository's shared Codex App Server. Importing
configuration never authorizes issue execution, host provisioning or session migration.

## Native defaults and issue overrides

With no top-level `codex` TOML block in the controlling issue body, leave model and
reasoning effort unset. The serving App Server resolves its native configuration
for the worktree. A resumed thread retains its saved selection, not today's local
default. There is no hardcoded model family, minimum effort or mandatory profile.

The optional block accepts only `model`, `model_reasoning_effort` and `profile`.
Explicit model/effort values take precedence over an optional installed profile;
unspecified fields remain native. The resolved explicit pair must be accepted by
the paginated live `model/list` catalog. An unavailable model/effort or missing
profile fails before a model turn; there is no silent fallback. Model names in
prose, comments, quoted examples or a parent issue do not configure a child.
Duplicate, malformed, unterminated and oversized blocks fail closed.

A parent's block configures its scheduler only. Every child selects independently.
On a Codex-owned issue the block applies to implementation and fresh audit. On a
Devin-owned issue it applies only to the independent Codex audit; it does not
select the model inside Devin. Use an explicitly suitable audit configuration
rather than assuming a low-cost implementation override becomes a stronger audit.

An explicit change applies only to the next authorized turn of an inactive
session. It never steers an active turn. Removing an override does not erase a
saved thread's previous choice. The former `CODEX_EXECUTION_PROFILE` and
`CODEX_AUDIT_PROFILE` repository variables are not consulted.

## Optional profiles and delegation

Profiles are optional existing files at `<effective CODEX_HOME>/<name>.config.toml`.
The launcher accepts only model/effort and an optional `[agents]` table whose sole
entry is `max_concurrent_threads_per_session = 3`. Profile names cannot contain
paths. Other permissions, authentication, instructions and native configuration
features cannot be granted through issue settings. Do not install or overwrite
profiles, change credentials or restart the shared server to select a model.

The auxiliary-thread cap of three is per session, not a host-wide budget and not
the epic's `max_parallel_workers`. Preserve one product implementation owner and
fresh independent review. Do not assume a helper's read-only role description is
an enforced isolation barrier. Heavy validation remains coordinated per host.

## Runtime and observability

The Actions launcher reads the current issue with short-lived authentication and
writes only validated settings to a mode-0600 snapshot. A failed issue read is
not an absent override. Detached clients remove Actions tokens and use the
existing shared App Server and persistent GitHub authentication. Helpers come
from the trusted published workflow revision, not a PR implementation worktree.

`model_policy_resolved` distinguishes native/override mode, observed model/effort
and intended turn overrides. A resume response describes the previous settings,
not confirmation that a subsequent override was applied. Null effort stays null.
Explicit new-thread settings must match the returned selection. An old loaded
thread may not adopt a newly requested auxiliary cap. Coordinate ownership:
there is no atomic cross-client lock between idle checking and turn launch.

Desktop project assignment is best-effort metadata only. A unique canonical
repository-root match can be assigned to an explicitly unassigned thread; manual
assignments and unknown fields are preserved. This cannot change cwd, model,
permissions, branch or audit isolation, and unsupported APIs are warnings.

## This repository's hosts and validation

Do not copy another repository's absolute paths, model availability, service
names, maintenance observations or issue holds as facts about this repository.
Verify the intended runner user's effective Python (3.11 or later for `tomllib`),
App Server version/home/socket, persistent clone and GitHub authentication when
provisioning is separately authorized. See `skills/codex-local-runner/SKILL.md`.
Application compute hosts and local caches are separate from coding-agent
selection; the router does not select or provision application infrastructure.

Run the offline lifecycle tests without provider credentials:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py' -v
```

Offline tests do not prove live-host readiness or model availability. Do not
activate a real issue, launch paid sessions, interrupt workloads or restart a host/server
as an import smoke test. Preserve existing local CI provisioning; workflow validation
uses a separate hosted job.

References: [App Server](https://developers.openai.com/codex/app-server/),
[native configuration](https://developers.openai.com/codex/config-advanced/).
