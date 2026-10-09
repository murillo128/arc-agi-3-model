"""Optional issue-owned model settings; otherwise leave native Codex defaults alone.

Only a top-level `codex` TOML fence is configuration. Issue text never controls
permissions, commands or paths. Actions credentials are used only by prepare-issue
before detaching; the runtime receives a bounded settings snapshot, not the token.
"""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib
import urllib.request


CAP_KEY = "agents.max_concurrent_threads_per_session"
SETTINGS_LIMIT = 8192
KEY_PATTERNS = {
    "profile": r"[A-Za-z0-9_-]{1,64}",
    "model": r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}",
    "model_reasoning_effort": r"[A-Za-z0-9_-]{1,32}",
}


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) - KEY_PATTERNS.keys():
        raise ValueError("Codex settings permit only profile, model and model_reasoning_effort")
    for key, value in settings.items():
        if not isinstance(value, str) or not re.fullmatch(KEY_PATTERNS[key], value):
            raise ValueError(f"Invalid Codex setting: {key}")
    return dict(settings)


def issue_settings(body):
    """Read one dedicated fence, ignoring quoted/indented/nested examples.

    Scan other Markdown fences too: a `codex` example inside a longer enclosing
    Markdown fence is documentation, not a runtime setting.
    """
    if body is None:
        return {}
    if not isinstance(body, str) or len(body.encode("utf-8")) > 1_048_576:
        raise ValueError("Invalid or oversized issue body")
    fence = None
    selected = False
    found = False
    lines = []
    for line in body.splitlines():
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*", line):
                fence = None
                selected = False
            elif selected:
                lines.append(line)
            continue
        opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
        if not opening:
            continue
        fence, info = opening.groups()
        selected = info.strip() == "codex"
        if selected:
            if found:
                raise ValueError("Expected at most one top-level codex settings block")
            found = True
    if selected:
        raise ValueError("Unterminated codex settings block")
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > SETTINGS_LIMIT:
        raise ValueError("Codex settings block exceeds 8 KiB")
    try:
        return validate_settings(tomllib.loads(text))
    except tomllib.TOMLDecodeError:
        raise ValueError("Malformed TOML in codex settings block") from None


def load_settings(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(SETTINGS_LIMIT + 1)
    if len(raw) > SETTINGS_LIMIT:
        raise ValueError("Oversized Codex settings snapshot")
    return validate_settings(json.loads(raw))


def prepare_issue(path):
    """Fetch current controlling issue using Actions auth, then retain settings only."""
    repo, number = os.environ["GITHUB_REPOSITORY"], os.environ["ISSUE_NUMBER"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid repository name")
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("Invalid controlling issue number")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{number}",
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "codex-issue-settings"},
    )
    # Failure to read the issue is not evidence that it has no override.
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError("Oversized issue response")
    issue = json.loads(raw)
    if (not isinstance(issue, dict) or issue.get("number") != int(number)
            or "pull_request" in issue or "body" not in issue):
        raise ValueError("Expected the controlling issue, not a pull request")
    settings = issue_settings(issue.get("body"))
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".codex-settings-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(settings, output, separators=(",", ":"))
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProfilePolicy:
    def __init__(self, client, codex_home, settings, role):
        if role not in {"execution", "audit"}:
            raise ValueError("Unknown Codex launcher role")
        self.client, self.role = client, role
        self.settings = validate_settings(settings)
        self.name = self.settings.get("profile")
        self.source = None
        self.overrides = {}
        if self.name:
            if not isinstance(codex_home, str) or not Path(codex_home).is_absolute():
                raise ValueError("App Server did not report an absolute CODEX_HOME")
            self.source = Path(codex_home) / f"{self.name}.config.toml"
            try:
                with self.source.open("rb") as handle:
                    raw = handle.read(SETTINGS_LIMIT + 1)
                if len(raw) > SETTINGS_LIMIT:
                    raise ValueError("Explicit Codex profile exceeds 8 KiB")
                config = tomllib.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
                raise ValueError(f"Cannot load explicit Codex profile {self.name}: {type(exc).__name__}") from None
            if set(config) - {"model", "model_reasoning_effort", "agents"}:
                raise ValueError("Launcher profiles permit only model, model_reasoning_effort and the auxiliary cap")
            if "agents" in config and config["agents"] != {"max_concurrent_threads_per_session": 3}:
                raise ValueError("Launcher profiles may only request auxiliary cap 3")
            self.overrides = validate_settings({k: v for k, v in config.items() if k != "agents"})
        self.overrides.update({k: v for k, v in self.settings.items() if k != "profile"})
        self.catalog = None
        self.observed_model = self.observed_effort = None
        if "model" in self.overrides:
            self.validate_target(self.overrides["model"], self.overrides.get("model_reasoning_effort"))

    def validate_target(self, model, effort):
        if self.catalog is None:
            self.catalog = {}
            cursor, seen = None, set()
            for _ in range(100):
                page = self.client.request("model/list", {"includeHidden": True, "cursor": cursor})
                for item in page.get("data", []):
                    self.catalog.setdefault(item.get("model"), set()).update(
                        value.get("reasoningEffort") for value in item.get("supportedReasoningEfforts", [])
                    )
                cursor = page.get("nextCursor")
                if not cursor:
                    break
                if not isinstance(cursor, str) or cursor in seen:
                    raise ValueError("model/list returned invalid pagination")
                seen.add(cursor)
            else:
                raise ValueError("model/list exceeded the page limit")
        if (not isinstance(model, str) or model not in self.catalog
                or effort is not None and effort not in self.catalog[model]):
            raise ValueError(f"Requested capability unavailable: {model!r} / {effort!r}; no fallback selected")

    def thread_options(self, *, resumed=False):
        options = {"config": {CAP_KEY: 3}}
        # Loaded threads may ignore resume config. Explicit changes are applied
        # at turn/start, after the existing active-thread guards have passed.
        if not resumed:
            if "model" in self.overrides:
                options.update(model=self.overrides["model"], allowProviderModelFallback=False)
            if "model_reasoning_effort" in self.overrides:
                options["config"]["model_reasoning_effort"] = self.overrides["model_reasoning_effort"]
        return options

    def confirm(self, response, *, resumed=False):
        self.observed_model = response.get("model")
        self.observed_effort = response.get("reasoningEffort")
        # Native mode must work without profile files, hardcoded floors or an
        # explicit reasoningEffort. Never manufacture a confirmation for null.
        if not self.overrides:
            return
        model = self.overrides.get("model", self.observed_model)
        effort = self.overrides.get("model_reasoning_effort", self.observed_effort)
        self.validate_target(model, effort)
        if not resumed:
            for key, observed in (("model", self.observed_model),
                                  ("model_reasoning_effort", self.observed_effort)):
                if key in self.overrides and observed != self.overrides[key]:
                    raise ValueError(f"App Server did not apply explicit {key}: requested "
                                     f"{self.overrides[key]!r}, observed {observed!r}. No turn was started.")

    def turn_options(self):
        options = {}
        if "model" in self.overrides:
            options["model"] = self.overrides["model"]
        if "model_reasoning_effort" in self.overrides:
            options["effort"] = self.overrides["model_reasoning_effort"]
        return options

    def log_confirmation(self, log, *, thread_id, cwd, resumed):
        # A resume response describes the OLD selection, not confirmation that
        # the next turn has already applied the issue override.
        log("model_policy_resolved", role=self.role,
            mode="issue-override" if self.settings else "native",
            profile=self.name, source=str(self.source) if self.source else None,
            observedModel=self.observed_model, observedEffort=self.observed_effort,
            turnOverrides=self.turn_options(), requestedMaxAuxiliaryThreads=3,
            threadId=thread_id, cwd=cwd, resumed=resumed)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "prepare-issue":
        raise SystemExit("usage: codex_profile.py prepare-issue <settings-json>")
    prepare_issue(sys.argv[2])
