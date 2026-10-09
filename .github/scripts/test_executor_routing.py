"""Offline tests: real Git/worktrees and a fake Devin executable; no model calls."""

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import devin_runner as runner
import devin_host_broker as host_broker
import local_issue_worktree as worktrees
from executor_control import (ControlError, GitHub, LEASE_MARKER, SESSION_MARKER,
                              devin_settings, executor, read_record, select_executor)

BODY = '```execution\nexecutor = "devin"\n```\n'
REPO = "owner/repo"
SCRIPTS = Path(__file__).resolve().parent
REAL_TMUX = shutil.which("tmux")


def comment(marker, data, cid=1):
    return {"id": cid, "user": {"login": "github-actions[bot]", "type": "Bot"},
            "body": marker + "\n```json\n" + json.dumps(data, sort_keys=True) + "\n```\n"}


class FakeGitHub(GitHub):
    def __init__(self, body=BODY, state="execution-ready"):
        super().__init__(REPO, "fake-actions-secret")
        self.current_issue = {"number": 7, "state": "open", "body": body,
                              "labels": [{"name": state}]}
        self.records, self.calls, self.branch = [], [], None

    def request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if path == "/issues/7":
            return copy.deepcopy(self.current_issue)
        if path == "/issues/8":
            return {**copy.deepcopy(self.current_issue), "number": 8}
        if path.startswith("/issues/7/comments?"):
            page = int(path.rsplit("=", 1)[1])
            return copy.deepcopy(self.records[(page - 1) * 100:page * 100])
        if path == "/git/ref/heads/codex/issue-7":
            return self.branch
        if method == "POST" and path == "/issues/7/comments":
            cid = max([c["id"] for c in self.records] + [0]) + 1
            self.records.append({**comment("", {}, cid), "body": payload["body"]})
            return copy.deepcopy(self.records[-1])
        if method == "PATCH" and path.startswith("/issues/comments/"):
            cid = int(path.rsplit("/", 1)[1])
            target = next(c for c in self.records if c["id"] == cid)
            target["body"] = payload["body"]
            return copy.deepcopy(target)
        raise AssertionError((method, path))

    def own(self):
        self.records.append(comment(LEASE_MARKER, {"executor": "devin"}))
        return self


class SettingsTests(unittest.TestCase):
    def test_codex_remains_default(self):
        for body in (None, "", "Use Devin in prose", '```codex\nmodel = "example"\n```', "```execution\n```"):
            self.assertEqual(executor(body), "codex")

    def test_explicit_executors(self):
        for value in ("codex", "devin"):
            self.assertEqual(executor(f'```execution\nexecutor = "{value}"\n```'), value)

    def test_quoted_and_nested_examples_do_not_select(self):
        for body in ("> " + BODY.replace("\n", "\n> "), "    " + BODY.replace("\n", "\n    "),
                     "````markdown\n" + BODY + "````", "~~~markdown\n" + BODY + "~~~"):
            self.assertEqual(executor(body), "codex")

    def test_duplicate_unterminated_invalid_and_oversized_blocks_fail(self):
        for body in (BODY + BODY, '```execution\nexecutor = "devin"', '```execution\nexecutor =\n```',
                     "x" * 1_048_577, 123):
            with self.subTest(body=str(body)[:40]), self.assertRaises(ControlError):
                executor(body)

    def test_unknown_executor_fields_and_values_fail(self):
        for text in ('executor = "other"', 'executor = false', 'command = "sh"', 'model = "example"'):
            with self.assertRaises(ControlError):
                executor("```execution\n" + text + "\n```")

    def test_devin_native_default_and_separate_model(self):
        self.assertEqual(devin_settings(BODY + '```codex\nmodel = "audit-model"\n```'), {})
        self.assertEqual(devin_settings('```devin\nmodel = "local-model"\n```'), {"model": "local-model"})

    def test_cloud_cost_mode_and_arbitrary_cli_arguments_rejected(self):
        for text in ('devin_mode = "normal"', 'max_acu_limit = 5', 'model_reasoning_effort = "xhigh"',
                     'command = "sh"', 'permission_mode = "dangerous"', 'model = "--cloud"',
                     'model = true', 'model = "a;touch /tmp/x"', 'model = ""'):
            with self.subTest(text=text), self.assertRaises(ControlError):
                devin_settings("```devin\n" + text + "\n```")


class RoutingTests(unittest.TestCase):
    def test_records_default_once(self):
        gh = FakeGitHub(body="")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(len(gh.records), 1)

    def test_selects_devin_and_refuses_hot_switch(self):
        gh = FakeGitHub()
        self.assertEqual(select_executor(gh, 7), "devin")
        gh.current_issue["body"] = ""
        with self.assertRaises(ControlError):
            select_executor(gh, 7)

    def test_delayed_events_do_not_release_holds(self):
        for state in ("queued", "blocked", "review-ready", "completed", "design-required", "investigation-required"):
            gh = FakeGitHub(state=state)
            self.assertEqual(select_executor(gh, 7, wait=True), "")
            self.assertEqual(gh.records, [])

    def test_closed_invalid_and_pr_targets(self):
        gh = FakeGitHub()
        gh.current_issue["state"] = "closed"
        self.assertEqual(select_executor(gh, 7), "")
        for change in ({"labels": []}, {"pull_request": {}}, {"number": 8}):
            gh = FakeGitHub()
            gh.current_issue.update(change)
            with self.assertRaises(ControlError):
                select_executor(gh, 7)

    def test_legacy_codex_and_cloud_are_not_adopted(self):
        for kind in ("branch", "dag", "active", "cloud"):
            gh = FakeGitHub()
            if kind == "branch":
                gh.branch = {"ref": "existing"}
            elif kind == "dag":
                gh.records.append({"body": "<!-- codex-epic-dag:v1 -->"})
            elif kind == "active":
                gh.current_issue["labels"] = [{"name": "in-progress"}]
            else:
                gh.records.append(comment(SESSION_MARKER, {"phase": "ready"}))
            with self.subTest(kind=kind), self.assertRaises(ControlError):
                select_executor(gh, 7, wait=True)

    def test_scheduler_reads_controlling_issue(self):
        gh = FakeGitHub(state="in-progress").own()
        self.assertEqual(select_executor(gh, 7), "")
        self.assertEqual(select_executor(gh, 7, wait=True), "devin")

    def test_all_comment_pages_are_checked(self):
        gh = FakeGitHub()
        gh.records = [{"id": i, "body": "ordinary"} for i in range(100)]
        gh.records.append(comment(LEASE_MARKER, {"executor": "codex"}, 101))
        with self.assertRaises(ControlError):
            select_executor(gh, 7)
        self.assertTrue(any("page=2" in path for _, path, _ in gh.calls))

    def test_duplicate_untrusted_and_malformed_records_fail(self):
        for kind in ("duplicate", "untrusted", "malformed"):
            gh = FakeGitHub().own()
            if kind == "duplicate":
                gh.records.append(copy.deepcopy(gh.records[0]))
            elif kind == "untrusted":
                gh.records[0]["user"]["login"] = "other"
            else:
                gh.records[0]["body"] = LEASE_MARKER + "\ninvalid"
            with self.assertRaises(ControlError):
                select_executor(gh, 7)


class ContextTests(unittest.TestCase):
    def body(self, fields):
        return [{"body": worktrees.CONTEXT_MARKER + "\n```yaml\n" + fields + "\n```"}]

    def test_default_and_canonical_pin(self):
        self.assertIsNone(worktrees.context([], 7))
        fields = 'epic_issue: 8\nintegration_branch: "codex/epic-8" # comment\nbase_sha: ' + "a" * 40
        self.assertEqual(worktrees.context(self.body(fields), 7)["base_sha"], "a" * 40)

    def test_duplicate_missing_alias_and_invalid_fields_fail(self):
        good = 'epic_issue: 8\nintegration_branch: integration\nbase_sha: ' + "a" * 40
        for text in (good + '\nepic_issue: 9', good.replace('epic_issue: 8', 'epic_issue: 7'),
                     good.replace('base_sha:', 'wrong_key:'), good.replace('a' * 40, '*alias'),
                     good.replace('integration_branch: integration', 'integration_branch: [x]')):
            # Invalid branch syntax is rejected by Git before preparation.
            if "[x]" in text:
                continue
            with self.assertRaises(ControlError):
                worktrees.context(self.body(text), 7)
        with self.assertRaises(ControlError):
            worktrees.context(self.body(good) * 2, 7)


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local-devin-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.remote = self.base / "durable", self.base / "remote.git"
        self.root.mkdir()
        self.command("git", "init", "-b", "main", self.root)
        self.command("git", "init", "--bare", self.remote)
        self.command("git", "-C", self.root, "config", "user.email", "test@example.invalid")
        self.command("git", "-C", self.root, "config", "user.name", "Offline test")
        (self.root / "file.txt").write_text("baseline\n")
        self.command("git", "-C", self.root, "add", "file.txt")
        self.command("git", "-C", self.root, "commit", "-m", "baseline")
        self.command("git", "-C", self.root, "remote", "add", "origin", f"https://github.com/{REPO}.git")
        self.command("git", "-C", self.root, "push", str(self.remote), "main")
        original_git = worktrees.git
        def local_git(root, *args, **kwargs):
            if args[:1] == ("fetch",):
                return original_git(root, "fetch", str(self.remote),
                                    "+refs/heads/*:refs/remotes/origin/*", **kwargs)
            return original_git(root, *args, **kwargs)
        self.git_patch = patch.object(worktrees, "git", side_effect=local_git)
        self.git_patch.start()
        self.addCleanup(self.git_patch.stop)

    def command(self, *args):
        value = subprocess.run([str(x) for x in args], text=True, capture_output=True)
        if value.returncode:
            self.fail(value.stderr)
        return value.stdout.strip()

    def prepare(self, activation=None):
        return worktrees.prepare(self.root, self.base / "worktrees", REPO, 7, "main", activation)


class WorktreeTests(GitFixture):
    def test_creates_real_separate_worktree_and_branch(self):
        path = self.prepare()
        self.assertNotEqual(path, self.root)
        self.assertEqual(self.command("git", "-C", path, "branch", "--show-current"), "codex/issue-7")
        self.assertTrue((path / ".git").is_file())

    def test_preserves_dirty_tracked_and_untracked_work(self):
        path = self.prepare()
        (path / "file.txt").write_text("unfinished\n")
        (path / "untracked.txt").write_text("keep\n")
        head = self.command("git", "-C", path, "rev-parse", "HEAD")
        self.assertEqual(self.prepare(), path)
        self.assertEqual((path / "file.txt").read_text(), "unfinished\n")
        self.assertTrue((path / "untracked.txt").exists())
        self.assertEqual(self.command("git", "-C", path, "rev-parse", "HEAD"), head)

    def test_adopts_registered_path_instead_of_creating_another(self):
        alternate = self.base / "other-location"
        self.command("git", "-C", self.root, "worktree", "add", "-b", "codex/issue-7", alternate)
        self.assertEqual(self.prepare(), alternate)

    def test_new_child_starts_at_exact_integration_pin(self):
        self.command("git", "-C", self.root, "checkout", "-b", "integration")
        (self.root / "file.txt").write_text("integration\n")
        self.command("git", "-C", self.root, "commit", "-am", "integration")
        pin = self.command("git", "-C", self.root, "rev-parse", "HEAD")
        self.command("git", "-C", self.root, "push", str(self.remote), "integration")
        self.command("git", "-C", self.root, "checkout", "main")
        path = self.prepare({"integration_branch": "integration", "base_sha": pin, "epic_issue": "8"})
        self.assertEqual(self.command("git", "-C", path, "rev-parse", "HEAD"), pin)

    def test_invalid_pin_or_branch_fails_without_issue_worktree(self):
        for activation in ({"integration_branch": "main", "base_sha": "a" * 40},
                           {"integration_branch": "bad..branch", "base_sha": "a" * 40}):
            with self.assertRaises(ControlError):
                self.prepare(activation)

    def test_unregistered_directory_is_never_deleted(self):
        path = self.base / "worktrees" / "owner-repo" / "issue-7"
        path.mkdir(parents=True)
        (path / "keep").write_text("important")
        with self.assertRaises(ControlError):
            self.prepare()
        self.assertTrue((path / "keep").exists())

    def test_wrong_origin_and_actions_paths_fail(self):
        with self.assertRaises(ControlError):
            worktrees.verify_repo(self.root, "different/repo")
        with self.assertRaises(ControlError):
            worktrees.durable_path(self.base / "_work" / "checkout")

    def test_wrong_worktree_branch_fails(self):
        path = self.prepare()
        with self.assertRaises(ControlError):
            worktrees.verify_worktree(self.root, path, REPO, "codex/issue-8")


class LocalPolicyTests(unittest.TestCase):
    def test_clean_environment_preserves_host_auth_not_actions_secrets(self):
        source = {"GH_TOKEN": "secret", "GITHUB_TOKEN": "secret", "CI": "true", "GITHUB_ACTIONS": "true",
                  "RUNNER_TRACKING_ID": "tracked", "ACTIONS_RUNTIME_TOKEN": "secret", "TMUX": "personal",
                  "CODEX_HOME": "private", "DEVIN_API_KEY": "cloud", "GIT_CONFIG_COUNT": "1",
                  "DEVIN_PERMISSION_MODE": "dangerous", "DEVIN_SANDBOX": "false", "DEVIN_MODEL": "other",
                  "PATH": "/bin", "HOME": "/home/runner", "SSH_AUTH_SOCK": "/ssh-agent"}
        value = runner.clean_env(source)
        self.assertNotIn("secret", str(value))
        self.assertEqual(value["RUNNER_TRACKING_ID"], "")
        self.assertEqual(value["SSH_AUTH_SOCK"], "/ssh-agent")
        self.assertNotIn("TMUX", value)
        self.assertNotIn("CODEX_HOME", value)
        self.assertNotIn("DEVIN_PERMISSION_MODE", value)
        self.assertNotIn("DEVIN_SANDBOX", value)
        self.assertNotIn("DEVIN_MODEL", value)

    def test_cli_resumes_only_explicit_id_with_unattended_sandbox_policy(self):
        job = {"prompt": "/tmp/prompt", "export": "/tmp/export", "config": "/tmp/config",
               "settings": {"model": "exact-id"}, "session_id": "local-session"}
        command = runner.cli_command("/bin/devin", job)
        self.assertIn("--print", command)
        self.assertIn("--resume", command)
        self.assertIn("exact-id", command)
        self.assertEqual(command[command.index("--permission-mode") + 1], "autonomous")
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--respect-workspace-trust") + 1], "false")
        self.assertEqual(command[command.index("--config") + 1], "/tmp/config")
        for forbidden in ("--cloud", "--continue", "dangerous", "smart"):
            self.assertNotIn(forbidden, command)

    def test_sandbox_policy_confines_writes_and_denies_credential_mutation(self):
        config = runner.permission_config()
        self.assertEqual(config["permissions"]["allow"],
                         ["Read(**)", "Write(**)", "Write(/tmp/**)"])
        self.assertNotIn("sandbox", config)
        for rule in ("Exec(sudo)", "Exec(su)", "Exec(gh auth login)",
                     "Exec(gh auth token)", "Exec(gh auth status --show-token)",
                     "Exec(devin auth login)",
                     "Exec(git config)", "Exec(git credential)",
                     "Read(~/.ssh/**)", "Write(~/.config/gh/**)",
                     "Read(~/.local/share/devin/credentials.toml)"):
            self.assertIn(rule, config["permissions"]["deny"])
        self.assertNotIn("Exec(gh auth status)", config["permissions"]["deny"])
        self.assertNotIn("Exec(devin auth status)", config["permissions"]["deny"])

    def test_export_requires_exact_session_and_supported_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.json"
            runner.atomic_json(path, {"session_id": "local-session"})
            runner.atomic_json(path, {"schema_version": "ATIF-v1.7", "session_id": "local-session",
                                      "steps": []})
            self.assertEqual(runner.read_export(path), ("local-session", []))
            with self.assertRaises(ControlError):
                runner.read_export(path, "other")
            runner.atomic_json(path, {"unknown_schema": "x"})
            with self.assertRaises(ControlError):
                runner.read_export(path)

    def test_structured_rejection_and_empty_resume_are_not_progress(self):
        rejected = {"source": "agent", "tool_calls": [
            {"tool_call_id": "one"}, {"tool_call_id": "two"}], "observation": {"results": [
            {"source_call_id": "one", "content": "Tool execution was rejected by the user"},
            {"source_call_id": "two", "content": "Tool call canceled because another tool call was rejected"},
        ]}}
        succeeded = {"source": "agent", "tool_calls": [
            {"tool_call_id": "three"}], "observation": {"results": [
            {"source_call_id": "three", "content":
             "Output from command: historical log says Tool execution was rejected by the user"},
        ]}}
        denied = {"source": "agent", "tool_calls": [
            {"tool_call_id": "four"}], "observation": {"results": [
            {"source_call_id": "four", "content":
             "Permission to run the command `gh auth status --hostname github.com` was denied. "
             "The user needs to approve command execution."},
        ]}}
        host_denied = {"source": "agent", "tool_calls": [
            {"tool_call_id": "five"}], "observation": {"results": [
            {"source_call_id": "five", "content": "Output from command in shell 123:\n"
             "Host command rejected: Unsupported host Git subcommand: -c\n\nExit code: 64"},
        ]}}
        self.assertEqual(runner.turn_evidence([rejected], 0), (False, True, False))
        self.assertEqual(runner.turn_evidence([rejected, succeeded], 1), (True, False, False))
        self.assertEqual(runner.turn_evidence([rejected], 1), (False, False, False))
        self.assertEqual(runner.turn_evidence([succeeded, denied], 0), (True, True, False))
        self.assertEqual(runner.turn_evidence([succeeded, host_denied], 0),
                         (True, False, True))

    def test_only_explicit_new_final_block_is_classified(self):
        final = {"source": "agent", "message": "Blocked during epic initialization by missing auth."}
        self.assertTrue(runner.agent_reported_block([final], 0))
        self.assertFalse(runner.agent_reported_block([final], 1))
        self.assertFalse(runner.agent_reported_block(
            [{"source": "agent", "message": "Work completed. A child remains blocked."}], 0))
        self.assertFalse(runner.agent_reported_block(
            [{"source": "agent", "message": "Blocked", "tool_calls": [{"tool_call_id": "one"}]}], 0))

    def test_prompt_contains_shared_workflow_and_no_cloud_handoff(self):
        text = runner.prompt(REPO, 7, "/trusted/run/devin_host_client.py")
        for required in ("SKILLFORGE_LOCAL_RUNNER=1", "codex-epic-scheduler", "spec-driven-codex-loop",
                         "codex-execution-context:v1", "review-ready", "Never merge",
                         "max_parallel_workers", "Use the shell exec tool for file edits",
                         "retry safe commands rejected by an earlier permission policy",
                         "No shell exec", "Do not call git or gh directly",
                         "/trusted/run/devin_host_client.py git ARGS"):
            self.assertIn(required, text)


class HostCommandTests(GitFixture):
    def setUp(self):
        super().setUp()
        self.worktree = self.prepare()
        self.job = {"root": str(self.root), "worktree": str(self.worktree),
                    "repo": REPO, "branch": "codex/issue-7", "number": "7"}

    def test_global_git_options_config_alias_and_wrong_refs_are_rejected(self):
        for args in (["-C", str(self.worktree), "config", "--global", "user.name", "x"],
                     ["-c", "alias.audit=!id", "audit"],
                     ["config", "--file", "/tmp/config", "x", "y"],
                     ["push", "origin", "HEAD"],
                     ["push", "origin", "HEAD:refs/heads/main"],
                     ["fetch", "origin", "refs/heads/main:refs/heads/main"],
                     ["add", "--", "../../outside"]):
            with self.subTest(args=args), self.assertRaises(host_broker.Rejected):
                host_broker.git_args(args, self.job, self.worktree)
        self.assertEqual(host_broker.git_args(
            ["push", "origin", "HEAD:refs/heads/codex/issue-7"], self.job,
            self.worktree)[3:], ["push", "origin", "HEAD:refs/heads/codex/issue-7"])

    def test_gh_rejects_auth_mutation_other_repo_and_host_file_upload(self):
        for args in (["auth", "token"], ["auth", "logout"],
                     ["api", "repos/other/repo/issues"],
                     ["api", "repos/owner/repo/issues", "-X", "POST"],
                     ["issue", "view", "7", "--repo", "other/repo"],
                     ["issue", "comment", "7", "--body-file", "/etc/passwd"]):
            with self.subTest(args=args), self.assertRaises(host_broker.Rejected):
                host_broker.gh_args(args, self.job, self.worktree)
        self.assertEqual(host_broker.gh_args(["issue", "view", "7", "--repo", REPO],
                                               self.job, self.worktree)[1:],
                         ["issue", "view", "7", "--repo", REPO])
        body = self.worktree / "pr-body.md"
        body.write_text("Updated PR description\n")
        patch_args = host_broker.gh_args(["pr", "patch", "12", "--body-file", str(body)],
                                          self.job, self.worktree)
        self.assertEqual(patch_args[:5], ["/usr/bin/gh", "api", "-X", "PATCH",
                                          f"repos/{REPO}/pulls/12"])
        self.assertIn("body=Updated PR description\n", patch_args)

    def test_broker_supports_bounded_read_only_issue_workflow_queries(self):
        for args in (["ls-files", "-m", "-o", "--exclude-standard"],
                     ["diff", "--name-only", "HEAD"],
                     ["rev-parse", "origin/codex/epic-issue-7"]):
            self.assertEqual(host_broker.git_args(args, self.job, self.worktree)[3:], args)
        for args in (["ls-files", "--", "../../outside"],
                     ["ls-files", "--eol", "/etc/passwd"]):
            with self.subTest(args=args), self.assertRaises(host_broker.Rejected):
                host_broker.git_args(args, self.job, self.worktree)
        pr_list = ["pr", "list", "--repo", REPO, "--head", "codex/issue-7",
                   "--state", "all", "--json", "number,title,state"]
        self.assertEqual(host_broker.gh_args(pr_list, self.job, self.worktree)[1:], pr_list)
        for args in (["pr", "list", "--repo", REPO],
                     ["pr", "list", "--repo", REPO, "--head", "main"],
                     ["pr", "list", "--repo", REPO, "--head", "codex/issue-7",
                      "--state", "unknown"]):
            with self.subTest(args=args), self.assertRaises(host_broker.Rejected):
                host_broker.gh_args(args, self.job, self.worktree)

    def test_socket_broker_executes_only_in_registered_worktree(self):
        turn = self.base / "turn"
        turn.mkdir()
        command = turn / "devin_host_client.py"
        shutil.copyfile(SCRIPTS / "devin_host_client.py", command)
        command.chmod(0o700)
        self.job["host_socket"] = str(turn / "host.sock")
        (turn / "job.json").write_text(json.dumps(self.job))
        broker = host_broker.HostBroker(self.job)
        broker.start()
        self.addCleanup(broker.close)
        accepted = subprocess.run([str(command), "git", "status", "--short"],
                                  cwd=self.worktree, capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        (self.worktree / "file.txt").write_text("committed through host command\n")
        for args in (("git", "add", "--", "file.txt"),
                     ("git", "commit", "-m", "host transport commit")):
            executed = subprocess.run([str(command), *args], cwd=self.worktree,
                                      capture_output=True, text=True)
            self.assertEqual(executed.returncode, 0, executed.stderr)
        self.assertEqual(self.command("git", "-C", self.worktree, "log", "-1", "--format=%s"),
                         "host transport commit")
        denied = subprocess.run([str(command), "git", "-c", "alias.audit=!id", "audit"],
                                cwd=self.worktree, capture_output=True, text=True)
        self.assertEqual(denied.returncode, 64)
        outside = subprocess.run([str(command), "git", "status"], cwd=self.root,
                                 capture_output=True, text=True)
        self.assertEqual(outside.returncode, 64)


class LaunchFixture(GitFixture):
    def setUp(self):
        super().setUp()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        cli = self.bin / "devin"
        cli.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys, time
args = sys.argv[1:]
if "--help" in args:
    print("--print --prompt-file --export --resume --config --sandbox --permission-mode --respect-workspace-trust")
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    sys.exit(0)
if os.environ.get("FAKE_TRUST_REJECTED"):
    print("Error: Refusing to run in an untrusted workspace: test", flush=True)
    sys.exit(1)
path = pathlib.Path(args[args.index("--export") + 1])
sid = args[args.index("--resume") + 1] if "--resume" in args else "local-test-session"
path.with_suffix(".observed.json").write_text(json.dumps({"env": dict(os.environ), "args": args, "cwd": os.getcwd()}))
time.sleep(float(os.environ.get("FAKE_DEVIN_DELAY", "0")))
if os.environ.get("FAKE_PERMISSION_REJECTED"):
    print("warning: rejected a tool call that requires confirmation. Running in non-interactive mode.", flush=True)
print("fake CLI completed", flush=True)
if not os.environ.get("FAKE_MISSING_EXPORT"):
    prior = []
    if "--resume" in args:
        exports = sorted(p for p in path.parent.parent.glob("run-*/conversation.json") if p != path)
        if exports:
            prior = json.loads(exports[-1].read_text())["steps"]
    steps = list(prior)
    if not os.environ.get("FAKE_NO_TOOL_PROGRESS"):
        rejection = os.environ.get("FAKE_STRUCTURED_REJECTION")
        if rejection == "approval":
            result = "Permission to run the command `gh auth status --hostname github.com` was denied. The user needs to approve command execution."
        elif rejection == "host":
            result = "Output from command in shell 123:\\nHost command rejected: Unsupported host Git subcommand: -c\\n\\nExit code: 64"
        elif rejection:
            result = "Tool execution was rejected by the user"
        else:
            result = "command completed"
        steps.append({"source": "agent", "tool_calls": [{"function_name": "exec", "tool_call_id": "call-1"}],
                      "observation": {"results": [{"source_call_id": "call-1", "content": result}]}})
    if os.environ.get("FAKE_AGENT_BLOCKED"):
        steps.append({"source": "agent", "message": "Blocked during epic initialization by missing auth."})
    schema = "bad-schema" if os.environ.get("FAKE_BAD_EXPORT") else "ATIF-v1.7"
    path.write_text(json.dumps({"schema_version": schema, "session_id": sid, "steps": steps}))
sys.exit(int(os.environ.get("FAKE_DEVIN_EXIT", "0")))
''')
        cli.chmod(0o755)
        for name in ("gh", "tmux", "bwrap", "socat"):
            stub = self.bin / name
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
        self.env_patch = patch.dict(os.environ, {"PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                                                 "GITHUB_TOKEN": "ephemeral-test-secret",
                                                 "RUNNER_TRACKING_ID": "tracked-test", "FAKE_DEVIN_DELAY": "0"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.gh = FakeGitHub().own()
        self.state_dir = self.base / "state" / "issue-7" / "devin"
        self.log_dir = self.base / "logs"
        self.processes, self.tmux_calls = {}, []
        original_run = runner.run
        def fake_tmux(args, **kwargs):
            if args[0] != "tmux":
                return original_run(args, **kwargs)
            self.tmux_calls.append((list(args), dict(kwargs.get("env", {}))))
            socket, operation = args[2], args[5]
            process = self.processes.get(socket)
            if operation == "list-panes":
                status = 1 if process is None else 0
                text = "0\n" if process is not None and process.poll() is None else "1\n"
                return subprocess.CompletedProcess(args, status, text, "")
            if operation == "kill-session":
                if process and process.poll() is None:
                    raise AssertionError("Tried to kill an active tmux session")
                self.processes.pop(socket, None)
                return subprocess.CompletedProcess(args, 0, "", "")
            if operation == "new-session":
                receipt = runner.read_json(self.state_dir / "state.json")
                self.assertEqual(receipt["phase"], "pending")
                self.assertIsNotNone(read_record(self.gh.records, runner.LOCAL_MARKER)[1])
                command = args[args.index("-c") + 2]
                process = subprocess.Popen(shlex.split(command), env=kwargs["env"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True)
                self.processes[socket] = process
                return subprocess.CompletedProcess(args, 0, "", "")
            raise AssertionError(args)
        self.transport_patch = patch.object(runner, "run", side_effect=fake_tmux)
        self.transport_patch.start()
        self.addCleanup(self.transport_patch.stop)
        self.addCleanup(self.finish_processes)

    def finish_processes(self):
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    def launch(self, run_id="101", **kwargs):
        return runner.launch(self.gh, 7, self.root, self.base / "worktrees", self.state_dir,
                             self.log_dir, "main", run_id, **kwargs)

    def receipt(self):
        return runner.read_json(self.state_dir / "state.json")

    def finished(self):
        self.finish_processes()
        return self.receipt()


class LifecycleTests(LaunchFixture):
    def test_creates_real_worktree_and_records_cli_completion(self):
        result = self.launch()
        record = self.finished()
        self.assertEqual(result["result"], "launched-local-cli")
        self.assertEqual(record["phase"], "finished")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["session_id"], "local-test-session")
        self.assertIn("fake CLI completed", Path(result["log"]).read_text())
        observed = runner.read_json(self.state_dir / "run-101" / "conversation.observed.json")
        self.assertEqual(observed["cwd"], record["worktree"])
        self.assertEqual(observed["env"]["SKILLFORGE_LOCAL_RUNNER"], "1")
        self.assertNotIn("ephemeral-test-secret", json.dumps(observed))
        self.assertEqual(observed["env"]["RUNNER_TRACKING_ID"], "")
        job = runner.read_json(self.state_dir / "run-101" / "job.json")
        config = runner.read_json(self.state_dir / "run-101" / "devin-config.json")
        self.assertNotIn("sandbox", config)
        self.assertTrue(Path(job["host_client"]).is_file())
        self.assertNotIn(str(self.base / "worktrees"), job["host_client"])
        self.assertFalse(Path(job["host_socket"]).exists())

    def test_replay_does_not_start_second_cli(self):
        self.launch()
        self.finished()
        self.assertEqual(self.launch()["result"], "already-acknowledged")
        self.assertEqual(sum(args[5] == "new-session" for args, _ in self.tmux_calls), 1)

    def test_new_turn_resumes_exact_session_and_preserves_unfinished_file(self):
        self.launch()
        record = self.finished()
        unfinished = Path(record["worktree"]) / "unfinished.txt"
        unfinished.write_text("keep")
        self.launch("102")
        self.finished()
        observed = runner.read_json(self.state_dir / "run-102" / "conversation.observed.json")
        self.assertIn("--resume", observed["args"])
        self.assertIn("local-test-session", observed["args"])
        self.assertTrue(unfinished.exists())

    def test_tampered_prior_export_refuses_resume_without_new_launch_bundle(self):
        self.launch()
        self.finished()
        export = self.state_dir / "run-101" / "conversation.json"
        value = runner.read_json(export)
        value["session_id"] = "another-session"
        runner.atomic_json(export, value)
        with self.assertRaisesRegex(ControlError, "different session"):
            self.launch("102")
        self.assertFalse((self.state_dir / "run-102").exists())

    def test_active_turn_is_not_steered(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        self.assertEqual(self.launch("102")["result"], "already-active")

    def test_scheduler_wait_is_bounded(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        with self.assertRaisesRegex(ControlError, "bounded"):
            self.launch("102", wait=True, wait_seconds=0)

    def test_hold_during_wait_prevents_followup(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        def hold(_):
            self.gh.current_issue["labels"] = [{"name": "blocked"}]
        self.assertEqual(self.launch("102", wait=True, sleep=hold)["result"], "skipped-current-state")

    def test_scheduler_resumes_after_previous_turn_exits(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.4"}):
            self.launch()
        def finish(_):
            self.finish_processes()
        self.assertEqual(self.launch("102", wait=True, sleep=finish)["result"], "launched-local-cli")
        self.assertEqual(self.finished()["session_id"], "local-test-session")

    def test_cloud_receipt_is_not_migrated(self):
        self.gh.records.append(comment(SESSION_MARKER, {"phase": "ready"}, 2))
        with self.assertRaisesRegex(ControlError, "cloud"):
            self.launch()
        self.assertEqual(self.processes, {})

    def test_wrong_host_refuses_launch(self):
        self.gh.records.append(comment(runner.LOCAL_MARKER, {"transport": "cli-tmux", "host": "other"}, 2))
        with self.assertRaisesRegex(ControlError, "another local host"):
            self.launch()

    def test_missing_local_record_is_not_a_new_session(self):
        self.launch()
        self.finished()
        (self.state_dir / "state.json").unlink()
        with self.assertRaisesRegex(ControlError, "missing"):
            self.launch("102")

    def test_codex_thread_state_blocks_local_devin(self):
        self.state_dir.parent.mkdir(parents=True)
        (self.state_dir.parent / "app-server-thread-id").write_text("codex-thread")
        with self.assertRaisesRegex(ControlError, "Codex session state"):
            self.launch()

    def test_unknown_or_incomplete_receipt_never_relaunches(self):
        self.launch()
        record = self.finished()
        record["phase"] = "pending"
        runner.atomic_json(self.state_dir / "state.json", record)
        with self.assertRaisesRegex(ControlError, "Incomplete"):
            self.launch("102")

    def test_cli_error_is_recorded_not_task_success(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_EXIT": "3", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["exit_code"], 3)
        with self.assertRaises(ControlError):
            self.launch()

    def test_permission_rejection_is_failure_even_when_cli_exits_zero(self):
        with patch.dict(os.environ, {"FAKE_PERMISSION_REJECTED": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["error"], "permission-rejection")

    def test_structured_permission_rejection_is_failure_without_log_warning(self):
        for diagnostic in ("1", "approval"):
            with self.subTest(diagnostic=diagnostic):
                with patch.dict(os.environ, {"FAKE_STRUCTURED_REJECTION": diagnostic,
                                          "FAKE_DEVIN_DELAY": "0.3"}):
                    self.launch(run_id="101" if diagnostic == "1" else "102")
                record = self.finished()
                self.assertEqual(record["phase"], "failed")
                self.assertEqual(record["exit_code"], 0)
                self.assertEqual(record["error"], "permission-rejection")

    def test_host_broker_rejection_is_failure_even_with_cli_exit_zero(self):
        with patch.dict(os.environ, {"FAKE_STRUCTURED_REJECTION": "host",
                                  "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["error"], "host-command-rejection")
        self.assertEqual(record["exit_code"], 0)

    def test_resume_with_no_new_tool_result_is_not_success(self):
        self.launch()
        self.finished()
        with patch.dict(os.environ, {"FAKE_NO_TOOL_PROGRESS": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch("102")
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["error"], "no-tool-progress")
        self.assertEqual(record["session_id"], "local-test-session")

    def test_explicit_block_after_successful_tool_is_not_finished(self):
        with patch.dict(os.environ, {"FAKE_AGENT_BLOCKED": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["error"], "agent-blocked")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["session_id"], "local-test-session")

    def test_missing_export_requires_reconciliation(self):
        with patch.dict(os.environ, {"FAKE_MISSING_EXPORT": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        self.assertEqual(self.finished()["phase"], "needs-reconciliation")
        with self.assertRaises(ControlError):
            self.launch("102")

    def test_malformed_export_requires_reconciliation(self):
        with patch.dict(os.environ, {"FAKE_BAD_EXPORT": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "needs-reconciliation")
        self.assertEqual(record["error"], "invalid-export")
        self.assertEqual(record["session_id"], "local-test-session")

    def test_workspace_trust_failure_is_distinct_from_cli_failure(self):
        transport_run = runner.run

        def wait_for_rejection(args, **kwargs):
            result = transport_run(args, **kwargs)
            if args[0] == "tmux" and args[5] == "new-session":
                # Exercise rejection before acknowledgement deterministically.
                # Observing "running" first is a valid asynchronous launch, not
                # evidence that the supervisor ignored the later trust failure.
                self.finish_processes()
            return result

        with patch.dict(os.environ, {"FAKE_TRUST_REJECTED": "1"}), \
                patch.object(runner, "run", side_effect=wait_for_rejection):
            with self.assertRaisesRegex(ControlError, "failed to start/finish"):
                self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "needs-reconciliation")
        self.assertEqual(record["error"], "workspace-trust-rejection")

    def test_tmux_receives_clean_environment_and_isolated_socket(self):
        self.launch()
        self.finished()
        args, env = next((a, e) for a, e in self.tmux_calls if a[5] == "new-session")
        self.assertTrue(args[2].startswith("sf-devin-"))
        self.assertIn("/dev/null", args)
        self.assertIn("remain-on-exit", args)
        self.assertEqual(env["RUNNER_TRACKING_ID"], "")
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("TMUX", env)


class RealTmuxTests(LaunchFixture):
    def test_real_tmux_detachment_and_exact_resume_with_fake_cli(self):
        if not REAL_TMUX:
            if os.environ.get("REQUIRE_TMUX_TEST") == "1":
                self.fail("tmux is required for the transport integration gate")
            self.skipTest("tmux is not installed; CI requires this real transport test")
        self.transport_patch.stop()
        # The fake executable remains Devin only; tmux is the actual OS binary.
        (self.bin / "tmux").unlink()
        (self.bin / "tmux").symlink_to(REAL_TMUX)
        result = self.launch()
        self.addCleanup(lambda: subprocess.run(runner.tmux_args(self.receipt()["socket"], "kill-server"),
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        def wait_final():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.receipt()["phase"] in runner.TERMINAL:
                    # Let supervisor leave the pane and release its lifetime lock.
                    time.sleep(0.2)
                    return
                time.sleep(0.05)
            self.fail("Local tmux CLI did not finish")
        wait_final()
        self.assertEqual(self.receipt()["phase"], "finished")
        panes = subprocess.run(runner.tmux_args(result["socket"], "list-panes", "-t", "=" + result["session"] + ":0",
                                                "-F", "#{pane_dead}"), text=True, capture_output=True)
        self.assertEqual(panes.stdout.strip(), "1")
        self.launch("102")
        wait_final()
        self.assertEqual(self.receipt()["session_id"], "local-test-session")
        observed = runner.read_json(self.state_dir / "run-102" / "conversation.observed.json")
        self.assertIn("--resume", observed["args"])
        self.assertNotIn("ephemeral-test-secret", json.dumps(observed))


class WorkflowTests(unittest.TestCase):
    def test_local_launcher_has_no_cloud_or_ephemeral_checkout(self):
        source = (SCRIPTS.parent / "workflows/devin-execute-ready.yml").read_text()
        self.assertIn("runs-on: [self-hosted, codex]", source)
        self.assertIn("SKILLFORGE_REPO_ROOT", source)
        self.assertIn('"$GITHUB_SHA:.github/scripts/$name"', source)
        self.assertIn("devin_host_broker.py", source)
        self.assertIn("devin_host_client.py", source)
        for forbidden in ("DEVIN_API_KEY", "DEVIN_ORG_ID", "secrets:", "actions/checkout"):
            self.assertNotIn(forbidden, source)

    def test_single_dispatcher_and_fixed_independent_audit(self):
        sources = {p.name: p.read_text() for p in (SCRIPTS.parent / "workflows").glob("*.yml")}
        if "codex-issue-state.yml" not in sources:
            self.skipTest("Full repository workflow set is checked in CI")
        self.assertEqual(
            [name for name, text in sources.items() if "types: [labeled" in text],
            ["codex-issue-state.yml"],
        )
        self.assertIn("types: [labeled, closed]", sources["codex-issue-state.yml"])
        dispatcher = sources["codex-issue-state.yml"]
        self.assertIn("needs.select-executor.outputs.executor == 'devin'", dispatcher)
        self.assertIn("uses: ./.github/workflows/codex-review-ready.yml", dispatcher)
        self.assertNotIn("DEVIN_API_KEY", dispatcher)


if __name__ == "__main__":
    unittest.main()