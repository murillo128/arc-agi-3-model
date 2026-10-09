import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("cleanup_issue_worktrees.py")
SPEC = importlib.util.spec_from_file_location("cleanup_issue_worktrees", MODULE_PATH)
cleanup_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup_module)


def run(*args, cwd=None):
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


class WorktreeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.worktree_root = self.root / "worktrees"
        self.run_root = self.root / "run"
        self.repository = "example/project"
        self.repo_key = "example-project"

        run("git", "init", "-b", "main", self.repo)
        run("git", "-C", self.repo, "config", "user.email", "test@example.com")
        run("git", "-C", self.repo, "config", "user.name", "Test")
        run("git", "-C", self.repo, "remote", "add", "origin", "https://github.com/example/project.git")
        (self.repo / "tracked.txt").write_text("base\n", encoding="utf-8")
        run("git", "-C", self.repo, "add", "tracked.txt")
        run("git", "-C", self.repo, "commit", "-m", "base")

    def tearDown(self):
        self.temp.cleanup()

    def add_issue(self, number):
        path = self.worktree_root / self.repo_key / f"issue-{number}"
        path.parent.mkdir(parents=True, exist_ok=True)
        run("git", "-C", self.repo, "worktree", "add", "-b", f"codex/issue-{number}", path, "main")
        return path

    def add_review(self, pr, attempt="run-1"):
        path = self.worktree_root / self.repo_key / "reviews" / f"pr-{pr}-{attempt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        head = run("git", "-C", self.repo, "rev-parse", "main").stdout.strip()
        run("git", "-C", self.repo, "worktree", "add", "--detach", path, head)
        return path

    def test_removes_only_closed_issue_and_its_reviews(self):
        issue = self.add_issue(12)
        review = self.add_review(34)
        other_issue = self.add_issue(13)
        other_review = self.add_review(35)

        removed, leftovers = cleanup_module.cleanup(
            self.repo,
            self.worktree_root,
            self.run_root,
            self.repository,
            12,
            [34],
            wait_seconds=0,
        )

        self.assertEqual(set(removed), {issue.resolve(), review.resolve()})
        self.assertEqual(leftovers, [])
        self.assertFalse(issue.exists())
        self.assertFalse(review.exists())
        self.assertTrue(other_issue.exists())
        self.assertTrue(other_review.exists())
        self.assertEqual(
            run("git", "-C", self.repo, "show-ref", "--verify", "refs/heads/codex/issue-12").returncode,
            0,
        )

    def test_active_review_lock_blocks_cleanup(self):
        issue = self.add_issue(12)
        review = self.add_review(34)
        lock_path = self.run_root / self.repo_key / "pr-review-34" / "active.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(cleanup_module.CleanupError):
                cleanup_module.cleanup(
                    self.repo,
                    self.worktree_root,
                    self.run_root,
                    self.repository,
                    12,
                    [34],
                    wait_seconds=0,
                )
            self.assertTrue(issue.exists())
            self.assertTrue(review.exists())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_uncommitted_changes_block_cleanup(self):
        for number, kind in enumerate(("modified", "staged", "untracked"), start=20):
            with self.subTest(kind=kind):
                issue = self.add_issue(number)
                path = issue / ("untracked.txt" if kind == "untracked" else "tracked.txt")
                path.write_text("unfinished work\n", encoding="utf-8")
                if kind == "staged":
                    run("git", "-C", issue, "add", "tracked.txt")
                with self.assertRaises(cleanup_module.CleanupError):
                    cleanup_module.cleanup(
                        self.repo, self.worktree_root, self.run_root,
                        self.repository, number, [], wait_seconds=0,
                    )
                self.assertEqual(path.read_text(encoding="utf-8"), "unfinished work\n")
                self.assertIn(
                    str(issue.resolve()),
                    run("git", "-C", self.repo, "worktree", "list", "--porcelain").stdout,
                )

    def test_uncommitted_review_changes_block_cleanup(self):
        review = self.add_review(34)
        path = review / "review-notes.txt"
        path.write_text("unpublished evidence\n", encoding="utf-8")
        with self.assertRaises(cleanup_module.CleanupError):
            cleanup_module.cleanup(
                self.repo, self.worktree_root, self.run_root,
                self.repository, 12, [34], wait_seconds=0,
            )
        self.assertEqual(path.read_text(encoding="utf-8"), "unpublished evidence\n")


if __name__ == "__main__":
    unittest.main()
