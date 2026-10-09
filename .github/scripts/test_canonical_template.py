"""Canonical-template safety regression; no host/model calls."""
from pathlib import Path
import unittest

class CanonicalTemplateTests(unittest.TestCase):
    def test_canonical_template_does_not_schedule_local_cleanup(self):
        source = (Path(__file__).resolve().parents[1] / "workflows/codex-issue-state.yml").read_text()
        cleanup = source.split("  cleanup-worktrees:\n", 1)[1].split("  select-executor:\n", 1)[0]
        self.assertIn("if: github.repository != 'murillo128/skillforge' && needs.route.outputs.cleanup_issue_number != ''", cleanup)
        self.assertIn("runs-on: [self-hosted, codex]", cleanup)

if __name__ == "__main__":
    unittest.main()
