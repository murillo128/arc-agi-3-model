"""Offline regressions for canonical epic discovery and deterministic waves."""

import copy
import importlib.util
from pathlib import Path
import unittest

import find_epic_parent as discovery

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "epic_wave_planner", ROOT / "skills/codex-epic-scheduler/scripts/plan_wave.py"
)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def child(number, state="queued", dependencies=(), mutex=()):
    return {"issue": number, "state": state,
            "depends_on": list(dependencies), "mutex": list(mutex)}


def snapshot(*children, limit=2):
    return {"execution_mode": "epic-dag", "parent_state": "in-progress",
            "max_parallel_workers": limit, "children": list(children)}


def parent(number=1, labels=None, **extra):
    return {"number": number, "title": "Epic", "state": "open",
            "labels": [{"name": value} for value in
                       (labels if labels is not None else ["in-progress"])], **extra}


def dag(*numbers):
    lines = ["execution_mode: epic-dag", "children:"]
    for number in numbers:
        lines += [f"  - issue: {number}", "    depends_on: []", "    mutex: []"]
    return {"body": discovery.DAG_MARKER + "\n```yaml\n" + "\n".join(lines) + "\n```"}


class WaveTests(unittest.TestCase):
    def test_wave_order_does_not_depend_on_input_order(self):
        result = planner.plan_wave(snapshot(child(30), child(10), child(20)))
        self.assertEqual(result["candidates"], [10, 20, 30])
        self.assertEqual(result["selected"], [10, 20])

    def test_active_states_consume_slots(self):
        result = planner.plan_wave(snapshot(child(1, "execution-ready"),
            child(2, "in-progress"), child(3), limit=3))
        self.assertEqual(result["active"], [1, 2])
        self.assertEqual(result["available_slots"], 1)
        self.assertEqual(result["selected"], [3])

    def test_over_capacity_never_activates_another_child(self):
        result = planner.plan_wave(snapshot(child(1, "in-progress"),
            child(2, "execution-ready"), child(3), limit=1))
        self.assertEqual(result["available_slots"], 0)
        self.assertEqual(result["selected"], [])

    def test_only_completed_satisfies_a_dependency(self):
        for state in sorted(planner.WORKFLOW_STATES):
            with self.subTest(state=state):
                result = planner.plan_wave(snapshot(child(1, state),
                    child(2, dependencies=[1]), limit=3))
                self.assertEqual(2 in result["selected"], state == "completed")

    def test_review_ready_releases_slot_without_completing_dependency(self):
        result = planner.plan_wave(snapshot(child(1, "review-ready"),
            child(2, dependencies=[1]), child(3), limit=1))
        self.assertEqual(result["active"], [])
        self.assertEqual(result["selected"], [3])

    def test_holds_are_never_candidates(self):
        result = planner.plan_wave(snapshot(child(1, "blocked"),
            child(2, "design-required"), child(3, "investigation-required"), child(4)))
        self.assertEqual(result["selected"], [4])

    def test_mutexes_cover_active_and_newly_selected_work(self):
        result = planner.plan_wave(snapshot(child(1, "in-progress", mutex=["schema"]),
            child(2, mutex=["schema"]), child(3, mutex=["runtime"]),
            child(4, mutex=["runtime"]), child(5), limit=4))
        self.assertEqual(result["selected"], [3, 5])

    def test_greedy_order_is_not_replaced_by_maximum_cardinality(self):
        result = planner.plan_wave(snapshot(child(1, mutex=["a", "b"]),
            child(2, mutex=["a"]), child(3, mutex=["b"])))
        self.assertEqual(result["selected"], [1])

    def test_repeat_snapshot_preserves_original_and_does_not_duplicate_wave(self):
        original = snapshot(child(1), child(2))
        before = copy.deepcopy(original)
        updated = planner.apply_wave(original, planner.plan_wave(original)["selected"])
        self.assertEqual(original, before)
        self.assertEqual(planner.plan_wave(updated)["selected"], [])
        self.assertEqual(planner.apply_wave(updated, [1, 2]), updated)

    def test_completed_epic_has_no_wave(self):
        result = planner.plan_wave(snapshot(child(1, "completed"), child(2, "completed")))
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["active"], [])

    def test_invalid_worker_limits_fail_closed(self):
        for value in (0, -1, True, "2", 1.5, None):
            with self.subTest(value=value), self.assertRaises(planner.ContractError):
                planner.plan_wave(snapshot(child(1), limit=value))

    def test_invalid_identity_dependency_and_mutex_fail_closed(self):
        examples = [snapshot(), snapshot(child(1), child(1)), snapshot(child(True)),
            snapshot(child(1, dependencies=[1])), snapshot(child(1, dependencies=[99])),
            snapshot(child(1), child(2, dependencies=[1, 1])),
            snapshot(child(1, mutex=[""])), snapshot(child(1, mutex=["x", "x"])),
            snapshot(child(1, "unknown"))]
        for value in examples:
            with self.subTest(value=value), self.assertRaises(planner.ContractError):
                planner.plan_wave(value)

    def test_dependency_cycles_fail_closed(self):
        with self.assertRaises(planner.ContractError):
            planner.plan_wave(snapshot(child(1, dependencies=[3]),
                child(2, dependencies=[1]), child(3, dependencies=[2])))

    def test_normalization_sorts_dependencies_and_mutexes(self):
        _, records = planner.validate_snapshot(snapshot(child(1), child(2),
            child(3, dependencies=[2, 1], mutex=["z", "a"])))
        self.assertEqual(records[3]["depends_on"], [1, 2])
        self.assertEqual(records[3]["mutex"], ["a", "z"])


class DiscoveryTests(unittest.TestCase):
    def test_only_canonical_comment_selects_parent(self):
        issue = parent(body="execution_mode: epic-dag\nchild_issues: [10]")
        self.assertIsNone(discovery.find_parent(10, [issue], {}))
        self.assertEqual(discovery.find_parent(10, [issue], {1: [dag(10)]}),
                         {"number": 1, "title": "Epic"})

    def test_unrelated_child_does_not_wake_parent(self):
        self.assertIsNone(discovery.find_parent(11, [parent()], {1: [dag(10)]}))

    def test_inactive_closed_pr_and_ambiguous_states_do_not_wake(self):
        cases = [parent(labels=["queued"]), parent(state="closed"),
            parent(pull_request={"url": "https://example.invalid/pr"}),
            parent(labels=["in-progress", "blocked"])]
        for issue in cases:
            with self.subTest(issue=issue):
                self.assertIsNone(discovery.find_parent(10, [issue], {1: [dag(10)]}))

    def test_multiple_active_parents_fail_closed(self):
        with self.assertRaises(discovery.DiscoveryError):
            discovery.find_parent(10, [parent(1), parent(2)],
                                  {1: [dag(10)], 2: [dag(10)]})

    def test_duplicate_canonical_comments_and_children_fail_closed(self):
        for comments in ([dag(10), dag(10)], [dag(10, 10)],
                         [{"body": discovery.DAG_MARKER + "\n" + dag(10)["body"]}]):
            with self.subTest(comments=comments), self.assertRaises(discovery.DiscoveryError):
                discovery.extract_canonical_epic_children(comments)

    def test_malformed_canonical_dag_fails_closed(self):
        with self.assertRaises(discovery.DiscoveryError):
            discovery.extract_canonical_epic_children([{"body": discovery.DAG_MARKER}])

    def test_paginated_comments_keep_canonical_record_on_later_page(self):
        comments = discovery._flatten_pages([[{"body": "ordinary comment"}], [dag(10)]])
        self.assertEqual(discovery.extract_canonical_epic_children(comments), {10})


if __name__ == "__main__":
    unittest.main()
