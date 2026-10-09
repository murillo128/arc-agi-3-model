#!/usr/bin/env python3
"""Pure validation and wave planning for the Codex epic scheduler."""

import argparse
import copy
import json
from pathlib import Path


WORKFLOW_STATES = {
    "queued",
    "execution-ready",
    "in-progress",
    "review-ready",
    "design-required",
    "investigation-required",
    "blocked",
    "completed",
}
ACTIVE_STATES = {"execution-ready", "in-progress"}


class ContractError(ValueError):
    """Raised when a normalized epic snapshot is unsafe to schedule."""


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("{} must be a positive integer".format(field))
    return value


def _unique_list(value, field):
    if not isinstance(value, list):
        raise ContractError("{} must be a list".format(field))
    try:
        unique_values = set(value)
    except TypeError as exc:
        raise ContractError("{} entries must be scalar values".format(field)) from exc
    if len(value) != len(unique_values):
        raise ContractError("{} contains duplicate values".format(field))
    return value


def validate_snapshot(snapshot):
    """Validate and normalize one observed canonical epic-DAG snapshot."""
    if not isinstance(snapshot, dict):
        raise ContractError("snapshot must be an object")
    if snapshot.get("execution_mode") != "epic-dag":
        raise ContractError("execution_mode must be epic-dag")
    if snapshot.get("parent_state") not in {"execution-ready", "in-progress"}:
        raise ContractError("parent_state must be execution-ready or in-progress")

    worker_limit = _positive_integer(snapshot.get("max_parallel_workers"), "max_parallel_workers")
    children = snapshot.get("children")
    if not isinstance(children, list) or not children:
        raise ContractError("children must be a non-empty list")

    records = {}
    for index, child in enumerate(children):
        field = "children[{}]".format(index)
        if not isinstance(child, dict):
            raise ContractError("{} must be an object".format(field))
        issue = _positive_integer(child.get("issue"), "{}.issue".format(field))
        if issue in records:
            raise ContractError("child issue #{} is declared more than once".format(issue))

        dependencies = _unique_list(child.get("depends_on"), "{}.depends_on".format(field))
        for dependency in dependencies:
            _positive_integer(dependency, "{}.depends_on entry".format(field))
        if issue in dependencies:
            raise ContractError("child issue #{} depends on itself".format(issue))

        mutexes = _unique_list(child.get("mutex", []), "{}.mutex".format(field))
        if any(not isinstance(mutex, str) or not mutex.strip() for mutex in mutexes):
            raise ContractError("{}.mutex entries must be non-empty strings".format(field))

        state = child.get("state")
        if state not in WORKFLOW_STATES:
            raise ContractError("{} has invalid workflow state {!r}".format(field, state))
        records[issue] = {
            "issue": issue,
            "depends_on": sorted(dependencies),
            "mutex": sorted(mutexes),
            "state": state,
        }

    declared = set(records)
    for issue, child in records.items():
        missing = set(child["depends_on"]) - declared
        if missing:
            raise ContractError(
                "child issue #{} references undeclared dependencies: {}".format(
                    issue, ", ".join("#{}".format(number) for number in sorted(missing))
                )
            )

    visiting = set()
    visited = set()

    def visit(issue):
        if issue in visiting:
            raise ContractError("epic dependency graph contains a cycle")
        if issue in visited:
            return
        visiting.add(issue)
        for dependency in records[issue]["depends_on"]:
            visit(dependency)
        visiting.remove(issue)
        visited.add(issue)

    for issue in sorted(records):
        visit(issue)

    return worker_limit, records


def plan_wave(snapshot):
    """Validate a normalized snapshot and return one deterministic wave."""
    worker_limit, records = validate_snapshot(snapshot)
    active = sorted(issue for issue, child in records.items() if child["state"] in ACTIVE_STATES)
    available_slots = max(0, worker_limit - len(active))
    held_mutexes = {mutex for issue in active for mutex in records[issue]["mutex"]}
    candidates = sorted(
        issue
        for issue, child in records.items()
        if child["state"] == "queued"
        and all(records[dependency]["state"] == "completed" for dependency in child["depends_on"])
    )

    selected = []
    selected_mutexes = set()
    for issue in candidates:
        if len(selected) >= available_slots:
            break
        mutexes = set(records[issue]["mutex"])
        if mutexes & (held_mutexes | selected_mutexes):
            continue
        selected.append(issue)
        selected_mutexes.update(mutexes)

    return {
        "validated_children": len(records),
        "active": active,
        "available_slots": available_slots,
        "candidates": candidates,
        "selected": selected,
    }


def apply_wave(snapshot, selected):
    """Model queued -> execution-ready for repeat-run/idempotency tests."""
    updated = copy.deepcopy(snapshot)
    selected_set = set(selected)
    for child in updated["children"]:
        if child["issue"] in selected_set and child["state"] == "queued":
            child["state"] = "execution-ready"
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="normalized epic snapshot JSON")
    args = parser.parse_args()
    with args.snapshot.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    print(json.dumps(plan_wave(snapshot), sort_keys=True))


if __name__ == "__main__":
    main()
