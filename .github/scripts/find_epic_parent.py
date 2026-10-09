#!/usr/bin/env python3
"""Find the unique active epic whose canonical DAG contains a changed child."""

import argparse
import json
import re
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
DAG_MARKER = "<!-- codex-epic-dag:v1 -->"
FENCED_BLOCK = re.compile(r"```(?:yaml|yml)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
EPIC_MODE = re.compile(r"^execution_mode\s*:\s*epic-dag\s*(?:#.*)?$", re.MULTILINE)
CHILDREN_HEADER = re.compile(r"^children\s*:\s*(?:#.*)?$")
CHILD_ISSUE = re.compile(r"^\s*-\s*(?:\{\s*)?issue\s*:\s*(\d+)\b")


class DiscoveryError(ValueError):
    """Raised when active-parent discovery cannot choose safely."""


def _flatten_pages(payload):
    if not isinstance(payload, list):
        raise DiscoveryError("GitHub JSON payload must be a list")
    if payload and all(isinstance(page, list) for page in payload):
        return [item for page in payload for item in page]
    return payload


def _label_names(issue):
    names = []
    for label in issue.get("labels", []):
        names.append(label.get("name") if isinstance(label, dict) else label)
    return {name for name in names if name}


def extract_canonical_epic_children(comments):
    """Return the canonical DAG child set, None when no DAG is published."""
    matches = []
    for comment in comments:
        body = (comment or {}).get("body") or ""
        if DAG_MARKER not in body:
            continue
        if body.count(DAG_MARKER) != 1:
            raise DiscoveryError("canonical epic DAG comment contains a duplicate marker")
        matches.append(body)

    if not matches:
        return None
    if len(matches) != 1:
        raise DiscoveryError("epic has more than one canonical DAG comment")

    blocks = [block for block in FENCED_BLOCK.findall(matches[0]) if EPIC_MODE.search(block)]
    if len(blocks) != 1:
        raise DiscoveryError("canonical epic DAG must contain exactly one epic-dag YAML block")

    lines = blocks[0].splitlines()
    header_indexes = [index for index, line in enumerate(lines) if CHILDREN_HEADER.match(line)]
    if len(header_indexes) != 1:
        raise DiscoveryError("canonical epic DAG must contain exactly one children list")

    children = []
    for line in lines[header_indexes[0] + 1 :]:
        if line and not line[0].isspace():
            break
        match = CHILD_ISSUE.match(line)
        if match:
            children.append(int(match.group(1)))
    if not children:
        raise DiscoveryError("canonical epic DAG children list is empty or unreadable")
    if len(children) != len(set(children)):
        raise DiscoveryError("canonical epic DAG contains duplicate child issues")
    return set(children)


def find_parent(changed_issue, issues, comments_by_issue):
    matches = []
    for issue in issues:
        state = issue.get("state")
        if issue.get("pull_request") or (state is not None and str(state).lower() != "open"):
            continue
        states = _label_names(issue) & WORKFLOW_STATES
        if states != {"in-progress"}:
            continue
        number = issue.get("number")
        children = extract_canonical_epic_children(comments_by_issue.get(number, []))
        if children is not None and changed_issue in children:
            matches.append(issue)

    if len(matches) > 1:
        numbers = ", ".join("#{}".format(issue["number"]) for issue in matches)
        raise DiscoveryError(
            "changed issue #{} belongs to multiple active epics: {}".format(changed_issue, numbers)
        )
    if not matches:
        return None
    return {"number": matches[0]["number"], "title": matches[0].get("title", "")}


def _load_comments(directory, issues):
    result = {}
    for issue in issues:
        number = issue.get("number")
        if not isinstance(number, int):
            continue
        path = directory / "{}.json".format(number)
        if not path.exists():
            result[number] = []
            continue
        with path.open(encoding="utf-8") as handle:
            result[number] = _flatten_pages(json.load(handle))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-issue", required=True, type=int)
    parser.add_argument("--issues-json", required=True, type=Path)
    parser.add_argument("--comments-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.issues_json.open(encoding="utf-8") as handle:
        issues = _flatten_pages(json.load(handle))
    comments = _load_comments(args.comments_dir, issues)
    print(json.dumps({"parent": find_parent(args.changed_issue, issues, comments)}, sort_keys=True))


if __name__ == "__main__":
    main()
