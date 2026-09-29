from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_spec(path: Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("top-level input must be a JSON object")
    if spec.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if spec.get("kind") not in {"collective", "flow"}:
        raise ValueError("kind must be either 'collective' or 'flow'")
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("name must use only letters, digits, '.', '_' and '-' and start alphanumerically")
    world_size = spec.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("nodes must be a non-empty array")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    return spec


def topological_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = spec["nodes"]
    by_id: dict[str, dict[str, Any]] = {}
    index: dict[str, int] = {}
    for position, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{position}] must be an object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not _NAME.fullmatch(node_id):
            raise ValueError(f"nodes[{position}].id is missing or invalid")
        if node_id in by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        dependencies = node.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
            raise ValueError(f"node {node_id}: depends_on must be an array of node ids")
        by_id[node_id] = node
        index[node_id] = position

    outgoing = {node_id: [] for node_id in by_id}
    indegree = {node_id: 0 for node_id in by_id}
    for node_id, node in by_id.items():
        seen: set[str] = set()
        for dependency in node.get("depends_on", []):
            if dependency not in by_id:
                raise ValueError(f"node {node_id}: unknown dependency {dependency}")
            if dependency == node_id:
                raise ValueError(f"node {node_id}: self dependency is not allowed")
            if dependency in seen:
                raise ValueError(f"node {node_id}: duplicate dependency {dependency}")
            seen.add(dependency)
            outgoing[dependency].append(node_id)
            indegree[node_id] += 1

    ready = deque(node_id for node_id in by_id if indegree[node_id] == 0)
    ordered: list[dict[str, Any]] = []
    while ready:
        node_id = ready.popleft()
        ordered.append(by_id[node_id])
        for successor in sorted(outgoing[node_id], key=index.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(ordered) != len(nodes):
        cyclic = [node_id for node_id, degree in indegree.items() if degree]
        raise ValueError(f"high-level graph contains a cycle involving: {', '.join(cyclic)}")
    return ordered


def as_nonnegative_int(value: Any, field: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value

