from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_spec(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"template extends cycle: {chain}")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("top-level input must be a JSON object")
    extends = spec.get("extends")
    if extends is not None:
        if not isinstance(extends, str) or not extends:
            raise ValueError("extends must be a non-empty relative path")
        base_path = (path.parent / extends).resolve()
        base = load_spec(base_path, (*_stack, path))
        if spec.get("kind", base["kind"]) != base["kind"]:
            raise ValueError("an extending template cannot change kind")
        if spec.get("world_size", base["world_size"]) != base["world_size"]:
            raise ValueError("an extending template cannot change world_size")
        extension_nodes = spec.get("nodes", [])
        if not isinstance(extension_nodes, list):
            raise ValueError("nodes must be an array")
        merged_defaults = {**base.get("defaults", {}), **spec.get("defaults", {})}
        spec = {
            **base,
            **{key: value for key, value in spec.items() if key not in {"extends", "nodes", "defaults"}},
            "defaults": merged_defaults,
            "nodes": [*base["nodes"], *extension_nodes],
        }
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
    flow_nodes: dict[int, str] = {}
    for node_id, node in by_id.items():
        op = node.get("op")
        if op == "delay":
            raise ValueError(
                f"node {node_id}: use collective_delay in a collective DAG or flow_delay in a flow DAG"
            )
        if op == "collective_delay" and spec["kind"] != "collective":
            raise ValueError(f"node {node_id}: collective_delay is only valid in a collective DAG")
        if op == "flow_delay":
            if spec["kind"] != "flow":
                raise ValueError(f"node {node_id}: flow_delay is only valid in a flow DAG")
            forbidden = [key for key in ("ranks", "depends_on", "rank_depends_on") if node.get(key)]
            if forbidden:
                raise ValueError(
                    f"node {node_id}: flow_delay is selected only by flow_id; remove {', '.join(forbidden)}"
                )
        if node.get("op") == "flow" and "flow_id" in node:
            flow_id = as_nonnegative_int(node["flow_id"], f"node {node_id}.flow_id")
            if flow_id in flow_nodes:
                raise ValueError(f"duplicate flow_id {flow_id} in nodes {flow_nodes[flow_id]} and {node_id}")
            flow_nodes[flow_id] = node_id

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

        rank_dependencies = node.get("rank_depends_on", {})
        if not isinstance(rank_dependencies, dict):
            raise ValueError(f"node {node_id}: rank_depends_on must be an object")
        for raw_rank, dependencies in rank_dependencies.items():
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"node {node_id}: invalid rank_depends_on rank {raw_rank!r}") from exc
            if str(rank) != str(raw_rank) or not 0 <= rank < spec["world_size"]:
                raise ValueError(f"node {node_id}: invalid rank_depends_on rank {raw_rank!r}")
            if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
                raise ValueError(f"node {node_id}: rank_depends_on[{raw_rank!r}] must be a node-id array")
            for dependency in dependencies:
                if dependency not in by_id:
                    raise ValueError(f"node {node_id}: unknown rank dependency {dependency}")
                if dependency == node_id:
                    raise ValueError(f"node {node_id}: self dependency is not allowed")
                if dependency not in seen:
                    seen.add(dependency)
                    outgoing[dependency].append(node_id)
                    indegree[node_id] += 1

        if node.get("op") == "flow_delay":
            flow_id = as_nonnegative_int(node.get("flow_id"), f"node {node_id}.flow_id")
            dependency = flow_nodes.get(flow_id)
            if dependency is None:
                raise ValueError(f"node {node_id}: flow_delay references unknown flow_id {flow_id}")
            if dependency not in seen:
                outgoing[dependency].append(node_id)
                indegree[node_id] += 1

    ready = deque(node_id for node_id in by_id if indegree[node_id] == 0)
    ordered: list[dict[str, Any]] = []
    while ready:
        # A flow_delay mutates the selected recv's existing successor edges.
        # Compile all ordinary DAG nodes first so adding a perturbation cannot
        # reorder baseline task IDs or change otherwise-unperturbed execution.
        position = next(
            (idx for idx, candidate in enumerate(ready) if by_id[candidate].get("op") != "flow_delay"),
            0,
        )
        ready.rotate(-position)
        node_id = ready.popleft()
        ready.rotate(position)
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

