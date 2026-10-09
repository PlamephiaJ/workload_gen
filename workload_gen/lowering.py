from __future__ import annotations

from typing import Any

from .model import OpType, Workload
from .spec import topological_nodes


def _flow_template_name(name: str) -> str:
    if name.endswith(".collective"):
        return name.removesuffix(".collective") + ".flow"
    return name + ".flow"


def lower_collective_to_flow_spec(
    source_spec: dict[str, Any], workload: Workload, *, name: str | None = None
) -> dict[str, Any]:
    """Export a compiled collective workload as an endpoint-exact flow DAG.

    Communication send/recv pairs become one lowered flow node. Local GOAL
    tasks become single-rank compute nodes. ``rank_depends_on`` retains the
    dependency of each individual flow endpoint, avoiding accidental edges on
    the other endpoint when two flow nodes share both ranks.
    """
    if workload.source_kind != "collective":
        raise ValueError("only a collective workload can be lowered to a flow template")

    task_to_node: dict[tuple[int, int], str] = {}
    nodes_by_id: dict[str, dict[str, Any]] = {}

    for flow_id, flow in sorted(workload.flows.items()):
        node_id = f"flow_{flow_id}"
        send_key = (int(flow["send_task"]["rank"]), int(flow["send_task"]["task_id"]))
        recv_key = (int(flow["recv_task"]["rank"]), int(flow["recv_task"]["task_id"]))
        send_task = workload.ranks[send_key[0]].tasks[send_key[1]]
        recv_task = workload.ranks[recv_key[0]].tasks[recv_key[1]]
        if send_task.cpu != recv_task.cpu or send_task.nic != recv_task.nic:
            raise ValueError(f"flow {flow_id}: endpoint cpu/nic assignments differ")

        excluded = {
            "flow_id",
            "src",
            "dst",
            "bytes",
            "tag",
            "message_sequence",
            "send_task",
            "recv_task",
            "post_delay",
            "dag_node_id",
        }
        metadata = {key: value for key, value in flow.items() if key not in excluded}
        node: dict[str, Any] = {
            "id": node_id,
            "op": "flow",
            "flow_id": flow_id,
            "src": flow["src"],
            "dst": flow["dst"],
            "bytes": flow["bytes"],
            "tag": flow["tag"],
            "cpu": send_task.cpu,
            "nic": send_task.nic,
            "lowered": True,
        }
        if metadata:
            node["metadata"] = metadata
        nodes_by_id[node_id] = node
        task_to_node[send_key] = node_id
        task_to_node[recv_key] = node_id

    for graph in workload.ranks:
        for task in graph.tasks:
            key = (graph.rank, task.id)
            if key in task_to_node:
                continue
            if task.op_type != OpType.CALC:
                raise ValueError(f"rank {graph.rank} task {task.id}: communication task lacks flow metadata")
            node_id = f"calc_r{graph.rank}_t{task.id}"
            node = {
                "id": node_id,
                "op": "compute",
                "ranks": [graph.rank],
                "duration_ns": task.size,
                "cpu": task.cpu,
            }
            if task.label:
                node["source_label"] = task.label
            nodes_by_id[node_id] = node
            task_to_node[key] = node_id

    rank_dependencies: dict[str, dict[str, set[str]]] = {
        node_id: {} for node_id in nodes_by_id
    }
    for graph in workload.ranks:
        for task in graph.tasks:
            node_id = task_to_node[(graph.rank, task.id)]
            dependencies = rank_dependencies[node_id].setdefault(str(graph.rank), set())
            for predecessor in task.dependencies:
                predecessor_node = task_to_node[(graph.rank, predecessor)]
                if predecessor_node != node_id:
                    dependencies.add(predecessor_node)

    for node_id, by_rank in rank_dependencies.items():
        compact = {
            rank: sorted(dependencies)
            for rank, dependencies in sorted(by_rank.items(), key=lambda item: int(item[0]))
            if dependencies
        }
        if compact:
            nodes_by_id[node_id]["rank_depends_on"] = compact

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "flow",
        "name": name or _flow_template_name(source_spec["name"]),
        "world_size": workload.world_size,
        "defaults": source_spec.get("defaults", {}),
        "lowered_from": {
            "name": source_spec["name"],
            "kind": "collective",
        },
    }
    unordered = {**result, "nodes": list(nodes_by_id.values())}
    result["nodes"] = topological_nodes(unordered)
    return result
