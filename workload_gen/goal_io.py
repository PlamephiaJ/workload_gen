from __future__ import annotations

import struct
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .model import OpType, RankGraph, Task, Workload


MAGIC_COOKIE = 4223
_NODE = struct.Struct("<IBIQIBBIIII")


def write_goal_text(workload: Workload, path: Path) -> None:
    lines = [f"num_ranks {workload.world_size}", ""]
    for graph in workload.ranks:
        lines.append(f"rank {graph.rank} {{")
        for task in graph.tasks:
            if task.op_type == OpType.SEND:
                lines.append(
                    f"l{task.id}: send {task.size}b to {task.peer} tag {task.tag} "
                    f"cpu {task.cpu} nic {task.nic}"
                )
            elif task.op_type == OpType.RECV:
                lines.append(
                    f"l{task.id}: recv {task.size}b from {task.peer} tag {task.tag} "
                    f"cpu {task.cpu} nic {task.nic}"
                )
            else:
                lines.append(f"l{task.id}: calc {task.size} cpu {task.cpu}")
        for task in graph.tasks:
            for predecessor in sorted(task.dependencies):
                lines.append(f"l{task.id} requires l{predecessor}")
        lines.extend(["}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _rank_blob(graph: RankGraph) -> bytes:
    depend_on_me: list[list[int]] = [[] for _ in graph.tasks]
    for task in graph.tasks:
        for predecessor in sorted(task.dependencies):
            depend_on_me[predecessor].append(task.id)

    roots = [task.id for task in graph.tasks if not task.dependencies]
    body = bytearray(struct.pack("<II", len(graph.tasks), len(roots)))
    for root in roots:
        body.extend(struct.pack("<I", root))

    appendix_index = 0
    for task, dependents in zip(graph.tasks, depend_on_me):
        body.extend(
            _NODE.pack(
                len(task.dependencies),
                int(task.op_type),
                task.peer,
                task.size,
                task.tag,
                task.cpu,
                task.nic,
                len(dependents),
                appendix_index,
                0,
                appendix_index + len(dependents),
            )
        )
        appendix_index += len(dependents)

    for dependents in depend_on_me:
        for dependent in dependents:
            body.extend(struct.pack("<I", dependent))
    return bytes(body)


def write_goal_binary(workload: Workload, path: Path) -> None:
    blobs = [_rank_blob(graph) for graph in workload.ranks]
    max_cpu = max((task.cpu for graph in workload.ranks for task in graph.tasks), default=0)
    max_nic = max((task.nic for graph in workload.ranks for task in graph.tasks), default=0)

    # Jump-table offsets are relative to the byte immediately after the cookie,
    # matching Atlas/LogGOPSim Parser.hpp::serialize_mmap.
    header_size = 4 + 1 + 1 + 16 * workload.world_size
    offsets: list[tuple[int, int]] = []
    cursor = header_size
    for blob in blobs:
        offsets.append((cursor, cursor + len(blob)))
        cursor += len(blob)

    output = bytearray(struct.pack("<QIBB", MAGIC_COOKIE, workload.world_size, max_cpu, max_nic))
    for start, end in offsets:
        output.extend(struct.pack("<QQ", start, end))
    for blob in blobs:
        output.extend(blob)
    path.write_bytes(output)


def read_goal_binary(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 14:
        raise ValueError("binary is too short for a GOAL header")
    magic, num_ranks, max_cpu, max_nic = struct.unpack_from("<QIBB", data, 0)
    if magic != MAGIC_COOKIE:
        raise ValueError(f"bad GOAL magic cookie: {magic}, expected {MAGIC_COOKIE}")
    table_end = 14 + 16 * num_ranks
    if len(data) < table_end:
        raise ValueError("truncated GOAL jump table")

    schedules: list[dict[str, Any]] = []
    for rank in range(num_ranks):
        relative_start, relative_end = struct.unpack_from("<QQ", data, 14 + rank * 16)
        start, end = 8 + relative_start, 8 + relative_end
        if not (table_end <= start <= end <= len(data)):
            raise ValueError(f"rank {rank}: invalid jump-table interval [{start}, {end})")
        num_nodes, num_roots = struct.unpack_from("<II", data, start)
        cursor = start + 8
        roots = list(struct.unpack_from(f"<{num_roots}I", data, cursor)) if num_roots else []
        cursor += 4 * num_roots
        nodes: list[dict[str, Any]] = []
        for node_id in range(num_nodes):
            if cursor + _NODE.size > end:
                raise ValueError(f"rank {rank}: truncated node table")
            fields = _NODE.unpack_from(data, cursor)
            cursor += _NODE.size
            nodes.append(
                {
                    "id": node_id,
                    "dependencies_count": fields[0],
                    "type": fields[1],
                    "peer": fields[2],
                    "size": fields[3],
                    "tag": fields[4],
                    "cpu": fields[5],
                    "nic": fields[6],
                    "num_dependents": fields[7],
                    "dependents_index": fields[8],
                    "num_start_dependents": fields[9],
                    "start_dependents_index": fields[10],
                }
            )
        appendix = list(struct.unpack_from(f"<{(end - cursor) // 4}I", data, cursor)) if cursor < end else []
        for node in nodes:
            begin = node["dependents_index"]
            finish = begin + node["num_dependents"]
            node["dependents"] = appendix[begin:finish]
        schedules.append({"rank": rank, "roots": roots, "nodes": nodes})
    return {"num_ranks": num_ranks, "max_cpu": max_cpu, "max_nic": max_nic, "schedules": schedules}


def validate_goal_binary(path: Path) -> dict[str, int]:
    parsed = read_goal_binary(path)
    sends: Counter[tuple[int, int, int, int]] = Counter()
    recvs: Counter[tuple[int, int, int, int]] = Counter()
    task_count = 0
    edge_count = 0

    for schedule in parsed["schedules"]:
        rank = schedule["rank"]
        nodes = schedule["nodes"]
        incoming = [0] * len(nodes)
        for node in nodes:
            if node["type"] not in {1, 2, 3}:
                raise ValueError(f"rank {rank}: unknown operation type {node['type']}")
            for dependent in node["dependents"]:
                if dependent >= len(nodes):
                    raise ValueError(f"rank {rank}: dependent offset {dependent} is out of range")
                incoming[dependent] += 1
                edge_count += 1
            if node["type"] == OpType.SEND:
                sends[(rank, node["peer"], node["size"], node["tag"])] += 1
            elif node["type"] == OpType.RECV:
                recvs[(node["peer"], rank, node["size"], node["tag"])] += 1

        declared = [node["dependencies_count"] for node in nodes]
        if incoming != declared:
            raise ValueError(f"rank {rank}: serialized incoming dependency counts are inconsistent")
        expected_roots = [idx for idx, count in enumerate(incoming) if count == 0]
        if schedule["roots"] != expected_roots:
            raise ValueError(f"rank {rank}: root table does not match dependency graph")

        queue = deque(expected_roots)
        visited = 0
        remaining = incoming[:]
        while queue:
            current = queue.popleft()
            visited += 1
            for dependent in nodes[current]["dependents"]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    queue.append(dependent)
        if visited != len(nodes):
            raise ValueError(f"rank {rank}: local task graph contains a cycle")
        task_count += len(nodes)

    if sends != recvs:
        missing_recvs = sends - recvs
        missing_sends = recvs - sends
        raise ValueError(
            "unmatched point-to-point operations: "
            f"missing recvs={dict(missing_recvs)}, missing sends={dict(missing_sends)}"
        )
    return {
        "ranks": parsed["num_ranks"],
        "tasks": task_count,
        "edges": edge_count,
        "flows": sum(sends.values()),
    }

