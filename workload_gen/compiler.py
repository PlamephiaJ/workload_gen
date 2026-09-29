from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .model import RankGraph, Span, Workload
from .spec import as_nonnegative_int, topological_nodes


@dataclass(frozen=True)
class FlowHandle:
    flow_id: int
    src: int
    dst: int
    send_task: int
    recv_task: int


def _split_bytes(total: int, pieces: int) -> list[int]:
    base, extra = divmod(total, pieces)
    return [base + (1 if idx < extra else 0) for idx in range(pieces)]


def _chunk_bytes(total: int, maximum: int) -> list[int]:
    if total == 0:
        return [0]
    return [min(maximum, total - offset) for offset in range(0, total, maximum)]


class Compiler:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.kind = spec["kind"]
        self.world_size = spec["world_size"]
        self.defaults = spec.get("defaults", {})
        self.graphs = [RankGraph(rank) for rank in range(self.world_size)]
        self.spans: dict[str, dict[int, Span]] = {}
        self.flows: dict[int, dict[str, Any]] = {}
        self.collectives: list[dict[str, Any]] = []
        self.next_flow_id = as_nonnegative_int(spec.get("flow_id_start", 0), "flow_id_start")
        self.message_sequence: defaultdict[tuple[int, int, int], int] = defaultdict(int)
        self.used_message_keys: set[tuple[int, int, int]] = set()
        self.used_collective_ids: set[int] = set()
        self.next_collective_id = 0

    def compile(self) -> Workload:
        for node in topological_nodes(self.spec):
            node_id = node["id"]
            op = node.get("op")
            if op in {"compute", "sink", "barrier"}:
                spans = self._compile_compute(node)
            elif self.kind == "flow" and op == "flow":
                spans = self._compile_direct_flow(node)
            elif self.kind == "collective" and op in {"alltoall", "allreduce"}:
                spans = self._compile_collective(node)
            else:
                supported = "compute/sink/barrier/flow" if self.kind == "flow" else "compute/sink/barrier/alltoall/allreduce"
                raise ValueError(f"node {node_id}: unsupported op {op!r}; supported: {supported}")

            for dependency in node.get("depends_on", []):
                predecessor_spans = self.spans[dependency]
                shared_ranks = sorted(set(spans) & set(predecessor_spans))
                if not shared_ranks:
                    raise ValueError(
                        f"edge {dependency} -> {node_id} has no shared rank; GOAL dependencies are rank-local"
                    )
                for rank in shared_ranks:
                    self.graphs[rank].require(spans[rank].start, predecessor_spans[rank].end)
            self.spans[node_id] = spans

        return Workload(self.spec["name"], self.kind, self.graphs, self.flows, self.collectives)

    def _value(self, node: dict[str, Any], key: str, fallback: Any = None) -> Any:
        if key in node:
            return node[key]
        return self.defaults.get(key, fallback)

    def _ranks(self, node: dict[str, Any]) -> list[int]:
        value = node.get("ranks", "all")
        if value == "all":
            return list(range(self.world_size))
        if not isinstance(value, list) or not value:
            raise ValueError(f"node {node['id']}: ranks must be 'all' or a non-empty integer array")
        if not all(isinstance(rank, int) and not isinstance(rank, bool) for rank in value):
            raise ValueError(f"node {node['id']}: ranks must contain integers")
        if len(set(value)) != len(value):
            raise ValueError(f"node {node['id']}: ranks contains duplicates")
        if any(rank < 0 or rank >= self.world_size for rank in value):
            raise ValueError(f"node {node['id']}: rank outside 0..{self.world_size - 1}")
        return value

    def _cpu_nic(self, node: dict[str, Any]) -> tuple[int, int]:
        cpu = as_nonnegative_int(self._value(node, "cpu", 0), f"node {node['id']}.cpu")
        nic = as_nonnegative_int(self._value(node, "nic", 0), f"node {node['id']}.nic")
        if cpu > 255 or nic > 255:
            raise ValueError(f"node {node['id']}: cpu and nic must fit uint8")
        return cpu, nic

    def _compile_compute(self, node: dict[str, Any]) -> dict[int, Span]:
        ranks = self._ranks(node)
        cpu, _ = self._cpu_nic(node)
        duration = node.get("duration_ns", 0)
        spans: dict[int, Span] = {}
        for rank in ranks:
            if isinstance(duration, dict):
                raw = duration.get(str(rank), duration.get(rank))
                if raw is None:
                    raise ValueError(f"node {node['id']}: duration_ns has no value for rank {rank}")
            else:
                raw = duration
            value = as_nonnegative_int(raw, f"node {node['id']}.duration_ns")
            task = self.graphs[rank].add_calc(value, cpu=cpu, label=node["id"])
            spans[rank] = Span(task, task)
        return spans

    def _new_anchors(self, node_id: str, ranks: list[int], cpu: int) -> dict[int, Span]:
        return {rank: self.graphs[rank].anchor_span(node_id, cpu=cpu) for rank in ranks}

    def _allocate_tag(self, src: int, dst: int, context: int) -> tuple[int, int]:
        if not 0 <= context <= 99_999:
            raise ValueError("collective context must be in 0..99999")
        key = (src, dst, context)
        sequence = self.message_sequence[key]
        self.message_sequence[key] += 1
        if sequence > 9_999:
            raise ValueError(
                f"Atlas-compatible message sequence overflow for src={src}, dst={dst}, context={context}"
            )
        tag = sequence * 100_000 + context
        return tag, sequence

    def _add_flow(
        self,
        src: int,
        dst: int,
        size: int,
        *,
        cpu: int,
        nic: int,
        context: int | None = None,
        explicit_tag: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FlowHandle:
        if src == dst:
            raise ValueError("a point-to-point flow cannot have identical src and dst")
        if not 0 <= src < self.world_size or not 0 <= dst < self.world_size:
            raise ValueError(f"flow endpoint outside 0..{self.world_size - 1}")
        as_nonnegative_int(size, "flow bytes")
        if explicit_tag is None:
            if context is None:
                tag = self.next_flow_id
                sequence = None
            else:
                tag, sequence = self._allocate_tag(src, dst, context)
        else:
            tag = as_nonnegative_int(explicit_tag, "flow tag")
            sequence = None
        if tag > 0xFFFFFFFF:
            raise ValueError("flow tag must fit uint32")
        message_key = (src, dst, tag)
        if message_key in self.used_message_keys:
            raise ValueError(f"duplicate message key src={src}, dst={dst}, tag={tag}")
        self.used_message_keys.add(message_key)

        flow_id = self.next_flow_id
        self.next_flow_id += 1
        send = self.graphs[src].add_send(dst, size, tag, cpu=cpu, nic=nic, label=f"flow:{flow_id}:send")
        recv = self.graphs[dst].add_recv(src, size, tag, cpu=cpu, nic=nic, label=f"flow:{flow_id}:recv")
        if self.kind == "collective":
            record = dict(metadata or {})
            record.update(
                {
                    "flow_id": flow_id,
                    "src": src,
                    "dst": dst,
                    "bytes": size,
                    "tag": tag,
                    "message_sequence": sequence,
                    "send_task": {"rank": src, "task_id": send},
                    "recv_task": {"rank": dst, "task_id": recv},
                }
            )
            self.flows[flow_id] = record
        return FlowHandle(flow_id, src, dst, send, recv)

    def _compile_direct_flow(self, node: dict[str, Any]) -> dict[int, Span]:
        src = as_nonnegative_int(node.get("src"), f"node {node['id']}.src")
        dst = as_nonnegative_int(node.get("dst"), f"node {node['id']}.dst")
        size = as_nonnegative_int(node.get("bytes"), f"node {node['id']}.bytes")
        cpu, nic = self._cpu_nic(node)
        maximum = as_nonnegative_int(self._value(node, "chunk_bytes", max(1, size)), "chunk_bytes", positive=True)
        pieces = _chunk_bytes(size, maximum)
        spans = self._new_anchors(node["id"], [src, dst], cpu)
        previous = {src: spans[src].start, dst: spans[dst].start}
        base_tag = node.get("tag")
        if base_tag is not None:
            base_tag = as_nonnegative_int(base_tag, f"node {node['id']}.tag")
        for chunk, chunk_size in enumerate(pieces):
            handle = self._add_flow(
                src,
                dst,
                chunk_size,
                cpu=cpu,
                nic=nic,
                explicit_tag=(base_tag + chunk if base_tag is not None else None),
            )
            self.graphs[src].require(handle.send_task, previous[src])
            self.graphs[dst].require(handle.recv_task, previous[dst])
            previous[src] = handle.send_task
            previous[dst] = handle.recv_task
        for rank in (src, dst):
            self.graphs[rank].require(spans[rank].end, previous[rank])
        return spans

    def _collective_identity(self, node: dict[str, Any]) -> tuple[int, int, int]:
        if "collective_id" in node:
            collective_id = as_nonnegative_int(node["collective_id"], f"node {node['id']}.collective_id")
        else:
            while self.next_collective_id in self.used_collective_ids:
                self.next_collective_id += 1
            collective_id = self.next_collective_id
            self.next_collective_id += 1
        if collective_id in self.used_collective_ids:
            raise ValueError(f"duplicate collective_id {collective_id}")
        self.used_collective_ids.add(collective_id)
        context = as_nonnegative_int(node.get("context", collective_id), f"node {node['id']}.context")
        ordering_domain = as_nonnegative_int(
            node.get("ordering_domain", collective_id), f"node {node['id']}.ordering_domain"
        )
        return collective_id, context, ordering_domain

    def _compile_collective(self, node: dict[str, Any]) -> dict[int, Span]:
        ranks = self._ranks(node)
        if len(ranks) < 2:
            raise ValueError(f"node {node['id']}: collective requires at least two ranks")
        cpu, nic = self._cpu_nic(node)
        collective_id, context, ordering_domain = self._collective_identity(node)
        op = node["op"]
        algorithm = node.get("algorithm", "direct" if op == "alltoall" else "ring")
        descriptor = {
            "collective_id": collective_id,
            "node_id": node["id"],
            "type": op,
            "algorithm": algorithm,
            "ranks": ranks,
            "context": context,
            "ordering_domain": ordering_domain,
        }
        if op == "alltoall":
            spans = self._lower_alltoall(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes_per_peer"] = node.get("bytes_per_peer")
        elif algorithm == "ring":
            spans = self._lower_ring_allreduce(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
        elif algorithm == "tree":
            spans = self._lower_tree_allreduce(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
        else:
            raise ValueError(f"node {node['id']}: allreduce algorithm must be 'ring' or 'tree'")
        self.collectives.append(descriptor)
        return spans

    def _flow_meta(
        self,
        node: dict[str, Any],
        collective_id: int,
        algorithm: str,
        phase: str,
        chunk: int,
        step: int,
        channel: int,
        ordering_domain: int,
        **extra: Any,
    ) -> dict[str, Any]:
        result = {
            "collective_id": collective_id,
            "node_id": node["id"],
            "type": node["op"],
            "algorithm": algorithm,
            "phase": phase,
            "chunk": chunk,
            "step": step,
            "channel": channel,
            "ordering_domain": ordering_domain + channel,
        }
        result.update(extra)
        return result

    def _finalize_collective(
        self, spans: dict[int, Span], ranks: list[int], first_internal_id: dict[int, int]
    ) -> None:
        for rank in ranks:
            internal = [
                task.id
                for task in self.graphs[rank].tasks[first_internal_id[rank] :]
                if task.id != spans[rank].end
            ]
            if not internal:
                self.graphs[rank].require(spans[rank].end, spans[rank].start)
            else:
                for task_id in internal:
                    self.graphs[rank].require(spans[rank].end, task_id)

    def _lower_alltoall(
        self,
        node: dict[str, Any],
        ranks: list[int],
        cpu: int,
        nic: int,
        collective_id: int,
        context: int,
        ordering_domain: int,
    ) -> dict[int, Span]:
        if node.get("algorithm", "direct") != "direct":
            raise ValueError(f"node {node['id']}: alltoall currently supports algorithm='direct'")
        size = as_nonnegative_int(node.get("bytes_per_peer"), f"node {node['id']}.bytes_per_peer")
        maximum = as_nonnegative_int(self._value(node, "chunk_bytes", max(1, size)), "chunk_bytes", positive=True)
        pieces = _chunk_bytes(size, maximum)
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}
        handles: dict[tuple[int, int, int], FlowHandle] = {}
        for src in ranks:
            for dst in ranks:
                if src == dst:
                    continue
                for chunk, chunk_size in enumerate(pieces):
                    handles[(src, dst, chunk)] = self._add_flow(
                        src,
                        dst,
                        chunk_size,
                        cpu=cpu,
                        nic=nic,
                        context=context,
                        metadata=self._flow_meta(
                            node,
                            collective_id,
                            "direct",
                            "exchange",
                            chunk,
                            ranks.index(dst),
                            0,
                            ordering_domain,
                            peer=dst,
                            chunks_per_peer=len(pieces),
                        ),
                    )

        for rank_index, rank in enumerate(ranks):
            ordered_tasks: list[int] = []
            for peer in ranks[:rank_index]:
                ordered_tasks.extend(handles[(peer, rank, chunk)].recv_task for chunk in range(len(pieces)))
                ordered_tasks.extend(handles[(rank, peer, chunk)].send_task for chunk in range(len(pieces)))
            for peer in ranks[rank_index + 1 :]:
                ordered_tasks.extend(handles[(rank, peer, chunk)].send_task for chunk in range(len(pieces)))
                ordered_tasks.extend(handles[(peer, rank, chunk)].recv_task for chunk in range(len(pieces)))
            previous = spans[rank].start
            for task_id in ordered_tasks:
                self.graphs[rank].require(task_id, previous)
                previous = task_id
        self._finalize_collective(spans, ranks, first_internal)
        return spans

    def _ring_orders(self, node: dict[str, Any], ranks: list[int], channels: int) -> list[list[int]]:
        value = node.get("ring_orders")
        if value is None:
            return [ranks[:] for _ in range(channels)]
        if not isinstance(value, list) or len(value) != channels:
            raise ValueError(f"node {node['id']}: ring_orders must have one rank permutation per channel")
        orders: list[list[int]] = []
        for order in value:
            if not isinstance(order, list) or len(order) != len(ranks) or set(order) != set(ranks):
                raise ValueError(f"node {node['id']}: each ring order must be a permutation of ranks")
            orders.append(order)
        return orders

    def _cost(self, node: dict[str, Any], prefix: str, size: int) -> int:
        rate = self._value(node, f"{prefix}_ns_per_byte", 0.0)
        overhead = self._value(node, f"{prefix}_overhead_ns", 0)
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate < 0:
            raise ValueError(f"node {node['id']}.{prefix}_ns_per_byte must be non-negative")
        overhead = as_nonnegative_int(overhead, f"node {node['id']}.{prefix}_overhead_ns")
        return overhead + math.ceil(float(rate) * size)

    def _lower_ring_allreduce(
        self,
        node: dict[str, Any],
        ranks: list[int],
        cpu: int,
        nic: int,
        collective_id: int,
        context: int,
        ordering_domain: int,
    ) -> dict[int, Span]:
        total = as_nonnegative_int(node.get("bytes"), f"node {node['id']}.bytes")
        channels = as_nonnegative_int(node.get("channels", 1), f"node {node['id']}.channels", positive=True)
        orders = self._ring_orders(node, ranks, channels)
        channel_sizes = _split_bytes(total, channels)
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}

        for channel, (order, channel_size) in enumerate(zip(orders, channel_sizes)):
            n = len(order)
            chunk_sizes = _split_bytes(channel_size, n)
            reduce_flows: dict[tuple[int, int], FlowHandle] = {}
            gather_flows: dict[tuple[int, int], FlowHandle] = {}
            for step in range(n - 1):
                for index, src in enumerate(order):
                    dst = order[(index + 1) % n]
                    chunk = (index - step - 1) % n
                    reduce_flows[(step, src)] = self._add_flow(
                        src,
                        dst,
                        chunk_sizes[chunk],
                        cpu=cpu,
                        nic=nic,
                        context=context,
                        metadata=self._flow_meta(
                            node,
                            collective_id,
                            "ring",
                            "reduce_scatter",
                            chunk,
                            step,
                            channel,
                            ordering_domain,
                            chunk_bytes=chunk_sizes[chunk],
                        ),
                    )
            for step in range(n - 1):
                for index, src in enumerate(order):
                    dst = order[(index + 1) % n]
                    chunk = (index - step) % n
                    gather_flows[(step, src)] = self._add_flow(
                        src,
                        dst,
                        chunk_sizes[chunk],
                        cpu=cpu,
                        nic=nic,
                        context=context,
                        metadata=self._flow_meta(
                            node,
                            collective_id,
                            "ring",
                            "allgather",
                            chunk,
                            step,
                            channel,
                            ordering_domain,
                            chunk_bytes=chunk_sizes[chunk],
                        ),
                    )

            for index, rank in enumerate(order):
                prev_rank = order[(index - 1) % n]
                first_send = reduce_flows[(0, rank)].send_task
                self.graphs[rank].require(first_send, spans[rank].start)
                for step in range(n - 1):
                    incoming = reduce_flows[(step, prev_rank)]
                    self.graphs[rank].require(incoming.recv_task, spans[rank].start)
                    duration = self._cost(node, "reduction", incoming_size := self.graphs[rank].tasks[incoming.recv_task].size)
                    if step == n - 2:
                        duration += self._cost(node, "copy", incoming_size)
                    calc = self.graphs[rank].add_calc(
                        duration, cpu=cpu, label=f"{node['id']}:ch{channel}:reduce:{step}"
                    )
                    self.graphs[rank].require(calc, incoming.recv_task)
                    next_send = (
                        reduce_flows[(step + 1, rank)].send_task
                        if step < n - 2
                        else gather_flows[(0, rank)].send_task
                    )
                    self.graphs[rank].require(next_send, calc)

                for step in range(n - 1):
                    incoming = gather_flows[(step, prev_rank)]
                    self.graphs[rank].require(incoming.recv_task, spans[rank].start)
                    if step < n - 2:
                        size = self.graphs[rank].tasks[incoming.recv_task].size
                        calc = self.graphs[rank].add_calc(
                            self._cost(node, "copy", size),
                            cpu=cpu,
                            label=f"{node['id']}:ch{channel}:copy:{step}",
                        )
                        self.graphs[rank].require(calc, incoming.recv_task)
                        self.graphs[rank].require(gather_flows[(step + 1, rank)].send_task, calc)
        self._finalize_collective(spans, ranks, first_internal)
        return spans

    def _tree_parents(self, node: dict[str, Any], ranks: list[int]) -> dict[int, int | None]:
        value = node.get("tree_parent")
        if value is None:
            return {rank: (None if index == 0 else ranks[(index - 1) // 2]) for index, rank in enumerate(ranks)}
        if not isinstance(value, dict):
            raise ValueError(f"node {node['id']}: tree_parent must be an object mapping rank to parent/null")
        parents: dict[int, int | None] = {}
        for rank in ranks:
            raw = value.get(str(rank), value.get(rank, "missing"))
            if raw == "missing" or (raw is not None and raw not in ranks):
                raise ValueError(f"node {node['id']}: invalid/missing tree parent for rank {rank}")
            if raw == rank:
                raise ValueError(f"node {node['id']}: rank {rank} cannot parent itself")
            parents[rank] = raw
        roots = [rank for rank, parent in parents.items() if parent is None]
        if len(roots) != 1:
            raise ValueError(f"node {node['id']}: tree must have exactly one root")
        for rank in ranks:
            seen: set[int] = set()
            current: int | None = rank
            while current is not None:
                if current in seen:
                    raise ValueError(f"node {node['id']}: tree_parent contains a cycle")
                seen.add(current)
                current = parents[current]
        return parents

    def _lower_tree_allreduce(
        self,
        node: dict[str, Any],
        ranks: list[int],
        cpu: int,
        nic: int,
        collective_id: int,
        context: int,
        ordering_domain: int,
    ) -> dict[int, Span]:
        total = as_nonnegative_int(node.get("bytes"), f"node {node['id']}.bytes")
        maximum = as_nonnegative_int(self._value(node, "chunk_bytes", max(1, total)), "chunk_bytes", positive=True)
        pieces = _chunk_bytes(total, maximum)
        parents = self._tree_parents(node, ranks)
        children: dict[int, list[int]] = {rank: [] for rank in ranks}
        for rank, parent in parents.items():
            if parent is not None:
                children[parent].append(rank)
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}

        for chunk, chunk_size in enumerate(pieces):
            upward: dict[int, FlowHandle] = {}
            downward: dict[int, FlowHandle] = {}
            for rank, parent in parents.items():
                if parent is None:
                    continue
                upward[rank] = self._add_flow(
                    rank,
                    parent,
                    chunk_size,
                    cpu=cpu,
                    nic=nic,
                    context=context,
                    metadata=self._flow_meta(
                        node,
                        collective_id,
                        "tree",
                        "reduce",
                        chunk,
                        0,
                        0,
                        ordering_domain,
                        tree_role="up",
                        parent=parent,
                        chunk_bytes=chunk_size,
                    ),
                )
                downward[rank] = self._add_flow(
                    parent,
                    rank,
                    chunk_size,
                    cpu=cpu,
                    nic=nic,
                    context=context,
                    metadata=self._flow_meta(
                        node,
                        collective_id,
                        "tree",
                        "broadcast",
                        chunk,
                        1,
                        0,
                        ordering_domain,
                        tree_role="down",
                        parent=parent,
                        chunk_bytes=chunk_size,
                    ),
                )

            for rank in ranks:
                reduced: list[int] = []
                for child in children[rank]:
                    recv = upward[child].recv_task
                    self.graphs[rank].require(recv, spans[rank].start)
                    calc = self.graphs[rank].add_calc(
                        self._cost(node, "reduction", chunk_size),
                        cpu=cpu,
                        label=f"{node['id']}:chunk{chunk}:reduce-from-{child}",
                    )
                    self.graphs[rank].require(calc, recv)
                    reduced.append(calc)

                parent = parents[rank]
                if parent is None:
                    down_ready = reduced or [spans[rank].start]
                else:
                    send_up = upward[rank].send_task
                    for predecessor in reduced or [spans[rank].start]:
                        self.graphs[rank].require(send_up, predecessor)
                    recv_down = downward[rank].recv_task
                    self.graphs[rank].require(recv_down, send_up)
                    down_ready = [recv_down]
                for child in children[rank]:
                    send_down = downward[child].send_task
                    for predecessor in down_ready:
                        self.graphs[rank].require(send_down, predecessor)
        self._finalize_collective(spans, ranks, first_internal)
        return spans


def compile_spec(spec: dict[str, Any]) -> Workload:
    return Compiler(spec).compile()

