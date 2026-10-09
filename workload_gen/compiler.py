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
        self.used_flow_ids: set[int] = set()
        self.used_collective_ids: set[int] = set()
        self.next_collective_id = 0

    def compile(self) -> Workload:
        for node in topological_nodes(self.spec):
            node_id = node["id"]
            op = node.get("op")
            if op == "flow_delay":
                if self.kind != "flow":
                    raise ValueError(f"node {node_id}: flow_delay is only valid in a flow DAG")
                spans = self._compile_flow_delay(node)
            elif op == "collective_delay":
                if self.kind != "collective":
                    raise ValueError(f"node {node_id}: collective_delay is only valid in a collective DAG")
                spans = self._compile_local_calc(node)
            elif op in {"compute", "sink", "barrier"}:
                spans = self._compile_local_calc(node)
            elif self.kind == "flow" and op == "flow":
                spans = self._compile_direct_flow(node)
            elif self.kind == "collective" and op in {
                "alltoall",
                "allreduce",
                "allgather",
                "reduce_scatter",
                "reducescatter",
                "reduce",
                "broadcast",
            }:
                spans = self._compile_collective(node)
            else:
                supported = (
                    "compute/flow_delay/sink/barrier/flow"
                    if self.kind == "flow"
                    else "compute/collective_delay/sink/barrier/"
                    "alltoall/allreduce/allgather/reduce_scatter/reduce/broadcast"
                )
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
            for raw_rank, dependencies in node.get("rank_depends_on", {}).items():
                rank = as_nonnegative_int(int(raw_rank), f"node {node_id}.rank_depends_on rank")
                if rank not in spans:
                    raise ValueError(f"node {node_id}: rank_depends_on rank {rank} is not part of the node")
                for dependency in dependencies:
                    predecessor_spans = self.spans[dependency]
                    if rank not in predecessor_spans:
                        raise ValueError(
                            f"node {node_id}: rank_depends_on predecessor {dependency} has no rank {rank}"
                        )
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

    def _compile_local_calc(self, node: dict[str, Any]) -> dict[int, Span]:
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
        explicit_flow_id: int | None = None,
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

        if explicit_flow_id is None:
            while self.next_flow_id in self.used_flow_ids:
                self.next_flow_id += 1
            flow_id = self.next_flow_id
            self.next_flow_id += 1
        else:
            flow_id = as_nonnegative_int(explicit_flow_id, "flow_id")
        if flow_id in self.used_flow_ids:
            raise ValueError(f"duplicate flow_id {flow_id}")
        self.used_flow_ids.add(flow_id)
        send = self.graphs[src].add_send(dst, size, tag, cpu=cpu, nic=nic, label=f"flow:{flow_id}:send")
        recv = self.graphs[dst].add_recv(src, size, tag, cpu=cpu, nic=nic, label=f"flow:{flow_id}:recv")
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
        if node.get("lowered", False):
            if node.get("lowered") is not True:
                raise ValueError(f"node {node['id']}.lowered must be boolean")
            metadata = node.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"node {node['id']}.metadata must be an object")
            handle = self._add_flow(
                src,
                dst,
                size,
                cpu=cpu,
                nic=nic,
                explicit_tag=node.get("tag"),
                explicit_flow_id=node.get("flow_id"),
                metadata={**metadata, "dag_node_id": node["id"]},
            )
            return {
                src: Span(handle.send_task, handle.send_task),
                dst: Span(handle.recv_task, handle.recv_task),
            }

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
                explicit_flow_id=(node.get("flow_id") + chunk if node.get("flow_id") is not None else None),
                metadata={"node_id": node["id"]},
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

    def _protocol(self, node: dict[str, Any]) -> str:
        protocol = str(self._value(node, "protocol", "simple")).lower()
        if protocol not in {"simple", "ll", "ll128"}:
            raise ValueError(f"node {node['id']}: protocol must be simple, ll, or ll128")
        return protocol

    def _wire_size(self, node: dict[str, Any], payload_bytes: int) -> int:
        protocol = self._protocol(node)
        if protocol in {"ll", "ll128"}:
            return ((payload_bytes + 7) // 8) * 8
        return payload_bytes

    def _root(self, node: dict[str, Any], ranks: list[int]) -> int:
        root = node.get("root", ranks[0])
        if not isinstance(root, int) or isinstance(root, bool) or root not in ranks:
            raise ValueError(f"node {node['id']}: root must be one of the collective ranks")
        return root

    def _compile_collective(self, node: dict[str, Any]) -> dict[int, Span]:
        if node["op"] == "reducescatter":
            node = {**node, "op": "reduce_scatter"}
        ranks = self._ranks(node)
        if len(ranks) < 2:
            raise ValueError(f"node {node['id']}: collective requires at least two ranks")
        cpu, nic = self._cpu_nic(node)
        collective_id, context, ordering_domain = self._collective_identity(node)
        op = node["op"]
        algorithm = node.get("algorithm", "direct" if op == "alltoall" else "ring")
        protocol = self._protocol(node)
        descriptor = {
            "collective_id": collective_id,
            "node_id": node["id"],
            "type": op,
            "algorithm": algorithm,
            "ranks": ranks,
            "context": context,
            "ordering_domain": ordering_domain,
            "protocol": protocol,
        }
        if op == "alltoall":
            if algorithm != "direct":
                raise ValueError(f"node {node['id']}: Atlas supports only direct AllToAll")
            if protocol != "simple":
                raise ValueError(f"node {node['id']}: Atlas AllToAll always lowers with Simple protocol")
            spans = self._lower_alltoall(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes_per_peer"] = node.get("bytes_per_peer")
        elif op == "allreduce" and algorithm == "ring":
            spans = self._lower_ring_allreduce(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
        elif op == "allreduce" and algorithm == "tree":
            spans = self._lower_tree_allreduce(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
        elif algorithm != "ring":
            raise ValueError(
                f"node {node['id']}: Atlas implements {op} only for ring; tree lowering is not implemented"
            )
        elif op == "allgather":
            spans = self._lower_ring_allgather(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes_per_rank"] = node.get("bytes_per_rank")
        elif op == "reduce_scatter":
            spans = self._lower_ring_reduce_scatter(
                node, ranks, cpu, nic, collective_id, context, ordering_domain
            )
            descriptor["bytes"] = node.get("bytes")
        elif op == "reduce":
            spans = self._lower_ring_reduce(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
            descriptor["root"] = self._root(node, ranks)
        elif op == "broadcast":
            spans = self._lower_ring_broadcast(node, ranks, cpu, nic, collective_id, context, ordering_domain)
            descriptor["bytes"] = node.get("bytes")
            descriptor["root"] = self._root(node, ranks)
        else:
            raise AssertionError(f"unhandled collective op: {op}")
        self.collectives.append(descriptor)
        return spans

    def _compile_flow_delay(self, node: dict[str, Any]) -> dict[int, Span]:
        flow_id = as_nonnegative_int(node.get("flow_id"), f"node {node['id']}.flow_id")
        flow = self.flows.get(flow_id)
        if flow is None:
            raise ValueError(f"node {node['id']}: unknown or not-yet-compiled flow_id {flow_id}")
        if "post_delay" in flow:
            raise ValueError(f"node {node['id']}: flow {flow_id} already has a flow_delay")
        recv_rank = flow["recv_task"]["rank"]
        local_node = {**node, "ranks": [recv_rank]}
        spans = self._compile_local_calc(local_node)
        delay_task = spans[recv_rank].start
        recv_task = flow["recv_task"]["task_id"]
        graph = self.graphs[recv_rank]

        # Put the delay directly on the receive-completion edge: every
        # existing local successor now waits for the delay instead of the
        # recv task, while the delay itself waits for that recv task.
        for task in graph.tasks[:-1]:
            if recv_task in task.dependencies:
                task.dependencies.remove(recv_task)
                task.dependencies.add(delay_task)
        graph.require(delay_task, recv_task)
        flow["post_delay"] = {
            "node_id": node["id"],
            "duration_ns": graph.tasks[delay_task].size,
            "rank": recv_rank,
            "task_id": delay_task,
        }
        flow_node_id = flow.get("dag_node_id")
        if flow_node_id in self.spans:
            original = self.spans[flow_node_id][recv_rank]
            self.spans[flow_node_id][recv_rank] = Span(original.start, delay_task)
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

    def _add_collective_flow(
        self,
        node: dict[str, Any],
        src: int,
        dst: int,
        payload_bytes: int,
        *,
        cpu: int,
        nic: int,
        context: int,
        metadata: dict[str, Any],
    ) -> FlowHandle:
        wire_bytes = self._wire_size(node, payload_bytes)
        enriched = {
            **metadata,
            "protocol": self._protocol(node),
            "payload_bytes": payload_bytes,
            "wire_bytes": wire_bytes,
        }
        return self._add_flow(
            src,
            dst,
            wire_bytes,
            cpu=cpu,
            nic=nic,
            context=context,
            metadata=enriched,
        )

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
                    handles[(src, dst, chunk)] = self._add_collective_flow(
                        node,
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
                    reduce_flows[(step, src)] = self._add_collective_flow(
                        node,
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
                    gather_flows[(step, src)] = self._add_collective_flow(
                        node,
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

    def _lower_ring_allgather(
        self,
        node: dict[str, Any],
        ranks: list[int],
        cpu: int,
        nic: int,
        collective_id: int,
        context: int,
        ordering_domain: int,
    ) -> dict[int, Span]:
        bytes_per_rank = as_nonnegative_int(
            node.get("bytes_per_rank"), f"node {node['id']}.bytes_per_rank"
        )
        channels = as_nonnegative_int(node.get("channels", 1), f"node {node['id']}.channels", positive=True)
        orders = self._ring_orders(node, ranks, channels)
        channel_sizes = _split_bytes(bytes_per_rank, channels)
        maximum = as_nonnegative_int(
            self._value(node, "chunk_bytes", max(1, bytes_per_rank)), "chunk_bytes", positive=True
        )
        in_place = node.get("in_place", True)
        if not isinstance(in_place, bool):
            raise ValueError(f"node {node['id']}.in_place must be boolean")
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}

        chunk_base = 0
        for channel, (order, channel_size) in enumerate(zip(orders, channel_sizes)):
            n = len(order)
            pieces = _chunk_bytes(channel_size, maximum)
            for piece, piece_size in enumerate(pieces):
                flows: dict[tuple[int, int], FlowHandle] = {}
                for step in range(n - 1):
                    for index, src in enumerate(order):
                        dst = order[(index + 1) % n]
                        owner = (index - step) % n
                        flows[(step, src)] = self._add_collective_flow(
                            node,
                            src,
                            dst,
                            piece_size,
                            cpu=cpu,
                            nic=nic,
                            context=context,
                            metadata=self._flow_meta(
                                node,
                                collective_id,
                                "ring",
                                "allgather",
                                chunk_base + piece * n + owner,
                                step,
                                channel,
                                ordering_domain,
                                owner_rank=order[owner],
                                piece=piece,
                                in_place=in_place,
                            ),
                        )

                for index, rank in enumerate(order):
                    prev_rank = order[(index - 1) % n]
                    first_send = flows[(0, rank)].send_task
                    if in_place:
                        self.graphs[rank].require(first_send, spans[rank].start)
                    else:
                        initial_copy = self.graphs[rank].add_calc(
                            self._cost(node, "copy", self.graphs[rank].tasks[first_send].size),
                            cpu=cpu,
                            label=f"{node['id']}:ch{channel}:piece{piece}:initial-copy",
                        )
                        self.graphs[rank].require(initial_copy, spans[rank].start)
                        self.graphs[rank].require(first_send, initial_copy)
                    for step in range(n - 1):
                        incoming = flows[(step, prev_rank)]
                        self.graphs[rank].require(incoming.recv_task, spans[rank].start)
                        if step < n - 2:
                            size = self.graphs[rank].tasks[incoming.recv_task].size
                            copy = self.graphs[rank].add_calc(
                                self._cost(node, "copy", size),
                                cpu=cpu,
                                label=f"{node['id']}:ch{channel}:piece{piece}:copy:{step}",
                            )
                            self.graphs[rank].require(copy, incoming.recv_task)
                            self.graphs[rank].require(flows[(step + 1, rank)].send_task, copy)
            chunk_base += n * len(pieces)
        self._finalize_collective(spans, ranks, first_internal)
        return spans

    def _lower_ring_reduce_scatter(
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
            flows: dict[tuple[int, int], FlowHandle] = {}
            for step in range(n - 1):
                for index, src in enumerate(order):
                    dst = order[(index + 1) % n]
                    chunk = (index - step - 1) % n
                    flows[(step, src)] = self._add_collective_flow(
                        node,
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
                            owner_rank=order[chunk],
                            chunk_bytes=chunk_sizes[chunk],
                        ),
                    )
            for index, rank in enumerate(order):
                prev_rank = order[(index - 1) % n]
                self.graphs[rank].require(flows[(0, rank)].send_task, spans[rank].start)
                for step in range(n - 1):
                    incoming = flows[(step, prev_rank)]
                    self.graphs[rank].require(incoming.recv_task, spans[rank].start)
                    size = self.graphs[rank].tasks[incoming.recv_task].size
                    duration = self._cost(node, "reduction", size)
                    if step == n - 2:
                        duration += self._cost(node, "copy", size)
                    reduce_calc = self.graphs[rank].add_calc(
                        duration,
                        cpu=cpu,
                        label=f"{node['id']}:ch{channel}:reduce:{step}",
                    )
                    self.graphs[rank].require(reduce_calc, incoming.recv_task)
                    if step < n - 2:
                        self.graphs[rank].require(flows[(step + 1, rank)].send_task, reduce_calc)
        self._finalize_collective(spans, ranks, first_internal)
        return spans

    def _lower_ring_reduce(
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
        root = self._root(node, ranks)
        channels = as_nonnegative_int(node.get("channels", 1), f"node {node['id']}.channels", positive=True)
        orders = self._ring_orders(node, ranks, channels)
        if any(root not in order for order in orders):
            raise ValueError(f"node {node['id']}: root is missing from a ring order")
        channel_sizes = _split_bytes(total, channels)
        maximum = as_nonnegative_int(
            self._value(node, "chunk_bytes", max(1, total)), "chunk_bytes", positive=True
        )
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}

        chunk = 0
        for channel, (order, channel_size) in enumerate(zip(orders, channel_sizes)):
            n = len(order)
            root_index = order.index(root)
            for piece, piece_size in enumerate(_chunk_bytes(channel_size, maximum)):
                edges: list[FlowHandle] = []
                for step in range(n - 1):
                    src = order[(root_index + 1 + step) % n]
                    dst = order[(root_index + 2 + step) % n]
                    edges.append(
                        self._add_collective_flow(
                            node,
                            src,
                            dst,
                            piece_size,
                            cpu=cpu,
                            nic=nic,
                            context=context,
                            metadata=self._flow_meta(
                                node,
                                collective_id,
                                "ring",
                                "reduce",
                                chunk,
                                step,
                                channel,
                                ordering_domain,
                                root=root,
                                piece=piece,
                            ),
                        )
                    )
                self.graphs[edges[0].src].require(edges[0].send_task, spans[edges[0].src].start)
                for step, handle in enumerate(edges):
                    rank = handle.dst
                    self.graphs[rank].require(handle.recv_task, spans[rank].start)
                    size = self.graphs[rank].tasks[handle.recv_task].size
                    duration = self._cost(node, "reduction", size)
                    if rank == root:
                        duration += self._cost(node, "copy", size)
                    reduce_calc = self.graphs[rank].add_calc(
                        duration,
                        cpu=cpu,
                        label=f"{node['id']}:ch{channel}:chunk{chunk}:reduce:{step}",
                    )
                    self.graphs[rank].require(reduce_calc, handle.recv_task)
                    if step + 1 < len(edges):
                        self.graphs[rank].require(edges[step + 1].send_task, reduce_calc)
                chunk += 1
        self._finalize_collective(spans, ranks, first_internal)
        return spans

    def _lower_ring_broadcast(
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
        root = self._root(node, ranks)
        channels = as_nonnegative_int(node.get("channels", 1), f"node {node['id']}.channels", positive=True)
        orders = self._ring_orders(node, ranks, channels)
        channel_sizes = _split_bytes(total, channels)
        maximum = as_nonnegative_int(
            self._value(node, "chunk_bytes", max(1, total)), "chunk_bytes", positive=True
        )
        spans = self._new_anchors(node["id"], ranks, cpu)
        first_internal = {rank: len(self.graphs[rank].tasks) for rank in ranks}

        chunk = 0
        for channel, (order, channel_size) in enumerate(zip(orders, channel_sizes)):
            n = len(order)
            root_index = order.index(root)
            for piece, piece_size in enumerate(_chunk_bytes(channel_size, maximum)):
                edges: list[FlowHandle] = []
                for step in range(n - 1):
                    src = order[(root_index + step) % n]
                    dst = order[(root_index + step + 1) % n]
                    edges.append(
                        self._add_collective_flow(
                            node,
                            src,
                            dst,
                            piece_size,
                            cpu=cpu,
                            nic=nic,
                            context=context,
                            metadata=self._flow_meta(
                                node,
                                collective_id,
                                "ring",
                                "broadcast",
                                chunk,
                                step,
                                channel,
                                ordering_domain,
                                root=root,
                                piece=piece,
                            ),
                        )
                    )
                self.graphs[root].require(edges[0].send_task, spans[root].start)
                for step, handle in enumerate(edges):
                    rank = handle.dst
                    self.graphs[rank].require(handle.recv_task, spans[rank].start)
                    if step + 1 < len(edges):
                        size = self.graphs[rank].tasks[handle.recv_task].size
                        copy = self.graphs[rank].add_calc(
                            self._cost(node, "copy", size),
                            cpu=cpu,
                            label=f"{node['id']}:ch{channel}:chunk{chunk}:copy:{step}",
                        )
                        self.graphs[rank].require(copy, handle.recv_task)
                        self.graphs[rank].require(edges[step + 1].send_task, copy)
                chunk += 1
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
                upward[rank] = self._add_collective_flow(
                    node,
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
                downward[rank] = self._add_collective_flow(
                    node,
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

