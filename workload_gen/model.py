from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class OpType(IntEnum):
    SEND = 1
    RECV = 2
    CALC = 3


@dataclass
class Task:
    id: int
    op_type: OpType
    size: int
    peer: int = 0
    tag: int = 0
    cpu: int = 0
    nic: int = 0
    dependencies: set[int] = field(default_factory=set)
    label: str | None = None


@dataclass
class Span:
    start: int
    end: int


class RankGraph:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.tasks: list[Task] = []

    def add_task(
        self,
        op_type: OpType,
        size: int,
        *,
        peer: int = 0,
        tag: int = 0,
        cpu: int = 0,
        nic: int = 0,
        label: str | None = None,
    ) -> int:
        if size < 0:
            raise ValueError(f"rank {self.rank}: task size/duration cannot be negative")
        if not 0 <= cpu <= 255 or not 0 <= nic <= 255:
            raise ValueError("cpu and nic must fit uint8 (0..255)")
        if not 0 <= tag <= 0xFFFFFFFF:
            raise ValueError("tag must fit uint32 (0..4294967295)")
        task_id = len(self.tasks)
        self.tasks.append(Task(task_id, op_type, size, peer, tag, cpu, nic, label=label))
        return task_id

    def add_calc(self, duration_ns: int, *, cpu: int = 0, label: str | None = None) -> int:
        return self.add_task(OpType.CALC, duration_ns, cpu=cpu, label=label)

    def add_send(
        self, peer: int, size: int, tag: int, *, cpu: int = 0, nic: int = 0, label: str | None = None
    ) -> int:
        return self.add_task(OpType.SEND, size, peer=peer, tag=tag, cpu=cpu, nic=nic, label=label)

    def add_recv(
        self, peer: int, size: int, tag: int, *, cpu: int = 0, nic: int = 0, label: str | None = None
    ) -> int:
        return self.add_task(OpType.RECV, size, peer=peer, tag=tag, cpu=cpu, nic=nic, label=label)

    def require(self, task_id: int, predecessor_id: int) -> None:
        if task_id == predecessor_id:
            raise ValueError(f"rank {self.rank}: task {task_id} cannot depend on itself")
        self.tasks[task_id].dependencies.add(predecessor_id)

    def anchor_span(self, name: str, *, cpu: int = 0) -> Span:
        start = self.add_calc(0, cpu=cpu, label=f"{name}:start")
        end = self.add_calc(0, cpu=cpu, label=f"{name}:end")
        return Span(start, end)


@dataclass
class Workload:
    name: str
    source_kind: str
    ranks: list[RankGraph]
    flows: dict[int, dict[str, Any]] = field(default_factory=dict)
    collectives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return len(self.ranks)

