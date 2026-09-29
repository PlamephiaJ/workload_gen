#!/usr/bin/env python3
"""Compare representative lowering traffic against Atlas nccl_generator_v2.

This intentionally compares the p2p transfer multiset (src, dst, bytes), not
task IDs or Atlas' profiling-derived local calc durations.  It imports Atlas
only for validation; the workload generator itself has no Atlas dependency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from workload_gen.compiler import compile_spec  # noqa: E402


SEND = re.compile(r"^l\d+: send (\d+)b to (\d+) tag \d+")


def _ours(spec: dict) -> Counter[tuple[int, int, int]]:
    workload = compile_spec(spec)
    return Counter((flow["src"], flow["dst"], flow["bytes"]) for flow in workload.flows.values())


def _reset_goal(goal) -> None:
    goal.GoalOpAtom.task_id_for_rank.clear()
    goal.GoalSend.send_message_id.clear()
    goal.GoalRecv.recv_message_id.clear()


def _reference_lines(operation_by_rank: list, mapping: dict, goal) -> Counter[tuple[int, int, int]]:
    _reset_goal(goal)
    transfers: Counter[tuple[int, int, int]] = Counter()
    for rank, operation in enumerate(operation_by_rank):
        goal_op, _ = operation.to_goal(mapping, 0, 0)
        for line in goal_op.generate_lines():
            match = SEND.match(line)
            if match:
                transfers[(rank, int(match.group(2)), int(match.group(1)))] += 1
    return transfers


def compare(atlas_root: Path) -> dict[str, dict[str, int]]:
    module_dir = atlas_root / "goal_gen" / "ai" / "nccl_generator_v2"
    if not module_dir.is_dir():
        raise ValueError(f"Atlas nccl_generator_v2 not found under {atlas_root}")
    sys.path.insert(0, str(module_dir))
    import goal  # type: ignore
    import nccl_primitives  # type: ignore
    from nccl_comm import (  # type: ignore
        AllGather,
        AllReduce,
        AllToAll,
        Broadcast,
        CollAlgo,
        CollChnlInfo,
        CollInfo,
        Communicator,
        NCCLProto,
        Reduce,
        ReduceScatter,
    )

    nccl_primitives.init_generation_flags(True, False, False)
    ranks = [0, 1, 2, 3]
    gpu_ids = [(f"node-{rank}", rank) for rank in ranks]
    mapping = {gpu_id: rank for rank, gpu_id in enumerate(gpu_ids)}
    comm = Communicator("validation", gpu_ids[:])
    for rank in ranks:
        comm.add_ring_topo(rank, (rank - 1) % 4, (rank + 1) % 4)
    parents = {0: None, 1: 0, 2: 0, 3: 1}
    children = {rank: [] for rank in ranks}
    for rank, parent in parents.items():
        if parent is not None:
            children[parent].append(rank)
    for rank in ranks:
        comm.add_tree_topo(rank, -1 if parents[rank] is None else parents[rank], children[rank])

    base = {
        "schema_version": 1,
        "kind": "collective",
        "world_size": 4,
        "defaults": {"reduction_overhead_ns": 0, "copy_overhead_ns": 0},
    }
    cases: list[tuple[str, dict, list]] = []

    alltoall_spec = {
        **base,
        "name": "compare_alltoall",
        "nodes": [
            {
                "id": "c",
                "op": "alltoall",
                "algorithm": "direct",
                "ranks": ranks,
                "bytes_per_peer": 1024,
                "chunk_bytes": 256,
                "collective_id": 1,
            }
        ],
    }
    alltoall_ref = [AllToAll(gpu_id, 1024, comm, 256, 1) for gpu_id in gpu_ids]
    cases.append(("alltoall_direct", alltoall_spec, alltoall_ref))

    ring_info = CollInfo(0, 0, CollAlgo.RING, NCCLProto.SIMPLE, 4096, 1, 1, 1, 1024)
    ring_channel = CollChnlInfo(4096, 1024, 4096, 1024, 0, 0, 0)
    ring_spec = {
        **base,
        "name": "compare_ring",
        "nodes": [
            {
                "id": "c",
                "op": "allreduce",
                "algorithm": "ring",
                "ranks": ranks,
                "bytes": 4096,
                "collective_id": 2,
            }
        ],
    }
    ring_ref = [AllReduce(gpu_id, comm, ring_info, [ring_channel], 2) for gpu_id in gpu_ids]
    cases.append(("allreduce_ring", ring_spec, ring_ref))

    tree_info = CollInfo(0, 0, CollAlgo.TREE, NCCLProto.SIMPLE, 2048, 1, 1, 1, 2048)
    tree_channel = CollChnlInfo(2048, 2048, 2048, 2048, 0, 0, 0)
    tree_spec = {
        **base,
        "name": "compare_tree",
        "nodes": [
            {
                "id": "c",
                "op": "allreduce",
                "algorithm": "tree",
                "ranks": ranks,
                "bytes": 2048,
                "chunk_bytes": 2048,
                "tree_parent": {str(rank): parent for rank, parent in parents.items()},
                "collective_id": 3,
            }
        ],
    }
    tree_ref = [AllReduce(gpu_id, comm, tree_info, [tree_channel], 3) for gpu_id in gpu_ids]
    cases.append(("allreduce_tree", tree_spec, tree_ref))

    allgather_info = CollInfo(0, 0, CollAlgo.RING, NCCLProto.SIMPLE, 1024, 1, 1, 1, 1024)
    allgather_spec = {
        **base,
        "name": "compare_allgather",
        "nodes": [
            {
                "id": "c",
                "op": "allgather",
                "algorithm": "ring",
                "protocol": "simple",
                "ranks": ranks,
                "bytes_per_rank": 1024,
                "collective_id": 4,
            }
        ],
    }
    allgather_ref = []
    for rank, gpu_id in enumerate(gpu_ids):
        channel = CollChnlInfo(1024, 1024, 1024, 1024, 0, rank * 1024, 0)
        allgather_ref.append(AllGather(gpu_id, comm, allgather_info, [channel], 4))
    cases.append(("allgather_ring", allgather_spec, allgather_ref))

    rs_info = CollInfo(0, 0, CollAlgo.RING, NCCLProto.SIMPLE, 4096, 1, 1, 1, 1024)
    rs_channel = CollChnlInfo(4096, 1024, 1024, 1024, 0, 0, 0)
    rs_spec = {
        **base,
        "name": "compare_reduce_scatter",
        "nodes": [
            {
                "id": "c",
                "op": "reduce_scatter",
                "algorithm": "ring",
                "protocol": "simple",
                "ranks": ranks,
                "bytes": 4096,
                "collective_id": 5,
            }
        ],
    }
    rs_ref = [ReduceScatter(gpu_id, comm, rs_info, [rs_channel], 5) for gpu_id in gpu_ids]
    cases.append(("reduce_scatter_ring", rs_spec, rs_ref))

    reduce_info = CollInfo(0, 0, CollAlgo.RING, NCCLProto.SIMPLE, 2048, 1, 1, 1, 2048)
    reduce_channel = CollChnlInfo(2048, 2048, 2048, 2048, 0, 0, 0)
    reduce_spec = {
        **base,
        "name": "compare_reduce",
        "nodes": [
            {
                "id": "c",
                "op": "reduce",
                "algorithm": "ring",
                "protocol": "simple",
                "ranks": ranks,
                "bytes": 2048,
                "root": 0,
                "collective_id": 6,
            }
        ],
    }
    reduce_ref = [Reduce(gpu_id, comm, reduce_info, [reduce_channel], 6) for gpu_id in gpu_ids]
    cases.append(("reduce_ring", reduce_spec, reduce_ref))

    broadcast_info = CollInfo(0, 0, CollAlgo.RING, NCCLProto.SIMPLE, 2048, 1, 1, 1, 2048)
    broadcast_channel = CollChnlInfo(2048, 2048, 2048, 2048, 0, 0, 0)
    broadcast_spec = {
        **base,
        "name": "compare_broadcast",
        "nodes": [
            {
                "id": "c",
                "op": "broadcast",
                "algorithm": "ring",
                "protocol": "simple",
                "ranks": ranks,
                "bytes": 2048,
                "root": 0,
                "collective_id": 7,
            }
        ],
    }
    broadcast_ref = [Broadcast(gpu_id, comm, broadcast_info, [broadcast_channel], 7) for gpu_id in gpu_ids]
    cases.append(("broadcast_ring", broadcast_spec, broadcast_ref))

    for protocol_name, atlas_protocol, collective_id in [
        ("ll", NCCLProto.LL, 8),
        ("ll128", NCCLProto.LL128, 9),
    ]:
        protocol_info = CollInfo(0, 0, CollAlgo.TREE, atlas_protocol, 2049, 1, 1, 1, 2049)
        protocol_channel = CollChnlInfo(2049, 2049, 2049, 2049, 0, 0, 0)
        protocol_spec = {
            **base,
            "name": f"compare_tree_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "allreduce",
                    "algorithm": "tree",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes": 2049,
                    "chunk_bytes": 2049,
                    "tree_parent": {str(rank): parent for rank, parent in parents.items()},
                    "collective_id": collective_id,
                }
            ],
        }
        protocol_ref = [
            AllReduce(gpu_id, comm, protocol_info, [protocol_channel], collective_id)
            for gpu_id in gpu_ids
        ]
        cases.append((f"allreduce_tree_{protocol_name}", protocol_spec, protocol_ref))

        ring_protocol_info = CollInfo(0, 0, CollAlgo.RING, atlas_protocol, 4100, 1, 1, 1, 1025)
        ring_protocol_channel = CollChnlInfo(4100, 1025, 4100, 1025, 0, 0, 0)
        ring_protocol_spec = {
            **base,
            "name": f"compare_ring_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "allreduce",
                    "algorithm": "ring",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes": 4100,
                    "collective_id": collective_id + 10,
                }
            ],
        }
        ring_protocol_ref = [
            AllReduce(gpu_id, comm, ring_protocol_info, [ring_protocol_channel], collective_id + 10)
            for gpu_id in gpu_ids
        ]
        cases.append((f"allreduce_ring_{protocol_name}", ring_protocol_spec, ring_protocol_ref))

        ag_protocol_info = CollInfo(0, 0, CollAlgo.RING, atlas_protocol, 1025, 1, 1, 1, 1025)
        ag_protocol_spec = {
            **base,
            "name": f"compare_allgather_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "allgather",
                    "algorithm": "ring",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes_per_rank": 1025,
                    "collective_id": collective_id + 20,
                }
            ],
        }
        ag_protocol_ref = []
        for rank, gpu_id in enumerate(gpu_ids):
            channel = CollChnlInfo(1025, 1025, 1025, 1025, 0, rank * 1025, 0)
            ag_protocol_ref.append(
                AllGather(gpu_id, comm, ag_protocol_info, [channel], collective_id + 20)
            )
        cases.append((f"allgather_ring_{protocol_name}", ag_protocol_spec, ag_protocol_ref))

        rs_protocol_info = CollInfo(0, 0, CollAlgo.RING, atlas_protocol, 4100, 1, 1, 1, 1025)
        rs_protocol_channel = CollChnlInfo(4100, 1025, 1025, 1025, 0, 0, 0)
        rs_protocol_spec = {
            **base,
            "name": f"compare_reduce_scatter_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "reduce_scatter",
                    "algorithm": "ring",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes": 4100,
                    "collective_id": collective_id + 30,
                }
            ],
        }
        rs_protocol_ref = [
            ReduceScatter(gpu_id, comm, rs_protocol_info, [rs_protocol_channel], collective_id + 30)
            for gpu_id in gpu_ids
        ]
        cases.append((f"reduce_scatter_ring_{protocol_name}", rs_protocol_spec, rs_protocol_ref))

        reduce_protocol_info = CollInfo(0, 0, CollAlgo.RING, atlas_protocol, 2049, 1, 1, 1, 2049)
        reduce_protocol_channel = CollChnlInfo(2049, 2049, 2049, 2049, 0, 0, 0)
        reduce_protocol_spec = {
            **base,
            "name": f"compare_reduce_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "reduce",
                    "algorithm": "ring",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes": 2049,
                    "root": 0,
                    "collective_id": collective_id + 40,
                }
            ],
        }
        reduce_protocol_ref = [
            Reduce(gpu_id, comm, reduce_protocol_info, [reduce_protocol_channel], collective_id + 40)
            for gpu_id in gpu_ids
        ]
        cases.append((f"reduce_ring_{protocol_name}", reduce_protocol_spec, reduce_protocol_ref))

        broadcast_protocol_info = CollInfo(
            0, 0, CollAlgo.RING, atlas_protocol, 2049, 1, 1, 1, 2049
        )
        broadcast_protocol_channel = CollChnlInfo(2049, 2049, 2049, 2049, 0, 0, 0)
        broadcast_protocol_spec = {
            **base,
            "name": f"compare_broadcast_{protocol_name}",
            "nodes": [
                {
                    "id": "c",
                    "op": "broadcast",
                    "algorithm": "ring",
                    "protocol": protocol_name,
                    "ranks": ranks,
                    "bytes": 2049,
                    "root": 0,
                    "collective_id": collective_id + 50,
                }
            ],
        }
        broadcast_protocol_ref = [
            Broadcast(
                gpu_id,
                comm,
                broadcast_protocol_info,
                [broadcast_protocol_channel],
                collective_id + 50,
            )
            for gpu_id in gpu_ids
        ]
        cases.append(
            (f"broadcast_ring_{protocol_name}", broadcast_protocol_spec, broadcast_protocol_ref)
        )

    report: dict[str, dict[str, int]] = {}
    for name, spec, reference_operations in cases:
        expected = _reference_lines(reference_operations, mapping, goal)
        actual = _ours(spec)
        if actual != expected:
            raise AssertionError(
                f"{name} lowering differs from Atlas\n"
                f"only ours: {actual - expected}\nonly Atlas: {expected - actual}"
            )
        report[name] = {"matched_transfers": sum(actual.values()), "unique_tuples": len(actual)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-root", type=Path, default=PROJECT_ROOT.parent / "atlahs", help="path to the Atlas/atlahs repository"
    )
    args = parser.parse_args()
    try:
        report = compare(args.atlas_root.resolve())
    except (AssertionError, ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

