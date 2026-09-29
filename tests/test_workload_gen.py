from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from workload_gen.cli import _verify_with_txt2bin
from workload_gen.compiler import compile_spec
from workload_gen.goal_io import validate_goal_binary, write_goal_binary, write_goal_text
from workload_gen.model import OpType
from workload_gen.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]


class WorkloadGenerationTests(unittest.TestCase):
    def _generate(self, template: str):
        spec = load_spec(ROOT / "templates" / template)
        workload = compile_spec(spec)
        temp = tempfile.TemporaryDirectory()
        directory = Path(temp.name)
        goal = directory / "workload.goal"
        binary = directory / "workload.bin"
        write_goal_text(workload, goal)
        write_goal_binary(workload, binary)
        stats = validate_goal_binary(binary)
        return temp, spec, workload, goal, binary, stats

    def test_flow_template(self) -> None:
        temp, _, workload, _, _, stats = self._generate("flow_dag.json")
        self.addCleanup(temp.cleanup)
        self.assertEqual(workload.source_kind, "flow")
        self.assertEqual(workload.flows, {})
        self.assertEqual(stats["ranks"], 4)
        self.assertEqual(stats["flows"], 6)

    def test_collective_template_metadata(self) -> None:
        temp, _, workload, _, _, stats = self._generate("collective_dag.json")
        self.addCleanup(temp.cleanup)
        self.assertEqual(len(workload.collectives), 3)
        self.assertEqual(stats["flows"], 96)
        self.assertEqual(len(workload.flows), 96)
        phases = {flow["phase"] for flow in workload.flows.values()}
        self.assertEqual(phases, {"exchange", "reduce_scatter", "allgather", "reduce", "broadcast"})
        for flow in workload.flows.values():
            self.assertIn("collective_id", flow)
            self.assertIn("ordering_domain", flow)
            self.assertIn("send_task", flow)
            self.assertIn("recv_task", flow)

    def test_all_atlas_supported_collectives_template(self) -> None:
        temp, _, workload, _, _, stats = self._generate("collective_all_supported.json")
        self.addCleanup(temp.cleanup)
        matrix = {(item["type"], item["algorithm"]) for item in workload.collectives}
        self.assertEqual(
            matrix,
            {
                ("alltoall", "direct"),
                ("allreduce", "ring"),
                ("allreduce", "tree"),
                ("allgather", "ring"),
                ("reduce_scatter", "ring"),
                ("reduce", "ring"),
                ("broadcast", "ring"),
            },
        )
        self.assertEqual(stats["flows"], 72)
        self.assertEqual(len(workload.flows), 72)
        for flow in workload.flows.values():
            self.assertIn(flow["protocol"], {"simple", "ll", "ll128"})
            self.assertEqual(flow["bytes"], flow["wire_bytes"])
            if flow["protocol"] in {"ll", "ll128"}:
                self.assertEqual(flow["wire_bytes"] % 8, 0)

    def test_delay_nodes_lower_to_rank_local_calc(self) -> None:
        cases = [
            (
                "flow",
                [
                    {"id": "transfer", "op": "flow", "src": 0, "dst": 1, "bytes": 32},
                    {
                        "id": "pause",
                        "op": "delay",
                        "ranks": [1],
                        "duration_ns": 75,
                        "depends_on": ["transfer"],
                    },
                ],
                {1: 75},
            ),
            (
                "collective",
                [
                    {
                        "id": "exchange",
                        "op": "allreduce",
                        "algorithm": "ring",
                        "ranks": "all",
                        "bytes": 32,
                    },
                    {
                        "id": "pause",
                        "op": "delay",
                        "ranks": "all",
                        "duration_ns": {"0": 75, "1": 125},
                        "depends_on": ["exchange"],
                    },
                ],
                {0: 75, 1: 125},
            ),
        ]
        for kind, nodes, expected in cases:
            with self.subTest(kind=kind):
                workload = compile_spec(
                    {
                        "schema_version": 1,
                        "kind": kind,
                        "name": f"{kind}_delay",
                        "world_size": 2,
                        "nodes": nodes,
                    }
                )
                for rank, duration in expected.items():
                    tasks = [task for task in workload.ranks[rank].tasks if task.label == "pause"]
                    self.assertEqual(len(tasks), 1)
                    self.assertEqual(tasks[0].op_type, OpType.CALC)
                    self.assertEqual(tasks[0].size, duration)
                    self.assertTrue(tasks[0].dependencies)

                with tempfile.TemporaryDirectory() as temp_dir:
                    binary = Path(temp_dir) / "workload.bin"
                    write_goal_binary(workload, binary)
                    stats = validate_goal_binary(binary)
                    self.assertEqual(stats["ranks"], 2)

    def test_atlas_unsupported_combinations_are_rejected(self) -> None:
        base = {
            "schema_version": 1,
            "kind": "collective",
            "name": "unsupported",
            "world_size": 2,
        }
        with self.assertRaisesRegex(ValueError, "only for ring"):
            compile_spec(
                {
                    **base,
                    "nodes": [
                        {"id": "c", "op": "allgather", "algorithm": "tree", "bytes_per_rank": 8}
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "always lowers with Simple"):
            compile_spec(
                {
                    **base,
                    "nodes": [
                        {"id": "c", "op": "alltoall", "algorithm": "direct", "protocol": "ll", "bytes_per_peer": 8}
                    ],
                }
            )

    def test_rank_disjoint_dependency_is_rejected(self) -> None:
        spec = {
            "schema_version": 1,
            "kind": "flow",
            "name": "bad",
            "world_size": 2,
            "nodes": [
                {"id": "a", "op": "compute", "ranks": [0], "duration_ns": 1},
                {"id": "b", "op": "compute", "ranks": [1], "duration_ns": 1, "depends_on": ["a"]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "no shared rank"):
            compile_spec(spec)

    def test_native_binary_is_atlas_txt2bin_identical(self) -> None:
        executable = ROOT.parent / "atlahs" / "sim" / "LogGOPSim" / "txt2bin"
        if not executable.is_file():
            self.skipTest("Atlas txt2bin is not available")
        temp, _, _, goal, binary, _ = self._generate("flow_dag.json")
        self.addCleanup(temp.cleanup)
        _verify_with_txt2bin(goal, binary, executable)

    def test_reference_lowering_comparison(self) -> None:
        atlas = ROOT.parent / "atlahs"
        if not atlas.is_dir():
            self.skipTest("Atlas source is not available")
        if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("scipy") is None:
            self.skipTest("optional Atlas validation dependencies numpy/scipy are not available")
        result = subprocess.run(
            [str(ROOT / "tools" / "compare_atlas_lowering.py"), "--atlas-root", str(atlas)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["alltoall_direct"]["matched_transfers"], 48)
        self.assertEqual(report["allreduce_ring"]["matched_transfers"], 24)
        self.assertEqual(report["allreduce_tree"]["matched_transfers"], 6)
        self.assertEqual(
            set(report),
            {
                "alltoall_direct",
                "allreduce_ring",
                "allreduce_tree",
                "allgather_ring",
                "reduce_scatter_ring",
                "reduce_ring",
                "broadcast_ring",
                "allreduce_tree_ll",
                "allreduce_ring_ll",
                "allgather_ring_ll",
                "reduce_scatter_ring_ll",
                "reduce_ring_ll",
                "broadcast_ring_ll",
                "allreduce_tree_ll128",
                "allreduce_ring_ll128",
                "allgather_ring_ll128",
                "reduce_scatter_ring_ll128",
                "reduce_ring_ll128",
                "broadcast_ring_ll128",
            },
        )


if __name__ == "__main__":
    unittest.main()

