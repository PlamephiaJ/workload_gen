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
from workload_gen.lowering import lower_collective_to_flow_spec
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
        self.assertEqual(len(workload.flows), 6)
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
                    {
                        "id": "transfer",
                        "op": "flow",
                        "flow_id": 7,
                        "src": 0,
                        "dst": 1,
                        "bytes": 32,
                    },
                    {
                        "id": "pause",
                        "op": "flow_delay",
                        "flow_id": 7,
                        "duration_ns": 75,
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
                        "op": "collective_delay",
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

    def test_flow_delay_rewires_recv_successors_by_flow_id(self) -> None:
        workload = compile_spec(
            {
                "schema_version": 1,
                "kind": "flow",
                "name": "internal_flow_delay",
                "world_size": 3,
                "nodes": [
                    {
                        "id": "exchange",
                        "op": "flow",
                        "flow_id": 41,
                        "src": 0,
                        "dst": 1,
                        "bytes": 32,
                    },
                    {
                        "id": "exchange_0_to_1_delay",
                        "op": "flow_delay",
                        "flow_id": 41,
                        "duration_ns": 2000,
                    },
                ],
            }
        )
        flow = workload.flows[41]
        delay_meta = flow["post_delay"]
        self.assertEqual(delay_meta["duration_ns"], 2000)
        self.assertEqual(delay_meta["rank"], 1)

        graph = workload.ranks[1]
        recv_task = flow["recv_task"]["task_id"]
        delay_task = delay_meta["task_id"]
        self.assertEqual(graph.tasks[delay_task].size, 2000)
        self.assertEqual(graph.tasks[delay_task].label, "exchange_0_to_1_delay")
        self.assertEqual(graph.tasks[delay_task].dependencies, {recv_task})
        self.assertFalse(
            any(recv_task in task.dependencies for task in graph.tasks if task.id != delay_task)
        )
        self.assertTrue(any(delay_task in task.dependencies for task in graph.tasks))

    def test_collective_lowering_exports_equivalent_flow_template(self) -> None:
        spec = load_spec(ROOT / "templates" / "exp2" / "exp2.a.1.collective.json")
        collective = compile_spec(spec)
        flow_spec = lower_collective_to_flow_spec(spec, collective)
        lowered = compile_spec(flow_spec)

        self.assertEqual(flow_spec["kind"], "flow")
        self.assertEqual(flow_spec["name"], "exp2.a.1.flow")
        self.assertEqual(set(lowered.flows), set(collective.flows))
        self.assertEqual(
            sum(len(graph.tasks) for graph in lowered.ranks),
            sum(len(graph.tasks) for graph in collective.ranks),
        )
        self.assertEqual(
            sum(len(task.dependencies) for graph in lowered.ranks for task in graph.tasks),
            sum(len(task.dependencies) for graph in collective.ranks for task in graph.tasks),
        )
        for flow_id, actual in lowered.flows.items():
            expected = collective.flows[flow_id]
            self.assertEqual(
                (actual["src"], actual["dst"], actual["bytes"], actual["tag"]),
                (expected["src"], expected["dst"], expected["bytes"], expected["tag"]),
            )

    def test_exp2_flow_perturbations_target_exact_ids(self) -> None:
        expected = {
            "exp2.c.1.flow.json": (35, "tree", 2000),
            "exp2.c.2.flow.json": (59, "direct", 2000),
            "exp2.c.3.flow.json": (22, "ring", 2000),
            "exp2.c.4.flow.json": (59, "direct", 5000),
        }
        for filename, (flow_id, algorithm, duration_ns) in expected.items():
            with self.subTest(template=filename):
                spec = load_spec(ROOT / "templates" / "exp2" / filename)
                self.assertEqual(sum(node.get("op") == "flow_delay" for node in spec["nodes"]), 1)
                self.assertFalse(any(node.get("op") == "collective_delay" for node in spec["nodes"]))
                workload = compile_spec(spec)
                delayed = [item for item in workload.flows.values() if "post_delay" in item]
                self.assertEqual(len(delayed), 1)
                self.assertEqual(delayed[0]["flow_id"], flow_id)
                self.assertEqual(delayed[0]["algorithm"], algorithm)
                self.assertEqual(delayed[0]["post_delay"]["duration_ns"], duration_ns)

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

    def test_delay_kinds_cannot_cross_dag_granularity(self) -> None:
        cases = [
            (
                "flow",
                {"id": "bad", "op": "collective_delay", "ranks": [0], "duration_ns": 1},
                "collective_delay is only valid in a collective DAG",
            ),
            (
                "collective",
                {"id": "bad", "op": "flow_delay", "flow_id": 0, "duration_ns": 1},
                "flow_delay is only valid in a flow DAG",
            ),
        ]
        for kind, node, message in cases:
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, message):
                    compile_spec(
                        {
                            "schema_version": 1,
                            "kind": kind,
                            "name": "bad_delay_kind",
                            "world_size": 2,
                            "nodes": [node],
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

