from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from . import __version__
from .compiler import compile_spec
from .goal_io import validate_goal_binary, write_goal_binary, write_goal_text
from .spec import load_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an HTSim GOAL binary from a collective-level or p2p-flow-level JSON DAG."
    )
    parser.add_argument("input", type=Path, help="input JSON template")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="root directory; a workload-name subdirectory is created (default: %(default)s)",
    )
    parser.add_argument("--no-goal-text", action="store_true", help="do not keep the readable workload.goal file")
    parser.add_argument("--validate-only", action="store_true", help="parse and compile input without writing output")
    parser.add_argument(
        "--verify-with-txt2bin",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="also convert workload.goal with Atlas txt2bin and require byte-identical output",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _find_txt2bin(value: str) -> Path:
    if value != "auto":
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"txt2bin does not exist: {path}")
        return path
    candidates = [
        PROJECT_ROOT.parent / "atlahs" / "sim" / "LogGOPSim" / "txt2bin",
        PROJECT_ROOT.parent / "HTSIM" / "htsim" / "sim" / "lgs" / "txt2bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("could not auto-locate txt2bin; pass its path explicitly")


def _verify_with_txt2bin(goal_path: Path, binary_path: Path, executable: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="workload-gen-") as temp_dir:
        reference = Path(temp_dir) / "reference.bin"
        result = subprocess.run(
            [str(executable), "-i", str(goal_path), "-o", str(reference)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ValueError(f"txt2bin verification failed ({result.returncode}): {detail}")
        actual = binary_path.read_bytes()
        expected = reference.read_bytes()
        if actual != expected:
            first = next((idx for idx, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)
            if first is None:
                first = min(len(actual), len(expected))
            raise ValueError(
                "native writer does not match txt2bin: "
                f"first difference at byte {first}, native={len(actual)} bytes, reference={len(expected)} bytes"
            )


def _write_metadata(
    spec: dict, workload, path: Path, stats: dict[str, int], *, goal_text_present: bool
) -> None:
    metadata = {
        "schema_version": 1,
        "generator": {"name": "htsim-workload-gen", "version": __version__},
        "workload": {
            "name": workload.name,
            "kind": workload.source_kind,
            "world_size": workload.world_size,
            "binary": "workload.bin",
            "goal_text": "workload.goal" if goal_text_present else None,
            "tasks": stats["tasks"],
            "dependency_edges": stats["edges"],
            "p2p_flows": stats["flows"],
        },
        "source_graph": {"defaults": spec.get("defaults", {}), "nodes": spec["nodes"]},
        "collectives": workload.collectives,
        "flows": {str(flow_id): workload.flows[flow_id] for flow_id in sorted(workload.flows)},
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.expanduser().resolve()
    spec = load_spec(input_path)
    workload = compile_spec(spec)
    if args.validate_only:
        return {
            "status": "valid",
            "name": workload.name,
            "kind": workload.source_kind,
            "ranks": workload.world_size,
        }

    output_dir = args.output_root.expanduser().resolve() / workload.name
    output_dir.mkdir(parents=True, exist_ok=True)
    goal_path = output_dir / "workload.goal"
    binary_path = output_dir / "workload.bin"
    write_goal_text(workload, goal_path)
    write_goal_binary(workload, binary_path)
    stats = validate_goal_binary(binary_path)

    verified_with: str | None = None
    if args.verify_with_txt2bin:
        txt2bin = _find_txt2bin(args.verify_with_txt2bin)
        _verify_with_txt2bin(goal_path, binary_path, txt2bin)
        verified_with = str(txt2bin)
    if args.no_goal_text:
        goal_path.unlink()

    metadata_path: Path | None = None
    if workload.source_kind == "collective":
        metadata_path = output_dir / "workload.meta.json"
        _write_metadata(spec, workload, metadata_path, stats, goal_text_present=not args.no_goal_text)

    return {
        "status": "generated",
        "name": workload.name,
        "kind": workload.source_kind,
        "output_dir": str(output_dir),
        "binary": str(binary_path),
        "goal_text": None if args.no_goal_text else str(goal_path),
        "metadata": str(metadata_path) if metadata_path else None,
        "verified_with_txt2bin": verified_with,
        **stats,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

