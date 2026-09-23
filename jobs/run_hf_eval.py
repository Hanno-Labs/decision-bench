#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.15.0",
#   "datasets>=4.0,<5",
#   "httpx[http2]>=0.28,<1",
#   "peft==0.21.0",
#   "pillow==12.3.0",
#   "pyarrow>=21,<22",
#   "pydantic>=2.11,<3",
#   "sentencepiece==0.2.2",
#   "torch==2.8.0",
#   "transformers==5.17.0",
#   "typer>=0.16,<1",
# ]
# ///
"""Portable HF Job harness for a DecisionBench local-model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--task-spec",
        type=Path,
        help="Task spec to evaluate; defaults to the development suite.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=("bosun", "cua-s1", "nimble", "nanojev", "openjev", "system-one"),
        default="bosun",
    )
    parser.add_argument("--model-repo")
    parser.add_argument("--model-revision")
    parser.add_argument("--base-revision")
    parser.add_argument("--expected-weights-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-characters-per-batch", type=int, default=1_048_576)
    parser.add_argument("--expected-successful-rows", type=int)
    parser.add_argument("--expected-error-rows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_spec = args.task_spec or args.source_dir / "task_specs/decisionbench-dev.toml"
    command_name = (
        "run-nimble-hf"
        if args.model_type == "nimble"
        else "run-public-hf"
        if args.model_type in {"cua-s1", "nanojev", "openjev", "system-one"}
        else "run-hf"
    )
    command = [
        sys.executable,
        "-m",
        "decision_bench.cli",
        command_name,
        str(task_spec),
        str(args.model_dir),
        str(args.output_dir),
        "--project-root",
        str(args.source_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-prompt-characters-per-batch",
        str(args.max_prompt_characters_per_batch),
    ]
    if args.model_type in {"cua-s1", "nanojev", "openjev", "system-one"}:
        if not args.model_repo or not args.model_revision:
            raise ValueError("public HF models require --model-repo and --model-revision")
        command.extend(
            [
                "--model-type",
                args.model_type,
                "--model-repo",
                args.model_repo,
                "--model-revision",
                args.model_revision,
            ]
        )
        if args.base_revision:
            command.extend(["--base-revision", args.base_revision])
        if args.expected_weights_sha256:
            command.extend(
                ["--expected-weights-sha256", args.expected_weights_sha256]
            )
    if args.smoke:
        command.append("--smoke")
    environment = dict(os.environ)
    source_path = str(args.source_dir / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else f"{source_path}:{environment['PYTHONPATH']}"
    )
    subprocess.run(command, check=True, env=environment)
    summary = read_object(args.output_dir / "summary.json")
    manifest = read_object(args.output_dir / "manifest.json")
    files = cast(dict[str, str], manifest["files"])
    for relative, expected in files.items():
        actual = sha256_file(args.output_dir / relative)
        if actual != expected:
            raise RuntimeError(f"durable artifact hash mismatch: {relative}")
    requested_rows = int(summary["requested_rows"])
    if args.expected_rows is not None and requested_rows != args.expected_rows:
        raise RuntimeError("DecisionBench row-count gate failed")
    if args.expected_rows is None and not args.smoke:
        raise RuntimeError("full DecisionBench runs require --expected-rows")
    if args.smoke and requested_rows <= 0:
        raise RuntimeError("DecisionBench smoke selected no rows")
    if args.model_type in {"cua-s1", "nimble", "nanojev", "openjev", "system-one"}:
        if summary["successful_rows"] + summary["error_rows"] != requested_rows:
            raise RuntimeError("DecisionBench recorded-row gate failed")
        if (
            args.expected_successful_rows is not None
            and summary["successful_rows"] != args.expected_successful_rows
        ):
            raise RuntimeError("DecisionBench successful-row gate failed")
        if (
            args.expected_successful_rows is not None
            and summary["error_rows"] != args.expected_error_rows
        ):
            raise RuntimeError("DecisionBench error-row gate failed")
    else:
        expected_successful_rows = args.expected_successful_rows or requested_rows
        if summary["successful_rows"] != expected_successful_rows:
            raise RuntimeError("DecisionBench successful-row gate failed")
        if summary["error_rows"] != args.expected_error_rows:
            raise RuntimeError("DecisionBench error-row gate failed")
    print(
        f"DECISION_BENCH_{args.model_type.upper().replace('-', '_')}_COMPLETE=1 "
        f"rows={summary['successful_rows']} "
        f"errors={summary['error_rows']} "
        f"accuracy={summary['metrics']['overall']['accuracy']:.9f} "
        f"ece={summary['metrics']['overall']['expected_calibration_error']:.9f} "
        f"manifest_sha256={sha256_file(args.output_dir / 'manifest.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
