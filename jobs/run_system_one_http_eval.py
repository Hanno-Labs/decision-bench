#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets>=4.0,<5",
#   "httpx[http2]>=0.28,<1",
#   "pyarrow>=21,<22",
#   "pydantic>=2.11,<3",
#   "typer>=0.16,<1",
# ]
# ///
"""Portable harness for a DecisionBench SystemOne HTTP evaluation."""

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
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default="xor")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--serving-bundle-sha256", required=True)
    parser.add_argument("--inference-image", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-successful-rows", type=int)
    parser.add_argument("--expected-error-rows", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_spec = args.task_spec or args.source_dir / "task_specs/decisionbench-dev.toml"
    command = [
        sys.executable,
        "-m",
        "decision_bench.cli",
        "run-system-one-http",
        str(task_spec),
        str(args.output_dir),
        "--project-root",
        str(args.source_dir),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--model-repo",
        args.model_repo,
        "--model-revision",
        args.model_revision,
        "--serving-bundle-sha256",
        args.serving_bundle_sha256,
        "--inference-image",
        args.inference_image,
        "--concurrency",
        str(args.concurrency),
    ]
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
        if sha256_file(args.output_dir / relative) != expected:
            raise RuntimeError(f"durable artifact hash mismatch: {relative}")
    requested_rows = int(summary["requested_rows"])
    successful_rows = int(summary["successful_rows"])
    error_rows = int(summary["error_rows"])
    if successful_rows + error_rows != requested_rows:
        raise RuntimeError("DecisionBench recorded-row gate failed")
    if args.expected_rows is not None and requested_rows != args.expected_rows:
        raise RuntimeError("DecisionBench row-count gate failed")
    if args.expected_rows is None and not args.smoke:
        raise RuntimeError("full DecisionBench runs require --expected-rows")
    if args.expected_successful_rows is not None:
        if successful_rows != args.expected_successful_rows:
            raise RuntimeError("DecisionBench successful-row gate failed")
        if args.expected_error_rows is None or error_rows != args.expected_error_rows:
            raise RuntimeError("DecisionBench error-row gate failed")
    elif not args.smoke:
        raise RuntimeError("full DecisionBench runs require expected success and error counts")
    if args.smoke and requested_rows <= 0:
        raise RuntimeError("DecisionBench smoke selected no rows")
    print(
        "DECISION_BENCH_SYSTEM_ONE_HTTP_COMPLETE=1 "
        f"rows={successful_rows} errors={error_rows} "
        f"accuracy={summary['metrics']['overall']['accuracy']:.9f} "
        f"ece={summary['metrics']['overall']['expected_calibration_error']:.9f} "
        f"manifest_sha256={sha256_file(args.output_dir / 'manifest.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
