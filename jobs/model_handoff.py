"""Validate the reviewed model contract handed from an adapter PR to a run."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

FIELDS = frozenset(
    {
        "schema_version",
        "issue_number",
        "runner",
        "runner_model_type",
        "model_repo",
        "model_revision",
        "result_model_type",
        "adapter",
        "probability_source",
    }
)
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{1,79}\Z")
RUNNER_MODEL_TYPE = re.compile(r"[a-z][a-z0-9_-]{1,79}\Z")
MODEL_REPO = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z"
)
REVISION = re.compile(r"[a-f0-9]{40}\Z")
RESULT_MODEL_TYPES = frozenset({"decision-model", "language-model", "classifier"})


def validate_handoff(value: Any, issue_number: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Model handoff has unexpected fields")
    if value["schema_version"] != 1 or value["issue_number"] != issue_number:
        raise ValueError("Model handoff does not match the issue")
    if value["runner"] != "public-hf":
        raise ValueError("Issue automation currently requires the public-hf runner")
    if not isinstance(value["runner_model_type"], str) or not RUNNER_MODEL_TYPE.fullmatch(
        value["runner_model_type"]
    ):
        raise ValueError("Invalid runner model type in handoff")
    if not isinstance(value["result_model_type"], str) or value[
        "result_model_type"
    ] not in RESULT_MODEL_TYPES:
        raise ValueError("Result model type is not supported by stage-result")
    for field in ("adapter", "probability_source"):
        if not isinstance(value[field], str) or not IDENTIFIER.fullmatch(value[field]):
            raise ValueError(f"Invalid model handoff field: {field}")
    if not isinstance(value["model_repo"], str) or not MODEL_REPO.fullmatch(
        value["model_repo"]
    ):
        raise ValueError("Invalid model repo in handoff")
    if not isinstance(value["model_revision"], str) or not REVISION.fullmatch(
        value["model_revision"]
    ):
        raise ValueError("Model revision must be an immutable 40-character commit")
    return value


def read_handoff(repo: Path, issue_number: int) -> dict[str, Any]:
    path = repo / "jobs" / "model_requests" / f"{issue_number}.json"
    if not path.is_file():
        raise RuntimeError(f"Missing model handoff: {path.relative_to(repo)}")
    value = validate_handoff(json.loads(path.read_text()), issue_number)
    runner_source = ast.parse((repo / "jobs" / "run_hf_eval.py").read_text())
    runner_types: set[str] = set()
    for node in ast.walk(runner_source):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--model-type"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                runner_types.update(ast.literal_eval(keyword.value))

    cli_source = ast.parse((repo / "src" / "decision_bench" / "cli.py").read_text())
    cli_types: set[str] = set()
    for node in ast.walk(cli_source):
        if not isinstance(node, ast.FunctionDef) or node.name != "run_public_hf":
            continue
        for argument in node.args.args:
            if argument.arg == "model_type" and argument.annotation is not None:
                cli_types.update(
                    item.value
                    for item in ast.walk(argument.annotation)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )

    if value["runner_model_type"] not in runner_types.intersection(cli_types):
        raise ValueError("Runner model type is not wired in both public-HF entrypoints")
    return value
