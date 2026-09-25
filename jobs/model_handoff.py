"""Validate the reviewed model contract handed from an adapter PR to a run."""

from __future__ import annotations

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
MODEL_REPO = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z"
)
REVISION = re.compile(r"[a-f0-9]{40}\Z")


def validate_handoff(value: Any, issue_number: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Model handoff has unexpected fields")
    if value["schema_version"] != 1 or value["issue_number"] != issue_number:
        raise ValueError("Model handoff does not match the issue")
    if value["runner"] != "public-hf":
        raise ValueError("Issue automation currently requires the public-hf runner")
    for field in (
        "runner_model_type",
        "result_model_type",
        "adapter",
        "probability_source",
    ):
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
    return validate_handoff(json.loads(path.read_text()), issue_number)
