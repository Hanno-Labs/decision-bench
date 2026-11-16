"""Quyet 1.2's public decision-distribution API, without prompt rewriting."""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.schemas import Candidate, DecisionExample, DecisionPrediction, Primitive

QUYET_ADAPTER = "quyet-public-api-v1"
QUYET_PROBABILITY_SOURCE = "quyet_raw_probabilities_default_temperature"


def _candidate_text(candidate: Candidate) -> str:
    if candidate.description is None or candidate.description == candidate.label:
        return candidate.label
    return f"{candidate.label}: {candidate.description}"


def _response_order(example: DecisionExample) -> list[Candidate]:
    if example.primitive is Primitive.ORDINAL_SCORING:
        return sorted(example.candidates, key=lambda c: float(c.ordinal_value or 0.0))
    return list(example.candidates)


def build_quyet_request(example: DecisionExample) -> dict[str, Any]:
    """Preserve candidate text; score levels use stable ascending ordinal order."""
    ordered = _response_order(example)
    ordinal = example.primitive is Primitive.ORDINAL_SCORING
    return {
        "state": example.state,
        "questions": {
            "decision": {
                "type": "score" if ordinal else "choice",
                "instructions": example.instruction,
                "criteria": (
                    [_candidate_text(c) for c in ordered]
                    if ordinal
                    else {c.id: _candidate_text(c) for c in ordered}
                ),
            }
        },
    }


def _audit_request(example: DecisionExample) -> dict[str, Any]:
    # The harness sorts JSON keys; keep the actual ordered API argument as text.
    return {
        "method": "raw_probabilities",
        "requests_json": json.dumps([build_quyet_request(example)], ensure_ascii=False),
        "strict": True,
        "candidate_ids": [c.id for c in example.candidates],
        "candidate_labels": [c.label for c in example.candidates],
        "response_candidate_ids": [c.id for c in _response_order(example)],
    }


class UnsupportedQuyetInput(ValueError):
    """Explicit unsupported input, with the attempted public request for auditing."""

    def __init__(self, example: DecisionExample, message: str) -> None:
        super().__init__(message)
        self.request = _audit_request(example)
        self.response: dict[str, Any] = {"error": message, "probabilities": None}


class QuyetOutputError(ValueError):
    """Malformed public response; never treated as an unsupported input."""

    def __init__(self, example: DecisionExample, response: Any, message: str) -> None:
        super().__init__(message)
        self.request = _audit_request(example)
        self.response = {"raw_probabilities": response, "error": message}


class QuyetDecisionModel:
    """Use public choice and score distributions aligned to benchmark candidate IDs."""

    def __init__(self, *, model_repo: str, model_revision: str, device: str = "cuda") -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise ValueError("Quyet requires a pinned 40-character Hub model revision")
        from huggingface_hub import hf_hub_download

        quyet = importlib.import_module("quyet")
        if quyet.__version__ != "1.2.0":
            raise ValueError("This adapter requires quyet==1.2.0")
        config = json.loads(
            Path(
                hf_hub_download(model_repo, "quyet_config.json", revision=model_revision)
            ).read_text()
        )
        limits = config.get("limits", {})
        self.max_options = int(limits.get("max_options", {}).get("choice", 10))
        self.max_score_options = int(limits.get("max_options", {}).get("score", 10))
        max_tokens = limits.get("max_input_tokens", config.get("max_len"))
        self.input_contract = {
            "policy_version": QUYET_ADAPTER,
            "truncated": False,
            "primitive_mapping": "choice for selection/binary; score in ascending ordinal order",
            "max_options": {"choice": self.max_options, "score": self.max_score_options},
            "max_input_tokens": max_tokens,
            "truncation_policy": "reject; public strict=True",
            "eligibility": ("valid public request, complete input within model limits"),
            "temperature_policy": "unchanged model default",
            "temperatures": config.get("temperatures", {}),
            "prefix_cache": False,
        }
        self.metadata = {
            "model_repo": model_repo,
            "model_revision": model_revision,
            "tokenizer_repo": model_repo,
            "tokenizer_revision": model_revision,
            "adapter": QUYET_ADAPTER,
            "probability_source": QUYET_PROBABILITY_SOURCE,
            "quyet_version": quyet.__version__,
            "device": device,
            "input_contract": self.input_contract,
            "latency_definition": "amortized public API batch seconds per row",
        }
        self.question_error = quyet.QuestionError
        self.model = quyet.load(
            model_repo, revision=model_revision, device=device, prefix_cache=False
        )

    def validate_example(self, example: DecisionExample) -> None:
        ordinal = example.primitive is Primitive.ORDINAL_SCORING
        limit = self.max_score_options if ordinal else self.max_options
        kind = "score" if ordinal else "choice"
        reason = None
        if not 2 <= len(example.candidates) <= limit:
            reason = f"{kind} supports 2 to {limit} candidates"
        elif not isinstance(example.state, (str, dict, list)):
            reason = "state must be a string, object or list"
        elif not example.instruction.strip():
            reason = "instruction must be nonempty"
        if reason is not None:
            raise UnsupportedQuyetInput(example, reason)

    def prompt_characters(self, example: DecisionExample) -> int:
        return len(json.dumps(build_quyet_request(example), ensure_ascii=False))

    def _probabilities(self, examples: list[DecisionExample]) -> list[Any]:
        try:
            result = self.model.raw_probabilities(
                [build_quyet_request(e) for e in examples], strict=True
            )
        except self.question_error as error:
            if len(examples) == 1:
                return [UnsupportedQuyetInput(examples[0], str(error))]
            middle = len(examples) // 2
            return self._probabilities(examples[:middle]) + self._probabilities(examples[middle:])
        except Exception as error:
            return [
                QuyetOutputError(
                    e,
                    {"error_type": type(error).__name__, "error": str(error)},
                    "public API failed",
                )
                for e in examples
            ]
        if not isinstance(result, list) or len(result) != len(examples):
            return [
                QuyetOutputError(e, result, "public response row count differs") for e in examples
            ]
        return result

    def predict_batch(
        self, examples: list[DecisionExample]
    ) -> list[HFDecisionResponse | UnsupportedQuyetInput | QuyetOutputError]:
        for example in examples:
            self.validate_example(example)
        start = time.perf_counter()
        raw = self._probabilities(examples)
        elapsed = (time.perf_counter() - start) / max(len(examples), 1)
        responses: list[HFDecisionResponse | UnsupportedQuyetInput | QuyetOutputError] = []
        for example, result in zip(examples, raw, strict=True):
            if isinstance(result, (UnsupportedQuyetInput, QuyetOutputError)):
                responses.append(result)
                continue
            values = result.get("decision") if isinstance(result, dict) else None
            if (
                not isinstance(result, dict)
                or set(result) != {"decision"}
                or not isinstance(values, list)
                or len(values) != len(example.candidates)
                or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in values)
                or not math.isclose(sum(values), 1.0, abs_tol=1e-6)
            ):
                responses.append(QuyetOutputError(example, result, "invalid public distribution"))
                continue
            by_id = dict(zip((c.id for c in _response_order(example)), values, strict=True))
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(
                        probabilities=[by_id[c.id] for c in example.candidates]
                    ),
                    request=_audit_request(example),
                    response={"raw_probabilities": result},
                    latency_seconds=elapsed,
                    input_contract=self.input_contract,
                )
            )
        return responses

    def predict(self, example: DecisionExample) -> HFDecisionResponse:
        result = self.predict_batch([example])[0]
        if isinstance(result, Exception):
            raise result
        return result
