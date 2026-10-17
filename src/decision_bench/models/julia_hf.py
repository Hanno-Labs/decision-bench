"""Native decision readout for SupersonicLabs/Julia-1."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.public_hf import (
    _align_probabilities,
    _binary_candidate_ids,
    _candidate_text,
    _sha256_file,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

MODEL_ID = "SupersonicLabs/Julia-1"
MODEL_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
TOKENIZER_REVISION = MODEL_REVISION
BASE_MODEL = "jhu-clsp/mmBERT-small"
WEIGHTS_FILE = "model.safetensors"
WEIGHTS_SHA256 = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
MAX_LENGTH = 8192
HEAD_LENGTH = 512
OPTION_TOKEN_LIMIT = 48
MIN_CANDIDATES = 2
MAX_CANDIDATES = 20
PROBABILITY_SOURCE = "softmax_over_native_option_logits"
POLICY_VERSION = "julia1-strict-encoding-v1"
PROMPT_CONTRACT = "julia-native-decision-row-v1"
NATIVE_TYPES: dict[Primitive, str] = {
    Primitive.BINARY_CLASSIFICATION: "noul",
    Primitive.CANDIDATE_SELECTION: "choice",
    Primitive.ORDINAL_SCORING: "score",
}


class UnsupportedJuliaInput(ValueError):
    """An example is outside Julia-1's published native input contract."""


class JuliaHFDecisionModel:
    """Run Julia-1 through its released resident decision runtime."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str,
        model_revision: str,
        expected_weights_sha256: str,
        device: str | None = None,
        max_length: int = MAX_LENGTH,
        head_length: int = HEAD_LENGTH,
    ) -> None:
        try:
            import torch
            from julia import load_model
        except ImportError as error:
            raise RuntimeError(
                "Julia-1 support requires the pinned supersonic-julia runtime, e.g. "
                "uv run --with 'supersonic-julia @ "
                "git+https://huggingface.co/SupersonicLabs/Julia-1"
                "@a85b127321d580d65176c89ced8273f305745d85'"
            ) from error
        if model_repo != MODEL_ID or model_revision != MODEL_REVISION:
            raise ValueError("Julia-1 requires its pinned model repository and revision")
        if expected_weights_sha256 != WEIGHTS_SHA256:
            raise ValueError("Julia-1 requires the published checkpoint SHA-256")
        weights = model_dir / WEIGHTS_FILE
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if _sha256_file(weights) != WEIGHTS_SHA256:
            raise ValueError("Julia-1 checkpoint SHA-256 does not match the pinned release")
        if (max_length, head_length) != (MAX_LENGTH, HEAD_LENGTH):
            raise ValueError(
                "Julia-1's published runtime pins max_length=8192 and head_length=512"
            )

        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.expected_weights_sha256 = expected_weights_sha256
        self.max_length = max_length
        self.head_length = head_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._engine = load_model(
            str(model_dir),
            device=self.device,
            max_length=max_length,
            head_length=head_length,
            strict_encoding=True,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "julia1_native_decision",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "tokenizer_revision": TOKENIZER_REVISION,
            "base_model": BASE_MODEL,
            "probability_source": PROBABILITY_SOURCE,
            "probability_interpretation": (
                "native softmax over every offered option; a conditional option "
                "preference, not calibrated confidence"
            ),
            "prompt_contract": PROMPT_CONTRACT,
            "native_types": {
                primitive.value: native_type for primitive, native_type in NATIVE_TYPES.items()
            },
            "max_length": self.max_length,
            "head_length": self.head_length,
            "option_token_limit": OPTION_TOKEN_LIMIT,
            "strict_encoding": True,
            "device": self.device,
            "input_truncation_policy": "reject_rows_over_strict_encoding_limits",
            "calibration": "not established; softmax is conditional option preference",
            "candidate_boundary": {
                "min": MIN_CANDIDATES,
                "max": MAX_CANDIDATES,
                "unique_ids": True,
                "noul_options": "ordered [false, true]",
                "effective_limit": (
                    "2-20 options within the 8,192-token strict encoding budget, "
                    "each option at most 48 tokens"
                ),
            },
            "eligibility_definition": (
                "2 <= candidate_count <= 20 with a strict-encoding-valid native row"
            ),
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        row, _ = self._prepare(example)
        rendered_state = row["state"]
        if not isinstance(rendered_state, str):
            rendered_state = json.dumps(rendered_state, ensure_ascii=False)
        return (
            len(row["question"])
            + sum(len(option) for option in row["options"])
            + len(rendered_state)
        )

    def validate_example(self, example: DecisionExample) -> None:
        row, _ = self._prepare(example)
        try:
            self._engine.encoding_info([row])
        except ValueError as error:
            raise UnsupportedJuliaInput(str(error)) from error

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        if not examples:
            return []
        prepared = [(example, *self._prepare(example)) for example in examples]
        rows = [row for _, row, _ in prepared]
        started = time.monotonic()
        raw_logits = self._engine.logits(rows)
        elapsed = time.monotonic() - started
        if len(raw_logits) != len(rows):
            raise RuntimeError("Julia returned an incorrect answer count")

        responses: list[HFDecisionResponse] = []
        for (example, row, native_ids), logits in zip(prepared, raw_logits, strict=True):
            options = row["options"]
            if len(logits) != len(options):
                raise RuntimeError(
                    "Julia returned a logit count that does not match the offered options"
                )
            native_logits = [float(value) for value in logits]
            if not all(math.isfinite(value) for value in native_logits):
                raise RuntimeError("Julia returned non-finite logits")
            native_probabilities = _softmax(native_logits)
            probabilities = _align_probabilities(example, native_ids, native_probabilities)
            encoding = self._engine.encoding_info([row])[0]
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "state": row["state"],
                        "question": row["question"],
                        "options": options,
                        "type": row["type"],
                        "native_candidate_ids": native_ids,
                    },
                    response={
                        "native_candidate_ids": native_ids,
                        "native_options": options,
                        "native_type": row["type"],
                        "native_logits": native_logits,
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(prepared),
                    input_contract={
                        "candidate_ids": [candidate.id for candidate in example.candidates],
                        "native_candidate_ids": native_ids,
                        "native_type": row["type"],
                        "option_token_counts": cast(list[int], encoding["optionTokens"]),
                        "original_input_tokens": int(encoding["tokens"]),
                        "final_input_tokens": int(encoding["tokens"]),
                        "max_input_tokens": self.max_length,
                        "head_length": self.head_length,
                        "option_token_limit": OPTION_TOKEN_LIMIT,
                        "truncated": False,
                        "policy_version": POLICY_VERSION,
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _prepare(self, example: DecisionExample) -> tuple[dict[str, Any], list[str]]:
        candidate_count = len(example.candidates)
        if not MIN_CANDIDATES <= candidate_count <= MAX_CANDIDATES:
            raise UnsupportedJuliaInput(
                f"candidate count {candidate_count} outside "
                f"[{MIN_CANDIDATES}, {MAX_CANDIDATES}]"
            )
        ids = [candidate.id for candidate in example.candidates]
        if len(set(ids)) != len(ids):
            raise UnsupportedJuliaInput("duplicate candidate ids are unsupported")

        rendered = {
            candidate.id: _candidate_text(candidate.label, candidate.description)
            for candidate in example.candidates
        }
        if any(not text.strip() for text in rendered.values()):
            raise UnsupportedJuliaInput("empty candidate renderings are unsupported")

        native_type = NATIVE_TYPES[example.primitive]
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            try:
                false_id, true_id = _binary_candidate_ids(example)
            except ValueError as error:
                raise UnsupportedJuliaInput(str(error)) from error
            native_ids = [false_id, true_id]
        else:
            native_ids = ids
        row = {
            "state": _native_state(example.state),
            "question": example.instruction,
            "options": [rendered[candidate_id] for candidate_id in native_ids],
            "type": native_type,
        }
        return row, native_ids


def _native_state(state: Any) -> Any:
    if isinstance(state, str | dict | list):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
