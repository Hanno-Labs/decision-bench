"""Prompt and strict response-schema contract for generic chat models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from decision_bench.schemas import Candidate, DecisionExample

PROMPT_VERSION = "generic-decision-distribution-v1"
TOP_LOGPROBS_PROMPT_VERSION = "generic-decision-top-logprobs-v1"
DECISION_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRST")
TEXT_TRUNCATION_POLICY_VERSION = "balanced-text-cap-v1"

SYSTEM_PROMPT = """You are a calibrated decision model.
Evaluate the supplied state against the instruction and the ordered candidate list.
Return one probability for every candidate, in exactly the same order as presented.
Candidates are mutually exclusive and collectively exhaustive for this task.
Use the candidate descriptions and ordinal values when present.
Do not return an explanation, candidate IDs, labels, or any fields outside the schema.
Your probabilities must each be between 0 and 1 and must sum to 1."""

TOP_LOGPROBS_SYSTEM_PROMPT = """You are a decision model.
Evaluate the supplied state against the instruction and candidate list.
Select exactly one allowed candidate symbol in the required response schema.
Do not explain your answer."""


@dataclass(frozen=True)
class TextTruncationReport:
    """Auditable result of fitting model-facing text to an input-token budget."""

    policy_version: str
    max_input_tokens: int
    original_input_tokens: int
    final_input_tokens: int
    original_text_characters: int
    final_text_characters: int
    truncated_text_fields: int
    text_field_character_cap: int | None

    @property
    def truncated(self) -> bool:
        return self.truncated_text_fields > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "max_input_tokens": self.max_input_tokens,
            "original_input_tokens": self.original_input_tokens,
            "final_input_tokens": self.final_input_tokens,
            "original_text_characters": self.original_text_characters,
            "final_text_characters": self.final_text_characters,
            "truncated_text_fields": self.truncated_text_fields,
            "text_field_character_cap": self.text_field_character_cap,
            "truncated": self.truncated,
        }


def fit_example_to_token_budget(
    example: DecisionExample,
    *,
    max_input_tokens: int,
    count_tokens: Callable[[DecisionExample], int],
) -> tuple[DecisionExample, TextTruncationReport]:
    """Fit text fields to a model limit without changing decision semantics.

    The frozen row remains untouched. Candidate IDs, candidate order, ordinal
    values, and gold fields are never changed. A single character cap is applied
    across instruction, state string values, labels, and descriptions so no one
    long field consumes the entire model budget.
    """

    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    original_tokens = count_tokens(example)
    original_texts = _example_texts(example)
    original_characters = sum(len(value) for value in original_texts)
    if original_tokens <= max_input_tokens:
        return example, TextTruncationReport(
            policy_version=TEXT_TRUNCATION_POLICY_VERSION,
            max_input_tokens=max_input_tokens,
            original_input_tokens=original_tokens,
            final_input_tokens=original_tokens,
            original_text_characters=original_characters,
            final_text_characters=original_characters,
            truncated_text_fields=0,
            text_field_character_cap=None,
        )

    largest_field = max((len(value) for value in original_texts), default=1)
    smallest = _example_with_text_cap(example, 1)
    smallest_tokens = count_tokens(smallest)
    if smallest_tokens > max_input_tokens:
        raise ValueError(
            "decision protocol and minimally preserved text exceed "
            f"max_input_tokens={max_input_tokens}: {smallest_tokens}"
        )

    best = smallest
    best_tokens = smallest_tokens
    best_cap = 1
    low = 2
    high = max(largest_field, 1)
    while low <= high:
        cap = (low + high) // 2
        candidate = _example_with_text_cap(example, cap)
        candidate_tokens = count_tokens(candidate)
        if candidate_tokens <= max_input_tokens:
            best = candidate
            best_tokens = candidate_tokens
            best_cap = cap
            low = cap + 1
        else:
            high = cap - 1

    final_texts = _example_texts(best)
    return best, TextTruncationReport(
        policy_version=TEXT_TRUNCATION_POLICY_VERSION,
        max_input_tokens=max_input_tokens,
        original_input_tokens=original_tokens,
        final_input_tokens=best_tokens,
        original_text_characters=original_characters,
        final_text_characters=sum(len(value) for value in final_texts),
        truncated_text_fields=sum(
            before != after for before, after in zip(original_texts, final_texts, strict=True)
        ),
        text_field_character_cap=best_cap,
    )


def _example_with_text_cap(example: DecisionExample, cap: int) -> DecisionExample:
    candidates = [
        Candidate(
            id=candidate.id,
            label=_truncate_text(candidate.label, cap),
            description=(
                _truncate_text(candidate.description, cap)
                if candidate.description is not None
                else None
            ),
            ordinal_value=candidate.ordinal_value,
        )
        for candidate in example.candidates
    ]
    return example.model_copy(
        update={
            "instruction": _truncate_text(example.instruction, cap),
            "state": _truncate_json_strings(example.state, cap),
            "candidates": candidates,
        },
        deep=True,
    )


def _truncate_json_strings(value: JsonValue, cap: int) -> JsonValue:
    if isinstance(value, str):
        return _truncate_text(value, cap)
    if isinstance(value, list):
        return [_truncate_json_strings(item, cap) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_json_strings(item, cap) for key, item in value.items()}
    return value


def _truncate_text(value: str, cap: int) -> str:
    if len(value) <= cap:
        return value
    if cap < 1:
        raise ValueError("text cap must be positive")
    if cap == 1:
        return "…"
    remaining = cap - 1
    left = (remaining + 1) // 2
    right = remaining - left
    return f"{value[:left]}…{value[-right:] if right else ''}"


def _example_texts(example: DecisionExample) -> list[str]:
    values = [example.instruction, *_json_texts(example.state)]
    for candidate in example.candidates:
        values.append(candidate.label)
        if candidate.description is not None:
            values.append(candidate.description)
    return values


def _json_texts(value: JsonValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _json_texts(item)]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _json_texts(value[key])]
    return []


def render_user_prompt(example: DecisionExample) -> str:
    """Render a stable model-facing JSON prompt."""

    payload: dict[str, Any] = {
        "primitive": example.primitive.value,
        "instruction": example.instruction,
        "state": example.state,
        "candidates": [
            {
                "index": index,
                "label": candidate.label,
                **(
                    {"description": candidate.description}
                    if candidate.description is not None
                    else {}
                ),
                **(
                    {"ordinal_value": candidate.ordinal_value}
                    if candidate.ordinal_value is not None
                    else {}
                ),
            }
            for index, candidate in enumerate(example.candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_top_logprobs_user_prompt(example: DecisionExample) -> str:
    """Render a compact prompt whose answer is one decision symbol."""

    if len(example.candidates) > len(DECISION_SYMBOLS):
        raise ValueError(
            f"top-logprobs protocol supports at most {len(DECISION_SYMBOLS)} candidates"
        )
    payload: dict[str, Any] = {
        "instruction": example.instruction,
        "state": example.state,
        "allowed_symbols": list(DECISION_SYMBOLS[: len(example.candidates)]),
        "candidates": [
            {
                "symbol": DECISION_SYMBOLS[index],
                "label": candidate.label,
                **(
                    {"description": candidate.description}
                    if candidate.description is not None
                    else {}
                ),
                **(
                    {"ordinal_value": candidate.ordinal_value}
                    if candidate.ordinal_value is not None
                    else {}
                ),
            }
            for index, candidate in enumerate(example.candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def response_format(candidate_count: int) -> dict[str, Any]:
    """Build an exact-length schema that stays compact at 255 candidates."""

    if not 2 <= candidate_count <= 255:
        raise ValueError("candidate_count must be between 2 and 255")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "decision_distribution",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "probabilities": {
                        "type": "array",
                        "description": "Probability for each candidate in input order.",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": candidate_count,
                        "maxItems": candidate_count,
                    }
                },
                "required": ["probabilities"],
                "additionalProperties": False,
            },
        },
    }


def build_openrouter_request(
    example: DecisionExample,
    *,
    model: str = "openai/gpt-5.6-luna",
    reasoning_effort: str = "medium",
    seed: int = 0,
) -> dict[str, Any]:
    """Build the fully serializable OpenRouter chat-completion request."""

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(example)},
        ],
        "response_format": response_format(len(example.candidates)),
        "provider": {"require_parameters": True},
        "reasoning": {"effort": reasoning_effort},
        "seed": seed,
    }


def build_openrouter_top_logprobs_request(
    example: DecisionExample,
    *,
    model: str,
    seed: int = 0,
    top_logprobs: int = 20,
) -> dict[str, Any]:
    """Build a one-token request that asks OpenRouter for native alternatives."""

    if not 1 <= top_logprobs <= 20:
        raise ValueError("top_logprobs must be between 1 and 20")
    if len(example.candidates) > top_logprobs:
        raise ValueError("candidate count exceeds requested top_logprobs")
    allowed_symbols = list(DECISION_SYMBOLS[: len(example.candidates)])
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": TOP_LOGPROBS_SYSTEM_PROMPT},
            {"role": "user", "content": render_top_logprobs_user_prompt(example)},
        ],
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "decision_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": allowed_symbols}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        },
        "seed": seed,
        "temperature": 0,
        "max_tokens": 16,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }


def prompt_sha256() -> str:
    """Return the stable hash recorded in benchmark run manifests."""

    material = f"{PROMPT_VERSION}\n{SYSTEM_PROMPT}".encode()
    return hashlib.sha256(material).hexdigest()


def top_logprobs_prompt_sha256() -> str:
    """Return the stable hash for the native top-logprobs protocol."""

    material = f"{TOP_LOGPROBS_PROMPT_VERSION}\n{TOP_LOGPROBS_SYSTEM_PROMPT}".encode()
    return hashlib.sha256(material).hexdigest()
