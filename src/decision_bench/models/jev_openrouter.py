"""Native OpenRouter Decisions adapter for TypeSafe Jev."""

from __future__ import annotations

import json
import os
import time
from typing import Any, cast

import httpx

from decision_bench.models.openrouter import OpenRouterResponse
from decision_bench.prompt import (
    TEXT_TRUNCATION_POLICY_VERSION,
    fit_example_to_token_budget,
)
from decision_bench.schemas import Candidate, DecisionExample, DecisionPrediction, Primitive

JEV_CONTRACT_VERSION = "openrouter-decisions-v2-arbitrary-binary-labels"
JEV_MAX_STATE_QUESTION_TOKENS = 32_000
JEV_INPUT_TOKEN_RESERVE = 2_048
JEV_TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"
JEV_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def jev_effective_input_token_budget(
    max_state_question_tokens: int,
    input_token_reserve: int,
) -> int:
    """Return the proxy-token budget below Jev's private serialization ceiling."""

    if input_token_reserve < 0:
        raise ValueError("input_token_reserve must be non-negative")
    if input_token_reserve >= max_state_question_tokens:
        raise ValueError("input_token_reserve must be smaller than max_state_question_tokens")
    return max_state_question_tokens - input_token_reserve


def build_jev_request(
    example: DecisionExample,
    *,
    model: str = "typesafe/jev-1.13",
) -> tuple[dict[str, Any], list[str]]:
    """Build a native Decisions request and its response-to-candidate mapping."""

    if example.primitive is Primitive.BINARY_CLASSIFICATION:
        true_outcome, false_outcome = _binary_outcomes(example)
        question: dict[str, Any] = {
            "type": "noul",
            "instructions": example.instruction,
            "criteria": {
                "true": _candidate_description(true_outcome),
                "false": _candidate_description(false_outcome),
            },
        }
        response_order = [candidate.id for candidate in example.candidates]
    elif example.primitive is Primitive.CANDIDATE_SELECTION:
        question = {
            "type": "choice",
            "instructions": example.instruction,
            "criteria": {
                candidate.id: _candidate_description(candidate) for candidate in example.candidates
            },
        }
        response_order = [candidate.id for candidate in example.candidates]
    else:
        ordered = sorted(
            example.candidates,
            key=lambda candidate: float(candidate.ordinal_value or 0.0),
        )
        question = {
            "type": "score",
            "instructions": example.instruction,
            "criteria": [_candidate_description(candidate) for candidate in ordered],
        }
        response_order = [candidate.id for candidate in ordered]
    return (
        {
            "model": model,
            "state": example.state,
            "questions": {"decision": question},
        },
        response_order,
    )


class JevOpenRouterDecisionModel:
    """Run DecisionBench rows through OpenRouter's native Decisions endpoint."""

    def __init__(
        self,
        *,
        model: str = "typesafe/jev-1.13",
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
        max_state_question_tokens: int = JEV_MAX_STATE_QUESTION_TOKENS,
        input_token_reserve: int = JEV_INPUT_TOKEN_RESERVE,
        tokenizer_model: str = JEV_TOKENIZER_MODEL,
        tokenizer_revision: str = JEV_TOKENIZER_REVISION,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Jev token-budget support requires transformers") from error
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if key is None:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.model = model
        self.max_retries = max_retries
        self.max_state_question_tokens = max_state_question_tokens
        self.input_token_reserve = input_token_reserve
        self.effective_input_token_budget = jev_effective_input_token_budget(
            max_state_question_tokens,
            input_token_reserve,
        )
        self.tokenizer_model = tokenizer_model
        self.tokenizer_revision = tokenizer_revision
        self._tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            tokenizer_model,
            revision=tokenizer_revision,
            use_fast=True,
        )
        self._client = httpx.Client(
            base_url="https://openrouter.ai",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_seconds,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "input_truncation_policy": TEXT_TRUNCATION_POLICY_VERSION,
            "max_state_question_tokens": self.max_state_question_tokens,
            "input_token_reserve": self.input_token_reserve,
            "effective_input_token_budget": self.effective_input_token_budget,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_revision": self.tokenizer_revision,
            "token_count_method": "qwen3_proxy_over_canonical_state_and_question_json",
            "token_limit_source": "typesafe_documented_state_plus_longest_question_limit",
        }

    def predict(self, example: DecisionExample) -> OpenRouterResponse:
        """Return Jev's native distribution aligned to benchmark candidate order."""

        model_example, truncation = fit_example_to_token_budget(
            example,
            max_input_tokens=self.effective_input_token_budget,
            count_tokens=self._state_question_tokens,
        )
        request, response_order = build_jev_request(model_example, model=self.model)
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                http_response = self._client.post("/api/alpha/decisions", json=request)
                if http_response.is_error:
                    detail = http_response.text[:2000]
                    raise httpx.HTTPStatusError(
                        f"OpenRouter returned {http_response.status_code}: {detail}",
                        request=http_response.request,
                        response=http_response,
                    )
                response = http_response.json()
                answer = response["answers"]["decision"]
                if not isinstance(answer, dict):
                    raise TypeError("Jev response answer is not an object")
                prediction = _prediction_from_answer(example, answer, response_order)
                return OpenRouterResponse(
                    prediction=prediction,
                    request=request,
                    response=response,
                    latency_seconds=time.monotonic() - started,
                    input_contract={
                        **truncation.as_dict(),
                        "tokenizer_model": self.tokenizer_model,
                        "tokenizer_revision": self.tokenizer_revision,
                        "documented_max_state_question_tokens": (
                            self.max_state_question_tokens
                        ),
                        "input_token_reserve": self.input_token_reserve,
                        "counted_surface": "canonical_json(state,question)",
                        "model_facing_example": model_example.model_dump(mode="json"),
                    },
                )
            except httpx.HTTPStatusError as error:
                last_error = error
                status_code = error.response.status_code
                if attempt == self.max_retries or status_code not in {
                    429,
                    500,
                    502,
                    503,
                    524,
                    529,
                }:
                    break
                time.sleep(min(2**attempt, 16))
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        assert last_error is not None
        raise last_error

    def _state_question_tokens(self, example: DecisionExample) -> int:
        request, _ = build_jev_request(example, model=self.model)
        material = {
            "state": request["state"],
            "question": request["questions"]["decision"],
        }
        serialized = json.dumps(
            material,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return len(self._tokenizer.encode(serialized, add_special_tokens=False))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JevOpenRouterDecisionModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _prediction_from_answer(
    example: DecisionExample,
    answer: dict[str, Any],
    response_order: list[str],
) -> DecisionPrediction:
    if example.primitive is Primitive.BINARY_CLASSIFICATION:
        yes_probability = float(answer["noul"])
        true_outcome, _ = _binary_outcomes(example)
        probabilities = [
            yes_probability if candidate.id == true_outcome.id else 1.0 - yes_probability
            for candidate in example.candidates
        ]
        return DecisionPrediction(probabilities=probabilities)

    raw_probabilities = answer.get("probabilities")
    if not isinstance(raw_probabilities, dict):
        raise TypeError("Jev response did not include a probability distribution")
    if example.primitive is Primitive.CANDIDATE_SELECTION:
        by_candidate_id = {
            candidate_id: float(raw_probabilities[candidate_id]) for candidate_id in response_order
        }
    else:
        by_candidate_id = {
            candidate_id: float(raw_probabilities[str(index)])
            for index, candidate_id in enumerate(response_order)
        }
    return DecisionPrediction(
        probabilities=[by_candidate_id[candidate.id] for candidate in example.candidates]
    )


def _binary_outcomes(example: DecisionExample) -> tuple[Candidate, Candidate]:
    if len(example.candidates) != 2:
        raise ValueError(f"binary row {example.row_id} does not contain exactly two candidates")
    return example.candidates[0], example.candidates[1]


def _candidate_description(candidate: Candidate) -> str:
    if candidate.description is None:
        return candidate.label
    return f"{candidate.label}: {candidate.description}"
