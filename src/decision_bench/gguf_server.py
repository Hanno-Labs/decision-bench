"""Serve a generic GGUF chat model through a Jev-shaped decision API.

The model is an ordinary llama.cpp-compatible chat model.  It is asked for a
probability distribution over the runtime candidates; the HTTP layer projects
that common distribution into ``choice``, ``score``, or ``noul`` answers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["choice", "score", "noul"]


class DecisionQuestionRequest(BaseModel):
    """One Jev-compatible runtime question."""

    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    instructions: str | Mapping[str, Any] | list[Any] = Field(min_length=1)
    criteria: Mapping[str, str | None] | list[str | Mapping[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> DecisionQuestionRequest:
        if self.type == "choice":
            if not isinstance(self.criteria, Mapping) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("choice criteria must be an object with 2 to 255 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("score criteria must be an array with 2 to 255 levels")
        elif self.criteria is not None and not isinstance(self.criteria, Mapping):
            raise ValueError("noul criteria, when supplied, must be an object")
        return self


class SystemOneRequest(BaseModel):
    """Request body for ``POST /v1/systemone``."""

    model_config = ConfigDict(extra="forbid")

    model: str = "gguf-decision-model"
    state: Any
    questions: dict[str, DecisionQuestionRequest] = Field(min_length=1)


class GgufDecisionEngine:
    """Run ordinary GGUF chat inference and normalize its probability output."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        seed: int = 0,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as error:  # pragma: no cover - exercised by CLI users
            raise RuntimeError(
                "GGUF serving requires the optional 'gguf' dependencies; "
                "install with `uv sync --extra gguf`"
            ) from error
        if not model_path.is_file():
            raise ValueError(f"GGUF model does not exist: {model_path}")
        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "verbose": False,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        self.model_path = model_path
        self.seed = seed
        self._llama = Llama(**kwargs)

    def predict(
        self,
        state: Any,
        question: DecisionQuestionRequest,
    ) -> dict[str, Any]:
        candidates = _candidates(question)
        prompt = _prompt(state, question, candidates)
        schema = {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": len(candidates),
                    "maxItems": len(candidates),
                }
            },
            "required": ["probabilities"],
            "additionalProperties": False,
        }
        response = cast(
            Mapping[str, Any],
            self._llama.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a calibrated decision model. Return only JSON "
                            "with one probability for every candidate, in order. "
                            "The probabilities must sum to 1. Do not explain."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": schema},
                temperature=0.0,
                seed=self.seed,
                max_tokens=max(32, len(candidates) * 8),
            ),
        )
        content = _response_content(response)
        raw = json.loads(content)
        if not isinstance(raw, Mapping):
            raise ValueError("GGUF model returned a non-object decision")
        probabilities = _normalize_probabilities(raw.get("probabilities"), len(candidates))
        return _project_answer(question, candidates, probabilities)


def create_app(engine: GgufDecisionEngine) -> Any:
    """Create a FastAPI app without importing FastAPI for library users."""

    try:
        from fastapi import FastAPI
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "GGUF HTTP serving requires the optional 'gguf' dependencies; "
            "install with `uv sync --extra gguf`"
        ) from error

    app = FastAPI(title="GGUF Decision API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": engine.model_path.name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict[str, Any]:
        answers = {
            question_id: engine.predict(request.state, question)
            for question_id, question in request.questions.items()
        }
        return {
            "model": request.model,
            "answers": answers,
            "usage": {"input_tokens": None, "output_tokens": 0},
        }

    return app


def _candidates(question: DecisionQuestionRequest) -> list[tuple[str, str]]:
    if question.type == "choice":
        assert isinstance(question.criteria, Mapping)
        return [(str(key), str(value or key)) for key, value in question.criteria.items()]
    if question.type == "score":
        assert isinstance(question.criteria, list)
        return [
            (str(index), _criterion_text(value))
            for index, value in enumerate(question.criteria)
        ]
    if isinstance(question.criteria, Mapping):
        true_text = str(question.criteria.get("true") or "true")
        false_text = str(question.criteria.get("false") or "false")
        return [("true", true_text), ("false", false_text)]
    return [("true", "yes"), ("false", "no")]


def _criterion_text(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    return value


def _prompt(
    state: Any,
    question: DecisionQuestionRequest,
    candidates: Sequence[tuple[str, str]],
) -> str:
    return json.dumps(
        {
            "state": state,
            "instructions": question.instructions,
            "type": question.type,
            "candidates": [
                {"id": candidate_id, "description": description}
                for candidate_id, description in candidates
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _response_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("GGUF model returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        raise ValueError("GGUF model returned no textual decision")
    return content


def _normalize_probabilities(value: Any, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"expected exactly {expected} probabilities")
    probabilities = [float(item) for item in value]
    if any(item < 0.0 or item > 1.0 for item in probabilities):
        raise ValueError("probabilities must be between 0 and 1")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probabilities must have a positive sum")
    return [item / total for item in probabilities]


def _project_answer(
    question: DecisionQuestionRequest,
    candidates: Sequence[tuple[str, str]],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    distribution = {
        candidate_id: probability
        for (candidate_id, _), probability in zip(candidates, probabilities, strict=True)
    }
    if question.type == "noul":
        return {"type": "noul", "noul": distribution["true"]}
    if question.type == "choice":
        selected = candidates[max(range(len(probabilities)), key=probabilities.__getitem__)][0]
        return {
            "type": "choice",
            "choice": selected,
            "probabilities": distribution,
            "confidence": max(probabilities),
        }
    score = sum(float(index) * probability for index, probability in enumerate(probabilities))
    return {
        "type": "score",
        "score": score,
        "legend": {str(index): description for index, (_, description) in enumerate(candidates)},
        "probabilities": distribution,
        "confidence": max(probabilities),
    }
