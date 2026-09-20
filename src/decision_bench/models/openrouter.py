"""OpenRouter adapter for ordinary chat models."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from decision_bench.prompt import build_openrouter_request
from decision_bench.schemas import DecisionExample, DecisionPrediction


class OpenRouterResponse(BaseModel):
    """Validated prediction plus raw reproducibility fields."""

    model_config = ConfigDict(extra="forbid")

    prediction: DecisionPrediction
    request: dict[str, Any]
    response: dict[str, Any]
    latency_seconds: float
    input_contract: dict[str, Any] | None = None


class OpenRouterDecisionModel:
    """Run DecisionBench rows through OpenRouter chat completions."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-5.6-luna",
        reasoning_effort: str = "medium",
        seed: int = 0,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if key is None:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_seconds,
        )

    def predict(self, example: DecisionExample) -> OpenRouterResponse:
        """Return a schema-validated probability distribution."""

        request = build_openrouter_request(
            example,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            seed=self.seed,
        )
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                http_response = self._client.post("/chat/completions", json=request)
                if http_response.is_error:
                    detail = http_response.text[:2000]
                    raise httpx.HTTPStatusError(
                        f"OpenRouter returned {http_response.status_code}: {detail}",
                        request=http_response.request,
                        response=http_response,
                    )
                response = http_response.json()
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("OpenRouter response content is not text")
                prediction = DecisionPrediction.model_validate(json.loads(content))
                if len(prediction.probabilities) != len(example.candidates):
                    raise ValueError("OpenRouter returned the wrong number of probabilities")
                return OpenRouterResponse(
                    prediction=prediction,
                    request=request,
                    response=response,
                    latency_seconds=time.monotonic() - started,
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenRouterDecisionModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
