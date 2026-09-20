"""OpenRouter adapter using native output-token log probabilities."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import httpx

from decision_bench.models.openrouter import OpenRouterResponse
from decision_bench.prompt import DECISION_SYMBOLS, build_openrouter_top_logprobs_request
from decision_bench.schemas import DecisionExample, DecisionPrediction


class IncompleteCandidateLogprobsError(ValueError):
    """Raised when OpenRouter omits one or more valid candidate tokens."""


class OpenRouterTopLogprobsDecisionModel:
    """Score up to 20 candidates from OpenRouter's native token logprobs."""

    def __init__(
        self,
        *,
        model: str,
        seed: int = 0,
        top_logprobs: int = 20,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if key is None:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.model = model
        self.seed = seed
        self.top_logprobs = top_logprobs
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_seconds,
        )

    def predict(self, example: DecisionExample) -> OpenRouterResponse:
        """Return candidate probabilities conditional on the valid symbol set."""

        request = build_openrouter_top_logprobs_request(
            example,
            model=self.model,
            seed=self.seed,
            top_logprobs=self.top_logprobs,
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
                prediction = prediction_from_top_logprobs(response, len(example.candidates))
                return OpenRouterResponse(
                    prediction=prediction,
                    request=request,
                    response=response,
                    latency_seconds=time.monotonic() - started,
                )
            except IncompleteCandidateLogprobsError:
                raise
            except httpx.HTTPStatusError as error:
                last_error = error
                if attempt == self.max_retries or error.response.status_code not in {
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

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenRouterTopLogprobsDecisionModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def prediction_from_top_logprobs(
    response: dict[str, Any],
    candidate_count: int,
) -> DecisionPrediction:
    """Extract a complete candidate distribution from one-token alternatives."""

    if not 2 <= candidate_count <= len(DECISION_SYMBOLS):
        raise ValueError("candidate_count must be between 2 and 20")
    choice = response["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("OpenRouter response content is not text")
    parsed_content = json.loads(content)
    if not isinstance(parsed_content, dict):
        raise TypeError("OpenRouter structured response is not an object")
    selected_symbol = parsed_content.get("choice")
    if not isinstance(selected_symbol, str):
        raise TypeError("OpenRouter structured response has no text choice")
    allowed_symbols = DECISION_SYMBOLS[:candidate_count]
    if selected_symbol not in allowed_symbols:
        raise ValueError(f"model returned invalid decision symbol: {content!r}")
    positions = choice["logprobs"]["content"]
    if not isinstance(positions, list):
        raise TypeError("output-token logprob positions are not a list")
    candidate_masses: dict[str, float] = {}
    for position in positions:
        generated_token = position["token"]
        if not isinstance(generated_token, str):
            raise TypeError("generated logprob token is not text")
        if _candidate_symbol(generated_token, allowed_symbols) != selected_symbol:
            continue
        alternatives = position["top_logprobs"]
        if not isinstance(alternatives, list):
            raise TypeError("top_logprobs is not a list")
        observed_tokens: set[str] = set()
        for alternative in alternatives:
            token = alternative["token"]
            if not isinstance(token, str):
                raise TypeError("top-logprob token is not text")
            symbol = _candidate_symbol(token, allowed_symbols)
            if symbol is not None:
                candidate_masses[symbol] = candidate_masses.get(symbol, 0.0) + math.exp(
                    float(alternative["logprob"])
                )
                observed_tokens.add(token)
        if generated_token not in observed_tokens:
            candidate_masses[selected_symbol] = candidate_masses.get(
                selected_symbol, 0.0
            ) + math.exp(float(position["logprob"]))
        break
    if not candidate_masses:
        raise ValueError("could not locate the structured choice token's logprobs")
    missing = [symbol for symbol in allowed_symbols if symbol not in candidate_masses]
    if missing:
        raise IncompleteCandidateLogprobsError(
            "OpenRouter omitted candidate symbols from top_logprobs: " + ",".join(missing)
        )
    return DecisionPrediction(
        probabilities=[candidate_masses[symbol] for symbol in allowed_symbols]
    )


def _candidate_symbol(token: str, allowed_symbols: tuple[str, ...]) -> str | None:
    normalized = token.strip().strip("\"'{}[],: ")
    if normalized in allowed_symbols:
        return normalized
    return None
