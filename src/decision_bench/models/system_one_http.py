"""HTTP adapter for a pinned Jev-compatible SystemOne serving bundle."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from decision_bench.models.jev_openrouter import (
    build_jev_request,
    prediction_from_jev_answer,
)
from decision_bench.models.openrouter import OpenRouterResponse
from decision_bench.schemas import DecisionExample

SYSTEM_ONE_HTTP_CONTRACT_VERSION = "jev-compatible-systemone-http-v1"
SYSTEM_ONE_HTTP_INPUT_POLICY_VERSION = "reject-over-serving-limit-v1"
XOR_MODEL_REPO = "juspay/xor"
XOR_MODEL_REVISION = "679decd4c669e5c37f4ac29dbd9957997424c876"
XOR_SERVING_BUNDLE_SHA256 = (
    "0a63473caaa3c6bfc8bc15fbab62f0a9a84c7ebf4ab6e06d0699891b7be6159b"
)
XOR_SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921"
)


class SystemOneHTTPDecisionModel:
    """Run rows through a local, pinned Jev-compatible SystemOne endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        model_repo: str,
        model_revision: str,
        serving_bundle_sha256: str,
        inference_image: str = XOR_SGLANG_IMAGE,
        max_candidates: int = 26,
        max_rendered_state_characters: int = 4 * 1024 * 1024,
        max_request_bytes: int = 8 * 1024 * 1024,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
    ) -> None:
        if max_candidates < 2:
            raise ValueError("max_candidates must be at least two")
        if max_rendered_state_characters < 1 or max_request_bytes < 1:
            raise ValueError("serving input limits must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.serving_bundle_sha256 = serving_bundle_sha256
        self.inference_image = inference_image
        self.max_candidates = max_candidates
        self.max_rendered_state_characters = max_rendered_state_characters
        self.max_request_bytes = max_request_bytes
        self.max_retries = max_retries
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "xor-serving-systemone-v1",
            "model": self.model,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "serving_bundle_sha256": self.serving_bundle_sha256,
            "inference_image": self.inference_image,
            "native_contract_version": SYSTEM_ONE_HTTP_CONTRACT_VERSION,
            "endpoint_base_url": self.base_url,
            "endpoint_path": "/v1/systemone",
            "max_candidates": self.max_candidates,
            "max_rendered_state_characters": self.max_rendered_state_characters,
            "max_request_bytes": self.max_request_bytes,
            "input_truncation_policy": SYSTEM_ONE_HTTP_INPUT_POLICY_VERSION,
            "probability_source": (
                "forward_reverse_option_letter_logprobs_calibrated_v1"
            ),
            "prediction_normalization": "divide_positive_finite_values_by_sum",
            "eligibility_definition": (
                f"candidate_count <= {self.max_candidates}; rendered state <= "
                f"{self.max_rendered_state_characters} characters; serialized request <= "
                f"{self.max_request_bytes} bytes"
            ),
        }

    def validate_example(self, example: DecisionExample) -> None:
        """Reject rows outside the published serving contract without truncation."""

        if len(example.candidates) > self.max_candidates:
            raise ValueError(
                f"SystemOne endpoint supports at most {self.max_candidates} candidates"
            )
        rendered_state = _render_state(example.state)
        if len(rendered_state) > self.max_rendered_state_characters:
            raise ValueError(
                "rendered state exceeds the SystemOne endpoint character limit: "
                f"{len(rendered_state)} > {self.max_rendered_state_characters}"
            )
        request, _ = build_jev_request(example, model=self.model)
        request_bytes = len(_serialize_request(request))
        if request_bytes > self.max_request_bytes:
            raise ValueError(
                "serialized request exceeds the SystemOne endpoint byte limit: "
                f"{request_bytes} > {self.max_request_bytes}"
            )

    def predict(self, example: DecisionExample) -> OpenRouterResponse:
        """Return the native SystemOne distribution in benchmark candidate order."""

        self.validate_example(example)
        request, response_order = build_jev_request(example, model=self.model)
        rendered_state_characters = len(_render_state(example.state))
        request_bytes = len(_serialize_request(request))
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                http_response = self._client.post("/v1/systemone", json=request)
                if http_response.is_error:
                    detail = http_response.text[:2000]
                    raise httpx.HTTPStatusError(
                        f"SystemOne endpoint returned {http_response.status_code}: {detail}",
                        request=http_response.request,
                        response=http_response,
                    )
                response = http_response.json()
                if not isinstance(response, dict):
                    raise TypeError("SystemOne response is not an object")
                answer = response["answers"]["decision"]
                if not isinstance(answer, dict):
                    raise TypeError("SystemOne response answer is not an object")
                prediction = prediction_from_jev_answer(example, answer, response_order)
                return OpenRouterResponse(
                    prediction=prediction,
                    request=request,
                    response=response,
                    latency_seconds=time.monotonic() - started,
                    input_contract={
                        "policy_version": SYSTEM_ONE_HTTP_INPUT_POLICY_VERSION,
                        "truncated": False,
                        "original_characters": rendered_state_characters,
                        "model_characters": rendered_state_characters,
                        "rendered_state_characters": rendered_state_characters,
                        "max_rendered_state_characters": (
                            self.max_rendered_state_characters
                        ),
                        "serialized_request_bytes": request_bytes,
                        "max_request_bytes": self.max_request_bytes,
                    },
                )
            except httpx.HTTPStatusError as error:
                last_error = error
                if attempt == self.max_retries or error.response.status_code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
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

    def __enter__(self) -> SystemOneHTTPDecisionModel:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _render_state(state: Any) -> str:
    """Mirror the released XOR serving bundle's state rendering."""

    if isinstance(state, str):
        return state
    messages = state.get("messages") if isinstance(state, dict) else state
    if isinstance(messages, list) and all(
        isinstance(message, dict) and "role" in message for message in messages
    ):
        return "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
    return json.dumps(state, indent=2)


def _serialize_request(request: dict[str, Any]) -> bytes:
    """Match httpx's compact UTF-8 JSON request encoding for the byte limit."""

    return json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
