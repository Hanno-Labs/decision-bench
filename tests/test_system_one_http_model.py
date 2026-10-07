from __future__ import annotations

import json

import httpx
import pytest

from decision_bench.models.system_one_http import (
    XOR_ADAPTER_NAME,
    XOR_PROBABILITY_SOURCE,
    SystemOneHTTPDecisionModel,
    UnsupportedSystemOneInput,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example(candidates: int) -> DecisionExample:
    return DecisionExample(
        row_id="row",
        task_name="task",
        primitive=Primitive.CANDIDATE_SELECTION,
        family="family",
        domain="test",
        instruction="Choose.",
        state="state",
        candidates=[Candidate(id=str(index), label=str(index)) for index in range(candidates)],
        gold_candidate_id="0",
        gold_probabilities=[1.0, *([0.0] * (candidates - 1))],
    )


def _model(**overrides: object) -> SystemOneHTTPDecisionModel:
    arguments: dict[str, object] = {
        "base_url": "http://127.0.0.1:1",
        "model": "model",
        "model_repo": "org/model",
        "model_revision": "0" * 40,
        "serving_bundle_sha256": "1" * 64,
    }
    arguments.update(overrides)
    return SystemOneHTTPDecisionModel(**arguments)  # type: ignore[arg-type]


def test_metadata_defaults_describe_the_xor_bundle() -> None:
    with _model() as model:
        metadata = model.metadata
    assert metadata["adapter"] == XOR_ADAPTER_NAME == "xor-serving-systemone-v1"
    assert metadata["probability_source"] == XOR_PROBABILITY_SOURCE
    assert metadata["max_candidates"] == 26


def test_metadata_records_another_served_model_identity() -> None:
    with _model(
        adapter_name="imajev-serving-systemone-v1",
        probability_source="trained_readout_4_rotation_mean_temperature_calibrated_v1",
        max_candidates=255,
    ) as model:
        metadata = model.metadata
        model.validate_example(_example(255))  # the benchmark's largest candidate set
    assert metadata["adapter"] == "imajev-serving-systemone-v1"
    assert (
        metadata["probability_source"]
        == "trained_readout_4_rotation_mean_temperature_calibrated_v1"
    )
    assert "candidate_count <= 255" in metadata["eligibility_definition"]


def test_default_candidate_limit_rejects_the_twenty_seventh_candidate() -> None:
    with _model() as model:
        model.validate_example(_example(26))
        with pytest.raises(UnsupportedSystemOneInput):
            model.validate_example(_example(27))


def test_candidate_limit_is_enforced_at_the_configured_value() -> None:
    with _model(max_candidates=100) as model:
        model.validate_example(_example(100))
        with pytest.raises(UnsupportedSystemOneInput):
            model.validate_example(_example(101))


def _answer_with(model: SystemOneHTTPDecisionModel, seen: list[httpx.Request]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {"decision": {"type": "choice", "probabilities": {"0": 0.75, "1": 0.25}}}
            },
        )

    headers = model._client.headers
    model._client.close()
    model._client = httpx.Client(
        base_url=model.base_url, headers=headers, transport=httpx.MockTransport(handler)
    )


def test_hosted_endpoint_sends_the_key_as_a_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_TEST_KEY", "test-only-secret")
    seen: list[httpx.Request] = []
    with _model(base_url="https://example.com", api_key_env="SYSTEM_ONE_TEST_KEY") as model:
        _answer_with(model, seen)
        response = model.predict(_example(2))
        metadata = model.metadata
    assert seen[0].headers["authorization"] == "Bearer test-only-secret"
    assert str(seen[0].url) == "https://example.com/v1/systemone"
    assert response.prediction.probabilities == [0.75, 0.25]
    assert metadata["endpoint_authentication"] == "bearer"
    assert "test-only-secret" not in json.dumps(metadata)
    assert "test-only-secret" not in json.dumps(response.request)


def test_local_endpoint_sends_no_credentials_by_default() -> None:
    seen: list[httpx.Request] = []
    with _model() as model:
        _answer_with(model, seen)
        model.predict(_example(2))
        metadata = model.metadata
    assert "authorization" not in seen[0].headers
    assert metadata["endpoint_authentication"] == "none"


def test_missing_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEM_ONE_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SYSTEM_ONE_TEST_KEY"):
        _model(base_url="https://example.com", api_key_env="SYSTEM_ONE_TEST_KEY")


@pytest.mark.parametrize(
    ("base_url", "allowed"),
    [
        ("https://example.com", True),
        ("http://127.0.0.1:30002", True),
        ("http://localhost:30002", True),
        ("http://[::1]:30002", True),
        ("http://example.com", False),
        ("http://10.0.0.5:30002", False),
    ],
)
def test_key_is_sent_only_over_https_or_to_loopback(
    monkeypatch: pytest.MonkeyPatch, base_url: str, allowed: bool
) -> None:
    monkeypatch.setenv("SYSTEM_ONE_TEST_KEY", "test-only-secret")
    if allowed:
        _model(base_url=base_url, api_key_env="SYSTEM_ONE_TEST_KEY").close()
    else:
        with pytest.raises(ValueError, match="HTTPS"):
            _model(base_url=base_url, api_key_env="SYSTEM_ONE_TEST_KEY")
