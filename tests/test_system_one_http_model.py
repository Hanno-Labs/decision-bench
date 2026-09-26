from __future__ import annotations

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
    return SystemOneHTTPDecisionModel(
        base_url="http://127.0.0.1:1",
        model="model",
        model_repo="org/model",
        model_revision="0" * 40,
        serving_bundle_sha256="1" * 64,
        **overrides,  # type: ignore[arg-type]
    )


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
