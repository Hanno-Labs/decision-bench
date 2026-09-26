from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from decision_bench.models.intern_decision_hf import (
    INTERN_DECISION_ANSWER_SYMBOLS,
    INTERN_DECISION_DECISION_TOKEN,
    INTERN_DECISION_MAX_OPTIONS,
    INTERN_DECISION_MODEL_REPO,
    INTERN_DECISION_MODEL_REVISION,
    INTERN_DECISION_SYSTEM_PROMPT,
    InternDecisionHFDecisionModel,
    _argmax,
    _options,
    _read_serving_constants,
    _scale_probabilities,
    build_intern_decision_request,
    compile_intern_decision_row,
    prediction_from_intern_decision_response,
)
from decision_bench.models.public_hf import (
    UnsupportedCandidateCount,
    UnsupportedInputLength,
)
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _choice_example() -> DecisionExample:
    return DecisionExample(
        row_id="row-choice",
        task_name="routing",
        primitive=Primitive.CANDIDATE_SELECTION,
        family="routing_triage",
        domain="financial",
        instruction="Route the request.",
        state={"request": "ATM kept my card."},
        candidates=[
            Candidate(id="cash", label="Cash support"),
            Candidate(id="card", label="Card support", description="Retained cards"),
            Candidate(id="fraud", label="Fraud support"),
        ],
        gold_candidate_id="card",
        gold_probabilities=[0.0, 1.0, 0.0],
    )


def _binary_example() -> DecisionExample:
    return DecisionExample(
        row_id="row-binary",
        task_name="refund",
        primitive=Primitive.BINARY_CLASSIFICATION,
        family="refund_eligibility",
        domain="retail",
        instruction="Is the customer eligible?",
        state={"order": "late"},
        candidates=[
            Candidate(id="yes", label="Eligible"),
            Candidate(id="no", label="Not eligible"),
        ],
        gold_candidate_id="yes",
        gold_probabilities=[1.0, 0.0],
    )


def _ordinal_example() -> DecisionExample:
    return DecisionExample(
        row_id="row-ordinal",
        task_name="severity",
        primitive=Primitive.ORDINAL_SCORING,
        family="severity_rating",
        domain="ops",
        instruction="Rate the severity.",
        state={"incident": "database down"},
        candidates=[
            Candidate(id="high", label="High", ordinal_value=2.0),
            Candidate(id="low", label="Low", ordinal_value=0.0),
            Candidate(id="medium", label="Medium", ordinal_value=1.0),
        ],
        gold_candidate_id="high",
        gold_probabilities=[1.0, 0.0, 0.0],
    )


class _InternTokenizerStub:
    """Deterministic CPU-only stand-in for the checkpoint chat tokenizer."""

    def __init__(self, marker_id: int = 909_090) -> None:
        self._marker_id = marker_id

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        **_: object,
    ) -> str:
        assert tokenize is False
        return "\n".join(str(message["content"]) for message in messages)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, list[list[int]]]:
        del add_special_tokens, return_tensors
        token_ids: list[int] = []
        cursor = 0
        marker = INTERN_DECISION_DECISION_TOKEN
        while cursor < len(text):
            if text.startswith(marker, cursor):
                token_ids.append(self._marker_id)
                cursor += len(marker)
            else:
                token_ids.append(ord(text[cursor]) % 997)
                cursor += 1
        return {"input_ids": [token_ids]}


def _model() -> InternDecisionHFDecisionModel:
    model = InternDecisionHFDecisionModel.__new__(InternDecisionHFDecisionModel)
    model.model_repo = INTERN_DECISION_MODEL_REPO
    model.model_revision = INTERN_DECISION_MODEL_REVISION
    model.max_length = 8_192
    model.device = "cpu"
    model.attn_implementation = "sdpa"
    model.temperature = 1.99241824
    model.native_model_name = "Intern-Decision-4B"
    model.serving_code_sha256 = "0" * 64
    model.tokenizer = _InternTokenizerStub()
    model.marker_id = model.tokenizer._marker_id
    model._symbol_token_ids = {
        symbol: 100 + index
        for index, symbol in enumerate(INTERN_DECISION_ANSWER_SYMBOLS)
    }
    return model


def test_read_serving_constants_parses_published_calibration(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.py"
    inference_path.write_text(
        "MODEL_NAME = 'Intern-Decision-4B'\nDEFAULT_TEMPERATURE = 1.99241824\n",
        encoding="utf-8",
    )
    temperature, name = _read_serving_constants(inference_path)
    assert temperature == pytest.approx(1.99241824)
    assert name == "Intern-Decision-4B"


def test_read_serving_constants_rejects_missing_calibration(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.py"
    inference_path.write_text("MODEL_NAME = 'Intern-Decision-4B'\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _read_serving_constants(inference_path)


def test_compile_preserves_candidate_ids_and_displayed_order() -> None:
    compiled = compile_intern_decision_row(
        {
            "state": {"request": "ATM kept my card."},
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Route the request.",
                    "criteria": {"cash": "Cash support", "card": "Card support"},
                }
            },
        }
    )
    assert compiled.fields == ("decision",)
    assert compiled.symbols["decision"] == ("A", "B")
    assert list(compiled.questions["decision"]["criteria"]) == ["cash", "card"]
    assert compiled.messages[0]["content"] == INTERN_DECISION_SYSTEM_PROMPT
    assert INTERN_DECISION_DECISION_TOKEN not in compiled.user_text
    assert "## State" in compiled.user_text
    assert json.loads(compiled.skeleton) == {"decision": INTERN_DECISION_DECISION_TOKEN}


def test_compile_rejects_reserved_marker_in_evidence() -> None:
    with pytest.raises(ValueError):
        compile_intern_decision_row(
            {
                "state": {"request": f"leak {INTERN_DECISION_DECISION_TOKEN}"},
                "questions": {"decision": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            }
        )


def test_compile_rejects_too_many_options() -> None:
    criteria = {
        f"c{index}": f"Candidate {index}"
        for index in range(INTERN_DECISION_MAX_OPTIONS + 1)
    }
    with pytest.raises(UnsupportedCandidateCount):
        compile_intern_decision_row(
            {"state": {}, "questions": {"decision": {"type": "choice", "criteria": criteria}}}
        )


def test_options_maps_noul_to_symbol_order() -> None:
    assert _options(
        {"type": "noul", "criteria": {"true": "Eligible", "false": "Not eligible"}}
    ) == [("no", "Not eligible"), ("yes", "Eligible")]


def test_build_request_preserves_declared_candidate_order() -> None:
    example = _choice_example()
    _, response_order, compiled = build_intern_decision_request(example)
    assert response_order == ["cash", "card", "fraud"]
    assert list(compiled.questions["decision"]["criteria"]) == ["cash", "card", "fraud"]
    assert compiled.symbols["decision"] == ("A", "B", "C")


def test_build_request_orders_ordinal_candidates_by_value() -> None:
    example = _ordinal_example()
    _, response_order, compiled = build_intern_decision_request(example)
    assert response_order == ["low", "medium", "high"]
    assert _options(compiled.questions["decision"]) == [
        ("0", "Low"),
        ("1", "Medium"),
        ("2", "High"),
    ]


def test_prediction_aligns_choice_distribution_to_candidate_order() -> None:
    example = _choice_example()
    _, response_order, _ = build_intern_decision_request(example)
    response = {
        "answers": {
            "decision": {
                "type": "choice",
                "probabilities": {"cash": 0.1, "card": 0.7, "fraud": 0.2},
                "confidence": 0.7,
                "choice": "card",
            }
        }
    }
    prediction = prediction_from_intern_decision_response(example, response, response_order)
    assert prediction.probabilities == pytest.approx([0.1, 0.7, 0.2])


def test_prediction_aligns_binary_noul_to_candidate_order() -> None:
    example = _binary_example()
    _, response_order, _ = build_intern_decision_request(example)
    response = {
        "answers": {
            "decision": {
                "type": "noul",
                "probabilities": {"no": 0.25, "yes": 0.75},
                "confidence": 0.75,
                "noul": 0.75,
            }
        }
    }
    prediction = prediction_from_intern_decision_response(example, response, response_order)
    assert prediction.probabilities == pytest.approx([0.75, 0.25])


def test_prediction_aligns_ordinal_distribution_to_declared_order() -> None:
    example = _ordinal_example()
    _, response_order, _ = build_intern_decision_request(example)
    response = {
        "answers": {
            "decision": {
                "type": "score",
                "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
                "confidence": 0.5,
                "score": 1.3,
            }
        }
    }
    prediction = prediction_from_intern_decision_response(example, response, response_order)
    assert prediction.probabilities == pytest.approx([0.5, 0.2, 0.3])


def test_prediction_rejects_missing_answers() -> None:
    example = _choice_example()
    with pytest.raises(TypeError):
        prediction_from_intern_decision_response(example, {"other": {}}, ["cash", "card", "fraud"])


def test_scale_probabilities_matches_published_formula_and_preserves_argmax() -> None:
    probabilities = {"A": 0.5, "B": 0.3, "C": 0.2}
    temperature = 1.99241824
    scaled = _scale_probabilities(probabilities, temperature)
    logs = {key: math.log(value) for key, value in probabilities.items()}
    maximum = max(logs.values())
    weights = {key: math.exp((value - maximum) / temperature) for key, value in logs.items()}
    total = sum(weights.values())
    expected = {key: weight / total for key, weight in weights.items()}
    assert scaled == pytest.approx(expected)
    assert _argmax(scaled) == _argmax(probabilities) == "A"
    assert sum(scaled.values()) == pytest.approx(1.0)


def test_scale_probabilities_identity_temperature() -> None:
    probabilities = {"yes": 0.75, "no": 0.25}
    assert _scale_probabilities(probabilities, 1.0) == probabilities


def test_validate_example_rejects_candidate_overflow() -> None:
    model = _model()
    candidates = [
        Candidate(id=f"c{index}", label=f"Candidate {index}")
        for index in range(INTERN_DECISION_MAX_OPTIONS + 1)
    ]
    example = DecisionExample(
        row_id="row-wide",
        task_name="wide",
        primitive=Primitive.CANDIDATE_SELECTION,
        family="wide",
        domain="ops",
        instruction="Pick one.",
        state={},
        candidates=candidates,
        gold_candidate_id="c0",
        gold_probabilities=[1.0] + [0.0] * (INTERN_DECISION_MAX_OPTIONS),
    )
    with pytest.raises(UnsupportedCandidateCount):
        model.validate_example(example)


def test_encode_marks_one_position_per_field_and_preserves_order() -> None:
    model = _model()
    input_ids, compiled, positions, response_order, request = model._encode(_choice_example())
    assert len(positions) == len(compiled.fields) == 1
    assert positions[0] >= 0
    assert response_order == ["cash", "card", "fraud"]
    assert request["model"] == INTERN_DECISION_MODEL_REPO
    assert input_ids[positions[0] + 1] == model.marker_id


def test_encode_rejects_over_long_input_without_truncation() -> None:
    model = _model()
    model.max_length = 8
    with pytest.raises(UnsupportedInputLength):
        model._encode(_choice_example())


def test_prompt_characters_counts_rendered_prompt() -> None:
    model = _model()
    lengths = model.prompt_characters(_choice_example())
    assert lengths > 0


def test_metadata_reports_pinned_native_contract() -> None:
    metadata = _model().metadata
    assert metadata["model"] == INTERN_DECISION_MODEL_REPO
    assert metadata["model_revision"] == INTERN_DECISION_MODEL_REVISION
    assert metadata["max_candidates"] == INTERN_DECISION_MAX_OPTIONS
    assert metadata["answer_symbols"] == INTERN_DECISION_ANSWER_SYMBOLS
    assert metadata["temperature"] == pytest.approx(1.99241824)
