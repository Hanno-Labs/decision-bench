from __future__ import annotations

import json

import pytest

from decision_bench.models.hf import render_hf_decision_prompt
from decision_bench.models.nimble_hf import build_nimble_input
from decision_bench.models.public_hf import (
    NanoJevHFDecisionModel,
    OpenJevHFDecisionModel,
    SystemOneHFDecisionModel,
    Tev1HFDecisionModel,
    UnsupportedCandidateCount,
    UnsupportedInputLength,
)
from decision_bench.prompt import fit_example_to_token_budget
from decision_bench.schemas import Candidate, DecisionExample, Primitive


def _example() -> DecisionExample:
    return DecisionExample(
        row_id="row-1",
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


class _Tev1TokenizerStub:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        **_: object,
    ) -> str | list[int]:
        rendered = json.dumps(messages)
        return list(range(len(rendered))) if tokenize else rendered


def _tev1_model() -> Tev1HFDecisionModel:
    model = Tev1HFDecisionModel.__new__(Tev1HFDecisionModel)
    model.tokenizer = _Tev1TokenizerStub()
    model._letter_token_ids = list(range(24))
    return model


def test_hf_prompt_uses_stable_slots_and_preserves_candidate_identity() -> None:
    tokens = [f"<|decision_{index:03d}|>" for index in range(256)]
    content, order, mapping = render_hf_decision_prompt(_example(), tokens, seed=7)
    repeated = render_hf_decision_prompt(_example(), tokens, seed=7)
    payload = json.loads(content)

    assert repeated == (content, order, mapping)
    assert sorted(order) == [0, 1, 2]
    assert sorted(mapping.values()) == [0, 1, 2]
    assert payload["schema"] == "bosun-decision-prompt-v3-stable-slots"
    assert payload["question"]["type"] == "choice"
    assert payload["question"]["criteria_fields"] == ["t", "o", "n", "d"]
    assert [row[0] for row in payload["question"]["criteria"]] == tokens[:3]
    assert {row[2] for row in payload["question"]["criteria"]} == {
        "Cash support",
        "Card support",
        "Fraud support",
    }


def test_hf_prompt_maps_primitive_names_to_native_contract() -> None:
    tokens = [f"<|decision_{index:03d}|>" for index in range(256)]
    binary = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": _example().candidates[:2],
            "gold_probabilities": [0.0, 1.0],
        }
    )
    ordinal = _example().model_copy(
        update={
            "primitive": Primitive.ORDINAL_SCORING,
            "candidates": [
                candidate.model_copy(update={"ordinal_value": float(index)})
                for index, candidate in enumerate(_example().candidates)
            ],
        }
    )

    binary_payload = json.loads(render_hf_decision_prompt(binary, tokens, seed=0)[0])
    ordinal_payload = json.loads(render_hf_decision_prompt(ordinal, tokens, seed=0)[0])
    assert binary_payload["question"]["type"] == "noul"
    assert ordinal_payload["question"]["type"] == "score"


def test_hf_prompt_factors_uniform_json_description_fields() -> None:
    tokens = [f"<|decision_{index:03d}|>" for index in range(256)]
    example = _example().model_copy(
        update={
            "candidates": [
                candidate.model_copy(
                    update={"description": json.dumps({"city": candidate.id, "rank": index})}
                )
                for index, candidate in enumerate(_example().candidates)
            ]
        }
    )
    payload = json.loads(render_hf_decision_prompt(example, tokens, seed=0)[0])

    assert payload["question"]["criteria_fields"] == ["t", "o", "n", "city", "rank"]
    assert all(len(row) == 5 for row in payload["question"]["criteria"])


def test_hf_prompt_truncation_preserves_slots_and_candidate_identity() -> None:
    tokens = [f"<|decision_{index:03d}|>" for index in range(256)]
    example = _example().model_copy(
        update={"state": {"request": "very long state " * 100}}
    )
    fitted, report = fit_example_to_token_budget(
        example,
        max_input_tokens=600,
        count_tokens=lambda value: len(
            render_hf_decision_prompt(value, tokens, seed=7)[0]
        ),
    )
    content, order, mapping = render_hf_decision_prompt(fitted, tokens, seed=7)

    assert len(content) <= 600
    assert sorted(order) == [0, 1, 2]
    assert set(mapping) == {"cash", "card", "fraud"}
    assert fitted.gold_candidate_id == example.gold_candidate_id
    assert report.truncated is True


def test_nimble_input_matches_native_boolean_contract() -> None:
    example = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="false", label="No", description="Criterion is absent."),
                Candidate(id="true", label="Yes", description="Criterion is present."),
            ],
            "gold_candidate_id": "true",
            "gold_probabilities": [0.0, 1.0],
        }
    )
    context, schema, output_ids = build_nimble_input(example)

    assert json.loads(context) == example.state
    assert schema["decision"] == {
        "description": "Route the request.",
        "type": "boolean",
        "choices": [False, True],
        "choice_descriptions": {
            "false": "No: Criterion is absent.",
            "true": "Yes: Criterion is present.",
        },
    }
    assert output_ids == ["false", "true"]


def test_nimble_input_maps_noncanonical_binary_ids_by_semantic_label() -> None:
    example = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="label-1", label="Yes", description="Outcome applies."),
                Candidate(id="label-0", label="No", description="Outcome does not apply."),
            ],
            "gold_candidate_id": "label-1",
            "gold_probabilities": [1.0, 0.0],
        }
    )
    _, schema, output_ids = build_nimble_input(example)

    assert schema["decision"]["choice_descriptions"] == {
        "false": "No: Outcome does not apply.",
        "true": "Yes: Outcome applies.",
    }
    assert output_ids == ["label-0", "label-1"]


def test_nimble_input_maps_ordinal_codes_back_to_candidate_ids() -> None:
    example = _example().model_copy(
        update={
            "primitive": Primitive.ORDINAL_SCORING,
            "candidates": [
                candidate.model_copy(update={"ordinal_value": float(index)})
                for index, candidate in enumerate(_example().candidates)
            ],
        }
    )
    _, schema, output_ids = build_nimble_input(example)

    assert schema["decision"]["choices"] == ["0", "1", "2"]
    assert output_ids == ["cash", "card", "fraud"]


def test_nanojev_input_uses_native_boolean_contract() -> None:
    example = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="label-1", label="Yes", description="Outcome applies."),
                Candidate(id="label-0", label="No", description="Outcome does not apply."),
            ],
            "gold_candidate_id": "label-1",
            "gold_probabilities": [1.0, 0.0],
        }
    )
    model = NanoJevHFDecisionModel.__new__(NanoJevHFDecisionModel)
    request, output_ids = model._build_request(example)

    assert output_ids == ["label-0", "label-1"]
    question = request["states"][0]["questions"]["decision"]
    assert question == {
        "type": "boolean",
        "instructions": "Route the request.",
        "criteria": {
            "false": "No: Outcome does not apply.",
            "true": "Yes: Outcome applies.",
        },
    }


def test_openjev_input_uses_noul_and_ordered_score_contracts() -> None:
    binary = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="false", label="No"),
                Candidate(id="true", label="Yes"),
            ],
            "gold_candidate_id": "true",
            "gold_probabilities": [0.0, 1.0],
        }
    )
    ordinal = _example().model_copy(
        update={
            "primitive": Primitive.ORDINAL_SCORING,
            "candidates": [
                candidate.model_copy(update={"ordinal_value": float(index)})
                for index, candidate in enumerate(_example().candidates)
            ],
        }
    )
    model = OpenJevHFDecisionModel.__new__(OpenJevHFDecisionModel)

    _, binary_question, binary_ids = model._build_input(binary)
    _, ordinal_question, ordinal_ids = model._build_input(ordinal)

    assert binary_question == {"type": "noul", "instructions": "Route the request."}
    assert binary_ids == ["false", "true"]
    assert ordinal_question["type"] == "score"
    assert ordinal_question["options"] == [
        "Cash support",
        "Card support: Retained cards",
        "Fraud support",
    ]
    assert ordinal_ids == ["cash", "card", "fraud"]


def test_system_one_input_uses_published_yes_no_order() -> None:
    example = _example().model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="false", label="No"),
                Candidate(id="true", label="Yes"),
            ],
            "gold_candidate_id": "true",
            "gold_probabilities": [0.0, 1.0],
        }
    )
    model = SystemOneHFDecisionModel.__new__(SystemOneHFDecisionModel)
    _, _, options, output_ids = model._build_input(example)

    assert options == ["yes", "no"]
    assert output_ids == ["true", "false"]


def test_tev1_preserves_candidate_order_for_all_primitives() -> None:
    choice = _example()
    binary = choice.model_copy(
        update={
            "primitive": Primitive.BINARY_CLASSIFICATION,
            "candidates": [
                Candidate(id="true", label="Yes"),
                Candidate(id="false", label="No"),
            ],
            "gold_candidate_id": "true",
            "gold_probabilities": [1.0, 0.0],
        }
    )
    ordinal = choice.model_copy(
        update={
            "primitive": Primitive.ORDINAL_SCORING,
            "candidates": [
                candidate.model_copy(update={"ordinal_value": float(index)})
                for index, candidate in enumerate(choice.candidates)
            ],
        }
    )
    model = _tev1_model()

    for example in (binary, choice, ordinal):
        prepared = model._prepare(example)
        options = json.loads(prepared["user_prompt"])["options"]
        assert [option["key"] for option in options] == [
            candidate.id for candidate in example.candidates
        ]
        assert [option["label"] for option in options] == list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(example.candidates)]
        )
        assert prepared["candidate_ids"] == [candidate.id for candidate in example.candidates]


def test_tev1_rejects_candidate_counts_above_native_limit_as_unsupported() -> None:
    example = _example().model_copy(
        update={
            "candidates": [
                Candidate(id=f"option-{index}", label=f"Option {index}") for index in range(25)
            ],
            "gold_candidate_id": "option-0",
            "gold_probabilities": [1.0] + [0.0] * 24,
        }
    )

    with pytest.raises(UnsupportedCandidateCount, match="at most 24 candidates"):
        _tev1_model().validate_example(example)


def test_tev1_marks_unfit_minimum_prompt_as_unsupported() -> None:
    example = _example().model_copy(
        update={
            "candidates": [
                Candidate(id=f"candidate-{index}-" + "x" * 100, label=f"Option {index}")
                for index in range(24)
            ],
            "gold_candidate_id": "candidate-0-" + "x" * 100,
            "gold_probabilities": [1.0] + [0.0] * 23,
        }
    )

    with pytest.raises(UnsupportedInputLength, match="minimally preserved text exceed"):
        _tev1_model().validate_example(example)
