"""The Arbit adapter renders rows as its reference run did and reads decide() faithfully."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from decision_bench.models import arbit_hf
from decision_bench.models.arbit_hf import (
    ADAPTER_NAME,
    MAX_INPUT_TOKENS,
    MODEL_REPO,
    MODEL_REVISION,
    PROBABILITY_SOURCE,
    TABLE_FIELDS_LINE,
    ArbitHFDecisionModel,
    UnsupportedArbitInputLength,
    render_arbit_request,
)
from decision_bench.results import _classify_errors
from decision_bench.schemas import Candidate, DecisionExample, Primitive
from decision_bench.scoring import score_prediction

FIXTURE = Path(__file__).parent / "fixtures" / "arbit_hf_official_rows.jsonl"
OFFICIAL_ROWS = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _row_id(row: dict[str, Any]) -> str:
    return str(row["row_id"])


class _ReplayModel:
    """Answers only the reference run's exact requests, with the outputs recorded for them."""

    serve_max_length = MAX_INPUT_TOKENS

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {row["request_sha256"]: row for row in rows}

    def _row(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        return self.rows[_canonical_sha256({"state": state, "questions": questions})]

    def count_tokens(self, state: str, questions: dict[str, Any]) -> int:
        return int(self._row(state, questions)["input_contract"]["cpu_counted_input_tokens"])

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        row = self._row(state, questions)
        return {
            "answers": {"decision": row["response"]["answer"]},
            "probabilities": {"decision": row["response"]["option_probabilities"]},
            "input_tokens": row["input_contract"]["model_input_tokens"],
            "max_length": MAX_INPUT_TOKENS,
            "truncated": False,
        }


class _FakeModel:
    """Option probabilities proportional to 1, 2, 3, ... in the order the model lists options."""

    serve_max_length = MAX_INPUT_TOKENS

    def __init__(
        self,
        *,
        tokens: int | Callable[[str], int] = 100,
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.tokens = tokens
        self.mutate = mutate
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def count_tokens(self, state: str, questions: dict[str, Any]) -> int:
        return self.tokens(state) if callable(self.tokens) else self.tokens

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((state, questions))
        question = questions["decision"]
        criteria = question["criteria"]
        if question["type"] == "noul":
            options = ["true", "false"]
        elif question["type"] == "choice":
            options = sorted(criteria)
        else:
            options = [str(index) for index in range(len(criteria))]
        total = len(options) * (len(options) + 1) / 2
        probabilities = {option: (index + 1) / total for index, option in enumerate(options)}
        answer = {
            "type": question["type"],
            "probabilities": {key: round(value, 4) for key, value in probabilities.items()},
        }
        result: dict[str, Any] = {
            "answers": {"decision": answer},
            "probabilities": {"decision": probabilities},
            "input_tokens": self.count_tokens(state, questions),
            "max_length": MAX_INPUT_TOKENS,
            "truncated": False,
        }
        if self.mutate is not None:
            self.mutate(result)
        return result


def _adapter(model: Any) -> ArbitHFDecisionModel:
    adapter = ArbitHFDecisionModel.__new__(ArbitHFDecisionModel)
    adapter.model = model
    adapter.model_repo = MODEL_REPO
    adapter.model_revision = MODEL_REVISION
    adapter.attn_implementation = "sdpa"
    adapter.serve_max_input_tokens = int(model.serve_max_length)
    adapter._runtime = {}
    return adapter


def _example(
    primitive: Primitive,
    labels: list[str],
    *,
    ids: list[str] | None = None,
    descriptions: list[str | None] | None = None,
    ordinal_values: list[float] | None = None,
    state: Any = None,
) -> DecisionExample:
    count = len(labels)
    candidate_ids = ids or [f"candidate-{index}" for index in range(count)]
    values = ordinal_values or [float(index) for index in range(count)]
    candidates = [
        Candidate(
            id=candidate_ids[index],
            label=label,
            description=descriptions[index] if descriptions is not None else None,
            ordinal_value=values[index] if primitive is Primitive.ORDINAL_SCORING else None,
        )
        for index, label in enumerate(labels)
    ]
    return DecisionExample(
        row_id=f"arbit-{primitive.value}-{count}",
        task_name="routing",
        primitive=primitive,
        family="routing_triage",
        domain="financial",
        instruction="Choose the option named in the state.",
        state=state if state is not None else {"best_option": labels[0], "note": "café"},
        candidates=candidates,
        gold_candidate_id=candidate_ids[0],
        gold_probabilities=[1.0] + [0.0] * (count - 1),
    )


@pytest.mark.parametrize("row", OFFICIAL_ROWS, ids=_row_id)
def test_arbit_renders_the_reference_request(row: dict[str, Any]) -> None:
    example = DecisionExample.model_validate(row["example"])
    request = render_arbit_request(example)
    contract, response = row["input_contract"], row["response"]

    rendered = {"state": request.state, "questions": request.questions}
    assert _canonical_sha256(rendered) == row["request_sha256"]
    assert list(request.option_keys) == response["candidate_option_keys"]
    assert list(request.candidate_ids) == response["candidate_ids"]
    assert request.question_type == contract["question_type"]
    assert request.instruction_verbatim == contract["instruction_verbatim"]
    fields = list(request.criteria_fields) if request.criteria_fields is not None else None
    assert fields == contract["criteria_fields"]
    assert (fields is not None) == contract["table_form"]
    assert request.prompt_characters == contract["original_text_characters"]


@pytest.mark.parametrize("row", OFFICIAL_ROWS, ids=_row_id)
def test_arbit_reproduces_the_reference_record(row: dict[str, Any]) -> None:
    example = DecisionExample.model_validate(row["example"])
    adapter = _adapter(_ReplayModel(OFFICIAL_ROWS))

    adapter.validate_example(example)
    [result] = adapter.predict_batch([example])

    assert _canonical_sha256(result.request) == row["request_sha256"]
    assert result.response == row["response"]
    assert result.input_contract == row["input_contract"]
    scored = score_prediction(example, result.prediction)
    assert scored.selected_candidate_id == row["selected_candidate_id"]


def test_reference_rows_cover_the_input_contract() -> None:
    contracts = [row["input_contract"] for row in OFFICIAL_ROWS]
    assert len(OFFICIAL_ROWS) == 50
    assert {contract["question_type"] for contract in contracts} == {"noul", "choice", "score"}
    assert sum(contract["table_form"] for contract in contracts) >= 8
    assert max(len(row["example"]["candidates"]) for row in OFFICIAL_ROWS) == 255
    assert max(contract["model_input_tokens"] for contract in contracts) == 26_211
    assert not any(contract["truncated"] for contract in contracts)


@pytest.mark.parametrize("true_first", [True, False])
def test_arbit_binary_rows_put_the_first_candidate_on_the_true_side(true_first: bool) -> None:
    labels, ids = ["Yes", "No"], ["true", "false"]
    if not true_first:
        labels, ids = labels[::-1], ids[::-1]
    example = _example(
        Primitive.BINARY_CLASSIFICATION,
        labels,
        ids=ids,
        descriptions=["The criterion is present.", None],
    )
    model = _FakeModel()

    [result] = _adapter(model).predict_batch([example])

    question = result.request["questions"]["decision"]
    assert question["type"] == "noul"
    assert question["criteria"] == {
        "true": f"{labels[0]}: The criterion is present.",
        "false": labels[1],
    }
    assert result.response["candidate_option_keys"] == ["true", "false"]
    assert result.response["probabilities"] == [1 / 3, 1 - 1 / 3]
    assert result.prediction.probabilities == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize("count", [77, 255])
def test_arbit_choice_keys_keep_display_order(count: int) -> None:
    labels = [f"Queue {count - index}" for index in range(count)]
    example = _example(Primitive.CANDIDATE_SELECTION, labels)
    model = _FakeModel()

    [result] = _adapter(model).predict_batch([example])

    keys = [f"{index + 1:03d} {label}" for index, label in enumerate(labels)]
    assert list(result.request["questions"]["decision"]["criteria"]) == keys
    assert result.response["candidate_option_keys"] == keys
    assert result.response["candidate_ids"] == [candidate.id for candidate in example.candidates]
    total = count * (count + 1) / 2
    assert result.response["probabilities"] == [(index + 1) / total for index in range(count)]
    assert result.input_contract is not None
    assert result.input_contract["question_type"] == "choice"


def test_arbit_score_levels_follow_display_order() -> None:
    example = _example(
        Primitive.ORDINAL_SCORING,
        ["Medium", "Low", "High"],
        descriptions=["Some risk.", None, "Severe risk."],
        ordinal_values=[1.0, 0.0, 2.0],
    )

    [result] = _adapter(_FakeModel()).predict_batch([example])

    question = result.request["questions"]["decision"]
    assert question["type"] == "score"
    assert question["criteria"] == ["Medium: Some risk.", "Low", "High: Severe risk."]
    assert result.response["candidate_option_keys"] == ["0", "1", "2"]
    assert result.response["probabilities"] == [1 / 6, 2 / 6, 3 / 6]


def test_arbit_table_form_keeps_every_value() -> None:
    records = [
        {"name": "Anna", "born": "1990-01-02", "city": "Delft", "aliases": ["A."]},
        {"name": "Bram", "born": None, "city": "Gouda", "aliases": []},
        {"name": "Cees", "born": "1971-12-30", "city": "Zeist", "aliases": ["C", "Ce"]},
    ]
    example = _example(
        Primitive.CANDIDATE_SELECTION,
        ["record A", "record B", "record C"],
        descriptions=[json.dumps(record) for record in records],
    )

    request = render_arbit_request(example)

    fields = ["aliases", "born", "city", "name"]
    assert request.criteria_fields == tuple(fields)
    question = request.questions["decision"]
    assert question["instructions"] == example.instruction + TABLE_FIELDS_LINE + json.dumps(fields)
    cells = list(question["criteria"].values())
    assert [dict(zip(fields, json.loads(cell), strict=True)) for cell in cells] == records


def test_arbit_passes_descriptions_verbatim_when_fields_differ() -> None:
    descriptions = [json.dumps({"a": 1}), json.dumps({"b": 2}), "plain text"]
    example = _example(
        Primitive.CANDIDATE_SELECTION, ["one", "two", "three"], descriptions=descriptions
    )

    request = render_arbit_request(example)

    assert request.criteria_fields is None
    assert request.questions["decision"]["instructions"] == example.instruction
    assert list(request.questions["decision"]["criteria"].values()) == descriptions


def test_arbit_state_uses_benchmark_storage_json() -> None:
    example = _example(
        Primitive.CANDIDATE_SELECTION, ["one", "two"], state={"z": [1, 2.5], "a": "café"}
    )

    request = render_arbit_request(example)

    assert request.state == '{"a":"café","z":[1,2.5]}'


def test_arbit_eligibility_boundary_rejects_without_truncating() -> None:
    example = _example(Primitive.CANDIDATE_SELECTION, ["one", "two", "three"])

    _adapter(_FakeModel(tokens=MAX_INPUT_TOKENS)).validate_example(example)
    [result] = _adapter(_FakeModel(tokens=MAX_INPUT_TOKENS)).predict_batch([example])
    assert result.input_contract is not None
    assert result.input_contract["model_input_tokens"] == MAX_INPUT_TOKENS
    assert result.input_contract["truncated"] is False

    over = _adapter(_FakeModel(tokens=MAX_INPUT_TOKENS + 1))
    with pytest.raises(UnsupportedArbitInputLength, match="rejected, not truncated"):
        over.validate_example(example)
    with pytest.raises(UnsupportedArbitInputLength):
        over.predict_batch([example])
    assert UnsupportedArbitInputLength.__name__.startswith("Unsupported")


def _drop_option(result: dict[str, Any]) -> None:
    key = sorted(result["probabilities"]["decision"])[0]
    del result["probabilities"]["decision"][key]
    del result["answers"]["decision"]["probabilities"][key]


def _add_option(result: dict[str, Any]) -> None:
    result["probabilities"]["decision"]["999 extra"] = 0.0
    result["answers"]["decision"]["probabilities"]["999 extra"] = 0.0


def _not_finite(result: dict[str, Any]) -> None:
    key = sorted(result["probabilities"]["decision"])[0]
    result["probabilities"]["decision"][key] = math.nan


def _not_normalized(result: dict[str, Any]) -> None:
    distribution = result["probabilities"]["decision"]
    for key in distribution:
        distribution[key] /= 2
        result["answers"]["decision"]["probabilities"][key] = round(distribution[key], 4)


def _rounded_disagrees(result: dict[str, Any]) -> None:
    key = sorted(result["probabilities"]["decision"])[0]
    result["answers"]["decision"]["probabilities"][key] += 0.01


def _rounded_not_a_number(result: dict[str, Any]) -> None:
    key = sorted(result["probabilities"]["decision"])[0]
    result["answers"]["decision"]["probabilities"][key] = math.nan


def _rounded_infinite(result: dict[str, Any]) -> None:
    key = sorted(result["probabilities"]["decision"])[0]
    result["answers"]["decision"]["probabilities"][key] = math.inf


def _truncated(result: dict[str, Any]) -> None:
    result["truncated"] = True


def _token_mismatch(result: dict[str, Any]) -> None:
    result["input_tokens"] -= 1


def _other_question(result: dict[str, Any]) -> None:
    result["answers"] = {"other": result["answers"]["decision"]}


@pytest.mark.parametrize(
    "mutate",
    [
        _drop_option,
        _add_option,
        _not_finite,
        _not_normalized,
        _rounded_disagrees,
        _rounded_not_a_number,
        _rounded_infinite,
        _truncated,
        _token_mismatch,
        _other_question,
    ],
)
def test_arbit_rejects_malformed_decisions(mutate: Callable[[dict[str, Any]], None]) -> None:
    example = _example(Primitive.CANDIDATE_SELECTION, ["one", "two", "three"])

    with pytest.raises(RuntimeError):
        _adapter(_FakeModel(mutate=mutate)).predict_batch([example])


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_repo": "someone/arbit-1.7b"},
        {"model_revision": "0" * 40},
        {"base_revision": "0" * 40},
        {"expected_weights_sha256": "0" * 64},
    ],
)
def test_arbit_requires_its_pinned_release(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Arbit requires"):
        ArbitHFDecisionModel(**overrides)


class _OneBadRowModel(_FakeModel):
    """Healthy decisions, except a malformed answer for one row and an exception for another."""

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        if "crash" in state:
            raise RuntimeError("inference failed")
        result = super().decide(state, questions)
        if "malformed" in state:
            _rounded_not_a_number(result)
        return result


def test_arbit_failing_rows_do_not_fail_their_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_bench.evaluate as evaluate

    labels = ["one", "two", "three"]
    states = ["healthy a", "malformed", "healthy b", "crash", "healthy c", "long"]
    examples = [
        _example(Primitive.CANDIDATE_SELECTION, labels).model_copy(
            update={"row_id": f"row-{index}", "state": {"text": state}}
        )
        for index, state in enumerate(states)
    ]
    model = _OneBadRowModel(tokens=lambda state: MAX_INPUT_TOKENS + 1 if "long" in state else 90)
    monkeypatch.setattr(evaluate, "ArbitHFDecisionModel", lambda **kwargs: _adapter(model))

    summary = evaluate.run_public_hf_evaluation(
        examples,
        tmp_path,
        model_dir=tmp_path,
        model_type="arbit",
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        base_revision=None,
        expected_weights_sha256=None,
        batch_size=8,
        max_prompt_characters_per_batch=100_000,
        attn_implementation="sdpa",
    )

    assert summary["batch_size"] == 1
    assert summary["successful_rows"] == 3
    assert _classify_errors(tmp_path / "raw.jsonl") == {"unsupported_rows": 1, "error_rows": 2}
    records = {
        record["row_id"]: record
        for record in map(json.loads, (tmp_path / "raw.jsonl").read_text().splitlines())
    }
    assert {row_id for row_id, record in records.items() if record["status"] == "ok"} == {
        "row-0",
        "row-2",
        "row-4",
    }
    assert records["row-1"]["error_type"] == "RuntimeError"
    assert "not finite" in records["row-1"]["error"]
    assert records["row-3"]["error"] == "inference failed"
    assert records["row-5"]["error_type"] == "UnsupportedArbitInputLength"
    for row_id in ("row-0", "row-2", "row-4"):
        example = next(example for example in examples if example.row_id == row_id)
        [alone] = _adapter(_FakeModel(tokens=90)).predict_batch([example])
        assert records[row_id]["request"] == alone.request
        assert records[row_id]["response"] == alone.response
        assert records[row_id]["input_contract"] == alone.input_contract


def test_arbit_runs_through_the_public_hf_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decision_bench.evaluate as evaluate

    model = _FakeModel(tokens=lambda state: MAX_INPUT_TOKENS + 1 if "long" in state else 120)
    adapter = _adapter(model)
    constructed: list[dict[str, Any]] = []

    def construct(**kwargs: Any) -> ArbitHFDecisionModel:
        constructed.append(kwargs)
        return adapter

    monkeypatch.setattr(evaluate, "ArbitHFDecisionModel", construct)
    examples = [
        _example(Primitive.BINARY_CLASSIFICATION, ["Yes", "No"], ids=["true", "false"]),
        _example(Primitive.CANDIDATE_SELECTION, ["one", "two", "three"]),
        _example(Primitive.ORDINAL_SCORING, ["low", "high"], state={"text": "long"}),
    ]
    summary = evaluate.run_public_hf_evaluation(
        examples,
        tmp_path,
        model_dir=tmp_path,
        model_type="arbit",
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        base_revision=None,
        expected_weights_sha256=None,
        batch_size=2,
        max_prompt_characters_per_batch=100_000,
        attn_implementation="sdpa",
    )

    assert constructed == [
        {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "base_revision": None,
            "expected_weights_sha256": None,
            "attn_implementation": "sdpa",
        }
    ]
    assert summary["adapter"] == ADAPTER_NAME
    assert summary["probability_source"] == PROBABILITY_SOURCE
    assert summary["model_revision"] == MODEL_REVISION
    assert summary["successful_rows"] == 2
    assert summary["inspect_provenance_complete"]
    assert len(model.requests) == 2
    assert _classify_errors(tmp_path / "raw.jsonl") == {"unsupported_rows": 1, "error_rows": 0}
    records = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    contracts = [record["input_contract"] for record in records if record["status"] == "ok"]
    assert all(contract["policy_version"] == arbit_hf.POLICY_VERSION for contract in contracts)
    assert all(contract["truncated"] is False for contract in contracts)
