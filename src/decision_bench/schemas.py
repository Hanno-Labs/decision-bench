"""Typed benchmark row and prediction contracts."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Primitive(StrEnum):
    """The decision primitive evaluated by a row."""

    BINARY_CLASSIFICATION = "binary_classification"
    CANDIDATE_SELECTION = "candidate_selection"
    ORDINAL_SCORING = "ordinal_scoring"


class Candidate(BaseModel):
    """One runtime-defined decision candidate."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    ordinal_value: float | None = None


class DecisionExample(BaseModel):
    """One frozen DecisionBench example."""

    model_config = ConfigDict(extra="forbid")

    row_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    primitive: Primitive
    family: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    state: JsonValue
    candidates: list[Candidate] = Field(min_length=2, max_length=255)
    gold_candidate_id: str = Field(min_length=1)
    gold_probabilities: list[float]
    source: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidates(self) -> DecisionExample:
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        if self.gold_candidate_id not in ids:
            raise ValueError("gold_candidate_id must name one candidate")
        if len(self.gold_probabilities) != len(ids):
            raise ValueError("gold_probabilities must align to candidate order")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.gold_probabilities
        ):
            raise ValueError("gold probabilities must be finite values from 0 to 1")
        if not math.isclose(sum(self.gold_probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("gold probabilities must sum to 1 within 1e-6")
        gold_index = max(range(len(ids)), key=self.gold_probabilities.__getitem__)
        if ids[gold_index] != self.gold_candidate_id:
            raise ValueError("gold_candidate_id must be an argmax of gold_probabilities")
        if self.primitive is Primitive.BINARY_CLASSIFICATION and len(ids) != 2:
            raise ValueError("binary classification requires exactly two candidates")
        if self.primitive is Primitive.ORDINAL_SCORING:
            values = [candidate.ordinal_value for candidate in self.candidates]
            if any(value is None for value in values):
                raise ValueError("ordinal scoring requires ordinal_value on every candidate")
            concrete = [float(value) for value in values if value is not None]
            if len(concrete) != len(set(concrete)):
                raise ValueError("ordinal candidates must have unique ordinal_value values")
        return self


class DecisionPrediction(BaseModel):
    """A calibrated distribution aligned to the input candidate order."""

    model_config = ConfigDict(extra="forbid")

    probabilities: list[float]

    @model_validator(mode="after")
    def validate_probabilities(self) -> DecisionPrediction:
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.probabilities
        ):
            raise ValueError("probabilities must be finite values from 0 to 1")
        total = sum(self.probabilities)
        if total <= 0.0:
            raise ValueError("probabilities must have a positive sum")
        self.probabilities = [value / total for value in self.probabilities]
        return self


class ScoredPrediction(BaseModel):
    """Prediction plus evaluator-derived values."""

    row_id: str
    probabilities: list[float]
    gold_probabilities: list[float]
    selected_candidate_id: str
    gold_candidate_id: str
    correct: bool
    expected_ordinal_score: float | None = None
