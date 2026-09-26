"""Native adapter for the released `interfaze-ai/lev` System One readout.

lev scores every supplied option in one forward pass instead of generating text.
Its release directory carries a LoRA adapter, a candidate-path head, a
calibration profile, and the tokenizer the label-token readout depends on. The
adapter below hands each benchmark row to that published inference path and keeps
only the probabilities lev returns over the row's own candidates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.public_hf import (
    _align_probabilities,
    _binary_candidate_ids,
    _candidate_text,
    _read_json,
    _sha256_file,
    _state_text,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

LEV_MODEL_REPO = "interfaze-ai/lev"
LEV_MODEL_REVISION = "f8ef71157ec06a7d3b6435bc0756f9d735c33748"
LEV_CODE_REPO = "Abhinavexists/lev"
LEV_CODE_REVISION = "cf104b69329302e4eac674a730c71f3511047db8"
LEV_BASE_MODEL = "Qwen/Qwen3.5-4B"
LEV_PROMPT_STYLE = "chat"
LEV_MIN_CANDIDATES = 2
LEV_MAX_CANDIDATES = 255
LEV_MIN_SCORE_LEVELS = 2
LEV_MAX_SCORE_LEVELS = 10
LEV_RATING_LEVELS = 9
LEV_INPUT_CONTRACT_VERSION = "lev-system-one-release-v1"
PROBABILITY_SOURCE = "lev_calibrated_option_distribution"


class UnsupportedLevInput(ValueError):
    """An example is outside lev's published input contract."""


class LevHFDecisionModel:
    """Run the released lev checkpoint through its System One readout."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str = LEV_MODEL_REPO,
        model_revision: str = LEV_MODEL_REVISION,
        expected_weights_sha256: str | None = None,
        expected_mode_b_head_sha256: str | None = None,
    ) -> None:
        try:
            import lev
        except ImportError as error:
            raise RuntimeError("lev support requires the decision-bench[lev] extra") from error
        if model_repo != LEV_MODEL_REPO or model_revision != LEV_MODEL_REVISION:
            raise ValueError("lev evaluation requires its pinned model and revision")
        for name, expected in (
            ("adapter_model.safetensors", expected_weights_sha256),
            ("mode_b_head.pt", expected_mode_b_head_sha256),
        ):
            if expected is None:
                continue
            path = model_dir / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"lev release {name} does not match its pinned SHA-256")

        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.manifest = _read_lev_manifest(model_dir)

        # The released loader reads `lev_release.json` for the base model and
        # prompt style, applies the adapter, and loads the Mode B head plus the
        # calibration profile that ships beside the weights.
        self.engine = lev.load(model_dir)
        config = self.engine.config
        if self.engine.mode_b_head is None:
            raise ValueError(
                "lev release is missing mode_b_head.pt and cannot route every question"
            )
        if str(self.manifest.get("base_model")) != LEV_BASE_MODEL:
            raise ValueError("lev release names an unexpected base model")
        if str(config.prompt_style) != LEV_PROMPT_STYLE:
            raise ValueError("lev release uses an unexpected prompt style")

        self.base_model = LEV_BASE_MODEL
        self.prompt_style = str(config.prompt_style)
        self.noul_readout = str(config.noul_readout)
        self.order_average = bool(config.order_average)
        self.max_label_options = config.max_label_options

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the pinned release identity recorded with every run."""

        return {
            "model_type": "lev_system_one_release",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "code_repo": LEV_CODE_REPO,
            "code_revision": LEV_CODE_REVISION,
            "release_step": self.manifest.get("step"),
            "base_model": self.base_model,
            "base_model_resolution": (
                "the release manifest names base_model and the pinned lev loader resolves it "
                "from the Hub without an adapter-level revision pin"
            ),
            "prompt_style": self.prompt_style,
            "noul_readout": self.noul_readout,
            "readout": "label_token_mode_a_with_candidate_path_mode_b_fallback",
            "order_averaging": self.order_average,
            "max_label_options": self.max_label_options,
            "adapter_sha256": _file_sha256(self.model_dir, "adapter_model.safetensors"),
            "adapter_config_sha256": _file_sha256(self.model_dir, "adapter_config.json"),
            "calibration_sha256": _file_sha256(self.model_dir, "calibration.json"),
            "mode_b_head_sha256": _file_sha256(self.model_dir, "mode_b_head.pt"),
            "release_manifest_sha256": _file_sha256(self.model_dir, "lev_release.json"),
            "probability_source": PROBABILITY_SOURCE,
            "input_truncation_policy": "no_published_limit_and_no_truncation",
            "prompt_characters_method": "rendered_state_characters_plus_serialized_question",
            "candidate_boundary": {
                "min": LEV_MIN_CANDIDATES,
                "max": LEV_MAX_CANDIDATES,
                "unique_labels": True,
                "binary_semantics": "row must expose unambiguous true/false candidates",
                "score_levels": {"min": LEV_MIN_SCORE_LEVELS, "max": LEV_MAX_SCORE_LEVELS},
                "noul_rating_levels": LEV_RATING_LEVELS,
            },
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        """Return a deterministic size proxy for memory-aware batch construction."""

        question = self._question(example)
        return len(_state_text(example)) + len(json.dumps(question, ensure_ascii=False))

    def validate_example(self, example: DecisionExample) -> None:
        self._question(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        responses: list[HFDecisionResponse] = []
        for example in examples:
            question = self._question(example)
            candidate_ids = [candidate.id for candidate in example.candidates]
            started = time.perf_counter()
            native = self.engine.system_one(example.state, {"decision": question})
            latency_seconds = time.perf_counter() - started
            answer = native.answers.get("decision")
            if answer is None:
                raise RuntimeError("lev returned no answer for the decision question")
            native_candidate_ids, native_probabilities, extra = self._native_distribution(
                example, answer
            )
            aligned = _align_probabilities(example, native_candidate_ids, native_probabilities)
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=aligned),
                    request={"state": example.state, "questions": {"decision": question}},
                    response={
                        "native_candidate_ids": native_candidate_ids,
                        "native_probabilities": native_probabilities,
                        "probabilities": aligned,
                        "raw_response": native.model_dump(mode="json"),
                        **extra,
                    },
                    latency_seconds=latency_seconds,
                    input_contract={
                        "policy_version": LEV_INPUT_CONTRACT_VERSION,
                        "primitive": example.primitive.value,
                        "question_type": question["type"],
                        "candidate_ids": candidate_ids,
                        "candidate_labels": [
                            candidate.label for candidate in example.candidates
                        ],
                        "prompt_style": self.prompt_style,
                        "noul_readout": self.noul_readout,
                        "order_averaging": self.order_average,
                        "truncation_policy": "no_published_limit_and_no_truncation",
                        "truncated": False,
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _question(self, example: DecisionExample) -> dict[str, Any]:
        """Build the lev System One question for one benchmark row."""

        candidates = example.candidates
        if not LEV_MIN_CANDIDATES <= len(candidates) <= LEV_MAX_CANDIDATES:
            raise UnsupportedLevInput(
                f"candidate count {len(candidates)} outside "
                f"[{LEV_MIN_CANDIDATES}, {LEV_MAX_CANDIDATES}]"
            )
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            try:
                false_id, true_id = _binary_candidate_ids(example)
            except ValueError as error:
                raise UnsupportedLevInput(str(error)) from error
            by_id = {candidate.id: candidate for candidate in candidates}
            return {
                "type": "noul",
                "instructions": example.instruction,
                "criteria": {
                    "true": _candidate_text(by_id[true_id].label, by_id[true_id].description),
                    "false": _candidate_text(by_id[false_id].label, by_id[false_id].description),
                },
            }
        if example.primitive is Primitive.CANDIDATE_SELECTION:
            labels = [candidate.label for candidate in candidates]
            if len(set(labels)) != len(labels):
                raise UnsupportedLevInput("lev choice keys require unique candidate labels")
            # lev renders the label as the option key, so only the description is extra text.
            return {
                "type": "choice",
                "instructions": example.instruction,
                "criteria": {
                    candidate.label: candidate.description for candidate in candidates
                },
            }
        levels = [
            _candidate_text(candidate.label, candidate.description) for candidate in candidates
        ]
        if not LEV_MIN_SCORE_LEVELS <= len(levels) <= LEV_MAX_SCORE_LEVELS:
            raise UnsupportedLevInput(
                f"lev score readout requires {LEV_MIN_SCORE_LEVELS}-{LEV_MAX_SCORE_LEVELS} "
                f"levels, got {len(levels)}"
            )
        return {"type": "score", "instructions": example.instruction, "criteria": levels}

    def _native_distribution(
        self, example: DecisionExample, answer: Any
    ) -> tuple[list[str], list[float], dict[str, Any]]:
        """Map lev's typed answer onto the row's candidates without reordering."""

        candidates = example.candidates
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            false_id, true_id = _binary_candidate_ids(example)
            probability_true = float(answer.noul)
            if not 0.0 <= probability_true <= 1.0:
                raise RuntimeError("lev returned a Noul probability outside [0, 1]")
            extra: dict[str, Any] = {}
            ratings = answer.probabilities
            if ratings:
                extra["native_rating_probabilities"] = {
                    str(level): float(value) for level, value in sorted(ratings.items())
                }
            return (
                [true_id, false_id],
                [probability_true, 1.0 - probability_true],
                extra,
            )
        if example.primitive is Primitive.CANDIDATE_SELECTION:
            by_label = {candidate.label: candidate.id for candidate in candidates}
            returned = {str(label): float(value) for label, value in answer.probabilities.items()}
            if set(returned) != set(by_label):
                raise RuntimeError("lev returned incomplete candidate probabilities")
            by_id = {by_label[label]: value for label, value in returned.items()}
            return (
                [candidate.id for candidate in candidates],
                [by_id[candidate.id] for candidate in candidates],
                {},
            )
        by_level = {int(level): float(value) for level, value in answer.probabilities.items()}
        if set(by_level) != set(range(len(candidates))):
            raise RuntimeError("lev returned incomplete score-level probabilities")
        return (
            [candidate.id for candidate in candidates],
            [by_level[index] for index in range(len(candidates))],
            {"native_score_levels": {str(level): by_level[level] for level in sorted(by_level)}},
        )


def _read_lev_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "lev_release.json"
    if not path.is_file():
        raise ValueError(f"lev model directory is missing lev_release.json: {model_dir}")
    return _read_json(path)


def _file_sha256(model_dir: Path, name: str) -> str | None:
    path = model_dir / name
    return _sha256_file(path) if path.is_file() else None
