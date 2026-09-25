"""Native classification readout for fastino/GLiNER2.5-Decide."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.schemas import DecisionExample, DecisionPrediction

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "0872ab149bd2f8a50ed5fc7ad8cfc3293e9a3bad"
TOKENIZER_REVISION = MODEL_REVISION
MAX_ENCODER_TOKENS = 512
PROBABILITY_SOURCE = "raw_classification_logits_softmax"
TASK_NAME = "decision"
MIN_CANDIDATES = 2
MAX_CANDIDATES = 255


class UnsupportedGLiNER25Input(ValueError):
    """An example is outside the model's published input contract."""


class GLiNER25DecideModel:
    """Adapter for the GLiNER2.5-Decide classification checkpoint."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str,
        model_revision: str,
    ) -> None:
        import torch
        from gliner2.classification import Classifier

        if model_repo != MODEL_ID or model_revision != MODEL_REVISION:
            raise ValueError("GLiNER2.5-Decide requires its pinned model and revision")
        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._classifier = Classifier.from_pretrained(str(model_dir))
        self._classifier.to(device=self.device).eval()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "gliner25_decide_classifier",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "tokenizer_revision": TOKENIZER_REVISION,
            "probability_source": PROBABILITY_SOURCE,
            "max_encoder_tokens": MAX_ENCODER_TOKENS,
            "task_name": TASK_NAME,
            "device": self.device,
            "input_truncation_policy": "reject_over_512_encoded_tokens",
            "candidate_boundary": {
                "min": MIN_CANDIDATES,
                "max": MAX_CANDIDATES,
                "unique_labels": True,
                "effective_limit": "512 encoded tokens including schema",
            },
            "calibration": "not established; softmax is conditional option preference",
        }

    def _format_input(self, example: DecisionExample) -> str:
        """Build the model-facing text from instruction and state."""
        parts = [example.instruction, "State: " + _render_state(example.state)]
        descriptions = [
            f"{candidate.label}: {candidate.description}"
            for candidate in example.candidates
            if candidate.description
        ]
        if descriptions:
            parts.append("Candidate descriptions:\n" + "\n".join(descriptions))
        return "\n\n".join(parts)

    def prompt_characters(self, example: DecisionExample) -> int:
        return len(self._format_input(example)) + sum(
            len(candidate.label) for candidate in example.candidates
        )

    def _build_schema(self, example: DecisionExample) -> Any:
        from gliner2.classification import ClassificationSchema

        labels = [candidate.label for candidate in example.candidates]
        schema = ClassificationSchema().single(TASK_NAME, labels)
        return schema

    def _prepare(self, example: DecisionExample) -> tuple[str, Any, int]:
        candidates = example.candidates
        if not (MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES):
            raise UnsupportedGLiNER25Input(
                f"candidate count {len(candidates)} outside [{MIN_CANDIDATES}, {MAX_CANDIDATES}]"
            )
        ids = [candidate.id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise UnsupportedGLiNER25Input("duplicate candidate ids are unsupported")
        labels = [candidate.label for candidate in candidates]
        if any(not label or not label.strip() for label in labels):
            raise UnsupportedGLiNER25Input("empty candidate labels are unsupported")
        if len(set(labels)) != len(labels):
            raise UnsupportedGLiNER25Input("duplicate candidate labels are unsupported")
        schema = self._build_schema(example)
        compiled = self._classifier.compile_schema(schema)
        text = self._format_input(example)
        batch = self._classifier.scorer.processor.collate_fn_inference(
            [(text, compiled.build())], max_len=None, error_policy="raise"
        )
        token_count = int(batch.input_ids.shape[1])
        if token_count > MAX_ENCODER_TOKENS:
            raise UnsupportedGLiNER25Input(
                f"encoded input is {token_count} tokens, over the "
                f"{MAX_ENCODER_TOKENS}-token encoder limit"
            )
        return text, compiled, token_count

    def validate_example(self, example: DecisionExample) -> None:
        self._prepare(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        from gliner2.classification import ClassificationConfig

        from decision_bench.models.public_hf import _align_probabilities

        responses: list[HFDecisionResponse] = []
        for example in examples:
            text, compiled, token_count = self._prepare(example)
            candidate_ids = [candidate.id for candidate in example.candidates]
            labels = [candidate.label for candidate in example.candidates]
            start = time.perf_counter()
            scores = self._classifier.score(
                text, compiled, config=ClassificationConfig(max_len=None)
            )
            latency_seconds = time.perf_counter() - start
            task_scores = scores.tasks[TASK_NAME]
            if len(task_scores) != len(labels) or set(task_scores) != set(labels):
                raise RuntimeError("GLiNER returned incomplete candidate scores")
            raw_logits = [float(scores.logit(TASK_NAME, label)) for label in labels]
            native_probabilities = [float(scores.probability(TASK_NAME, label)) for label in labels]
            aligned_probabilities = _align_probabilities(
                example, candidate_ids, native_probabilities
            )
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=aligned_probabilities),
                    request={
                        "text": text,
                        "schema": compiled.build(),
                        "candidate_ids": candidate_ids,
                        "labels": labels,
                    },
                    response={
                        "native_candidate_ids": candidate_ids,
                        "native_logits": raw_logits,
                        "native_probabilities": native_probabilities,
                        "probabilities": aligned_probabilities,
                    },
                    latency_seconds=latency_seconds,
                    input_contract={
                        "candidate_ids": candidate_ids,
                        "labels": labels,
                        "policy_version": "gliner25-decide-exact-encoder-v1",
                        "original_input_tokens": token_count,
                        "final_input_tokens": token_count,
                        "max_input_tokens": MAX_ENCODER_TOKENS,
                        "truncated": False,
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses


def _render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
