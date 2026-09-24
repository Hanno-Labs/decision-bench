"""Native adapter for MoJev's packed candidate scorer."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.public_hf import _align_probabilities
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

MOJEV_MODEL_REPO = "MoLeMo-Lab/mojev"
MOJEV_MODEL_REVISION = "0c8695b6252f4205907433d4e196a94f032e60c3"
MOJEV_CODE_REVISION = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MOJEV_CONTEXT_TOKENS = 16_384
MOJEV_MAX_OPTIONS = 255


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _state_text(example: DecisionExample) -> str:
    return _render(example.state)


def _option_texts(example: DecisionExample) -> tuple[list[str], list[str]]:
    """Return the upstream MoJev menu strings and their DecisionBench IDs."""

    if example.primitive in {
        Primitive.BINARY_CLASSIFICATION,
        Primitive.CANDIDATE_SELECTION,
    }:
        options = [
            candidate.label
            if candidate.description in (None, "", candidate.label)
            else f"{candidate.label}: {_render(candidate.description)}"
            for candidate in example.candidates
        ]
        return options, [candidate.id for candidate in example.candidates]
    else:
        # MoJev's native score contract passes ordered score levels as strings.
        options = [_render(candidate.label) for candidate in example.candidates]
    return options, [candidate.id for candidate in example.candidates]


class MoJevHFDecisionModel:
    """Run MoJev through its published schema, packing, and softmax contract."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str = MOJEV_MODEL_REPO,
        model_revision: str = MOJEV_MODEL_REVISION,
        context_tokens: int = MOJEV_CONTEXT_TOKENS,
    ) -> None:
        try:
            import torch
            from mojev.data import Example
            from mojev.full import move, packed_collate, sort_candidates
            from mojev.schema import Field, Schema
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("MoJev support requires the decision-bench[mojev] extra") from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("MoJev evaluation requires a BF16-capable CUDA GPU")

        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.context_tokens = context_tokens
        self._torch = torch
        self._Example = Example
        self._Field = Field
        self._Schema = Schema
        self._move = move
        self._packed_collate = packed_collate
        self._sort_candidates = sort_candidates

        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        ).to("cuda").eval()

        # MoJev creates its additive tree mask in float32, while this encoder
        # runs in bf16. Qwen3.5 requires the attention bias to match the query
        # dtype. Keep the same mask semantics and use a finite bf16 floor.
        encoder_dtype = next(self.model.encoder.parameters()).dtype
        original_build_mask = self.model.build_mask

        def build_mask_for_encoder(
            context_span: Any,
            field_span: Any,
            option_span: Any,
        ) -> Any:
            mask = original_build_mask(context_span, field_span, option_span)
            mask = mask.to(dtype=encoder_dtype)
            return mask.masked_fill(
                torch.isneginf(mask), torch.finfo(encoder_dtype).min
            )

        self.model.build_mask = build_mask_for_encoder
        self.processor = cast(
            Any,
            AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
            ),
        )
        self.tokenizer = self.processor.tokenizer
        configured_context = int(self.model.config.context_tokens)
        if context_tokens < 1 or context_tokens > configured_context:
            raise ValueError(
                "MoJev context_tokens must be positive and no larger than the model's "
                f"configured context ({configured_context})"
            )
        self.context_tokens = context_tokens
        self.configured_context_tokens = configured_context

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "mojev_packed_candidate_scorer",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "code_revision": MOJEV_CODE_REVISION,
            "base_model": getattr(self.model.config, "encoder_name", None),
            "parameter_count": 854_036_544,
            "max_candidates": MOJEV_MAX_OPTIONS,
            "context_tokens": self.context_tokens,
            "configured_context_tokens": self.configured_context_tokens,
            "probability_source": "native_candidate_logits_softmax",
            "input_truncation_policy": (
                "mojev_packed_collate_tokenizer_truncation_at_context_tokens_v1"
            ),
            "model_dtype_policy": "published_mixed_bfloat16_encoder_float32_readout",
            "attention_mask_dtype_policy": "encoder_dtype_finite_floor_v1",
        }

    def batch_key(
        self, example: DecisionExample
    ) -> tuple[str, str, int, tuple[str, ...]]:
        options, _ = _option_texts(example)
        order = self._sort_candidates(options)
        sorted_options = tuple(options[index] for index in order)
        # Choice prompts list at most eight options. Above that, MoJev's prompt
        # contains the instruction and cardinality but omits option names.
        prompt_options = sorted_options if len(sorted_options) <= 8 else ()
        return example.task_name, example.instruction, len(sorted_options), prompt_options

    def prompt_characters(self, example: DecisionExample) -> int:
        options, _ = _option_texts(example)
        return (
            len(_state_text(example))
            + len(example.instruction)
            + sum(len(option) for option in options)
        )

    def validate_example(self, example: DecisionExample) -> None:
        options, _ = _option_texts(example)
        if not 2 <= len(options) <= MOJEV_MAX_OPTIONS:
            raise ValueError(f"MoJev supports 2 to {MOJEV_MAX_OPTIONS} candidates")
        if len(set(options)) != len(options):
            raise ValueError("MoJev candidate text must be unique")

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        if not examples:
            return []
        started = time.monotonic()
        batch_key = self.batch_key(examples[0])
        if any(self.batch_key(example) != batch_key for example in examples[1:]):
            raise ValueError("MoJev batches must share the same field prompt")
        prepared: list[dict[str, Any]] = []
        sorted_menus: list[tuple[str, ...]] = []
        for example in examples:
            options, candidate_ids = _option_texts(example)
            order = self._sort_candidates(options)
            sorted_options = tuple(options[index] for index in order)
            sorted_ids = [candidate_ids[index] for index in order]
            sorted_menus.append(sorted_options)
            prepared.append(
                {
                    "example": example,
                    "state": _state_text(example),
                    "options": options,
                    "sorted_options": sorted_options,
                    "native_candidate_ids": sorted_ids,
                }
            )

        # The shared evaluator groups compatible instructions/menus. Each row
        # then keeps its own state and candidate order, matching MoJev serving.
        schema = self._Schema(
            (
                self._Field(
                    examples[0].task_name,
                    "choice",
                    sorted_menus[0],
                    examples[0].instruction,
                ),
            )
        )
        examples_for_model = [
            self._Example(
                context=row["state"],
                labels=(0,),
                options=(row["sorted_options"],),
            )
            for row in prepared
        ]
        collate = self._packed_collate(
            self.tokenizer,
            schema,
            self.context_tokens,
            processor=self.processor,
        )
        model_batch = self._move(collate(examples_for_model), self._torch.device("cuda"))

        with self._torch.inference_mode():
            logits = self.model(model_batch).float()
        elapsed = time.monotonic() - started

        responses: list[HFDecisionResponse] = []
        for index, row in enumerate(prepared):
            example = cast(DecisionExample, row["example"])
            native_candidate_ids = cast(list[str], row["native_candidate_ids"])
            option_count = len(native_candidate_ids)
            native_logits_tensor = logits[index, 0, :option_count].double()
            native_probabilities = cast(
                list[float], native_logits_tensor.softmax(-1).tolist()
            )
            probabilities = _align_probabilities(
                example,
                native_candidate_ids,
                native_probabilities,
            )
            token_count = int(model_batch["packed_mask"][index].sum().item())
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "context": row["state"],
                        "field": example.task_name,
                        "instruction": example.instruction,
                        "candidate_options": row["options"],
                        "sorted_candidate_options": row["sorted_options"],
                        "candidate_ids_in_native_order": native_candidate_ids,
                        "packed_input_tokens": token_count,
                    },
                    response={
                        "native_candidate_ids": native_candidate_ids,
                        "native_logits": cast(list[float], native_logits_tensor.tolist()),
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(examples),
                    input_contract={
                        "policy_version": (
                            "mojev_packed_collate_tokenizer_truncation_at_context_tokens_v1"
                        ),
                        "context_tokens": self.context_tokens,
                        "packed_input_tokens": token_count,
                        "truncated": None,
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses
