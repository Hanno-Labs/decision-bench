"""Native Hugging Face adapter for Bespoke-Nimble-9B."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.prompt import (
    TEXT_TRUNCATION_POLICY_VERSION,
    TextTruncationReport,
    fit_example_to_token_budget,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

NIMBLE_CONTRACT_VERSION = "schema_candidate_classification_v1"
NIMBLE_MAX_CHOICES = 26


def build_nimble_input(example: DecisionExample) -> tuple[str, dict[str, Any], list[str]]:
    """Translate one frozen row into Nimble's documented context/schema contract."""

    if not 2 <= len(example.candidates) <= NIMBLE_MAX_CHOICES:
        raise ValueError(f"Nimble supports 2 to {NIMBLE_MAX_CHOICES} candidates")
    context = (
        example.state
        if isinstance(example.state, str)
        else json.dumps(example.state, ensure_ascii=False)
    )
    if not context.strip():
        raise ValueError("Nimble requires a nonempty context")

    descriptions = [
        _candidate_description(candidate.label, candidate.description)
        for candidate in example.candidates
    ]
    candidate_ids = [candidate.id for candidate in example.candidates]
    field: dict[str, Any] = {"description": example.instruction}
    if example.primitive is Primitive.BINARY_CLASSIFICATION:
        by_id = dict(zip(candidate_ids, descriptions, strict=True))
        false_id, true_id = _binary_candidate_ids(example)
        field.update(
            type="boolean",
            choices=[False, True],
            choice_descriptions={"false": by_id[false_id], "true": by_id[true_id]},
        )
        output_candidate_ids = [false_id, true_id]
    elif example.primitive is Primitive.ORDINAL_SCORING:
        choices = [str(index) for index in range(len(example.candidates))]
        field.update(
            type="enum",
            choices=choices,
            choice_descriptions=dict(zip(choices, descriptions, strict=True)),
        )
        output_candidate_ids = candidate_ids
    else:
        field.update(
            type="enum",
            choices=candidate_ids,
            choice_descriptions=dict(zip(candidate_ids, descriptions, strict=True)),
        )
        output_candidate_ids = candidate_ids
    return context, {"decision": field}, output_candidate_ids


def _binary_candidate_ids(example: DecisionExample) -> tuple[str, str]:
    """Map frozen binary identities onto Nimble's false/true output order."""

    by_id = {candidate.id.casefold(): candidate.id for candidate in example.candidates}
    if set(by_id) == {"false", "true"}:
        return by_id["false"], by_id["true"]
    if set(by_id) == {"label-0", "label-1"}:
        return by_id["label-0"], by_id["label-1"]

    by_label = {
        candidate.label.strip().casefold(): candidate.id
        for candidate in example.candidates
    }
    false_id = by_label.get("no") or by_label.get("false")
    true_id = by_label.get("yes") or by_label.get("true")
    if false_id is None or true_id is None or false_id == true_id:
        raise ValueError(
            "Nimble boolean rows require unambiguous false/true or no/yes semantics"
        )
    return false_id, true_id


class NimbleHFDecisionModel:
    """Run the published Nimble LoRA through its exact one-token scoring contract."""

    def __init__(
        self,
        *,
        model_dir: Path,
        adapter_repo: str,
        adapter_revision: str,
        expected_adapter_sha256: str,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Nimble support requires the decision-bench[nimble] extra"
            ) from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Nimble evaluation requires a BF16-capable CUDA GPU")

        self.model_dir = model_dir
        self.adapter_repo = adapter_repo
        self.adapter_revision = adapter_revision
        self.attn_implementation = attn_implementation
        self._torch = torch
        self.contract = _read_json(model_dir / "schema_config.json")
        if self.contract.get("task") != NIMBLE_CONTRACT_VERSION:
            raise RuntimeError("unsupported Nimble task contract")
        self.max_length = int(self.contract["max_length"])
        if self.max_length != 2_048:
            raise RuntimeError(f"unexpected Nimble max_length={self.max_length}")

        adapter_path = model_dir / "adapter_model.safetensors"
        actual_adapter_sha256 = _sha256_file(adapter_path)
        if actual_adapter_sha256 != expected_adapter_sha256:
            raise RuntimeError("Nimble adapter SHA-256 does not match the pinned release")
        prompt_path = model_dir / "parallel_schema.py"
        prompt_sha256 = _sha256_file(prompt_path)
        if prompt_sha256 != self.contract.get("prompt_code_sha256"):
            raise RuntimeError("Nimble prompt code does not match schema_config.json")
        prompt_module = _load_module(prompt_path)
        self._prepare_prompts = cast(Callable[..., Any], prompt_module.prepare_prompts)

        self.tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            model_dir, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError("Nimble tokenizer does not declare a padding token")
        base_model: Any = cast(Any, Qwen3_5ForConditionalGeneration).from_pretrained(
            str(self.contract["model"]),
            revision=str(self.contract["revision"]),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).to("cuda")
        base_model.config.use_cache = False
        self.model = PeftModel.from_pretrained(base_model, model_dir).eval()
        self._adapter_sha256 = actual_adapter_sha256
        self._prompt_sha256 = prompt_sha256

    @property
    def metadata(self) -> dict[str, Any]:
        """Return immutable checkpoint and serving metadata for the run summary."""

        return {
            "model_type": "bespoke_nimble_native_candidate_logits",
            "adapter_repo": self.adapter_repo,
            "adapter_revision": self.adapter_revision,
            "adapter_sha256": self._adapter_sha256,
            "schema_config_sha256": _sha256_file(self.model_dir / "schema_config.json"),
            "prompt_code_sha256": self._prompt_sha256,
            "contract_version": NIMBLE_CONTRACT_VERSION,
            "base_model": self.contract["model"],
            "base_revision": self.contract["revision"],
            "probability_source": "native_candidate_logits_fp32_softmax",
            "temperature": 1.0,
            "temperature_fitted": False,
            "max_runtime_choices": NIMBLE_MAX_CHOICES,
            "max_length": self.max_length,
            "attn_implementation": self.attn_implementation,
            "input_truncation_policy": TEXT_TRUNCATION_POLICY_VERSION,
            "token_count_method": "published_nimble_prompt_and_checkpoint_tokenizer",
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        """Return deterministic model-facing text size for batch construction."""

        fitted, _ = self._fit(example)
        context, schema, _ = build_nimble_input(fitted)
        return len(context) + len(json.dumps(schema, ensure_ascii=False))

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        """Return native candidate distributions for one CUDA batch."""

        if not examples:
            return []
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        for example in examples:
            fitted, truncation = self._fit(example)
            context, schema, output_candidate_ids = build_nimble_input(fitted)
            prepared = self._prepare_prompts(
                self.tokenizer,
                context,
                schema,
                self.max_length,
                system_role=True,
            )
            if prepared.names != ["decision"]:
                raise RuntimeError("Nimble prompt compiler returned unexpected fields")
            input_ids = [int(value) for value in prepared.full_ids[0]]
            candidate_token_ids = [int(value) for value in prepared.candidate_ids[0]]
            if len(candidate_token_ids) != len(example.candidates):
                raise RuntimeError("Nimble candidate-token count changed")
            rows.append(
                {
                    "example": example,
                    "fitted": fitted,
                    "truncation": truncation,
                    "context": context,
                    "schema": schema,
                    "output_candidate_ids": output_candidate_ids,
                    "input_ids": input_ids,
                    "candidate_token_ids": candidate_token_ids,
                }
            )

        max_input_length = max(len(cast(list[int], row["input_ids"])) for row in rows)
        pad_id = int(self.tokenizer.pad_token_id)
        input_tensor = self._torch.tensor(
            [
                [pad_id] * (max_input_length - len(cast(list[int], row["input_ids"])))
                + cast(list[int], row["input_ids"])
                for row in rows
            ],
            dtype=self._torch.long,
            device="cuda",
        )
        attention_mask = self._torch.tensor(
            [
                [0] * (max_input_length - len(cast(list[int], row["input_ids"])))
                + [1] * len(cast(list[int], row["input_ids"]))
                for row in rows
            ],
            dtype=self._torch.long,
            device="cuda",
        )
        candidate_tensor = self._torch.tensor(
            [
                cast(list[int], row["candidate_token_ids"])
                + [0] * (NIMBLE_MAX_CHOICES - len(cast(list[int], row["candidate_token_ids"])))
                for row in rows
            ],
            dtype=self._torch.long,
            device="cuda",
        )
        candidate_mask = self._torch.tensor(
            [
                [True] * len(cast(list[int], row["candidate_token_ids"]))
                + [False]
                * (NIMBLE_MAX_CHOICES - len(cast(list[int], row["candidate_token_ids"])))
                for row in rows
            ],
            dtype=self._torch.bool,
            device="cuda",
        )
        with self._torch.inference_mode(), self._torch.autocast(
            "cuda", dtype=self._torch.bfloat16
        ):
            vocabulary_logits = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
            selected_logits = vocabulary_logits.gather(1, candidate_tensor).masked_fill(
                ~candidate_mask, -self._torch.inf
            )
        selected_logits = selected_logits.cpu()
        elapsed = time.monotonic() - started

        responses: list[HFDecisionResponse] = []
        for index, row in enumerate(rows):
            example = cast(DecisionExample, row["example"])
            candidate_count = len(example.candidates)
            logits_tensor = selected_logits[index, :candidate_count].double()
            if not self._torch.isfinite(logits_tensor).all():
                raise RuntimeError("Nimble returned non-finite candidate logits")
            probability_tensor = logits_tensor.softmax(-1)
            logits = cast(list[float], logits_tensor.tolist())
            native_probabilities = cast(list[float], probability_tensor.tolist())
            output_candidate_ids = cast(list[str], row["output_candidate_ids"])
            by_candidate_id = dict(zip(output_candidate_ids, native_probabilities, strict=True))
            probabilities = [by_candidate_id[candidate.id] for candidate in example.candidates]
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                raise RuntimeError("Nimble probabilities do not sum to one")
            truncation = cast(TextTruncationReport, row["truncation"])
            fitted = cast(DecisionExample, row["fitted"])
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "context": row["context"],
                        "schema": row["schema"],
                        "input_ids": row["input_ids"],
                        "input_length": len(cast(list[int], row["input_ids"])),
                        "candidate_token_ids": row["candidate_token_ids"],
                    },
                    response={
                        "native_candidate_ids": output_candidate_ids,
                        "native_logits": logits,
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(rows),
                    input_contract={
                        **truncation.as_dict(),
                        "counted_surface": "published_nimble_chat_prompt",
                        "model_facing_example": fitted.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _fit(self, example: DecisionExample) -> tuple[DecisionExample, TextTruncationReport]:
        return fit_example_to_token_budget(
            example,
            max_input_tokens=self.max_length,
            count_tokens=self._compiled_prompt_tokens,
        )

    def _compiled_prompt_tokens(self, example: DecisionExample) -> int:
        context, schema, _ = build_nimble_input(example)
        prepared = self._prepare_prompts(
            self.tokenizer,
            context,
            schema,
            1_000_000_000,
            system_role=True,
        )
        return max(len(cast(list[int], row)) for row in prepared.full_ids)


def _candidate_description(label: str, description: str | None) -> str:
    if description is None or description == label:
        return label
    return f"{label}: {description}"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("decision_bench_nimble_parallel_schema", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Nimble prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
