"""Local Hugging Face adapter for native decision-token models."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from decision_bench.prompt import (
    TEXT_TRUNCATION_POLICY_VERSION,
    TextTruncationReport,
    fit_example_to_token_budget,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

HF_DECISION_CONTRACT_VERSION = "bosun-decision-prompt-v3-stable-slots"


class HFDecisionResponse(BaseModel):
    """Validated prediction plus exact local model inputs and outputs."""

    model_config = ConfigDict(extra="forbid")

    prediction: DecisionPrediction
    request: dict[str, Any]
    response: dict[str, Any]
    latency_seconds: float
    input_contract: dict[str, Any] | None = None


def render_hf_decision_prompt(
    example: DecisionExample,
    decision_tokens: Sequence[str],
    *,
    seed: int,
) -> tuple[str, list[int], dict[str, int]]:
    """Compile one benchmark row into Bosun's stable-slot prompt contract."""

    candidate_count = len(example.candidates)
    if not 2 <= candidate_count <= 255:
        raise ValueError("HF decision models support 2 to 255 candidates")
    if len(decision_tokens) < candidate_count:
        raise ValueError("model does not expose enough decision tokens")

    presentation_order = list(range(candidate_count))
    order_digest = hashlib.sha256(
        f"{seed}:{example.row_id}:candidate-order".encode()
    ).digest()
    random.Random(int.from_bytes(order_digest[:8], "big")).shuffle(presentation_order)
    candidate_to_slot = {
        example.candidates[candidate_index].id: slot
        for slot, candidate_index in enumerate(presentation_order)
    }
    mapped_criteria: list[dict[str, Any]] = []
    for slot, candidate_index in enumerate(presentation_order):
        candidate = example.candidates[candidate_index]
        mapped_criteria.append(
            {
                "t": decision_tokens[slot],
                "o": candidate_index,
                "n": candidate.label,
                "d": candidate.description or "",
            }
        )
    criteria_fields, criteria = _compact_mapped_criteria(mapped_criteria)
    primitive = {
        Primitive.BINARY_CLASSIFICATION: "noul",
        Primitive.CANDIDATE_SELECTION: "choice",
        Primitive.ORDINAL_SCORING: "score",
    }[example.primitive]
    content = _canonical_json(
        {
            "schema": HF_DECISION_CONTRACT_VERSION,
            "state": example.state,
            "question": {
                "instructions": example.instruction,
                "type": primitive,
                "criteria_fields": criteria_fields,
                "criteria": criteria,
            },
        }
    )
    return content, presentation_order, candidate_to_slot


class HFDecisionModel:
    """Run a local Transformers/PEFT decision-token checkpoint in batches."""

    def __init__(
        self,
        *,
        model_dir: Path,
        seed: int = 0,
        max_length: int = 22_528,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from safetensors.torch import load_file
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "HF model support requires the decision-bench[hf] extra"
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for HF decision-model evaluation")

        self.model_dir = model_dir
        self.seed = seed
        self.max_length = max_length
        self.attn_implementation = attn_implementation
        self._torch = torch
        self._verify_artifact()
        serving = _read_json(model_dir / "serving.json")
        if serving.get("schema_version") != HF_DECISION_CONTRACT_VERSION:
            raise RuntimeError("unsupported HF decision prompt contract")
        self.serving = serving
        self.decision_tokens = [str(value) for value in serving["decision_tokens"]]
        self.decision_token_ids = [int(value) for value in serving["decision_token_ids"]]
        if len(self.decision_tokens) != 256 or len(self.decision_token_ids) != 256:
            raise RuntimeError("expected exactly 256 decision tokens")

        self.tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            model_dir / "tokenizer", use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        actual_token_ids = [
            int(self.tokenizer.convert_tokens_to_ids(token)) for token in self.decision_tokens
        ]
        if actual_token_ids != self.decision_token_ids:
            raise RuntimeError("tokenizer decision-token IDs do not match serving.json")
        base_model: Any = cast(Any, AutoModelForCausalLM).from_pretrained(
            str(serving["base_model"]),
            revision=str(serving["base_revision"]),
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        original_vocab_size = len(self.tokenizer) - len(self.decision_tokens)
        base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        decision_rows = load_file(model_dir / "decision_embeddings.safetensors")
        input_weight = cast(Any, base_model.get_input_embeddings()).weight
        output_weight = cast(Any, base_model.get_output_embeddings()).weight
        with torch.no_grad():
            input_weight[
                original_vocab_size : original_vocab_size + 256
            ].copy_(decision_rows["input_embeddings"])
            output_weight[
                original_vocab_size : original_vocab_size + 256
            ].copy_(decision_rows["output_embeddings"])
        self.model = PeftModel.from_pretrained(base_model, model_dir / "adapter")
        self.model.eval().to("cuda")
        self._decision_token_tensor = torch.tensor(
            self.decision_token_ids,
            device="cuda",
            dtype=torch.long,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return immutable checkpoint and serving metadata for the run summary."""

        return {
            "model_type": "hf_decision_tokens",
            "model_dir": str(self.model_dir),
            "contract_version": HF_DECISION_CONTRACT_VERSION,
            "base_model": self.serving["base_model"],
            "base_revision": self.serving["base_revision"],
            "selected_epoch": self.serving["selected_epoch"],
            "model_manifest_sha256": _sha256_file(self.model_dir / "manifest.json"),
            "serving_sha256": _sha256_file(self.model_dir / "serving.json"),
            "probability_source": "native_full_candidate_logits",
            "decision_token_assignment": self.serving["decision_token_assignment"],
            "max_runtime_choices": self.serving["max_runtime_choices"],
            "seed": self.seed,
            "max_length": self.max_length,
            "attn_implementation": self.attn_implementation,
            "input_truncation_policy": TEXT_TRUNCATION_POLICY_VERSION,
            "token_count_method": "checkpoint_tokenizer_over_final_chat_prompt",
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        """Return deterministic prompt size for memory-aware batch construction."""

        model_example, _ = fit_example_to_token_budget(
            example,
            max_input_tokens=self.max_length,
            count_tokens=self._compiled_prompt_tokens,
        )
        content, _, _ = render_hf_decision_prompt(
            model_example, self.decision_tokens, seed=self.seed
        )
        return len(content)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        """Return full candidate distributions for one CUDA batch."""

        if not examples:
            return []
        started = time.monotonic()
        prompts: list[str] = []
        contents: list[str] = []
        presentation_orders: list[list[int]] = []
        candidate_to_slots: list[dict[str, int]] = []
        model_examples: list[DecisionExample] = []
        truncation_reports: list[TextTruncationReport] = []
        for example in examples:
            model_example, truncation = fit_example_to_token_budget(
                example,
                max_input_tokens=self.max_length,
                count_tokens=self._compiled_prompt_tokens,
            )
            content, presentation_order, candidate_to_slot = render_hf_decision_prompt(
                model_example,
                self.decision_tokens,
                seed=self.seed,
            )
            prompt = self._chat_prompt(content)
            contents.append(content)
            prompts.append(str(prompt))
            presentation_orders.append(presentation_order)
            candidate_to_slots.append(candidate_to_slot)
            model_examples.append(model_example)
            truncation_reports.append(truncation)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        if max(lengths) > self.max_length:
            raise RuntimeError(
                f"compiled prompt length {max(lengths)} exceeds max_length={self.max_length}"
            )
        input_ids = input_ids.to("cuda", non_blocking=True)
        attention_mask = attention_mask.to("cuda", non_blocking=True)
        with self._torch.no_grad(), self._torch.autocast(
            device_type="cuda", dtype=self._torch.bfloat16
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            decision_logits = outputs.logits[:, -1, :].index_select(
                -1, self._decision_token_tensor
            )
        input_rows = cast(list[list[int]], input_ids.detach().cpu().tolist())
        logit_rows = cast(list[list[float]], decision_logits.float().cpu().tolist())
        elapsed = time.monotonic() - started
        responses: list[HFDecisionResponse] = []
        for index, example in enumerate(examples):
            candidate_count = len(example.candidates)
            slot_logits = logit_rows[index][:candidate_count]
            slot_probabilities = _softmax(slot_logits)
            probabilities = [0.0] * candidate_count
            for slot, candidate_index in enumerate(presentation_orders[index]):
                probabilities[candidate_index] = slot_probabilities[slot]
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "content": contents[index],
                        "prompt": prompts[index],
                        "input_ids": input_rows[index],
                        "input_length": lengths[index],
                        "candidate_to_slot": candidate_to_slots[index],
                    },
                    response={
                        "slot_logits": slot_logits,
                        "slot_probabilities": slot_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(examples),
                    input_contract={
                        **truncation_reports[index].as_dict(),
                        "counted_surface": "final_chat_prompt",
                        "model_facing_example": model_examples[index].model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _chat_prompt(self, content: str) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Choose exactly one supplied decision token. Return only "
                        "that token. Do not explain the answer."
                    ),
                },
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return str(prompt)

    def _compiled_prompt_tokens(self, example: DecisionExample) -> int:
        content, _, _ = render_hf_decision_prompt(
            example,
            self.decision_tokens,
            seed=self.seed,
        )
        prompt = self._chat_prompt(content)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return len(token_ids)

    def _verify_artifact(self) -> None:
        manifest = _read_json(self.model_dir / "manifest.json")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("model manifest has no file hash mapping")
        for relative, expected in files.items():
            path = self.model_dir / str(relative)
            if not path.is_file():
                raise RuntimeError(f"model artifact file is missing: {relative}")
            actual = _sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"model artifact hash mismatch: {relative}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _compact_mapped_criteria(
    mapped_criteria: Sequence[dict[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    parsed_descriptions: list[dict[str, Any]] = []
    description_fields: list[str] | None = None
    for criterion in mapped_criteria:
        try:
            parsed = json.loads(str(criterion["d"]))
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            description_fields = None
            break
        fields = sorted(cast(dict[str, Any], parsed))
        if description_fields is None:
            description_fields = fields
        elif fields != description_fields:
            description_fields = None
            break
        parsed_descriptions.append(cast(dict[str, Any], parsed))
    if description_fields is not None and len(parsed_descriptions) == len(mapped_criteria):
        return (
            ["t", "o", "n", *description_fields],
            [
                [
                    criterion["t"],
                    criterion["o"],
                    criterion["n"],
                    *(parsed[field] for field in description_fields),
                ]
                for criterion, parsed in zip(
                    mapped_criteria, parsed_descriptions, strict=True
                )
            ],
        )
    return (
        ["t", "o", "n", "d"],
        [
            [criterion["t"], criterion["o"], criterion["n"], criterion["d"]]
            for criterion in mapped_criteria
        ],
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]
