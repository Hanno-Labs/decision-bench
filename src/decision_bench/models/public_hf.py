"""Native adapters for public Jev-shaped Hugging Face decision models."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.prompt import (
    TEXT_TRUNCATION_POLICY_VERSION,
    TextTruncationReport,
    fit_example_to_token_budget,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive

NANOJEV_MAX_LENGTH = 512
OPENJEV_MAX_LENGTH = 512
SYSTEM_ONE_MAX_LENGTH = 384


def _state_text(example: DecisionExample) -> str:
    if isinstance(example.state, str):
        return example.state
    return json.dumps(example.state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _candidate_text(label: str, description: str | None) -> str:
    if description is None or description == label:
        return label
    return f"{label}: {description}"


def _binary_candidate_ids(example: DecisionExample) -> tuple[str, str]:
    by_id = {candidate.id.casefold(): candidate.id for candidate in example.candidates}
    if set(by_id) == {"false", "true"}:
        return by_id["false"], by_id["true"]
    if set(by_id) == {"label-0", "label-1"}:
        return by_id["label-0"], by_id["label-1"]
    by_label = {
        candidate.label.strip().casefold(): candidate.id for candidate in example.candidates
    }
    false_id = by_label.get("no") or by_label.get("false")
    true_id = by_label.get("yes") or by_label.get("true")
    if false_id is None or true_id is None or false_id == true_id:
        raise ValueError("binary row does not expose unambiguous false/true semantics")
    return false_id, true_id


def _align_probabilities(
    example: DecisionExample,
    native_candidate_ids: Sequence[str],
    native_probabilities: Sequence[float],
) -> list[float]:
    if len(native_candidate_ids) != len(native_probabilities):
        raise RuntimeError("native candidate/probability lengths differ")
    by_id = dict(zip(native_candidate_ids, native_probabilities, strict=True))
    if len(by_id) != len(native_candidate_ids):
        raise RuntimeError("native candidate IDs are not unique")
    probabilities = [float(by_id[candidate.id]) for candidate in example.candidates]
    if any(not math.isfinite(value) for value in probabilities):
        raise RuntimeError("model returned non-finite probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise RuntimeError("model probabilities do not sum to one")
    return probabilities


def _token_lists_sha256(rows: Sequence[Sequence[int]]) -> str:
    material = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(material).hexdigest()


class NanoJevHFDecisionModel:
    """Run NanoJev through the released parallel candidate-path head."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str,
        model_revision: str,
        expected_weights_sha256: str,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from safetensors.torch import load_file
            from torch import nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("NanoJev support requires the decision-bench[hf] extra") from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("NanoJev evaluation requires a BF16-capable CUDA GPU")

        weights_path = model_dir / "best.safetensors"
        actual_weights_sha256 = _sha256_file(weights_path)
        if actual_weights_sha256 != expected_weights_sha256:
            raise RuntimeError("NanoJev weights do not match the pinned release")
        run_config = _read_json(model_dir / "config.json")
        if int(run_config.get("max_length", 0)) != NANOJEV_MAX_LENGTH:
            raise RuntimeError("unexpected NanoJev max_length")

        class DecisionModel(nn.Module):
            def __init__(self, backbone: Any, set_head: str) -> None:
                super().__init__()
                hidden = int(backbone.config.hidden_size)
                self.backbone = backbone
                self.norm = nn.LayerNorm(hidden)
                self.scalar = nn.Linear(hidden, 1)
                self.set_head = set_head
                if set_head == "attention":
                    self.set_project = nn.Linear(hidden + 1, 128)
                    self.set_attention = nn.MultiheadAttention(
                        128, 4, dropout=0.0, batch_first=True
                    )
                    self.set_output = nn.Linear(128, 1)

            def forward(self, examples: Sequence[dict[str, Any]], pad_token: int) -> Any:
                paths = [ids for example in examples for ids in example["leaf_tokens"]]
                device = self.scalar.weight.device
                lengths = torch.tensor([len(ids) for ids in paths], device=device)
                width = int(lengths.max())
                tokens = torch.full(
                    (len(paths), width), pad_token, dtype=torch.long, device=device
                )
                for index, ids in enumerate(paths):
                    tokens[index, : len(ids)] = torch.tensor(ids, device=device)
                attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
                hidden = self.backbone(
                    input_ids=tokens, attention_mask=attention, use_cache=False
                ).last_hidden_state
                leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]
                kmax = max(len(example["candidate_ids"]) for example in examples)
                values = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
                valid = torch.zeros(
                    (len(examples), kmax), dtype=torch.bool, device=device
                )
                offset = 0
                for index, example in enumerate(examples):
                    count = len(example["leaf_tokens"])
                    values[index, :count] = leaves[offset : offset + count]
                    valid[index, : len(example["candidate_ids"])] = True
                    offset += count
                values = self.norm(values)
                logits = self.scalar(values).squeeze(-1).float()
                choice = torch.tensor(
                    [
                        index
                        for index, example in enumerate(examples)
                        if example["type"] == "choice"
                    ],
                    device=device,
                )
                if self.set_head == "attention" and len(choice):
                    log_k = (
                        valid[choice]
                        .sum(-1)
                        .float()
                        .log()[:, None, None]
                        .expand(-1, kmax, 1)
                    )
                    projected = self.set_project(
                        torch.cat([values[choice], log_k.to(values.dtype)], dim=-1)
                    )
                    mixed, _ = self.set_attention(
                        projected,
                        projected,
                        projected,
                        key_padding_mask=~valid[choice],
                        need_weights=False,
                    )
                    delta = self.set_output(torch.tanh(projected + mixed)).squeeze(-1).float()
                    logits = logits.index_add(0, choice, delta)
                output = []
                for index, example in enumerate(examples):
                    if example["type"] == "boolean":
                        output.append(
                            torch.nn.functional.pad(
                                torch.stack([logits[index, 0] * 0, logits[index, 0]]),
                                (0, kmax - 2),
                            )
                        )
                    else:
                        output.append(logits[index])
                return torch.stack(output).masked_fill(~valid, -1e9)

        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.attn_implementation = attn_implementation
        self.run_config = run_config
        self._torch = torch
        self.tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_dir / "tokenizer", local_files_only=True
            ),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        body_config = AutoConfig.from_pretrained(
            model_dir / "backbone_config", local_files_only=True, trust_remote_code=False
        )
        body_config.use_cache = False
        backbone = AutoModel.from_config(  # type: ignore[no-untyped-call]
            body_config,
            attn_implementation=attn_implementation,
            trust_remote_code=False,
        ).float()
        model = DecisionModel(backbone, str(run_config["set_head"]))
        model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        self.model = model.to(device="cuda", dtype=torch.float32).eval()
        self._weights_sha256 = actual_weights_sha256

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "nanojev_parallel_candidate_paths",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "weights_sha256": self._weights_sha256,
            "config_sha256": _sha256_file(self.model_dir / "config.json"),
            "base_model": self.run_config["model"],
            "base_revision": self.run_config["resolved_model_revision"],
            "max_length": NANOJEV_MAX_LENGTH,
            "temperature": 1.0,
            "probability_source": "native_candidate_logits_fp32_softmax",
            "attn_implementation": self.attn_implementation,
            "input_truncation_policy": TEXT_TRUNCATION_POLICY_VERSION,
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        fitted, _ = self._fit(example)
        request, _ = self._build_request(fitted)
        return len(json.dumps(request, ensure_ascii=False))

    def validate_example(self, example: DecisionExample) -> None:
        self._fit(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        if not examples:
            return []
        started = time.monotonic()
        prepared: list[dict[str, Any]] = []
        for example in examples:
            fitted, truncation = self._fit(example)
            request, output_candidate_ids = self._build_request(fitted)
            native = self._prepare(request)
            native.update(
                example=example,
                fitted=fitted,
                truncation=truncation,
                request=request,
                output_candidate_ids=output_candidate_ids,
            )
            prepared.append(native)
        with self._torch.inference_mode(), self._torch.autocast(
            "cuda", dtype=self._torch.bfloat16
        ):
            logits = self.model(prepared, int(self.tokenizer.pad_token_id)).float().cpu()
        elapsed = time.monotonic() - started
        responses: list[HFDecisionResponse] = []
        for index, row in enumerate(prepared):
            example = cast(DecisionExample, row["example"])
            count = len(cast(list[str], row["candidate_ids"]))
            native_logits = cast(list[float], logits[index, :count].double().tolist())
            native_probabilities = cast(
                list[float], logits[index, :count].double().softmax(-1).tolist()
            )
            output_candidate_ids = cast(list[str], row["output_candidate_ids"])
            probabilities = _align_probabilities(
                example, output_candidate_ids, native_probabilities
            )
            leaf_tokens = cast(list[list[int]], row["leaf_tokens"])
            truncation = cast(TextTruncationReport, row["truncation"])
            fitted = cast(DecisionExample, row["fitted"])
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "payload": row["request"],
                        "candidate_path_lengths": [len(values) for values in leaf_tokens],
                        "candidate_path_token_ids_sha256": _token_lists_sha256(leaf_tokens),
                    },
                    response={
                        "native_candidate_ids": output_candidate_ids,
                        "native_logits": native_logits,
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(prepared),
                    input_contract={
                        **truncation.as_dict(),
                        "counted_surface": "nanojev_released_candidate_paths",
                        "model_facing_example": fitted.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _fit(self, example: DecisionExample) -> tuple[DecisionExample, TextTruncationReport]:
        if example.primitive is Primitive.ORDINAL_SCORING and len(example.candidates) > 10:
            raise ValueError("NanoJev score supports at most 10 ordered levels")
        return fit_example_to_token_budget(
            example,
            max_input_tokens=NANOJEV_MAX_LENGTH,
            count_tokens=self._compiled_prompt_tokens,
        )

    def _compiled_prompt_tokens(self, example: DecisionExample) -> int:
        request, _ = self._build_request(example)
        prepared = self._prepare(request)
        return max(len(values) for values in cast(list[list[int]], prepared["leaf_tokens"]))

    def _build_request(self, example: DecisionExample) -> tuple[dict[str, Any], list[str]]:
        question: dict[str, Any] = {"instructions": example.instruction}
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            false_id, true_id = _binary_candidate_ids(example)
            descriptions = {
                candidate.id: _candidate_text(candidate.label, candidate.description)
                for candidate in example.candidates
            }
            question.update(
                type="boolean",
                criteria={"false": descriptions[false_id], "true": descriptions[true_id]},
            )
            output_candidate_ids = [false_id, true_id]
        elif example.primitive is Primitive.ORDINAL_SCORING:
            question.update(
                type="score",
                criteria=[
                    _candidate_text(candidate.label, candidate.description)
                    for candidate in example.candidates
                ],
            )
            output_candidate_ids = [candidate.id for candidate in example.candidates]
        else:
            question.update(
                type="choice",
                criteria={
                    candidate.id: _candidate_text(candidate.label, candidate.description)
                    for candidate in example.candidates
                },
            )
            output_candidate_ids = [candidate.id for candidate in example.candidates]
        return {
            "states": [
                {
                    "id": example.row_id,
                    "state": _state_text(example),
                    "questions": {"decision": question},
                }
            ]
        }, output_candidate_ids

    def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        row = cast(dict[str, Any], cast(list[Any], request["states"])[0])
        question = cast(dict[str, Any], cast(dict[str, Any], row["questions"])["decision"])
        kind = str(question["type"])
        if kind == "boolean":
            candidate_ids = ["false", "true"]
            texts = ["The proposition is true."]
        elif kind == "choice":
            criteria = cast(dict[str, str], question["criteria"])
            candidate_ids = list(criteria)
            texts = [f"{key}: {criteria[key]}" for key in candidate_ids]
        else:
            criteria_list = cast(list[str], question["criteria"])
            candidate_ids = [str(index) for index in range(len(criteria_list))]
            texts = criteria_list
        segments = [
            f"State:\n{row['state']}\n",
            f"Question type: {kind}\nQuestion:\n{question['instructions']}\n",
        ]
        if kind == "boolean":
            boolean_criteria = cast(dict[str, str], question["criteria"])
            for key, label in (("false", "False"), ("true", "True")):
                segments[1] += f"{label} criterion: {boolean_criteria[key]}\n"
        prefix = list(
            chain.from_iterable(
                self.tokenizer.encode(text, add_special_tokens=False) for text in segments
            )
        )
        leaves = [
            prefix
            + self.tokenizer.encode(
                f"Candidate:\n{text}\nDecision:", add_special_tokens=False
            )
            + [int(self.tokenizer.eos_token_id)]
            for text in texts
        ]
        return {
            "type": kind,
            "candidate_ids": candidate_ids,
            "leaf_tokens": leaves,
        }


class OpenJevHFDecisionModel:
    """Run the released DeBERTa OpenJev bundle through its grouped span head."""

    def __init__(self, *, model_dir: Path, model_repo: str, model_revision: str) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("OpenJev support requires the decision-bench[hf] extra") from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("OpenJev evaluation requires a BF16-capable CUDA GPU")
        sys.path.insert(0, str(model_dir))
        try:
            from typed_decisions.open_jev import OpenJev
        finally:
            sys.path.pop(0)
        self.engine = OpenJev.from_pretrained(str(model_dir), device="cuda")
        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self._torch = torch

    @property
    def metadata(self) -> dict[str, Any]:
        config = cast(dict[str, Any], self.engine.config)
        return {
            "model_type": "open_jev_deberta_grouped_span_head",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "model_weights_sha256": _sha256_file(self.model_dir / "model.safetensors"),
            "head_sha256": _sha256_file(self.model_dir / "head.safetensors"),
            "config_sha256": _sha256_file(self.model_dir / "open_jev_config.json"),
            "temperature": float(config.get("temperature", 1.0)),
            "max_length": int(config.get("max_len", OPENJEV_MAX_LENGTH)),
            "max_state_tokens": int(config.get("max_state_tokens", 256)),
            "probability_source": "native_grouped_logits_temperature_softmax",
            "input_truncation_policy": TEXT_TRUNCATION_POLICY_VERSION,
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        fitted, _ = self._fit(example)
        state, question, _ = self._build_input(fitted)
        return len(state) + len(question["instructions"]) + sum(
            len(value) for value in cast(list[str], question.get("options", []))
        )

    def validate_example(self, example: DecisionExample) -> None:
        self._fit(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        if not examples:
            return []
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        items: list[tuple[str, list[Any]]] = []
        for example in examples:
            fitted, truncation = self._fit(example)
            state, question, output_candidate_ids = self._build_input(fitted)
            native_question = self.engine._question(0, question)
            items.append((state, [native_question]))
            rows.append(
                {
                    "example": example,
                    "fitted": fitted,
                    "truncation": truncation,
                    "state": state,
                    "question": question,
                    "output_candidate_ids": output_candidate_ids,
                }
            )
        batch = self.engine.collator(items, self.engine.device)
        with self._torch.inference_mode():
            logits = self.engine.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["opt_pos"],
                batch["opt_mask"],
                batch["q_pos"],
                batch["seg"],
            ).float()
        temperature = float(self.engine.model.temperature)
        probabilities_tensor = (logits / temperature).softmax(-1).cpu()
        logits = logits.cpu()
        input_ids = batch["input_ids"].cpu()
        attention_mask = batch["attention_mask"].cpu()
        elapsed = time.monotonic() - started
        responses: list[HFDecisionResponse] = []
        for index, row in enumerate(rows):
            example = cast(DecisionExample, row["example"])
            output_candidate_ids = cast(list[str], row["output_candidate_ids"])
            count = len(output_candidate_ids)
            native_probabilities = cast(
                list[float], probabilities_tensor[index, 0, :count].double().tolist()
            )
            probabilities = _align_probabilities(
                example, output_candidate_ids, native_probabilities
            )
            length = int(attention_mask[index].sum().item())
            token_ids = cast(list[int], input_ids[index, :length].tolist())
            truncation = cast(TextTruncationReport, row["truncation"])
            fitted = cast(DecisionExample, row["fitted"])
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "state": row["state"],
                        "question": row["question"],
                        "input_ids": token_ids,
                        "input_length": length,
                    },
                    response={
                        "native_candidate_ids": output_candidate_ids,
                        "native_logits": cast(
                            list[float], logits[index, 0, :count].double().tolist()
                        ),
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(rows),
                    input_contract={
                        **truncation.as_dict(),
                        "counted_surface": "open_jev_grouped_sequence",
                        "model_facing_example": fitted.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _fit(self, example: DecisionExample) -> tuple[DecisionExample, TextTruncationReport]:
        if example.primitive is Primitive.ORDINAL_SCORING and len(example.candidates) > 10:
            raise ValueError("OpenJev score supports at most 10 ordered levels")
        return fit_example_to_token_budget(
            example,
            max_input_tokens=OPENJEV_MAX_LENGTH,
            count_tokens=self._compiled_prompt_tokens,
        )

    def _compiled_prompt_tokens(self, example: DecisionExample) -> int:
        state, question, _ = self._build_input(example)
        native_question = self.engine._question(0, question)
        tokenizer = self.engine.tok
        count = 2 + min(
            len(tokenizer(state, add_special_tokens=False)["input_ids"]),
            int(self.engine.collator.max_state),
        )
        count += 1 + len(
            tokenizer(native_question.instructions, add_special_tokens=False)["input_ids"]
        )
        for option in native_question.options:
            count += 1 + len(tokenizer(option, add_special_tokens=False)["input_ids"])
        return count + 1

    def _build_input(self, example: DecisionExample) -> tuple[str, dict[str, Any], list[str]]:
        state = _state_text(example)
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            false_id, true_id = _binary_candidate_ids(example)
            return state, {"type": "noul", "instructions": example.instruction}, [
                false_id,
                true_id,
            ]
        options = [
            _candidate_text(candidate.label, candidate.description)
            for candidate in example.candidates
        ]
        kind = (
            "score"
            if example.primitive is Primitive.ORDINAL_SCORING
            else "choice"
        )
        return state, {
            "type": kind,
            "instructions": example.instruction,
            "options": options,
        }, [candidate.id for candidate in example.candidates]


class SystemOneHFDecisionModel:
    """Run the published Qwen3.5 scalar scorer over every supplied option."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str,
        model_revision: str,
        base_revision: str,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer
            from transformers import (  # type: ignore[attr-defined]
                Qwen3_5TextForSequenceClassification,
            )
        except ImportError as error:
            raise RuntimeError(
                "System One support requires the decision-bench[hf] extra"
            ) from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("System One evaluation requires a BF16-capable CUDA GPU")
        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.base_revision = base_revision
        self.attn_implementation = attn_implementation
        self._torch = torch
        self.tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_dir, local_files_only=True
            ),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = Qwen3_5TextForSequenceClassification.from_pretrained(
            "Qwen/Qwen3.5-4B-Base",
            revision=base_revision,
            num_labels=1,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        base_model.config.eos_token_id = self.tokenizer.eos_token_id
        text_config = base_model.config.get_text_config()
        text_config.pad_token_id = self.tokenizer.pad_token_id
        text_config.eos_token_id = self.tokenizer.eos_token_id
        text_config.num_labels = 1
        base_model.num_labels = 1
        self.model = PeftModel.from_pretrained(base_model, model_dir).to("cuda").eval()
        metrics = _read_json(model_dir / "metrics.json")
        self.temperature = float(metrics["temperature"])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "system_one_qwen35_scalar_option_scorer",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "adapter_sha256": _sha256_file(self.model_dir / "adapter_model.safetensors"),
            "adapter_config_sha256": _sha256_file(self.model_dir / "adapter_config.json"),
            "metrics_sha256": _sha256_file(self.model_dir / "metrics.json"),
            "base_model": "Qwen/Qwen3.5-4B-Base",
            "base_revision": self.base_revision,
            "max_length": SYSTEM_ONE_MAX_LENGTH,
            "temperature": self.temperature,
            "probability_source": "native_scalar_logits_temperature_softmax",
            "attn_implementation": self.attn_implementation,
            "input_truncation_policy": "published_tail_preserving_option_sequence_v5",
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        state, question, options, _ = self._build_input(example)
        return len(state) + len(question) + sum(len(option) for option in options)

    def validate_example(self, example: DecisionExample) -> None:
        self._build_input(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        if not examples:
            return []
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        sequences: list[list[int]] = []
        for example in examples:
            state, question, options, output_candidate_ids = self._build_input(example)
            token_rows = [self._encode(state, question, option) for option in options]
            sequences.extend(token_rows)
            rows.append(
                {
                    "example": example,
                    "state": state,
                    "question": question,
                    "options": options,
                    "output_candidate_ids": output_candidate_ids,
                    "token_rows": token_rows,
                }
            )
        width = min(max(len(values) for values in sequences), SYSTEM_ONE_MAX_LENGTH)
        input_ids = self._torch.full(
            (len(sequences), width),
            int(self.tokenizer.pad_token_id),
            dtype=self._torch.long,
        )
        attention_mask = self._torch.zeros(
            (len(sequences), width), dtype=self._torch.long
        )
        for index, values in enumerate(sequences):
            values = values[:width]
            input_ids[index, : len(values)] = self._torch.tensor(values)
            attention_mask[index, : len(values)] = 1
        with self._torch.inference_mode():
            logits = self.model(
                input_ids=input_ids.to("cuda"),
                attention_mask=attention_mask.to("cuda"),
            ).logits.squeeze(-1).float().cpu()
        elapsed = time.monotonic() - started
        responses: list[HFDecisionResponse] = []
        offset = 0
        for row in rows:
            example = cast(DecisionExample, row["example"])
            output_candidate_ids = cast(list[str], row["output_candidate_ids"])
            count = len(output_candidate_ids)
            native_logits_tensor = logits[offset : offset + count].double()
            offset += count
            native_probabilities = cast(
                list[float], (native_logits_tensor / self.temperature).softmax(-1).tolist()
            )
            probabilities = _align_probabilities(
                example, output_candidate_ids, native_probabilities
            )
            token_rows = cast(list[list[int]], row["token_rows"])
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={
                        "state": row["state"],
                        "question": row["question"],
                        "options": row["options"],
                        "option_sequence_lengths": [len(values) for values in token_rows],
                        "option_sequence_token_ids_sha256": _token_lists_sha256(token_rows),
                    },
                    response={
                        "native_candidate_ids": output_candidate_ids,
                        "native_logits": cast(list[float], native_logits_tensor.tolist()),
                        "native_probabilities": native_probabilities,
                        "probabilities": probabilities,
                    },
                    latency_seconds=elapsed / len(rows),
                    input_contract={
                        "policy_version": "published_tail_preserving_option_sequence_v5",
                        "max_input_tokens": SYSTEM_ONE_MAX_LENGTH,
                        "original_input_tokens": max(len(values) for values in token_rows),
                        "final_input_tokens": max(
                            min(len(values), SYSTEM_ONE_MAX_LENGTH) for values in token_rows
                        ),
                        "truncated": any(
                            len(values) > SYSTEM_ONE_MAX_LENGTH for values in token_rows
                        ),
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _build_input(
        self, example: DecisionExample
    ) -> tuple[str, str, list[str], list[str]]:
        state = _state_text(example)
        if example.primitive is Primitive.BINARY_CLASSIFICATION:
            false_id, true_id = _binary_candidate_ids(example)
            return state, example.instruction, ["yes", "no"], [true_id, false_id]
        return state, example.instruction, [
            _candidate_text(candidate.label, candidate.description)
            for candidate in example.candidates
        ], [candidate.id for candidate in example.candidates]

    def _encode(self, state: str, question: str, option: str) -> list[int]:
        tail = self.tokenizer(
            f"\n\nQuestion:\n{question}\n\nOption:\n{option}",
            add_special_tokens=False,
        )["input_ids"]
        if len(tail) >= SYSTEM_ONE_MAX_LENGTH:
            return cast(list[int], tail[-SYSTEM_ONE_MAX_LENGTH:])
        head = self.tokenizer(
            f"State:\n{state}", add_special_tokens=False
        )["input_ids"]
        return cast(list[int], head[: SYSTEM_ONE_MAX_LENGTH - len(tail)] + tail)


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
