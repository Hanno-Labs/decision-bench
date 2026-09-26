"""Native adapter for the ``internlm/Intern-Decision`` masked-symbol model family.

Intern-Decision checkpoints score a whole decision schema in one causal forward
pass. Each field's options are mapped to single-token symbols, the assistant
skeleton carries one ``<decision>`` placeholder per field, and the masked
next-token logits immediately before each placeholder are softmaxed over that
field's candidate symbols. The checkpoint then applies its published
probability calibration.

This adapter reproduces the readout published with the checkpoint's
``inference.py`` for a single DecisionBench ``decision`` field, without
executing checkpoint code. It reuses the shared Jev request builder and Jev
answer mapper so candidate IDs and displayed order are preserved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import string
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.models.jev_openrouter import (
    build_jev_request,
    prediction_from_jev_answer,
)
from decision_bench.models.public_hf import (
    UnsupportedCandidateCount,
    UnsupportedInputLength,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction

INTERN_DECISION_CONTRACT_VERSION = "intern-decision-masked-symbol-softmax-v1"
INTERN_DECISION_INPUT_POLICY_VERSION = "reject-over-checkpoint-limit-v1"
INTERN_DECISION_DECISION_TOKEN = "<decision>"
INTERN_DECISION_ANSWER_SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits
INTERN_DECISION_MAX_OPTIONS = len(INTERN_DECISION_ANSWER_SYMBOLS)
INTERN_DECISION_MAX_LENGTH = 8_192
INTERN_DECISION_SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in the user message "
    "to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, "
    "B, C, ...) from its listed options and return one valid JSON object mapping each field name "
    "to its chosen symbol. Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
INTERN_DECISION_MODEL_REPO = "internlm/Intern-Decision-4B"
INTERN_DECISION_MODEL_REVISION = "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd"
INTERN_DECISION_BASE_MODEL = "Qwen/Qwen3.5-4B"

_TEMPERATURE_PATTERN = re.compile(
    r"^DEFAULT_TEMPERATURE\s*=\s*([0-9eE.+-]+)\s*$", re.MULTILINE
)
_MODEL_NAME_PATTERN = re.compile(r"^MODEL_NAME\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)


@dataclass(frozen=True)
class CompiledInternDecisionRow:
    """The published chat messages, skeleton, and symbol mapping for one row."""

    fields: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]
    questions: dict[str, Any]
    messages: list[dict[str, Any]]
    user_text: str
    skeleton: str


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _options(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return ordered (option value, description) pairs for one field."""

    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping):
            raise ValueError("choice criteria must be an object")
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        if isinstance(criteria, Mapping):
            return [(str(key), str(value)) for key, value in criteria.items()]
        raise ValueError("score criteria must be a list or object")
    if kind == "noul":
        descriptions = criteria if isinstance(criteria, Mapping) else {}
        yes = next(
            (
                str(descriptions[key])
                for key in descriptions
                if str(key).lower() in {"yes", "true", "1"}
            ),
            "The answer is yes (affirmative, or align with the claim).",
        )
        no = next(
            (
                str(descriptions[key])
                for key in descriptions
                if str(key).lower() in {"no", "false", "0"}
            ),
            "The answer is no (negative, or disagree with the claim).",
        )
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def compile_intern_decision_row(row: Mapping[str, Any]) -> CompiledInternDecisionRow:
    """Reproduce the checkpoint's published chat-skeleton compiler."""

    questions = row.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty object")
    fields: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        if not isinstance(question, Mapping):
            raise ValueError(f"question {field!r} is not an object")
        options = _options(question)
        if not options:
            raise ValueError(f"question {field!r} has no options")
        if len(options) > INTERN_DECISION_MAX_OPTIONS:
            raise UnsupportedCandidateCount(
                f"Intern-Decision supports at most {INTERN_DECISION_MAX_OPTIONS} options"
            )
        field_name = str(field)
        fields.append(field_name)
        field_symbols = tuple(INTERN_DECISION_ANSWER_SYMBOLS[: len(options)])
        symbols[field_name] = field_symbols
        schema_lines.append(f"{field_name}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(field_symbols, options, strict=True):
            schema_lines.append(f"    {symbol} = {value}: {description}")
    state = _json(row.get("state"))
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    if INTERN_DECISION_DECISION_TOKEN in user_text:
        raise ValueError("Reserved decision marker appears in input evidence")
    skeleton = json.dumps(
        dict.fromkeys(fields, INTERN_DECISION_DECISION_TOKEN), ensure_ascii=False, indent=4
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": INTERN_DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return CompiledInternDecisionRow(
        fields=tuple(fields),
        symbols=symbols,
        questions={str(field): dict(question) for field, question in questions.items()},
        messages=messages,
        user_text=user_text,
        skeleton=skeleton,
    )


def build_intern_decision_request(
    example: DecisionExample,
    *,
    model_repo: str = INTERN_DECISION_MODEL_REPO,
) -> tuple[dict[str, Any], list[str], CompiledInternDecisionRow]:
    """Translate one frozen row into the checkpoint's documented request contract."""

    request, response_order = build_jev_request(example, model=model_repo)
    row = {"state": request["state"], "questions": request["questions"]}
    compiled = compile_intern_decision_row(row)
    return request, response_order, compiled


def prediction_from_intern_decision_response(
    example: DecisionExample,
    response: Mapping[str, Any],
    response_order: Sequence[str],
) -> DecisionPrediction:
    """Map a published Intern-Decision response onto benchmark candidates."""

    if not isinstance(response, Mapping):
        raise TypeError("response is not an object")
    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeError("response did not include an answers object")
    answer = answers.get("decision")
    if not isinstance(answer, Mapping):
        raise TypeError("response did not include the decision answer")
    return prediction_from_jev_answer(example, dict(answer), list(response_order))


def _scale_probabilities(
    probabilities: Mapping[str, float], temperature: float
) -> dict[str, float]:
    """Apply the checkpoint's published temperature calibration in log space."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not probabilities:
        raise ValueError("empty probability distribution")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities.values()
    ):
        raise ValueError("invalid probability distribution")
    if abs(sum(probabilities.values()) - 1.0) > 0.001:
        raise ValueError("probabilities must sum to one")
    if temperature == 1.0:
        return dict(probabilities)
    logs = {
        value: math.log(probability) if probability else -math.inf
        for value, probability in probabilities.items()
    }
    maximum = max(logs.values())
    weights = {
        value: math.exp((log_probability - maximum) / temperature)
        for value, log_probability in logs.items()
    }
    total = sum(weights.values())
    scaled = {value: weight / total for value, weight in weights.items()}
    if _argmax(scaled) != _argmax(probabilities):
        raise ArithmeticError("temperature scaling changed the argmax")
    return scaled


def _argmax(probabilities: Mapping[str, float]) -> str:
    return min(probabilities, key=lambda value: (-probabilities[value], value))


def _read_serving_constants(inference_path: Path) -> tuple[float, str]:
    """Read the published temperature and model name without executing serving code."""

    text = inference_path.read_text(encoding="utf-8")
    temperature_match = _TEMPERATURE_PATTERN.search(text)
    name_match = _MODEL_NAME_PATTERN.search(text)
    if temperature_match is None or name_match is None:
        raise RuntimeError(f"inference.py does not declare its calibration: {inference_path}")
    temperature = float(temperature_match.group(1))
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("inference.py declares an invalid temperature")
    return temperature, name_match.group(1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _output_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], Sequence):
        if len(encoded) != 1:
            raise RuntimeError("tokenizer returned more than one token sequence")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


class InternDecisionHFDecisionModel:
    """Run a pinned Intern-Decision checkpoint through its masked-symbol readout."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_repo: str = INTERN_DECISION_MODEL_REPO,
        model_revision: str = INTERN_DECISION_MODEL_REVISION,
        max_length: int = INTERN_DECISION_MAX_LENGTH,
        attn_implementation: str = "sdpa",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Intern-Decision support requires the decision-bench[hf] extra"
            ) from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Intern-Decision evaluation requires a BF16-capable CUDA GPU")
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")
        if max_length < 1:
            raise ValueError("max_length must be positive")

        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.max_length = max_length
        self.attn_implementation = attn_implementation
        self.device = device
        self._torch = torch
        self.temperature, self.native_model_name = _read_serving_constants(
            model_dir / "inference.py"
        )
        self.serving_code_sha256 = _sha256_file(model_dir / "inference.py")

        self.tokenizer = cast(Any, AutoTokenizer).from_pretrained(
            model_dir, local_files_only=True
        )
        if INTERN_DECISION_DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise RuntimeError("checkpoint must include its trained decision tokenizer")
        self.marker_id = int(
            self.tokenizer.convert_tokens_to_ids(INTERN_DECISION_DECISION_TOKEN)
        )
        self._symbol_token_ids = self._resolve_symbol_token_ids()
        self.model = cast(Any, Qwen3_5ForConditionalGeneration).from_pretrained(
            model_dir,
            dtype=getattr(torch, dtype),
            local_files_only=True,
            attn_implementation=attn_implementation,
        ).to(device).eval()

    def _resolve_symbol_token_ids(self) -> dict[str, int]:
        token_ids: dict[str, int] = {}
        for symbol in INTERN_DECISION_ANSWER_SYMBOLS:
            encoded = _output_ids(self.tokenizer.encode(symbol, add_special_tokens=False))
            if len(encoded) != 1 or self.tokenizer.decode(encoded) != symbol:
                raise RuntimeError(
                    f"candidate symbol {symbol!r} does not map to a single token"
                )
            token_ids[symbol] = encoded[0]
        if len(set(token_ids.values())) != len(token_ids):
            raise RuntimeError("candidate symbols do not map to distinct tokens")
        return token_ids

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "intern_decision_masked_symbol_softmax",
            "model": self.model_repo,
            "model_revision": self.model_revision,
            "native_model_name": self.native_model_name,
            "base_model": INTERN_DECISION_BASE_MODEL,
            "serving_code_sha256": self.serving_code_sha256,
            "contract_version": INTERN_DECISION_CONTRACT_VERSION,
            "probability_source": (
                "masked_softmax_over_candidate_symbol_logits_then_published_calibration"
            ),
            "probability_interpretation": (
                "masked-symbol distribution over the offered options with the checkpoint's "
                "published temperature calibration; a conditional option preference, not a "
                "ground-truth confidence"
            ),
            "temperature": self.temperature,
            "temperature_source": "checkpoint_inference.py:DEFAULT_TEMPERATURE",
            "decision_token": INTERN_DECISION_DECISION_TOKEN,
            "answer_symbols": INTERN_DECISION_ANSWER_SYMBOLS,
            "max_candidates": INTERN_DECISION_MAX_OPTIONS,
            "max_questions": 1,
            "max_length": self.max_length,
            "attn_implementation": self.attn_implementation,
            "input_truncation_policy": INTERN_DECISION_INPUT_POLICY_VERSION,
            "eligibility_definition": (
                f"candidate_count <= {INTERN_DECISION_MAX_OPTIONS} and encoded input <= "
                f"{self.max_length} tokens; images are unsupported"
            ),
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        """Return the rendered model-facing prompt length used for batch sizing."""

        _, _, compiled = build_intern_decision_request(example, model_repo=self.model_repo)
        return len(self._render_text(compiled))

    def validate_example(self, example: DecisionExample) -> None:
        """Reject rows outside the checkpoint contract without truncating them."""

        candidate_count = len(example.candidates)
        if candidate_count > INTERN_DECISION_MAX_OPTIONS:
            raise UnsupportedCandidateCount(
                f"Intern-Decision supports at most {INTERN_DECISION_MAX_OPTIONS} candidates"
            )
        if candidate_count < 2:
            raise ValueError("Intern-Decision requires at least two candidates")
        self._encode(example)

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        """Return the calibrated native distribution for each row."""

        if not examples:
            return []
        started = time.monotonic()
        responses: list[HFDecisionResponse] = []
        for example in examples:
            input_ids, compiled, positions, response_order, request = self._encode(example)
            result, native = self._score(input_ids, compiled, positions)
            prediction = prediction_from_intern_decision_response(
                example, result, response_order
            )
            responses.append(
                HFDecisionResponse(
                    prediction=prediction,
                    request={
                        "jev_request": request,
                        "messages": compiled.messages,
                        "assistant_skeleton": compiled.skeleton,
                        "input_token_ids": input_ids,
                        "input_tokens": len(input_ids),
                        "marker_positions": positions,
                    },
                    response={**result, "native": native},
                    latency_seconds=(time.monotonic() - started) / len(examples),
                    input_contract={
                        "policy_version": INTERN_DECISION_INPUT_POLICY_VERSION,
                        "truncated": False,
                        "original_tokens": len(input_ids),
                        "model_tokens": len(input_ids),
                        "max_input_tokens": self.max_length,
                        "temperature": self.temperature,
                        "counted_surface": "intern_decision_published_chat_skeleton",
                        "model_facing_example": example.model_dump(mode="json"),
                    },
                )
            )
        return responses

    def _encode(
        self, example: DecisionExample
    ) -> tuple[list[int], CompiledInternDecisionRow, list[int], list[str], dict[str, Any]]:
        request, response_order, compiled = build_intern_decision_request(
            example, model_repo=self.model_repo
        )
        text = self._render_text(compiled)
        input_ids = _output_ids(
            self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        )
        if len(input_ids) > self.max_length:
            raise UnsupportedInputLength(
                f"row {example.row_id} has {len(input_ids)} tokens, above {self.max_length}; "
                "truncation is forbidden"
            )
        positions = [index - 1 for index, token in enumerate(input_ids) if token == self.marker_id]
        if len(positions) != len(compiled.fields) or any(position < 0 for position in positions):
            raise RuntimeError("decision marker count or position mismatch")
        return input_ids, compiled, positions, response_order, request

    def _render_text(self, compiled: CompiledInternDecisionRow) -> str:
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                compiled.messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
                add_vision_id=True,
            ),
        )

    def _score(
        self,
        input_ids: list[int],
        compiled: CompiledInternDecisionRow,
        positions: list[int],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self._torch
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_tensor,
                use_cache=False,
                logits_to_keep=position_tensor,
            ).logits[0]
        inference_ms = (time.perf_counter() - started) * 1000

        answers: dict[str, Any] = {}
        native_fields: dict[str, Any] = {}
        for index, field in enumerate(compiled.fields):
            question = compiled.questions[field]
            options = _options(question)
            values = [value for value, _ in options]
            symbols = compiled.symbols[field]
            symbol_token_ids = [self._symbol_token_ids[symbol] for symbol in symbols]
            symbol_logits = logits[index, symbol_token_ids].float()
            probabilities = cast(list[float], symbol_logits.softmax(-1).cpu().tolist())
            distribution = dict(zip(values, probabilities, strict=True))
            best = _argmax(distribution)
            kind = question.get("type")
            answer: dict[str, Any] = {
                "type": kind,
                "probabilities": distribution,
                "confidence": distribution[best],
            }
            if kind == "noul":
                answer["noul"] = distribution["yes"]
            elif kind == "score":
                answer["score"] = sum(float(value) * distribution[value] for value in values)
                answer["legend"] = dict(options)
            else:
                answer["choice"] = best
            answers[field] = answer
            native_fields[field] = {
                "option_values": values,
                "symbols": list(symbols),
                "symbol_token_ids": symbol_token_ids,
                "symbol_logits": cast(list[float], symbol_logits.cpu().tolist()),
                "native_probabilities": probabilities,
            }

        result: dict[str, Any] = {
            "answers": answers,
            "usage": {"input_tokens": len(input_ids), "output_tokens": len(answers)},
            "timing": {"inference_ms": round(inference_ms, 2)},
        }
        result = self._calibrate(result)
        for answer in result["answers"].values():
            answer["source"] = "local"
            answer["decision"] = _argmax(answer["probabilities"])
        result["usage"]["decision_count"] = len(result["answers"])
        result["model"] = self.native_model_name
        result["backend"] = "hf"
        native = {
            "fields": native_fields,
            "temperature": self.temperature,
            "marker_positions": positions,
        }
        return result, native

    def _calibrate(self, result: dict[str, Any]) -> dict[str, Any]:
        scaled = copy.deepcopy(result)
        for answer in scaled["answers"].values():
            probabilities = _scale_probabilities(answer["probabilities"], self.temperature)
            answer["probabilities"] = probabilities
            best = _argmax(probabilities)
            answer["confidence"] = probabilities[best]
            if answer["type"] == "choice":
                answer["choice"] = best
            elif answer["type"] == "noul":
                answer["noul"] = probabilities["yes"]
            elif answer["type"] == "score":
                answer["score"] = sum(
                    float(value) * probability for value, probability in probabilities.items()
                )
        scaled["calibration"] = {
            "method": "temperature-scaling",
            "temperature": self.temperature,
        }
        return scaled
