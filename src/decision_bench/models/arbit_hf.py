"""Native adapter for litillabs/arbit-1.7b through its Transformers remote code.

Arbit is Qwen3-1.7B with a LoRA adapter and a joint schema head of Cloudflare's Clef design. Its
repository ships the loader as remote code: ``AutoModel.from_pretrained(..., revision=...,
trust_remote_code=True)`` downloads the pinned base, applies the adapter and loads the head. The
model answers a typed decision request (``state`` plus ``questions``) with a probability for every
offered option in one forward pass, and ``decide()`` returns those probabilities unrounded.

Each benchmark row becomes one question, id ``decision``, built from the model-visible fields only:

    binary_classification  noul    criteria {"true": candidates[0], "false": candidates[1]},
                                   each as "<label>: <description>"; P(candidates[0]) = p(true)
    candidate_selection    choice  option key "<3-digit display index> <label>", value =
                                   description; the model lists choice options sorted by key,
                                   which is display order
    ordinal_scoring        score   criteria list in display order as "<label>: <description>";
                                   level i is candidates[i]

The instruction is passed verbatim. When every candidate description is a JSON object with the
same keys, each description becomes the JSON array of its values in sorted-key order, and the
field list is given once on its own line after the instruction; every value is kept. The state is
passed in the benchmark's storage JSON form (sorted keys, compact separators, unicode kept). A row
is eligible when the model's own ``count_tokens`` measures the rendered request at no more than
27,648 tokens; longer rows are rejected, never truncated.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from decision_bench.models.hf import HFDecisionResponse
from decision_bench.schemas import Candidate, DecisionExample, DecisionPrediction, Primitive

ADAPTER_NAME = "arbit-hf-remote-code-v1"
MODEL_TYPE = "arbit_hf_remote_code_clef"
MODEL_REPO = "litillabs/arbit-1.7b"
MODEL_REVISION = "d8ca47d20f5f406317994c4df096b7bedfefb5de"
BASE_MODEL = "unsloth/Qwen3-1.7B"
BASE_REVISION = "6262b50d6c1f8ee5e4ac750d710c33603bfc2a0c"
PROBABILITY_SOURCE = "native_joint_head_softmax"
RUNTIME = "transformers-remote-code"
LOADER = "transformers.AutoModel.from_pretrained(trust_remote_code=True)"
MAX_INPUT_TOKENS = 27_648
QUESTION_ID = "decision"
POLICY_VERSION = "system-one-render-v4"
TRUNCATION_POLICY = "reject-over-limit, no truncation"
# Name of the counted layout in the reference run's rows: Clef's encode_record, text only.
COUNTED_SURFACE = "unsloth_clef_encode_record"
TABLE_FIELDS_LINE = "\n\nCANDIDATE FIELDS: "
STATE_SERIALIZATION = 'json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)'
QUESTION_TYPES: dict[Primitive, str] = {
    Primitive.BINARY_CLASSIFICATION: "noul",
    Primitive.CANDIDATE_SELECTION: "choice",
    Primitive.ORDINAL_SCORING: "score",
}
WEIGHTS_FILE = "checkpoint/adapter_model.safetensors"
# SHA-256 at MODEL_REVISION of the loader code and of the checkpoint files it loads.
RELEASE_SHA256 = {
    "config.json": "d83c9e85a1ff71ac6a6d8a73e3f09120934e300d910170997edd0327f06c5137",
    "configuration_arbit.py": "0ef09495ff959e633da2f44aaff69814673e3edcb8be134b238809c373a2c937",
    "modeling_arbit.py": "04b4d5077e8d1edad8b971bccad3371a86f3d7b27c9c6bbb73b39aff45d996be",
    "clef/joint_schema_model.py": (
        "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
    ),
    WEIGHTS_FILE: "0167368adfbe03e48593e7b2b7deba4979b92c05cf389975e530d6ab2cd43655",
    "checkpoint/adapter_config.json": (
        "af624a3c1f35381e9f593188a69ff5e98acf30005d0744762e9243a5aeb6ee8d"
    ),
    "checkpoint/joint_head.safetensors": (
        "d3602ec7b5e30e998ce03e30056ef9053572592889bc9e24ed2f3f80f0d92282"
    ),
    "checkpoint/unsloth_decision_config.json": (
        "8eea3f505d22e3a6e73a2ab9ca1a199405196b2559268da291c1a9aa5ee31b53"
    ),
    "checkpoint/tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
}
REMOTE_CODE_FILES = (
    "config.json",
    "configuration_arbit.py",
    "modeling_arbit.py",
    "clef/joint_schema_model.py",
)
PROBABILITY_SOURCE_DETAIL = (
    "ArbitDecisionModel.decide (Clef joint schema head): per-option logits divided by the served "
    "per-question-type temperature, softmaxed; unrounded values as returned next to the 4-decimal "
    "answer. noul: P(candidates[0]) = p(true), P(candidates[1]) = 1 - p(true). choice: "
    'p(option "NNN <label>") for the candidate at display index NNN. score: p(level "i") for the '
    "candidate at display index i (the full per-level distribution, not the expected-score scalar)."
)
TRUNCATION_POLICY_DETAIL = (
    "No benchmark text is cut, dropped or rewritten. A row whose rendered request is longer than "
    f"{MAX_INPUT_TOKENS:,} tokens by ArbitDecisionModel.count_tokens is rejected before "
    "inference. The model reads up to max(max_seq_length, 16384) tokens, 27,648 for this release, "
    "so every eligible row is read in full; decide() reports the input length and whether the "
    "state was cut, and both are checked on every row. input_contract.truncated is always false."
)
ELIGIBILITY_DEFINITION = (
    "A row is eligible when it renders as one decision question (2 to 255 candidates; binary rows "
    f"have exactly 2) and its rendered request is at most {MAX_INPUT_TOKENS:,} tokens as counted "
    "by ArbitDecisionModel.count_tokens, the length of the Clef encode_record input the model "
    "reads. Ineligible rows raise UnsupportedArbitInput or UnsupportedArbitInputLength and count "
    "as unsupported. All three primitives are supported; candidates are passed in full, in "
    "displayed order."
)


class UnsupportedArbitInput(ValueError):
    """The row cannot be rendered as one Arbit question without changing it."""


class UnsupportedArbitInputLength(UnsupportedArbitInput):
    """The rendered request is longer than the eligibility limit; it is rejected, not truncated."""


@dataclass(frozen=True)
class ArbitRequest:
    """One benchmark row as Arbit's typed request, with the option of every candidate."""

    state: str
    questions: dict[str, dict[str, Any]]
    questions_json: str
    question_type: str
    candidate_ids: tuple[str, ...]
    option_keys: tuple[str, ...]
    instruction_verbatim: str
    criteria_fields: tuple[str, ...] | None

    @property
    def question(self) -> dict[str, Any]:
        return self.questions[QUESTION_ID]

    @property
    def option_ids(self) -> list[str]:
        """Option IDs in the order the model lists them."""

        criteria = self.question["criteria"]
        if self.question_type == "noul":
            return ["true", "false"]
        if self.question_type == "choice":
            return sorted(str(key) for key in criteria)
        return [str(index) for index in range(len(criteria))]

    @property
    def prompt_characters(self) -> int:
        return len(self.state) + len(self.questions_json)


def render_arbit_request(example: DecisionExample) -> ArbitRequest:
    """Render a benchmark row into the exact state and questions Arbit receives."""

    candidates = example.candidates
    if len(candidates) < 2:
        raise UnsupportedArbitInput("Arbit needs at least two candidates")
    question_type = QUESTION_TYPES[example.primitive]
    instruction = example.instruction
    table = _description_table(candidates)
    fields: tuple[str, ...] | None = None
    if table is not None:
        fields, descriptions = table
        instruction += TABLE_FIELDS_LINE + json.dumps(list(fields), ensure_ascii=False)
    else:
        descriptions = [candidate.description or "" for candidate in candidates]

    criteria: dict[str, str] | list[str]
    if example.primitive is Primitive.BINARY_CLASSIFICATION:
        if len(candidates) != 2:
            raise UnsupportedArbitInput("binary rows need exactly two candidates")
        keys = ["true", "false"]
        criteria = {
            "true": _criterion(candidates[0].label, descriptions[0]),
            "false": _criterion(candidates[1].label, descriptions[1]),
        }
    elif example.primitive is Primitive.CANDIDATE_SELECTION:
        keys = [f"{index + 1:03d} {candidate.label}" for index, candidate in enumerate(candidates)]
        criteria = dict(zip(keys, descriptions, strict=True))
    else:
        keys = [str(index) for index in range(len(candidates))]
        criteria = [
            _criterion(candidate.label, description)
            for candidate, description in zip(candidates, descriptions, strict=True)
        ]
    questions = {
        QUESTION_ID: {"type": question_type, "instructions": instruction, "criteria": criteria}
    }
    request = ArbitRequest(
        state=_storage_json(example.state),
        questions=questions,
        questions_json=json.dumps(questions, ensure_ascii=False),
        question_type=question_type,
        candidate_ids=tuple(candidate.id for candidate in candidates),
        option_keys=tuple(keys),
        instruction_verbatim=example.instruction,
        criteria_fields=fields,
    )
    if len(set(keys)) != len(keys) or request.option_ids != keys:
        raise UnsupportedArbitInput("model option order differs from the displayed candidate order")
    return request


def candidate_probabilities(
    request: ArbitRequest, option_probabilities: Mapping[str, float]
) -> list[float]:
    """Map the model's option probabilities to one probability per candidate, in order."""

    probabilities = {str(key): float(value) for key, value in option_probabilities.items()}
    expected = request.option_ids
    if set(probabilities) != set(expected) or len(probabilities) != len(expected):
        raise RuntimeError(
            f"{request.question_type} answer has {len(probabilities)} options that do not match "
            f"the {len(expected)} rendered options"
        )
    _check_distribution(list(probabilities.values()), "model option probabilities")
    if request.question_type == "noul":
        true_probability = probabilities["true"]
        mapped = [true_probability, 1.0 - true_probability]
    else:
        mapped = [probabilities[key] for key in request.option_keys]
    if len(mapped) != len(request.candidate_ids):
        raise RuntimeError("mapped probabilities do not cover every candidate")
    _check_distribution(mapped, "mapped probabilities")
    return mapped


class ArbitHFDecisionModel:
    """Run Arbit through its released Transformers remote-code loader and decide()."""

    def __init__(
        self,
        *,
        model_repo: str = MODEL_REPO,
        model_revision: str = MODEL_REVISION,
        base_revision: str | None = None,
        expected_weights_sha256: str | None = None,
        attn_implementation: str = "sdpa",
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        if model_repo != MODEL_REPO or model_revision != MODEL_REVISION:
            raise ValueError("Arbit requires its pinned model repository and revision")
        if base_revision not in (None, BASE_REVISION):
            raise ValueError("Arbit requires its pinned base revision")
        if expected_weights_sha256 not in (None, RELEASE_SHA256[WEIGHTS_FILE]):
            raise ValueError("Arbit requires the published adapter weights SHA-256")
        try:
            import peft
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("Arbit support requires the decision-bench[hf] extra") from error

        # Check the pinned files before the remote code runs.
        for name, expected in RELEASE_SHA256.items():
            path = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION))
            if _sha256_file(path) != expected:
                raise RuntimeError(f"Arbit release file {name} does not match the pinned revision")

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if dtype is None:
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        if dtype not in {"bfloat16", "float32"}:
            raise ValueError("Arbit runs in bfloat16 or float32")
        self.model_repo = model_repo
        self.model_revision = model_revision
        self.attn_implementation = attn_implementation
        self.model = cast(Any, AutoModel).from_pretrained(
            MODEL_REPO,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            # Transformers' PEFT probe reads `main` unless given the revision.
            adapter_kwargs={"revision": MODEL_REVISION},
            dtype=getattr(torch, dtype),
            device=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if resolved_revision != MODEL_REVISION:
            raise RuntimeError(f"loaded Arbit revision {resolved_revision}, not {MODEL_REVISION}")
        self.serve_max_input_tokens = int(self.model.serve_max_length)
        if self.serve_max_input_tokens < MAX_INPUT_TOKENS:
            raise RuntimeError(
                f"the model reads {self.serve_max_input_tokens} tokens, fewer than the "
                f"{MAX_INPUT_TOKENS}-token eligibility limit"
            )
        self._runtime = {
            "model_resolved_revision": resolved_revision,
            "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "decision_config": json.loads(json.dumps(self.model.decision_config, default=str)),
            "load_max_seq_length": self.model.config.max_seq_length,
            "runtime_device": str(self.model.device),
            "cuda_device": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "backbone_dtype": dtype,
            "head_dtype": str(self.model.head.hidden_norm.weight.dtype).removeprefix("torch."),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": ADAPTER_NAME,
            "model_type": MODEL_TYPE,
            "model": self.model_repo,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "tokenizer_repo": self.model_repo,
            "tokenizer_revision": self.model_revision,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "runtime": RUNTIME,
            "loader": LOADER,
            "remote_code_sha256": {name: RELEASE_SHA256[name] for name in REMOTE_CODE_FILES},
            "adapter_weights_sha256": RELEASE_SHA256[WEIGHTS_FILE],
            "adapter_config_sha256": RELEASE_SHA256["checkpoint/adapter_config.json"],
            "joint_head_sha256": RELEASE_SHA256["checkpoint/joint_head.safetensors"],
            "decision_config_sha256": RELEASE_SHA256["checkpoint/unsloth_decision_config.json"],
            "model_tokenizer_sha256": RELEASE_SHA256["checkpoint/tokenizer.json"],
            "probability_source": PROBABILITY_SOURCE,
            "probability_source_detail": PROBABILITY_SOURCE_DETAIL,
            "prediction_normalization": "native_probability_distribution",
            "render_policy_version": POLICY_VERSION,
            "render_question_id": QUESTION_ID,
            "render_state_serialization": STATE_SERIALIZATION,
            "input_truncation_policy": TRUNCATION_POLICY,
            "input_truncation_policy_detail": TRUNCATION_POLICY_DETAIL,
            "input_token_counter": "ArbitDecisionModel.count_tokens",
            "max_input_tokens": MAX_INPUT_TOKENS,
            "serve_max_input_tokens": self.serve_max_input_tokens,
            "eligibility_definition": ELIGIBILITY_DEFINITION,
            "supported_primitives": [primitive.value for primitive in QUESTION_TYPES],
            "attn_implementation": self.attn_implementation,
            **self._runtime,
        }

    def prompt_characters(self, example: DecisionExample) -> int:
        return render_arbit_request(example).prompt_characters

    def validate_example(self, example: DecisionExample) -> None:
        self._count_tokens(render_arbit_request(example))

    def predict_batch(self, examples: Sequence[DecisionExample]) -> list[HFDecisionResponse]:
        responses: list[HFDecisionResponse] = []
        for example in examples:
            request = render_arbit_request(example)
            counted_tokens = self._count_tokens(request)
            started = time.monotonic()
            result = self.model.decide(request.state, request.questions)
            answer, option_probabilities, input_tokens = _read_decision(result, counted_tokens)
            latency_seconds = time.monotonic() - started
            probabilities = candidate_probabilities(request, option_probabilities)
            characters = request.prompt_characters
            responses.append(
                HFDecisionResponse(
                    prediction=DecisionPrediction(probabilities=probabilities),
                    request={"state": request.state, "questions": request.questions},
                    response={
                        "answer": answer,
                        "option_probabilities": option_probabilities,
                        "candidate_ids": list(request.candidate_ids),
                        "candidate_option_keys": list(request.option_keys),
                        "probabilities": probabilities,
                    },
                    latency_seconds=latency_seconds,
                    input_contract={
                        "policy_version": POLICY_VERSION,
                        "input_truncation_policy": TRUNCATION_POLICY,
                        "counted_surface": COUNTED_SURFACE,
                        "max_input_tokens": MAX_INPUT_TOKENS,
                        "original_input_tokens": input_tokens,
                        "final_input_tokens": input_tokens,
                        "cpu_counted_input_tokens": counted_tokens,
                        "model_input_tokens": input_tokens,
                        "serve_max_input_tokens": self.serve_max_input_tokens,
                        "original_text_characters": characters,
                        "final_text_characters": characters,
                        "truncated_text_fields": 0,
                        "truncated": False,
                        "question_id": QUESTION_ID,
                        "question_type": request.question_type,
                        "instruction_verbatim": request.instruction_verbatim,
                        "criteria_fields": (
                            list(request.criteria_fields)
                            if request.criteria_fields is not None
                            else None
                        ),
                        "table_form": request.criteria_fields is not None,
                    },
                )
            )
        return responses

    def _count_tokens(self, request: ArbitRequest) -> int:
        count = int(self.model.count_tokens(request.state, request.questions))
        if count > MAX_INPUT_TOKENS:
            raise UnsupportedArbitInputLength(
                f"rendered request is {count} tokens, over the {MAX_INPUT_TOKENS}-token limit; "
                "rejected, not truncated"
            )
        return count


def _read_decision(
    result: Any, counted_tokens: int
) -> tuple[dict[str, Any], dict[str, float], int]:
    if not isinstance(result, dict):
        raise RuntimeError("Arbit decide() did not return an object")
    answers = result.get("answers")
    distributions = result.get("probabilities")
    if not isinstance(answers, dict) or not isinstance(distributions, dict):
        raise RuntimeError("Arbit decide() returned no answers or probabilities")
    if set(answers) != {QUESTION_ID} or set(distributions) != {QUESTION_ID}:
        raise RuntimeError("Arbit decide() did not answer exactly the decision question")
    answer = answers[QUESTION_ID]
    distribution = distributions[QUESTION_ID]
    if not isinstance(answer, dict) or not isinstance(distribution, dict):
        raise RuntimeError("Arbit decide() returned a malformed answer")
    probabilities = {str(key): float(value) for key, value in distribution.items()}
    rounded = answer.get("probabilities") or {}
    if set(map(str, rounded)) != set(probabilities) or any(
        abs(round(value, 4) - float(rounded[key])) > 1e-9 for key, value in probabilities.items()
    ):
        raise RuntimeError("unrounded probabilities disagree with the answer's 4-decimal ones")
    input_tokens = result.get("input_tokens")
    if input_tokens != counted_tokens or result.get("truncated") is not False:
        raise RuntimeError(
            f"input of {counted_tokens} tokens was read as {input_tokens} "
            f"(truncated: {result.get('truncated')})"
        )
    return answer, probabilities, counted_tokens


def _description_table(
    candidates: Sequence[Candidate],
) -> tuple[tuple[str, ...], list[str]] | None:
    """Field list and per-candidate value arrays when every description is a same-keyed object."""

    parsed: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            value = json.loads(str(candidate.description))
        except ValueError:
            return None
        if not isinstance(value, dict) or not value:
            return None
        if not all(isinstance(key, str) for key in value):
            return None
        parsed.append(value)
    fields = sorted(parsed[0])
    if any(sorted(value) != fields for value in parsed):
        return None
    cells = [
        json.dumps(
            [value[field] for field in fields],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for value in parsed
    ]
    return tuple(fields), cells


def _criterion(label: str, description: str) -> str:
    return f"{label}: {description}" if description else label


def _storage_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _check_distribution(values: Sequence[float], name: str) -> None:
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise RuntimeError(f"{name} are outside [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
        raise RuntimeError(f"{name} sum to {sum(values)!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
