"""Resumable evaluation of an external decision model."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from decision_bench.models import (
    HFDecisionModel,
    JevOpenRouterDecisionModel,
    NanoJevHFDecisionModel,
    NimbleHFDecisionModel,
    OpenJevHFDecisionModel,
    OpenRouterDecisionModel,
    OpenRouterTopLogprobsDecisionModel,
    SystemOneHFDecisionModel,
)
from decision_bench.models.jev_openrouter import JEV_CONTRACT_VERSION
from decision_bench.prompt import (
    PROMPT_VERSION,
    TOP_LOGPROBS_PROMPT_VERSION,
    prompt_sha256,
    top_logprobs_prompt_sha256,
)
from decision_bench.schemas import DecisionExample
from decision_bench.scoring import negative_log_likelihood, score_prediction

ECE_BINS = 15


def select_smoke_examples(examples: list[DecisionExample]) -> list[DecisionExample]:
    """Cover every family, primitive, and major candidate-count regime."""

    selected: dict[str, DecisionExample] = {}
    for example in examples:
        selected.setdefault(f"family:{example.family}", example)
        selected.setdefault(f"primitive:{example.primitive.value}", example)
        if len(example.candidates) in {2, 4, 8, 16, 32, 64, 77, 128, 151, 255}:
            selected.setdefault(f"candidates:{len(example.candidates)}", example)
    return list({example.row_id: example for example in selected.values()}.values())


def run_openrouter_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model: str,
    reasoning_effort: str,
    seed: int,
    concurrency: int,
    reasoning_family_effort: str | None = None,
) -> dict[str, Any]:
    """Evaluate rows concurrently with append-only raw output and resumability."""

    with OpenRouterDecisionModel(
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_family_effort=reasoning_family_effort,
        seed=seed,
    ) as decision_model:
        return _run_evaluation(
            examples,
            output_dir,
            decision_model=decision_model,
            concurrency=concurrency,
            metadata={
                "model": model,
                "reasoning_effort": reasoning_effort,
                "reasoning_family_effort": reasoning_family_effort,
                "reasoning_effort_policy": (
                    "reasoning-family-override-v1"
                    if reasoning_family_effort is not None
                    else "uniform-v1"
                ),
                "seed": seed,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha256(),
                "prediction_normalization": "divide_positive_finite_values_by_sum",
            },
        )


def run_jev_openrouter_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model: str,
    concurrency: int,
    max_state_question_tokens: int = 32_000,
    input_token_reserve: int = 2_048,
    tokenizer_model: str = "Qwen/Qwen3-0.6B",
    tokenizer_revision: str = "c1899de289a04d12100db370d81485cdf75e47ca",
) -> dict[str, Any]:
    """Evaluate Jev through OpenRouter's native Decisions endpoint."""

    with JevOpenRouterDecisionModel(
        model=model,
        max_state_question_tokens=max_state_question_tokens,
        input_token_reserve=input_token_reserve,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
    ) as decision_model:
        return _run_evaluation(
            examples,
            output_dir,
            decision_model=decision_model,
            concurrency=concurrency,
            metadata={
                "model": model,
                "native_contract_version": JEV_CONTRACT_VERSION,
                "prediction_normalization": "native_probability_distribution",
                **decision_model.metadata,
            },
        )


def run_openrouter_top_logprobs_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model: str,
    seed: int,
    concurrency: int,
    top_logprobs: int = 20,
    benchmark_rows: int | None = None,
) -> dict[str, Any]:
    """Evaluate eligible rows from native one-token OpenRouter logprobs."""

    with OpenRouterTopLogprobsDecisionModel(
        model=model,
        seed=seed,
        top_logprobs=top_logprobs,
    ) as decision_model:
        return _run_evaluation(
            examples,
            output_dir,
            decision_model=decision_model,
            concurrency=concurrency,
            metadata={
                "model": model,
                "seed": seed,
                "prompt_version": TOP_LOGPROBS_PROMPT_VERSION,
                "prompt_sha256": top_logprobs_prompt_sha256(),
                "probability_source": "native_top_logprobs_conditional",
                "top_logprobs": top_logprobs,
                "benchmark_rows": benchmark_rows if benchmark_rows is not None else len(examples),
                "eligible_rows": len(examples),
                "ineligible_rows": (benchmark_rows - len(examples))
                if benchmark_rows is not None
                else 0,
                "eligibility_definition": (
                    f"candidate_count <= {top_logprobs}; complete candidate token coverage required"
                ),
            },
        )


def run_hf_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model_dir: Path,
    seed: int,
    batch_size: int,
    max_prompt_characters_per_batch: int,
    max_length: int,
    attn_implementation: str,
    benchmark_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a native local HF decision-token checkpoint in CUDA batches."""

    decision_model = HFDecisionModel(
        model_dir=model_dir,
        seed=seed,
        max_length=max_length,
        attn_implementation=attn_implementation,
    )
    return _run_hf_batches(
        examples,
        output_dir,
        decision_model=decision_model,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        metadata={**decision_model.metadata, **(benchmark_metadata or {})},
    )


def run_nimble_hf_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model_dir: Path,
    adapter_repo: str,
    adapter_revision: str,
    expected_adapter_sha256: str,
    batch_size: int,
    max_prompt_characters_per_batch: int,
    attn_implementation: str,
    benchmark_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Nimble natively while retaining its unsupported rows as failures."""

    decision_model = NimbleHFDecisionModel(
        model_dir=model_dir,
        adapter_repo=adapter_repo,
        adapter_revision=adapter_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        attn_implementation=attn_implementation,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    completed = _recorded_row_ids(raw_path)
    eligible = [example for example in examples if len(example.candidates) <= 26]
    unsupported = [example for example in examples if len(example.candidates) > 26]
    with raw_path.open("a") as raw_handle:
        for example in unsupported:
            if example.row_id in completed:
                continue
            raw_handle.write(
                json.dumps(
                    {
                        "status": "error",
                        "row_id": example.row_id,
                        "primitive": example.primitive.value,
                        "family": example.family,
                        "domain": example.domain,
                        "candidate_count": len(example.candidates),
                        "example": example.model_dump(mode="json"),
                        "error_type": "UnsupportedCandidateCount",
                        "error": "Bespoke-Nimble-9B supports at most 26 choices",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())

    summary = _run_hf_batches(
        eligible,
        output_dir,
        decision_model=decision_model,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        metadata={**decision_model.metadata, **(benchmark_metadata or {})},
    )
    successful_rows = int(summary["successful_rows"])
    eligible_accuracy = float(summary["metrics"]["overall"]["accuracy"])
    correct_rows = round(eligible_accuracy * successful_rows)
    summary.update(
        {
            "requested_rows": len(examples),
            "eligible_rows": len(eligible),
            "ineligible_rows": len(unsupported),
            "coverage": successful_rows / len(examples),
            "benchmark_accuracy_counting_unsupported_as_incorrect": correct_rows
            / len(examples),
            "ineligible_definition": "candidate_count > 26",
            "raw_sha256": _sha256_file(raw_path),
        }
    )
    _write_run_artifacts(output_dir, summary)
    return summary


def run_public_hf_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    model_dir: Path,
    model_type: Literal["nanojev", "openjev", "system-one"],
    model_repo: str,
    model_revision: str,
    base_revision: str | None,
    expected_weights_sha256: str | None,
    batch_size: int,
    max_prompt_characters_per_batch: int,
    attn_implementation: str,
    benchmark_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a public Jev-shaped model with its published native contract."""

    decision_model: (
        NanoJevHFDecisionModel | OpenJevHFDecisionModel | SystemOneHFDecisionModel
    )
    if model_type == "nanojev":
        if expected_weights_sha256 is None:
            raise ValueError("NanoJev requires an expected weights SHA-256")
        decision_model = NanoJevHFDecisionModel(
            model_dir=model_dir,
            model_repo=model_repo,
            model_revision=model_revision,
            expected_weights_sha256=expected_weights_sha256,
            attn_implementation=attn_implementation,
        )
    elif model_type == "openjev":
        decision_model = OpenJevHFDecisionModel(
            model_dir=model_dir,
            model_repo=model_repo,
            model_revision=model_revision,
        )
    else:
        if base_revision is None:
            raise ValueError("System One requires a pinned base revision")
        decision_model = SystemOneHFDecisionModel(
            model_dir=model_dir,
            model_repo=model_repo,
            model_revision=model_revision,
            base_revision=base_revision,
            attn_implementation=attn_implementation,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    recorded = _recorded_row_ids(raw_path)
    eligible: list[DecisionExample] = []
    with raw_path.open("a") as raw_handle:
        for example in examples:
            if example.row_id in recorded:
                continue
            try:
                decision_model.validate_example(example)
            except Exception as error:
                raw_handle.write(
                    json.dumps(
                        {
                            "status": "error",
                            "row_id": example.row_id,
                            "primitive": example.primitive.value,
                            "family": example.family,
                            "domain": example.domain,
                            "candidate_count": len(example.candidates),
                            "example": example.model_dump(mode="json"),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:
                eligible.append(example)
        raw_handle.flush()
        os.fsync(raw_handle.fileno())

    summary = _run_hf_batches(
        eligible,
        output_dir,
        decision_model=decision_model,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        metadata={**decision_model.metadata, **(benchmark_metadata or {})},
    )
    successful_rows = int(summary["successful_rows"])
    eligible_accuracy = float(summary["metrics"]["overall"]["accuracy"])
    correct_rows = round(eligible_accuracy * successful_rows)
    summary.update(
        {
            "requested_rows": len(examples),
            "eligible_rows": successful_rows,
            "ineligible_rows": int(summary["error_rows"]),
            "coverage": successful_rows / len(examples),
            "benchmark_accuracy_counting_unsupported_as_incorrect": correct_rows
            / len(examples),
            "raw_sha256": _sha256_file(raw_path),
        }
    )
    _write_run_artifacts(output_dir, summary)
    return summary


def _run_hf_batches(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    decision_model: (
        HFDecisionModel
        | NimbleHFDecisionModel
        | NanoJevHFDecisionModel
        | OpenJevHFDecisionModel
        | SystemOneHFDecisionModel
    ),
    batch_size: int,
    max_prompt_characters_per_batch: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    completed = _completed_row_ids(raw_path)
    pending = [example for example in examples if example.row_id not in completed]
    batches = _hf_batches(
        pending,
        decision_model=decision_model,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
    )
    started = time.time()
    completed_count = 0
    with raw_path.open("a") as raw_handle:
        for batch in batches:
            results = decision_model.predict_batch(batch)
            records = [
                _successful_record(example, result)
                for example, result in zip(batch, results, strict=True)
            ]
            for record in records:
                raw_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
            completed_count += len(batch)
            elapsed = max(time.time() - started, 1e-9)
            print(
                "DECISION_BENCH_PROGRESS "
                f"completed={completed_count} total={len(pending)} "
                f"rows_per_second={completed_count / elapsed:.3f}",
                flush=True,
            )
    summary = summarize_raw(raw_path)
    summary.update(metadata)
    summary.update(
        {
            "requested_rows": len(examples),
            "raw_sha256": _sha256_file(raw_path),
            "batch_size": batch_size,
            "max_prompt_characters_per_batch": max_prompt_characters_per_batch,
        }
    )
    _write_run_artifacts(output_dir, summary)
    return summary


def _hf_batches(
    examples: list[DecisionExample],
    *,
    decision_model: (
        HFDecisionModel
        | NimbleHFDecisionModel
        | NanoJevHFDecisionModel
        | OpenJevHFDecisionModel
        | SystemOneHFDecisionModel
    ),
    batch_size: int,
    max_prompt_characters_per_batch: int,
) -> list[list[DecisionExample]]:
    if batch_size < 1 or max_prompt_characters_per_batch < 1:
        raise ValueError("HF batch limits must be positive")
    ordered = sorted(
        examples,
        key=lambda example: (len(example.candidates), decision_model.prompt_characters(example)),
    )
    batches: list[list[DecisionExample]] = []
    current: list[DecisionExample] = []
    current_characters = 0
    current_candidate_count = 0
    for example in ordered:
        characters = decision_model.prompt_characters(example)
        candidate_count = len(example.candidates)
        if current and (
            len(current) >= batch_size
            or current_characters + characters > max_prompt_characters_per_batch
            or candidate_count != current_candidate_count
        ):
            batches.append(current)
            current = []
            current_characters = 0
        if not current:
            current_candidate_count = candidate_count
        current.append(example)
        current_characters += characters
    if current:
        batches.append(current)
    return batches


def _run_evaluation(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    decision_model: (
        OpenRouterDecisionModel
        | JevOpenRouterDecisionModel
        | OpenRouterTopLogprobsDecisionModel
    ),
    concurrency: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    completed = _completed_row_ids(raw_path)
    pending = [example for example in examples if example.row_id not in completed]
    started = time.time()
    progress_every = max(10, min(100, max(len(pending) // 100, 1)))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_examples: dict[Future[dict[str, Any]], DecisionExample] = {
            executor.submit(_evaluate_one, decision_model, example): example for example in pending
        }
        with raw_path.open("a") as raw_handle:
            for completed_count, future in enumerate(as_completed(future_examples), start=1):
                example = future_examples[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "status": "error",
                        "row_id": example.row_id,
                        "task_name": example.task_name,
                        "primitive": example.primitive.value,
                        "family": example.family,
                        "domain": example.domain,
                        "candidate_count": len(example.candidates),
                        "example": example.model_dump(mode="json"),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                raw_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                raw_handle.flush()
                if completed_count % progress_every == 0 or completed_count == len(pending):
                    os.fsync(raw_handle.fileno())
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        "DECISION_BENCH_PROGRESS "
                        f"completed={completed_count} total={len(pending)} "
                        f"rows_per_second={completed_count / elapsed:.3f}",
                        flush=True,
                    )
    summary = summarize_raw(raw_path)
    summary.update(metadata)
    summary.update({"requested_rows": len(examples), "raw_sha256": _sha256_file(raw_path)})
    _write_run_artifacts(output_dir, summary)
    return summary


def _evaluate_one(
    decision_model: (
        OpenRouterDecisionModel
        | JevOpenRouterDecisionModel
        | OpenRouterTopLogprobsDecisionModel
    ),
    example: DecisionExample,
) -> dict[str, Any]:
    result = decision_model.predict(example)
    return _successful_record(example, result)


def _successful_record(example: DecisionExample, result: Any) -> dict[str, Any]:
    scored = score_prediction(example, result.prediction)
    record = {
        "status": "ok",
        "row_id": example.row_id,
        "task_name": example.task_name,
        "primitive": example.primitive.value,
        "family": example.family,
        "domain": example.domain,
        "candidate_count": len(example.candidates),
        "example": example.model_dump(mode="json"),
        "request": result.request,
        "response": result.response,
        "latency_seconds": result.latency_seconds,
        "scored": scored.model_dump(mode="json"),
        "negative_log_likelihood": negative_log_likelihood(example, result.prediction),
    }
    if result.input_contract is not None:
        record["input_contract"] = result.input_contract
    return record


def summarize_raw(raw_path: Path) -> dict[str, Any]:
    """Aggregate successful raw rows without hiding failures."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latest_by_row_id: dict[str, dict[str, Any]] = {}
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            latest_by_row_id[str(record["row_id"])] = record
    errors = 0
    for record in latest_by_row_id.values():
        if record["status"] != "ok":
            errors += 1
            continue
        groups["overall"].append(record)
        task_name = _record_dimension(record, "task_name")
        if task_name is not None:
            groups[f"task:{task_name}"].append(record)
        groups[f"primitive:{record['primitive']}"].append(record)
        groups[f"family:{record['family']}"].append(record)
        groups[f"domain:{record['domain']}"].append(record)
        candidate_count = int(record.get("candidate_count", len(record["scored"]["probabilities"])))
        groups[f"candidate_count:{candidate_count}"].append(record)
    input_contracts = [
        record["input_contract"]
        for record in groups["overall"]
        if isinstance(record.get("input_contract"), dict)
    ]
    return {
        "successful_rows": len(groups["overall"]),
        "error_rows": errors,
        "model_input_truncation": {
            "reported_rows": len(input_contracts),
            "truncated_rows": sum(
                bool(contract.get("truncated")) for contract in input_contracts
            ),
            "policy_versions": sorted(
                {
                    str(contract["policy_version"])
                    for contract in input_contracts
                    if "policy_version" in contract
                }
            ),
        },
        "metrics": {
            name: {
                "rows": len(records),
                "accuracy": sum(bool(record["scored"]["correct"]) for record in records)
                / len(records),
                "mean_negative_log_likelihood": sum(
                    float(record["negative_log_likelihood"]) for record in records
                )
                / len(records),
                "mean_latency_seconds": sum(float(record["latency_seconds"]) for record in records)
                / len(records),
                "expected_calibration_error": expected_calibration_error(records),
                "ece_bins": ECE_BINS,
            }
            for name, records in sorted(groups.items())
            if records
        },
    }


def _record_dimension(record: dict[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is None and isinstance(record.get("example"), dict):
        value = record["example"].get(name)
    return str(value) if value is not None else None


def expected_calibration_error(
    records: list[dict[str, Any]],
    *,
    bins: int = ECE_BINS,
) -> float:
    """Return equal-width top-label ECE over successful predictions."""

    if bins < 1:
        raise ValueError("bins must be positive")
    if not records:
        raise ValueError("records must not be empty")
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    correctness_sums = [0.0] * bins
    for record in records:
        probabilities = record["scored"]["probabilities"]
        confidence = max(float(value) for value in probabilities)
        bin_index = min(int(confidence * bins), bins - 1)
        counts[bin_index] += 1
        confidence_sums[bin_index] += confidence
        correctness_sums[bin_index] += float(bool(record["scored"]["correct"]))
    total = len(records)
    error = 0.0
    for count, confidence_sum, correctness_sum in zip(
        counts,
        confidence_sums,
        correctness_sums,
        strict=True,
    ):
        if count == 0:
            continue
        mean_confidence = confidence_sum / count
        mean_accuracy = correctness_sum / count
        error += (count / total) * abs(mean_accuracy - mean_confidence)
    return error


def refresh_summary(output_dir: Path) -> dict[str, Any]:
    """Recompute aggregate metrics from saved raw rows without inference."""

    raw_path = output_dir / "raw.jsonl"
    summary_path = output_dir / "summary.json"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    existing: dict[str, Any] = {}
    if summary_path.is_file():
        loaded = json.loads(summary_path.read_text())
        if not isinstance(loaded, dict):
            raise TypeError("summary.json must contain an object")
        existing = loaded
    summary = {
        **existing,
        **summarize_raw(raw_path),
        "raw_sha256": _sha256_file(raw_path),
        "ece_definition": "15-bin equal-width top-label ECE over successful rows",
    }
    _write_run_artifacts(output_dir, summary)
    return summary


def _write_run_artifacts(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "decision-bench-run-v1",
        "requested_rows": summary.get(
            "requested_rows", int(summary["successful_rows"]) + int(summary["error_rows"])
        ),
        "successful_rows": summary["successful_rows"],
        "error_rows": summary["error_rows"],
        "files": {
            "raw.jsonl": _sha256_file(output_dir / "raw.jsonl"),
            "summary.json": _sha256_file(summary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _completed_row_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    completed: set[str] = set()
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(str(record["row_id"]))
    return completed


def _recorded_row_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    recorded: set[str] = set()
    with raw_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            recorded.add(str(record["row_id"]))
    return recorded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
