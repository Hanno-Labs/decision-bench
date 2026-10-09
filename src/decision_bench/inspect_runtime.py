"""Inspect execution for native decision probabilities and compatible HF batches."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelCall,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.scorer import Metric, SampleScore, exact, metric
from inspect_ai.solver import generate
from inspect_ai.tool import ToolChoice, ToolInfo

from decision_bench.schemas import DecisionExample

Record = dict[str, Any]


def audited_records(log_dir: Path) -> dict[str, Record]:
    """Return only records for which Inspect persisted the completed sample score."""
    records: dict[str, Record] = {}
    for path in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            record = (sample.output.metadata or {}).get("raw_record")
            exact_score = (sample.scores or {}).get("exact")
            if isinstance(record, dict) and exact_score is not None:
                correct = record["status"] == "ok" and record["scored"]["correct"]
                if (exact_score.value == "C") == correct:
                    records[str(record["row_id"])] = record
            # Preserve resume support for logs made before the built-in scorer.
            for score in (sample.scores or {}).values():
                record = (score.metadata or {}).get("raw_record")
                if isinstance(record, dict):
                    records[str(record["row_id"])] = record
    return records


class UnsupportedCandidateCount(ValueError):
    """An adapter cannot represent this row's full candidate set."""


def candidate_target(candidate_id: str) -> str:
    """Avoid collisions caused by exact()'s case and punctuation normalization."""
    return "dbid" + candidate_id.encode("utf-8").hex()


def decision_sample(example: DecisionExample) -> Sample:
    """Keep labels and source annotations out of the model-visible input."""
    payload = example.model_dump(
        mode="json", exclude={"gold_candidate_id", "gold_probabilities", "source"}
    )
    return Sample(
        id=example.row_id,
        input=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        target=candidate_target(example.gold_candidate_id),
        metadata={"example": example.model_dump(mode="json")},
    )


@modelapi(name="decision-bench")
class NativeDecisionAPI(ModelAPI):
    """Bridge native adapter I/O into Inspect's model-call transcript."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.adapter: Any = None
        self.examples: dict[str, DecisionExample] = {}
        self.records: dict[str, Record] = {}
        self.validation_errors: dict[str, Exception] = {}
        self.record_result: Callable[[DecisionExample, Any], Record]
        self.batches: list[list[DecisionExample]] | None = None
        self.batch_indices: dict[str, int] = {}
        self.batch_tasks: dict[int, asyncio.Task[None]] = {}
        self.batch_remaining: dict[int, int] = {}
        self.batch_lock = asyncio.Lock()
        self.responses: dict[str, Any] = {}
        self.raw_path: Path
        self.total = 0
        self.completed = 0
        self.started = time.monotonic()
        self.last_checkpoint_at = self.started
        self.checkpoint_dir: Path | None = None
        self.checkpoint_interval_seconds = 120

    async def _predict_batch(self, index: int) -> None:
        assert self.batches is not None
        batch = self.batches[index]
        async with self.batch_lock:
            results = await asyncio.to_thread(self.adapter.predict_batch, batch)
        self.responses.update(
            {example.row_id: result for example, result in zip(batch, results, strict=True)}
        )

    def _write_record(self, record: Record) -> None:
        # Persist before scoring so interrupted/unscored samples remain resumable.
        with self.raw_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed += 1
        now = time.monotonic()
        if self.checkpoint_dir is not None and (
            now - self.last_checkpoint_at >= self.checkpoint_interval_seconds
            or self.completed == self.total
        ):
            from decision_bench.evaluate import _write_raw_checkpoint

            _write_raw_checkpoint(self.raw_path, self.checkpoint_dir)
            self.last_checkpoint_at = now
        progress_every = max(10, min(100, max(self.total // 100, 1)))
        if self.completed % progress_every == 0 or self.completed == self.total:
            elapsed = max(now - self.started, 1e-9)
            print(
                "DECISION_BENCH_PROGRESS "
                f"completed={self.completed} total={self.total} "
                f"rows_per_second={self.completed / elapsed:.3f}",
                flush=True,
            )

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput, ModelCall]:
        payload = json.loads(input[-1].text)
        row_id = str(payload["row_id"])
        example = self.examples[row_id]
        result: Any = None
        try:
            if row_id in self.validation_errors:
                raise self.validation_errors[row_id]
            if self.batches is None:
                result = await asyncio.to_thread(self.adapter.predict, example)
            else:
                index = self.batch_indices[row_id]
                if index not in self.batch_tasks:
                    # Dispatch a full planned batch without waiting for peer samples.
                    self.batch_tasks[index] = asyncio.create_task(self._predict_batch(index))
                try:
                    await self.batch_tasks[index]
                    result = self.responses.pop(row_id)
                finally:
                    self.batch_remaining[index] -= 1
                    if self.batch_remaining[index] == 0:
                        del self.batch_tasks[index]
            record = self.record_result(example, result)
            record["evaluation_framework"] = "inspect-ai"
            completion = candidate_target(record["scored"]["selected_candidate_id"])
        except Exception as error:
            record = error_record(example, error)
            # Valid targets always start with dbid; this cannot match any target.
            completion = "dberror"
        self.records[row_id] = record
        self._write_record(record)
        output = ModelOutput.from_content(self.model_name, completion)
        output.metadata = {"raw_record": record}
        if result is None:
            return output
        return output, ModelCall.create(request=result.request, response=result.response)


@metric(scores="unreduced")
def decision_metrics() -> Metric:
    """Compute DecisionBench metrics from native records, including all-row accuracy."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        api = cast(NativeDecisionAPI, get_model().api)
        records = [api.records[str(item.sample_id)] for item in scores]
        for item, record in zip(scores, records, strict=True):
            correct = record["status"] == "ok" and record["scored"]["correct"]
            if (item.score.value == "C") != correct:
                raise RuntimeError("Inspect exact-match score disagrees with native decision score")
        successful = [record for record in records if record["status"] == "ok"]
        correct = sum(item.score.value == "C" for item in scores)
        count = len(records)
        supported = len(successful)
        # Share the canonical implementation rather than a framework-default ECE.
        from decision_bench.evaluate import expected_calibration_error

        return {
            "accuracy": correct / count if count else 0.0,
            "supported_accuracy": correct / supported if supported else float("nan"),
            "coverage": supported / count if count else 0.0,
            "negative_log_likelihood": (
                sum(float(record["negative_log_likelihood"]) for record in successful) / supported
                if supported
                else float("nan")
            ),
            "expected_calibration_error": (
                expected_calibration_error(successful) if supported else float("nan")
            ),
        }

    return compute


def error_record(example: DecisionExample, error: Exception) -> Record:
    return {
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
        "evaluation_framework": "inspect-ai",
    }


def run_inspect(
    examples: list[DecisionExample],
    output_dir: Path,
    *,
    adapter: Any,
    record_result: Callable[[DecisionExample, Any], Record],
    concurrency: int,
    metadata: Record,
    batches: list[list[DecisionExample]] | None = None,
    validation_errors: dict[str, Exception] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_interval_seconds: int = 120,
) -> Record:
    """Run pending rows as one Inspect Task; emit a native .eval audit artifact."""
    if concurrency < 1:
        raise ValueError("evaluation concurrency must be positive")
    if not examples:
        return {}
    model_name = metadata.get("model_repo") or metadata.get("model")
    if not model_name:
        model_dir = metadata.get("model_dir")
        model_name = Path(model_dir).name if model_dir else type(adapter).__name__
    api = NativeDecisionAPI(str(model_name))
    api.adapter = adapter
    api.examples = {example.row_id: example for example in examples}
    api.record_result = record_result
    api.validation_errors = validation_errors or {}
    api.batches = batches
    api.raw_path = output_dir / "raw.jsonl"
    api.total = len(examples)
    api.checkpoint_dir = checkpoint_dir
    api.checkpoint_interval_seconds = checkpoint_interval_seconds
    if batches is not None:
        api.batch_indices = {
            example.row_id: index for index, batch in enumerate(batches) for example in batch
        }
        api.batch_remaining = {index: len(batch) for index, batch in enumerate(batches)}
        examples = [example for batch in batches for example in batch] + [
            example for example in examples if example.row_id in api.validation_errors
        ]
    task = Task(
        name="DecisionBench",
        dataset=[decision_sample(example) for example in examples],
        solver=generate(),
        scorer=exact(),
        metrics=[decision_metrics()],
        metadata={**metadata, "ece_definition": "15-bin equal-width top-label ECE"},
    )
    logs = inspect_eval(
        task,
        model=Model(api, GenerateConfig(max_retries=0, max_connections=concurrency, cache=False)),
        max_samples=concurrency,
        log_dir=str(output_dir / "inspect"),
        display="none",
        log_samples=True,
        log_model_api=True,
        fail_on_error=False,
        retry_on_error=0,
    )
    if len(logs) != 1 or logs[0].status != "success":
        raise RuntimeError("Inspect evaluation did not complete; check its log and resume")
    log = logs[0]
    if log.results is None or log.results.completed_samples != len(examples):
        raise RuntimeError("Inspect log does not account for every pending row")
    return {
        "evaluation_framework": "inspect-ai",
        "evaluation_framework_version": version("inspect_ai"),
    }
