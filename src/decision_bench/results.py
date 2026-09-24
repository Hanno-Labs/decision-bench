"""Reviewed result records, local caching, and submission staging."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RESULTS_REPOSITORY = "https://github.com/Hanno-Labs/decision-bench-results.git"
ENGLISH_SUITE_VIEW = "suite:DecisionBench(eng, v1)"
ModelType = Literal["decision-model", "language-model", "classifier"]


class ModelMetadata(BaseModel):
    """Immutable model identity and leaderboard display metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    model_type: ModelType
    url: str | None = None
    adapter: str = Field(min_length=1)
    probability_source: str = Field(min_length=1)
    open_weights: bool | None = None
    parameter_count: int | None = Field(default=None, ge=0)


class ArtifactReference(BaseModel):
    """Content-addressed pointer to complete durable run artifacts."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ViewMetrics(BaseModel):
    """Metrics for one benchmark, task, family, domain, or primitive view."""

    model_config = ConfigDict(extra="forbid")

    rows: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_negative_log_likelihood: float | None = Field(default=None, ge=0.0)
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_latency_seconds: float | None = Field(default=None, ge=0.0)


class DecisionBenchResult(BaseModel):
    """One reviewed model result for an immutable benchmark release."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["decision-bench-result-v1"] = "decision-bench-result-v1"
    benchmark_name: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    dataset_repo: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    task_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ModelMetadata
    requested_rows: int = Field(ge=1)
    successful_rows: int = Field(ge=0)
    unsupported_rows: int = Field(ge=0)
    error_rows: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    primary_accuracy: float = Field(ge=0.0, le=1.0)
    supported_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_negative_log_likelihood: float | None = Field(default=None, ge=0.0)
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_latency_seconds: float | None = Field(default=None, ge=0.0)
    views: dict[str, ViewMetrics]
    artifact: ArtifactReference | None = None
    submitted_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> DecisionBenchResult:
        classified_rows = self.successful_rows + self.unsupported_rows + self.error_rows
        if classified_rows != self.requested_rows:
            raise ValueError(
                "successful_rows + unsupported_rows + error_rows must equal requested_rows"
            )
        expected_coverage = self.successful_rows / self.requested_rows
        if abs(self.coverage - expected_coverage) > 1e-9:
            raise ValueError("coverage must equal successful_rows / requested_rows")
        if self.supported_accuracy is not None:
            expected_primary = self.supported_accuracy * self.coverage
            if abs(self.primary_accuracy - expected_primary) > 1e-6:
                raise ValueError(
                    "primary_accuracy must count unsupported and error rows as misses"
                )
        return self


class ResultCache:
    """Load, sync, and stage records in the official results-repository layout."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        remote_url: str = DEFAULT_RESULTS_REPOSITORY,
    ) -> None:
        self.cache_path = Path(cache_path or Path.home() / ".cache/decision-bench/results")
        self.remote_url = remote_url

    def sync(self) -> Path:
        """Clone the official results repository or fast-forward an existing cache."""

        if (self.cache_path / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(self.cache_path), "pull", "--ff-only"], check=True
            )
        else:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", self.remote_url, str(self.cache_path)], check=True
            )
        return self.cache_path

    def load_results(self) -> list[DecisionBenchResult]:
        """Validate and return every reviewed result record in the cache."""

        results_root = self.cache_path / "results"
        if not results_root.is_dir():
            return []
        return [
            DecisionBenchResult.model_validate_json(path.read_text())
            for path in sorted(results_root.glob("*/*/*.json"))
            if path.name != "model_meta.json"
        ]

    def to_records(self, *, view: str = "overall") -> list[dict[str, Any]]:
        """Return flat rows suitable for tables or dataframe construction."""

        records: list[dict[str, Any]] = []
        for result in self.load_results():
            metrics = result.views.get(view)
            records.append(
                {
                    "model": result.model.name,
                    "revision": result.model.revision,
                    "model_type": result.model.model_type,
                    "benchmark": result.benchmark_name,
                    "benchmark_version": result.benchmark_version,
                    "view": view,
                    "primary_accuracy": result.primary_accuracy
                    if view == "overall"
                    else None,
                    "supported_accuracy": result.supported_accuracy
                    if view == "overall"
                    else (metrics.accuracy if metrics is not None else None),
                    "coverage": result.coverage if view == "overall" else None,
                    "successful_rows": result.successful_rows
                    if view == "overall"
                    else (metrics.rows if metrics is not None else None),
                    "error_rows": result.error_rows if view == "overall" else None,
                    "unsupported_rows": result.unsupported_rows if view == "overall" else None,
                    "probability_source": result.model.probability_source,
                    "adapter": result.model.adapter,
                    "artifact_uri": result.artifact.uri if result.artifact else None,
                }
            )
        return records

    def stage_result(
        self,
        run_dir: str | Path,
        *,
        model: ModelMetadata,
        artifact_uri: str | None = None,
        dataset_revision: str,
        benchmark_name: str = "DecisionBench",
        benchmark_version: str = "1.0",
        dataset_repo: str = "Hanno-Labs/decision-bench",
        task_spec_sha256: str | None = None,
    ) -> Path:
        """Validate a completed run and write its compact canonical result record."""

        run_path = Path(run_dir)
        summary_path = run_path / "summary.json"
        manifest_path = run_path / "manifest.json"
        raw_path = run_path / "raw.jsonl"
        for required in (summary_path, manifest_path, raw_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        summary = _load_object(summary_path)
        manifest = _load_object(manifest_path)
        expected_files = manifest.get("files")
        if not isinstance(expected_files, dict):
            raise ValueError("manifest files must be an object")
        for name, path in (("summary.json", summary_path), ("raw.jsonl", raw_path)):
            if expected_files.get(name) != _sha256_file(path):
                raise ValueError(f"manifest hash mismatch for {name}")

        requested_rows = int(summary["requested_rows"])
        successful_rows = int(summary["successful_rows"])
        classified_errors = _classify_errors(raw_path)
        unsupported_rows = classified_errors["unsupported_rows"]
        error_rows = classified_errors["error_rows"]
        if successful_rows + unsupported_rows + error_rows != requested_rows:
            raise ValueError("raw rows do not account for every requested benchmark row")
        overall = _object(summary["metrics"]).get("overall")
        if not isinstance(overall, dict):
            raise ValueError("summary metrics.overall must be an object")
        supported_accuracy = float(overall["accuracy"]) if successful_rows else None
        coverage = successful_rows / requested_rows
        primary_accuracy = (supported_accuracy or 0.0) * coverage
        task_hash = task_spec_sha256 or str(summary.get("task_spec_sha256", ""))
        if len(task_hash) != 64:
            raise ValueError("a 64-character task_spec_sha256 is required")

        views = {
            str(name): _view_metrics(metrics)
            for name, metrics in _object(summary["metrics"]).items()
        }
        suite_metrics = _non_reasoning_suite_metrics(raw_path)
        if suite_metrics is not None:
            views[ENGLISH_SUITE_VIEW] = suite_metrics

        record = DecisionBenchResult(
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            dataset_repo=dataset_repo,
            dataset_revision=dataset_revision,
            task_spec_sha256=task_hash,
            model=model,
            requested_rows=requested_rows,
            successful_rows=successful_rows,
            unsupported_rows=unsupported_rows,
            error_rows=error_rows,
            coverage=coverage,
            primary_accuracy=primary_accuracy,
            supported_accuracy=supported_accuracy,
            mean_negative_log_likelihood=_optional_float(
                overall.get("mean_negative_log_likelihood")
            ),
            expected_calibration_error=_optional_float(
                overall.get("expected_calibration_error")
            ),
            mean_latency_seconds=_optional_float(overall.get("mean_latency_seconds")),
            views=views,
            artifact=(
                ArtifactReference(
                    uri=artifact_uri,
                    manifest_sha256=_sha256_file(manifest_path),
                    summary_sha256=_sha256_file(summary_path),
                    raw_sha256=_sha256_file(raw_path),
                )
                if artifact_uri is not None
                else None
            ),
            submitted_at=datetime.now(UTC),
        )

        model_dir = self.cache_path / "results" / _safe_name(model.name) / model.revision
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_meta.json").write_text(
            json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        result_path = model_dir / f"{_safe_name(benchmark_name)}.json"
        payload = record.model_dump(mode="json")
        if payload["artifact"] is None:
            payload.pop("artifact")
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        return result_path

    def submit_result(
        self,
        result_path: Path,
        *,
        create_pr: bool = False,
    ) -> dict[str, str]:
        """Commit a staged record and optionally open a GitHub pull request."""

        if not create_pr:
            return {"path": str(result_path)}
        relative = result_path.relative_to(self.cache_path)
        model_name = result_path.parents[1].name
        branch = f"results/{model_name}-{datetime.now(UTC):%Y%m%d%H%M%S}"
        subprocess.run(["git", "-C", str(self.cache_path), "switch", "-c", branch], check=True)
        paths = [relative, relative.parent / "model_meta.json"]
        subprocess.run(
            ["git", "-C", str(self.cache_path), "add", *(str(path) for path in paths)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.cache_path), "commit", "-m", f"Add {model_name} results"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.cache_path), "push", "-u", "origin", branch], check=True
        )
        completed = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                "Hanno-Labs/decision-bench-results",
                "--fill",
            ],
            cwd=self.cache_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return {"path": str(result_path), "pr_url": completed.stdout.strip()}


def _load_object(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text()))


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return {str(key): item for key, item in value.items()}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        raise TypeError(f"expected a number, got {type(value)!r}")
    return float(value)


def _view_metrics(value: object) -> ViewMetrics:
    metrics = _object(value)
    return ViewMetrics.model_validate(
        {
            name: metrics[name]
            for name in ViewMetrics.model_fields
            if name in metrics
        }
    )


def _classify_errors(raw_path: Path) -> dict[str, int]:
    latest_by_row_id: dict[str, dict[str, Any]] = {}
    with raw_path.open() as handle:
        for line in handle:
            record = _object(json.loads(line))
            latest_by_row_id[str(record["row_id"])] = record
    unsupported_rows = 0
    error_rows = 0
    for record in latest_by_row_id.values():
        if record.get("status") == "ok":
            continue
        error_type = str(record.get("error_type", ""))
        if error_type.startswith("Unsupported") or (
            error_type == "ValueError"
            and record.get("error") == "SystemOne endpoint supports at most 26 candidates"
        ):
            unsupported_rows += 1
        else:
            error_rows += 1
    return {"unsupported_rows": unsupported_rows, "error_rows": error_rows}


def _non_reasoning_suite_metrics(raw_path: Path) -> ViewMetrics | None:
    """Compute the English suite metrics from successful non-reasoning rows.

    ECE is nonlinear, so it cannot be reconstructed by subtracting or
    averaging the overall and reasoning-family ECE values. The raw artifact is
    the authoritative source for this suite view.
    """

    latest_by_row_id: dict[str, dict[str, Any]] = {}
    with raw_path.open() as handle:
        for line in handle:
            record = _object(json.loads(line))
            latest_by_row_id[str(record["row_id"])] = record

    records = [
        record
        for record in latest_by_row_id.values()
        if record.get("status") == "ok"
        and _record_dimension(record, "family") != "reasoning"
        and isinstance(record.get("scored"), dict)
        and "negative_log_likelihood" in record
        and "latency_seconds" in record
    ]
    if not records:
        return None
    return ViewMetrics(
        rows=len(records),
        accuracy=sum(bool(record["scored"]["correct"]) for record in records)
        / len(records),
        mean_negative_log_likelihood=sum(
            float(record["negative_log_likelihood"]) for record in records
        )
        / len(records),
        expected_calibration_error=_expected_calibration_error(records),
        mean_latency_seconds=sum(
            float(record["latency_seconds"]) for record in records
        )
        / len(records),
    )


def _record_dimension(record: dict[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is None and isinstance(record.get("example"), dict):
        value = record["example"].get(name)
    return str(value) if value is not None else None


def _expected_calibration_error(
    records: list[dict[str, Any]],
    *,
    bins: int = 15,
) -> float:
    """Return equal-width top-label ECE over successful predictions."""

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
    return sum(
        (count / total) * abs(
            correctness_sum / count - confidence_sum / count
        )
        for count, confidence_sum, correctness_sum in zip(
            counts, confidence_sums, correctness_sums, strict=True
        )
        if count
    )


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
