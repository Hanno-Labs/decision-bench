"""DecisionBench public API."""

from decision_bench.benchmark import Benchmark, BenchmarkSpec, get_benchmark, get_benchmarks
from decision_bench.data import DatasetSpec, load_examples
from decision_bench.prompt import PROMPT_VERSION, build_openrouter_request
from decision_bench.results import (
    DecisionBenchResult,
    ModelMetadata,
    ModelType,
    ResultCache,
    ViewMetrics,
)
from decision_bench.schemas import DecisionExample, DecisionPrediction, Primitive
from decision_bench.task_spec import DecisionTask, TaskMetadata, get_task, get_tasks, register_task

__all__ = [
    "PROMPT_VERSION",
    "Benchmark",
    "BenchmarkSpec",
    "DatasetSpec",
    "DecisionBenchResult",
    "DecisionExample",
    "DecisionPrediction",
    "DecisionTask",
    "ModelMetadata",
    "ModelType",
    "Primitive",
    "ResultCache",
    "TaskMetadata",
    "ViewMetrics",
    "build_openrouter_request",
    "get_benchmark",
    "get_benchmarks",
    "get_task",
    "get_tasks",
    "load_examples",
    "register_task",
]
