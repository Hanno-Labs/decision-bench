"""DecisionBench command-line interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from decision_bench.benchmark import Benchmark, get_benchmark
from decision_bench.data import resolve_local_path
from decision_bench.evaluate import (
    refresh_summary,
    run_hf_evaluation,
    run_jev_openrouter_evaluation,
    run_nimble_hf_evaluation,
    run_openrouter_evaluation,
    run_openrouter_top_logprobs_evaluation,
    run_public_hf_evaluation,
    select_smoke_examples,
)
from decision_bench.prompt import prompt_sha256
from decision_bench.results import ModelMetadata, ModelType, ResultCache

app = typer.Typer(no_args_is_help=True)


@app.command("inspect")
def inspect_task(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    """Validate a task spec and sample its external dataset."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = [
        example.model_dump(mode="json") for example in benchmark.examples[:limit]
    ]
    typer.echo(
        json.dumps(
            {
                "benchmark": benchmark.spec.model_dump(mode="json"),
                "tasks": [task.metadata.model_dump(mode="json") for task in benchmark.tasks],
                "prompt_sha256": prompt_sha256(),
                "sample": examples,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("run-openrouter")
def run_openrouter(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "openai/gpt-5.6-luna",
    reasoning_effort: Annotated[str, typer.Option()] = "minimal",
    reasoning_family_effort: Annotated[str | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1, max=256)] = 32,
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a resumable OpenRouter evaluation and preserve complete raw I/O."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = list(benchmark.examples)
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_openrouter_evaluation(
        examples,
        output_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_family_effort=reasoning_family_effort,
        seed=seed,
        concurrency=concurrency,
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-openrouter-top-logprobs")
def run_openrouter_top_logprobs(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "openai/gpt-4o-mini",
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1, max=256)] = 32,
    top_logprobs: Annotated[int, typer.Option(min=1, max=20)] = 20,
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run eligible rows using native OpenRouter output-token logprobs."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    all_examples = list(benchmark.examples)
    examples = [example for example in all_examples if len(example.candidates) <= top_logprobs]
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_openrouter_top_logprobs_evaluation(
        examples,
        output_dir,
        model=model,
        seed=seed,
        concurrency=concurrency,
        top_logprobs=top_logprobs,
        benchmark_rows=len(all_examples),
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("summarize-run")
def summarize_run(
    output_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Recompute metrics from saved raw responses without running inference."""

    summary = refresh_summary(output_dir)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("stage-result")
def stage_result(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    results_repo_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    model_id: Annotated[str, typer.Option()],
    model_revision: Annotated[str, typer.Option()],
    dataset_revision: Annotated[str, typer.Option()],
    model_type: Annotated[ModelType, typer.Option()],
    artifact_uri: Annotated[str | None, typer.Option()] = None,
    adapter: Annotated[str, typer.Option()] = "decision-bench",
    probability_source: Annotated[str, typer.Option()] = "complete_candidate_distribution",
    model_url: Annotated[str | None, typer.Option()] = None,
    open_weights: Annotated[bool | None, typer.Option()] = None,
    parameter_count: Annotated[int | None, typer.Option(min=0)] = None,
    benchmark_name: Annotated[str, typer.Option()] = "DecisionBench",
    benchmark_version: Annotated[str, typer.Option()] = "1.0",
    task_spec_sha256: Annotated[str | None, typer.Option()] = None,
    create_pr: Annotated[bool, typer.Option()] = False,
) -> None:
    """Stage one validated result record and optionally open a PR."""

    cache = ResultCache(results_repo_dir)
    result_path = cache.stage_result(
        run_dir,
        model=ModelMetadata(
            name=model_id,
            revision=model_revision,
            model_type=model_type,
            url=model_url,
            adapter=adapter,
            probability_source=probability_source,
            open_weights=open_weights,
            parameter_count=parameter_count,
        ),
        artifact_uri=artifact_uri,
        dataset_revision=dataset_revision,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        task_spec_sha256=task_spec_sha256,
    )
    typer.echo(json.dumps(cache.submit_result(result_path, create_pr=create_pr), indent=2))


@app.command("list-results")
def list_results(
    results_repo_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    view: Annotated[str, typer.Option()] = "overall",
) -> None:
    """Print reviewed results from a local results-repository checkout."""

    typer.echo(json.dumps(ResultCache(results_repo_dir).to_records(view=view), indent=2))


@app.command("leaderboard")
def leaderboard(
    results_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "../decision-bench-results"
    ),
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 7860,
    share: Annotated[bool, typer.Option()] = False,
) -> None:
    """Launch the interactive leaderboard from reviewed result records."""

    from decision_bench.leaderboard import launch

    launch(results_dir, host=host, port=port, share=share)


@app.command("run-hf")
def run_hf(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 0,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    max_prompt_characters_per_batch: Annotated[int, typer.Option(min=1)] = 1_048_576,
    max_length: Annotated[int, typer.Option(min=1)] = 22_528,
    attn_implementation: Annotated[str, typer.Option()] = "sdpa",
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a local Hugging Face decision-token checkpoint."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = list(benchmark.examples)
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_hf_evaluation(
        examples,
        output_dir,
        model_dir=model_dir,
        seed=seed,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        max_length=max_length,
        attn_implementation=attn_implementation,
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-nimble-hf")
def run_nimble_hf(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    adapter_repo: Annotated[str, typer.Option()] = "bespokelabs/Bespoke-Nimble-9B",
    adapter_revision: Annotated[str, typer.Option()] = (
        "594dfdcfb6f94e3d0c0db7535180d3c71689169a"
    ),
    expected_adapter_sha256: Annotated[str, typer.Option()] = (
        "ba7e28acb97f973e80fa51f3aa6fc6f75ea4081b89632ed45d8e5f3a1d7bfa6b"
    ),
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    max_prompt_characters_per_batch: Annotated[int, typer.Option(min=1)] = 524_288,
    attn_implementation: Annotated[str, typer.Option()] = "sdpa",
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run Bespoke-Nimble-9B through its pinned native candidate-logit contract."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = list(benchmark.examples)
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_nimble_hf_evaluation(
        examples,
        output_dir,
        model_dir=model_dir,
        adapter_repo=adapter_repo,
        adapter_revision=adapter_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        attn_implementation=attn_implementation,
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-public-hf")
def run_public_hf(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    model_type: Annotated[
        Literal["nanojev", "openjev", "system-one"], typer.Option()
    ],
    model_repo: Annotated[str, typer.Option()],
    model_revision: Annotated[str, typer.Option()],
    project_root: Annotated[Path | None, typer.Option()] = None,
    base_revision: Annotated[str | None, typer.Option()] = None,
    expected_weights_sha256: Annotated[str | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    max_prompt_characters_per_batch: Annotated[int, typer.Option(min=1)] = 262_144,
    attn_implementation: Annotated[str, typer.Option()] = "sdpa",
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run a public Jev-shaped HF model through its published native contract."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = list(benchmark.examples)
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_public_hf_evaluation(
        examples,
        output_dir,
        model_dir=model_dir,
        model_type=model_type,
        model_repo=model_repo,
        model_revision=model_revision,
        base_revision=base_revision,
        expected_weights_sha256=expected_weights_sha256,
        batch_size=batch_size,
        max_prompt_characters_per_batch=max_prompt_characters_per_batch,
        attn_implementation=attn_implementation,
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("run-jev-openrouter")
def run_jev_openrouter(
    spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    project_root: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "typesafe/jev-1.13",
    concurrency: Annotated[int, typer.Option(min=1, max=256)] = 32,
    max_state_question_tokens: Annotated[int, typer.Option(min=1)] = 32_000,
    input_token_reserve: Annotated[int, typer.Option(min=0)] = 2_048,
    tokenizer_model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    tokenizer_revision: Annotated[str, typer.Option()] = (
        "c1899de289a04d12100db370d81485cdf75e47ca"
    ),
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run Jev through OpenRouter's native Decisions endpoint."""

    resolved_root = project_root if project_root is not None else Path.cwd()
    benchmark = get_benchmark(spec_path, project_root=resolved_root)
    examples = list(benchmark.examples)
    if smoke:
        examples = select_smoke_examples(examples)
    summary = run_jev_openrouter_evaluation(
        examples,
        output_dir,
        model=model,
        concurrency=concurrency,
        max_state_question_tokens=max_state_question_tokens,
        input_token_reserve=input_token_reserve,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        benchmark_metadata=_benchmark_metadata(benchmark),
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("serve-gguf")
def serve_gguf(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8080,
    n_ctx: Annotated[int, typer.Option(min=1)] = 8192,
    n_gpu_layers: Annotated[int, typer.Option()] = -1,
    n_threads: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Serve any chat-capable GGUF through /v1/systemone."""

    import uvicorn

    from decision_bench.gguf_server import GgufDecisionEngine, create_app

    engine = GgufDecisionEngine(
        model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        seed=seed,
    )
    uvicorn.run(create_app(engine), host=host, port=port)


def _benchmark_metadata(benchmark: Benchmark) -> dict[str, object]:
    metadata: dict[str, object] = {
        "benchmark_name": benchmark.spec.name,
        "benchmark_version": benchmark.spec.version,
        "benchmark_tasks": [task.metadata.name for task in benchmark.tasks],
        "datasets": [dataset.model_dump(mode="json") for dataset in benchmark.datasets],
    }
    if benchmark.spec_path is not None:
        metadata["task_spec_sha256"] = _sha256_file(benchmark.spec_path)

    local_datasets = [dataset for dataset in benchmark.datasets if dataset.backend == "local"]
    if len(local_datasets) == 1:
        dataset_path = resolve_local_path(
            local_datasets[0].path,
            project_root=benchmark.project_root,
        )
        metadata["dataset_sha256"] = _sha256_file(dataset_path)
        manifest_path = dataset_path.parent / "manifest.json"
        if manifest_path.is_file():
            metadata["dataset_manifest_sha256"] = _sha256_file(manifest_path)
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    app()
