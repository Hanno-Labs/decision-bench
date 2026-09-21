from pathlib import Path

from decision_bench.benchmark import Benchmark
from decision_bench.schemas import Candidate, DecisionExample, Primitive
from decision_bench.task_spec import TaskSpec


def test_benchmark_views_keep_row_identity() -> None:
    spec = TaskSpec.model_validate(
        {
            "name": "test",
            "version": "1",
            "description": "test benchmark",
            "license": "test",
            "languages": ["eng-Latn"],
            "primitives": ["candidate_selection"],
            "families": ["legal", "routing"],
            "domains": ["legal"],
            "dataset": {"backend": "local", "path": "unused.parquet"},
        }
    )
    example = DecisionExample(
        row_id="row-1",
        task_name="contracts",
        primitive=Primitive.CANDIDATE_SELECTION,
        family="legal",
        domain="legal",
        instruction="Select the clause.",
        state={"text": "Example"},
        candidates=[Candidate(id="a", label="A"), Candidate(id="b", label="B")],
        gold_candidate_id="a",
        gold_probabilities=[1.0, 0.0],
    )
    benchmark = Benchmark(Path("spec.toml"), spec, (example,))

    assert benchmark.select(family="legal").examples[0].row_id == "row-1"
    assert benchmark.select(family="routing").examples == ()
