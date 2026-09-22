<h1 align="center">DecisionBench</h1>

---

<h3 align="center">The evaluation ecosystem for decision models</h3>

<p align="center">
  <a href="https://github.com/Hanno-Labs/decision-bench/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Hanno-Labs/decision-bench.svg?color=green"></a>
  <a href="https://github.com/Hanno-Labs/decision-bench/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Hanno-Labs/decision-bench/actions/workflows/ci.yml/badge.svg"></a>
</p>

<h4 align="center">
  <a href="https://ubiquitous-bassoon-zzmjggp.pages.github.io/installation/">Installation</a> ·
  <a href="https://ubiquitous-bassoon-zzmjggp.pages.github.io/">Documentation</a> ·
  <a href="https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard">Leaderboard</a> ·
  <a href="https://github.com/Hanno-Labs/decision-bench-results">Results</a> ·
  <a href="#citing">Citing</a>
</h4>

## Installation

```bash
pip install git+https://github.com/Hanno-Labs/decision-bench.git
```

```bash
uv add git+https://github.com/Hanno-Labs/decision-bench.git
```

## Example Usage

Load the pinned benchmark and run a model through the Python API. See the
[evaluation guide](https://ubiquitous-bassoon-zzmjggp.pages.github.io/get_started/usage/running_the_evaluation/)
for every supported model surface.

```python
from pathlib import Path

from decision_bench import get_benchmark
from decision_bench.evaluate import run_openrouter_evaluation

benchmark = get_benchmark("DecisionBench")
summary = run_openrouter_evaluation(
    list(benchmark.examples), Path("results/luna"),
    model="openai/gpt-5.6-luna", reasoning_effort="minimal",
    seed=0, concurrency=32,
)
```

You can also use the CLI:

```bash
decision-bench run-openrouter task_specs/decisionbench-dev.toml results/luna \
  --model openai/gpt-5.6-luna --reasoning-effort minimal
```

## Overview

| | |
|---|---|
| 📈 [Leaderboard] | Compare reviewed results and filter by task, family, domain, or primitive |
| 🏃 [Get Started] | Install DecisionBench and run the frozen suite |
| 📋 [Tasks and Views] | Understand the 23,900 rows, nine families, three primitives, and reasoning track |
| 🤖 [Models] | Add or use model adapters with explicit probability and coverage contracts |
| 📊 [Results] | Load, inspect, and submit reproducible results |
| 🧪 [Evaluation] | Learn the metrics, artifacts, and comparability rules |
| 🤝 [Contributing] | Add models, tasks, benchmarks, and result records |

[Leaderboard]: https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard
[Get Started]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/
[Tasks and Views]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/overview/tasks/
[Models]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/overview/models/
[Results]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/get_started/usage/loading_results/
[Evaluation]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/get_started/usage/running_the_evaluation/
[Contributing]: https://ubiquitous-bassoon-zzmjggp.pages.github.io/contributing/

## Contribute

Choose the path that matches what you want to bring to DecisionBench:

<table>
  <tr>
    <td width="50%">
      <h3 align="center"><a href="docs/contributing/adding_a_model.md">🤖 Add a Model →</a></h3>
      <p>Add a compatibility adapter so DecisionBench can evaluate a new decision model.</p>
    </td>
    <td width="50%">
      <h3 align="center"><a href="docs/contributing/submitting_results.md">📊 Submit Results →</a></h3>
      <p>Run a supported model and submit its reviewed, reproducible scores.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3 align="center"><a href="docs/contributing/adding_a_task.md">🧩 Add a Task →</a></h3>
      <p>Contribute one dataset-backed decision problem with labels, provenance, and tests.</p>
    </td>
    <td width="50%">
      <h3 align="center"><a href="docs/contributing/adding_a_benchmark.md">🗂️ Add a Benchmark →</a></h3>
      <p>Curate existing tasks into a named evaluation for a domain or purpose.</p>
    </td>
  </tr>
</table>

## Citing

DecisionBench is under active development. Until the benchmark paper is published, cite the
repository and the individual datasets listed in the
[task catalog](https://ubiquitous-bassoon-zzmjggp.pages.github.io/overview/tasks/). Machine-readable
citation metadata lives in [`CITATION.cff`](CITATION.cff).
