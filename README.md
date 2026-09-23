<h1 align="center">DecisionBench</h1>

---

<h3 align="center">The evaluation ecosystem for decision models</h3>

<p align="center">
  <a href="https://github.com/Hanno-Labs/decision-bench/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Hanno-Labs/decision-bench.svg?color=green"></a>
  <a href="https://github.com/Hanno-Labs/decision-bench/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Hanno-Labs/decision-bench/actions/workflows/ci.yml/badge.svg"></a>
</p>

<h4 align="center">
  <a href="https://hanno-labs.github.io/decision-bench/installation/">Installation</a> ·
  <a href="https://hanno-labs.github.io/decision-bench/">Documentation</a> ·
  <a href="https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard">Leaderboard</a> ·
  <a href="https://github.com/Hanno-Labs/decision-bench-results">Results</a> ·
  <a href="https://github.com/Hanno-Labs/decision-bench/issues">Issues</a> ·
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

Smoke-test a supported decision model on the pinned benchmark. This example uses
[Bosun v3.1 0.6B](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b), whose
native decision-token readout is supported directly by `run-hf`.

```bash
hf download Hanno-Labs/bosun-v3.1-0.6b \
  --revision aaa9dd06d4d6501b33df61942472fed9284bc5e6 \
  --local-dir models/bosun-v3.1-0.6b

decision-bench run-hf task_specs/decisionbench-dev.toml \
  models/bosun-v3.1-0.6b results/bosun-v3.1-0.6b --smoke
```

Before running anything else, see the complete
[supported adapters and models](https://hanno-labs.github.io/decision-bench/overview/models/).
If your model is listed, use its runner; only add an adapter when its native
decision readout is not already supported.

## Overview

| | |
|---|---|
| 📈 [Leaderboard] | Compare reviewed results and filter by task, family, domain, or primitive |
| 🏃 [Get Started] | Install DecisionBench and run the frozen suite |
| 📋 [Tasks and Views] | Understand the 23,900 rows, nine families, three primitives, and reasoning track |
| 🤖 [Models] | See supported adapters and models, or add a new native readout contract |
| 📊 [Results] | Load, inspect, and submit reproducible results |
| 🧪 [Evaluation] | Learn the metrics, artifacts, and comparability rules |
| 🤝 [Contributing] | Add models, tasks, benchmarks, and result records |

[Leaderboard]: https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard
[Get Started]: https://hanno-labs.github.io/decision-bench/
[Tasks and Views]: https://hanno-labs.github.io/decision-bench/overview/tasks/
[Models]: https://hanno-labs.github.io/decision-bench/overview/models/
[Results]: https://hanno-labs.github.io/decision-bench/get_started/usage/loading_results/
[Evaluation]: https://hanno-labs.github.io/decision-bench/get_started/usage/running_the_evaluation/
[Contributing]: https://hanno-labs.github.io/decision-bench/contributing/

## Contribute

Report bugs and request features for any DecisionBench component in the
[central issue tracker](https://github.com/Hanno-Labs/decision-bench/issues).
Send code changes to the repository that owns that component.

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
[task catalog](https://hanno-labs.github.io/decision-bench/overview/tasks/). Machine-readable
citation metadata lives in [`CITATION.cff`](CITATION.cff).
