# Contributing

DecisionBench spans several repositories and Hugging Face resources. Start in the
repository that owns the change:

Report bugs and request features for any part of the ecosystem in the
[DecisionBench issue tracker](https://github.com/Hanno-Labs/decision-bench/issues).
Open code pull requests in the repository that owns the change.

| Part | What belongs there | Local path after setup |
| --- | --- | --- |
| [DecisionBench](https://github.com/Hanno-Labs/decision-bench) | Benchmark runtime, model adapters, evaluation, scoring, task specs, and documentation | This checkout |
| [Results](https://github.com/Hanno-Labs/decision-bench-results) | Reviewed, append-only model result records | `repos/decision-bench-results/` |
| [Leaderboard](https://github.com/Hanno-Labs/decision-bench-leaderboard) | Public leaderboard frontend and result explorer | `repos/decision-bench-leaderboard/` |
| [Jev compatible server](https://github.com/Hanno-Labs/jev-compatible-server) | Jev-compatible API server | `repos/jev-compatible-server/` |
| [Dataset](https://huggingface.co/datasets/Hanno-Labs/decision-bench) | Frozen benchmark rows consumed by the evaluation | Hugging Face; not cloned by setup |
| [Leaderboard Space](https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard) | Hosted leaderboard | The leaderboard checkout has the Space as its `origin` remote |

The private [data-generation repository](https://github.com/Hanno-Labs/decision-bench-data-gen)
contains dataset generation, import, validation, and publishing jobs. It is not
part of the default local setup. Contributors with access can add it through
`mise.local.toml` as described below.

## Local development setup

Clone the main DecisionBench repository, enter it, and run:

```bash
mise trust
mise run setup
```

`setup` runs the `clone` task in `mise.toml`. It clones the three default
companion GitHub repositories into `repos/` and skips checkouts that are already
there. You can rerun `mise run clone` when a companion checkout is missing.

To add private or other machine-specific repositories, create an ignored
`mise.local.toml` with one Git URL per line in `DECISION_BENCH_EXTRA_REPOS`:

```toml
[env]
DECISION_BENCH_EXTRA_REPOS = "git@github.com:Hanno-Labs/decision-bench-data-gen.git"
```

For more than one extra repository, use a multiline TOML string with one URL per
line. Then run `mise trust mise.local.toml` and `mise run setup`. The default
setup does not require access to the private data-generation repository.

The `repos/` directory is ignored by the main repository, so each companion
keeps its own Git history, branch, and working tree. Run Git commands from the
checkout you intend to change.

The dataset is loaded from its pinned Hugging Face revision by the benchmark;
the setup task does not copy dataset files. This local leaderboard checkout has
both GitHub and Hugging Face Space remotes; a fresh `mise run setup` clone starts
with the GitHub remote only. Check the target remote before pushing.

## Contributing to the benchmark

Use the guides for [adding a model](docs/contributing/adding_a_model.md),
[adding a task](docs/contributing/adding_a_task.md),
[adding a benchmark](docs/contributing/adding_a_benchmark.md), and
[submitting results](docs/contributing/submitting_results.md).

Run formatting, linting, typing, and tests in the main benchmark checkout before
submitting a change:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

For changes in a companion repository, follow that repository's own README and
checks. The leaderboard README covers its Node.js development and verification
commands; the data-generation README covers generation and validation.
