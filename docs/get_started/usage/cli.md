# Command Line

```bash
decision-bench --help
decision-bench inspect --help
decision-bench run-hf --help
decision-bench run-public-hf --help
decision-bench run-openrouter --help
decision-bench summarize-run --help
decision-bench stage-result --help
decision-bench list-results --help
decision-bench leaderboard --help
```

`run-hf` evaluates DecisionBench-native Hugging Face checkpoints. Architecture-specific public Hub
models use their corresponding adapter, including the adapters exposed by `run-public-hf`.

`stage-result` validates a completed run, creates the canonical result record under a local
`decision-bench-results` checkout, and optionally opens a GitHub pull request.
