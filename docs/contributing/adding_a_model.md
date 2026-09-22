# Add a Model

The normal model contribution starts with a model published on the Hugging Face Hub. Pin both the
model and tokenizer to immutable commit revisions, then add the smallest adapter needed to expose
the model's native decision distribution.

Adapters live in `src/decision_bench/models/`. A model adapter must:

1. preserve candidate IDs and their displayed order;
2. return a complete probability distribution, or an explicit unsupported/error result;
3. save the exact model-facing input and raw response for every row;
4. declare the model and tokenizer revisions, adapter name, probability source, input limits,
   truncation policy, and eligibility boundary; and
5. reject rows it cannot score without silently changing the benchmark question.

Use `run-hf` when the repository publishes DecisionBench's native checkpoint contract. For another
architecture, add its implementation under `src/decision_bench/models/`, expose a CLI command in
`src/decision_bench/cli.py`, and call the shared evaluation machinery in
`src/decision_bench/evaluate.py`. Existing `run-public-hf` adapters are concrete examples of pinned
Hub models with architecture-specific readout contracts.

Add adapter tests covering all three primitives, nontrivial candidate counts, malformed output, and
the model's supported/unsupported boundary. Run a smoke evaluation and inspect its saved raw rows
before running the complete benchmark. Hosted APIs are a secondary path and must record the provider
route and request settings.

When the adapter and its tests are ready, open a pull request against
[`Hanno-Labs/decision-bench`](https://github.com/Hanno-Labs/decision-bench). Model code and result
records are reviewed separately; after the adapter merges, run the pinned model and
[submit its result](submitting_results.md).
