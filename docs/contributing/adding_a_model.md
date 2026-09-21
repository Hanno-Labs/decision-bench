# Add a Model

Adapters live in `src/decision_bench/models/`. A model adapter must preserve candidate IDs and order,
return a complete probability distribution or an explicit unsupported/error result, save exact
model-facing input and raw response, and declare probability provenance, input limits, truncation,
and eligibility.

Add adapter tests covering all three primitives, nontrivial candidate counts, malformed output, and
the model's supported/unsupported boundary. Public models must pin immutable model and tokenizer
revisions. Hosted APIs must record the provider route and request settings.
