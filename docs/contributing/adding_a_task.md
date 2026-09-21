# Add a Task

Task discovery, transformation, generation, import, and artifact validation live in
[`Hanno-Labs/decision-bench-data-gen`](https://github.com/Hanno-Labs/decision-bench-data-gen).

Every task must define a primitive, family, domain, provenance, commercial-compatible license,
candidate semantics, gold candidate, and gold probability distribution. Generation and evaluation
splits must be frozen before the task enters a benchmark release. Existing releases are never
silently rewritten.
