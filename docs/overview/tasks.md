# Tasks

A DecisionBench task is an independently versioned evaluation unit. It owns a pinned Hugging Face
dataset reference, license, languages, primitive, family, domain, and an optional deterministic
transform into `DecisionExample`. Benchmarks are named collections of task IDs; they do not own the
datasets.

The nine use-case families are:

1. document and record classification;
2. routing and triage;
3. rubric scoring and prioritization;
4. retrieval and verification;
5. bounded extraction;
6. entity alignment;
7. guardrails and moderation;
8. function, agent, and skill routing; and
9. document workflows.

The current 43 tasks share the consolidated DecisionBench 1.0 dataset because that is how the first
release was produced. Future tasks can reference their own public HF datasets at pinned commits.
`family`, `domain`, `primitive`, candidate count, and reasoning metadata support narrower views. The
20,000 applied rows, 2,700 task expansion rows, and 1,200 reasoning rows are one release—not competing
“core” and “expanded” targets.
