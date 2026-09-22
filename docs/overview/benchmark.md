# Benchmark

DecisionBench 1.0 is a named collection of 43 registered tasks containing 23,900 English examples
over nine families and three primitives. Candidate sets range from 2 to 255. The release combines
applied decision tasks with a reasoning track while keeping task identity visible for filtering and
separate aggregates.

Those first-release tasks share the consolidated dataset
[`Hanno-Labs/decision-bench`](https://huggingface.co/datasets/Hanno-Labs/decision-bench) at revision
`b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443`. New tasks may own independent pinned datasets; a
benchmark composes task IDs rather than requiring one canonical data repository.
