# Test fixtures

## `arbit_hf_official_rows.jsonl`

Fifty rows from the reference run of `arbit-hf-remote-code-v1` (`litillabs/arbit-1.7b` at
`d8ca47d20f5f406317994c4df096b7bedfefb5de`) on the DecisionBench 1.0 eval split
(`Hanno-Labs/decision-bench` at `071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb`), run with this
repository's evaluator at `89aaa85`. They come from that run's `raw.jsonl` (SHA-256
`292fefe858a991ff4ed6f594b811b8c67f13a00e56c56ee8720f8dd61d549143`).

Selection: the longest row by model input tokens, then the first eight table-form rows, then one
row per task in turn, in the order of `sha256("arbit-hf-adapter-identity-v1:<row_id>")`. The rows
cover 41 tasks, all three primitives, 2 to 255 candidates and inputs of up to 26,211 tokens.

Each line keeps only what `tests/test_arbit_hf_model.py` reads:

- `example`: the benchmark row, without `source`;
- `request_sha256`: SHA-256 of the recorded model request (`state` and `questions`) as canonical
  JSON (sorted keys, compact separators, unicode kept);
- `response` and `input_contract`: as recorded;
- `selected_candidate_id`: the scored selection.
