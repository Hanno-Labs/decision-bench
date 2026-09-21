# Models

DecisionBench compares model behavior through explicit readout contracts:

| Surface | Probability source |
|---|---|
| Native decision-token model | Masked softmax over valid decision tokens |
| Structured chat model | Schema-constrained generated probability vector |
| Top-logprobs chat model | Conditional next-token probabilities when all candidates are returned |
| Jev-compatible API | Provider Noul, Choice, or Score distribution |
| Public candidate scorer | Model-native logits mapped to benchmark candidates |

The leaderboard never treats these as interchangeable metadata. Each result records its adapter,
prompt, probability source, coverage, eligibility, and truncation contract.
