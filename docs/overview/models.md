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

## Supported adapters and models

Check this table before adding an adapter. It lists the model-native contracts the runner already
knows how to evaluate. A listed model should use its existing runner; an unlisted native contract is
the reason to [add a model adapter](../contributing/adding_a_model.md).

| Runner | Supported model or surface | Native readout contract |
|---|---|---|
| `run-hf` | [`Hanno-Labs/bosun-v3.1-0.6b`](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b), [`Hanno-Labs/bosun-v3.1-1.7b`](https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b) | Masked softmax over valid decision tokens |
| `run-nimble-hf` | [`bespokelabs/Bespoke-Nimble-9B`](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) | Published candidate-token logits; up to 26 choices |
| `run-public-hf --model-type nanojev` | [`C-Tianyu/NanoJev`](https://huggingface.co/C-Tianyu/NanoJev) | Published parallel candidate-path head |
| `run-public-hf --model-type openjev` | [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) | Published grouped-span head |
| `run-public-hf --model-type system-one` | [`pngwn/system-one-qwen3.5-4b-scorer`](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer) | Published candidate scorer |
| `run-jev-openrouter` | [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13) | OpenRouter Decisions API distributions |
| `run-openrouter` | An OpenRouter chat model that can produce the required JSON-schema probability vector | Structured probability vector |
| `run-openrouter-top-logprobs` | An OpenRouter chat model that returns every required candidate in top-logprobs | Conditional next-token probabilities |

The first five rows run pinned local Hugging Face weights. The OpenRouter rows are hosted service
surfaces: record the provider model name, request settings, and dated service snapshot when you
submit results. `decision-bench --help` documents all runners; use the runner's `--help` for its
required paths and model-specific options.
