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
| `run-public-hf --model-type cua-s1` | [`cua-ai/cua-s1-4b-0.2`](https://huggingface.co/cua-ai/cua-s1-4b-0.2) text adapter | Published final-position option-letter logits; up to 26 candidates |
| `run-public-hf --model-type gliner25` | [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | Exclusive classification logits over candidate labels, softmaxed as conditional option preference; rows over 512 encoded tokens or with `(` in a candidate label are unsupported |
| `run-public-hf --model-type lev` | [`interfaze-ai/lev`](https://huggingface.co/interfaze-ai/lev) | Released System One readout: calibrated softmax over the supplied options from a single forward pass (letter-token readout, candidate-path head above the label-token cap); score rows are limited to lev's 2–10 levels |
| `run-public-hf --model-type mojev` | [`MoLeMo-Lab/mojev`](https://huggingface.co/MoLeMo-Lab/mojev) | Published packed candidate logits; up to 255 candidates |
| `run-public-hf --model-type nanojev` | [`C-Tianyu/NanoJev`](https://huggingface.co/C-Tianyu/NanoJev) | Published parallel candidate-path head |
| `run-public-hf --model-type openjev` | [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) | Published grouped-span head |
| `run-public-hf --model-type system-one` | [`pngwn/system-one-qwen3.5-4b-scorer`](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer) | Published candidate scorer |
| `run-public-hf --model-type tev1` | [`togethercomputer/Tev1-4B-experimental`](https://huggingface.co/togethercomputer/Tev1-4B-experimental) | Softmax over A–X next-token logits; up to 24 choices |
| `run-jev-openrouter` | [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13) | OpenRouter Decisions API distributions |
| `run-system-one-http` | [`juspay/xor`](https://huggingface.co/juspay/xor) | Released Jev-compatible SystemOne API; forward/reverse option-letter logprobs with published calibration; up to 26 candidates |
| `run-openrouter` | An OpenRouter chat model that can produce the required JSON-schema probability vector | Structured probability vector |
| `run-openrouter-top-logprobs` | An OpenRouter chat model that returns every required candidate in top-logprobs | Conditional next-token probabilities |

The local Hugging Face rows run pinned weights. MoJev's adapter pins model revision
`0c8695b6252f4205907433d4e196a94f032e60c3` and packing code revision
`a74d58cd19ec573e83e8e27f9fecd837b8d830fb`. The OpenRouter rows are hosted service
surfaces: record the provider model name, request settings, and dated service snapshot when you
submit results. `decision-bench --help` documents all runners; use the runner's `--help` for its
required paths and model-specific options.

The lev runner pins the released adapter revision
`f8ef71157ec06a7d3b6435bc0756f9d735c33748` and the inference code revision
`cf104b69329302e4eac674a730c71f3511047db8` (`lev` 0.1.1 from `Abhinavexists/lev`). The release
names `Qwen/Qwen3.5-4B` as its base model and the adapter layer loads that base without an
adapter-level revision pin. Recorded release hashes are the LoRA weights
`64c718974d7ed0c9b22a0304060669fa3289ff89b57e572d08b9e9a0dc0623c1`, the candidate-path head
`27eedf7bb9e20d66432882b91297ef6e1ce3d23d81e94ed105ffadba77c6d3ad`, and the tokenizer
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`. The runner hands each row to
the release's own System One inference path (bundled calibration, forward/reverse order averaging)
and keeps the returned distribution over exactly the row's candidates; rows with more than 255
candidates, choice rows with duplicate labels, binary rows without unambiguous true/false semantics,
and score rows outside lev's 2–10 published levels stay explicit unsupported rows, and no row is
truncated. The runner accepts `--expected-weights-sha256` to verify the LoRA weights against the
pinned release hash before scoring.

The registered Cua-S1 release is adapter revision
`16818868b0cc7813808aae4e87b417657046ab79` on base revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; its text adapter SHA-256 is
`9b59c5aed96171a50b26526613766bbf44347a5c7af70f81efe6bcc6e9dbfb0e`. DecisionBench's frozen
rows contain textual state rather than screenshots, so this runner uses the release's text adapter.

The XOR runner pins model revision `679decd4c669e5c37f4ac29dbd9957997424c876` and serving
bundle SHA-256 `0a63473caaa3c6bfc8bc15fbab62f0a9a84c7ebf4ab6e06d0699891b7be6159b`.
The released bundle pins SGLang image digest
`sha256:6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921`. The
runner sends the benchmark's Noul, Choice, and Score requests to `/v1/systemone`. Rows with more
than 26 candidates, rendered state over 4 MiB, or serialized requests over 8 MiB remain explicit
unsupported rows; the adapter never truncates them.
