# Installation

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/Hanno-Labs/decision-bench.git
```

Or add it to a uv project:

```bash
uv add git+https://github.com/Hanno-Labs/decision-bench.git
```

Hugging Face checkpoints require the `hf` extra; GGUF serving requires `gguf`; the local
leaderboard requires `leaderboard`.

```bash
uv add "decision-bench[hf,leaderboard] @ git+https://github.com/Hanno-Labs/decision-bench.git"
```

The benchmark dataset is public. Authenticate only when the model you are evaluating is gated or
private:

```bash
hf auth login
```
