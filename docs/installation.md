# Installation

Install directly from the private GitHub repository with pip:

```bash
pip install git+https://github.com/Hanno-Labs/decision-bench.git
```

Or add it to a uv project:

```bash
uv add git+https://github.com/Hanno-Labs/decision-bench.git
```

Native Hugging Face checkpoints require the `hf` extra; GGUF serving requires `gguf`; the local
leaderboard requires `leaderboard`.

```bash
uv add "decision-bench[hf,leaderboard] @ git+https://github.com/Hanno-Labs/decision-bench.git"
```

Authenticate with Hugging Face before loading the private dataset:

```bash
hf auth login
```
