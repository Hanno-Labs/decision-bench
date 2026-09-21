# Run the Leaderboard

Launch the same Gradio application used by the hosted Space:

```bash
decision-bench leaderboard --results-dir ../decision-bench-results
```

The leaderboard reads reviewed result records. It can filter by benchmark release and by the views
materialized in each run: overall, task, family, domain, primitive, candidate count, and reasoning
track. Coverage and errors stay visible beside accuracy.

The hosted instance is
[`Hanno-Labs/decision-bench-leaderboard`](https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard).
