# Load Results

`ResultCache` reads reviewed result records from a local checkout of
[`Hanno-Labs/decision-bench-results`](https://github.com/Hanno-Labs/decision-bench-results).

```python
from decision_bench import ResultCache

cache = ResultCache("../decision-bench-results")
results = cache.load_results()
leaderboard = cache.to_records(view="overall")
```

Sync the official result repository into the cache:

```python
cache = ResultCache()
cache.sync()
results = cache.load_results()
```

Result records contain reviewable aggregates and immutable pointers to complete row-level artifacts;
the multi-hundred-megabyte raw responses remain in durable object storage rather than Git.
