# Add a Benchmark

A benchmark release is an immutable task specification plus a pinned dataset revision. Adding a
release requires:

1. validating every row against the public schema;
2. freezing row IDs, source provenance, and dataset hashes;
3. publishing a task specification with task, family, domain, primitive, and reasoning metadata;
4. evaluating reference models through their declared native contracts; and
5. adding reviewed records to the results repository and menuing the release in the leaderboard.
