.PHONY: build-docs serve-docs test lint typecheck run-leaderboard

build-docs:
	uv run --no-sync --group docs zensical build --clean --strict

serve-docs:
	uv run --no-sync --group docs zensical serve

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy src

run-leaderboard:
	uv run --no-sync --extra leaderboard decision-bench leaderboard
