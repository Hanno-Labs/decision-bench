"""Interactive leaderboard built from reviewed DecisionBench result records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from decision_bench.results import RESULT_TAG_HELP, ResultCache


def leaderboard_rows(results_dir: str | Path, *, view: str = "overall") -> list[dict[str, Any]]:
    """Load one flat table view from a reviewed results checkout."""

    rows = ResultCache(results_dir).to_records(view=view)
    return sorted(
        rows,
        key=lambda row: (
            row["primary_accuracy"] is not None,
            row["primary_accuracy"] or -1.0,
        ),
        reverse=True,
    )


def create_app(results_dir: str | Path) -> Any:
    """Create the Gradio leaderboard application."""

    import gradio as gr
    import pandas as pd

    cache = ResultCache(results_dir)
    results = cache.load_results()
    views = sorted({name for result in results for name in result.views}) or ["overall"]

    def table(view: str) -> pd.DataFrame:
        frame = pd.DataFrame(leaderboard_rows(results_dir, view=view))
        if frame.empty:
            return frame
        for column in ("primary_accuracy", "supported_accuracy", "coverage"):
            if column in frame:
                frame[column] = frame[column].map(
                    lambda value: round(float(value) * 100, 2) if value is not None else None
                )
        return frame

    with gr.Blocks(title="DecisionBench Leaderboard") as app:
        gr.Markdown(
            "# DecisionBench Leaderboard\n"
            "Reviewed results over immutable benchmark releases. Errors and unsupported rows count "
            "as misses in the primary score."
        )
        view = gr.Dropdown(views, value="overall", label="Task, family, domain, or primitive view")
        gr.HTML(
            '<div>Tags <span tabindex="0" style="cursor:help; border-bottom:1px dotted" '
            f'aria-label="compact: {RESULT_TAG_HELP["compact"]}" '
            f'title="compact: {RESULT_TAG_HELP["compact"]}">?</span></div>'
        )
        leaderboard = gr.DataFrame(value=table("overall"), interactive=False)
        view.change(table, inputs=view, outputs=leaderboard)
        gr.Markdown(
            "Result records: [Hanno-Labs/decision-bench-results]"
            "(https://github.com/Hanno-Labs/decision-bench-results)"
        )
    return app


def launch(results_dir: str | Path, *, host: str, port: int, share: bool) -> None:
    """Launch the local leaderboard server."""

    create_app(results_dir).launch(server_name=host, server_port=port, share=share)
