"""Hugging Face Space entrypoint for the DecisionBench leaderboard."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

DATA_PATH = Path(__file__).with_name("leaderboard.parquet")
DISPLAY_COLUMNS = [
    "model",
    "primary_accuracy",
    "supported_accuracy",
    "coverage",
    "successful_rows",
    "unsupported_rows",
    "error_rows",
    "adapter",
    "probability_source",
    "revision",
]


def load_rows() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        return pd.DataFrame(columns=["view_kind", "view_name", *DISPLAY_COLUMNS])
    return pd.read_parquet(DATA_PATH)


ROWS = load_rows()
VIEW_KINDS = sorted(ROWS["view_kind"].dropna().unique().tolist()) if not ROWS.empty else []


def view_names(view_kind: str) -> gr.Dropdown:
    names = sorted(
        ROWS.loc[ROWS["view_kind"] == view_kind, "view_name"].dropna().unique().tolist()
    )
    return gr.Dropdown(choices=names, value=names[0] if names else None)


def table(view_kind: str, view_name: str) -> pd.DataFrame:
    selected = ROWS.loc[
        (ROWS["view_kind"] == view_kind) & (ROWS["view_name"] == view_name)
    ].copy()
    for column in ("primary_accuracy", "supported_accuracy", "coverage"):
        if column in selected:
            selected[column] = selected[column].map(
                lambda value: round(float(value) * 100, 2) if pd.notna(value) else None
            )
    columns = [column for column in DISPLAY_COLUMNS if column in selected]
    sort_column = (
        "primary_accuracy"
        if selected["primary_accuracy"].notna().any()
        else "supported_accuracy"
    )
    return selected[columns].sort_values(sort_column, ascending=False, na_position="last")


initial_kind = "benchmark" if "benchmark" in VIEW_KINDS else (VIEW_KINDS[0] if VIEW_KINDS else "")
initial_names = (
    sorted(
        ROWS.loc[ROWS["view_kind"] == initial_kind, "view_name"].dropna().unique().tolist()
    )
    if initial_kind
    else []
)
initial_name = initial_names[0] if initial_names else ""

with gr.Blocks(title="DecisionBench Leaderboard") as demo:
    gr.Markdown(
        "# DecisionBench Leaderboard\n"
        "Reviewed results on immutable DecisionBench releases. Unsupported and error rows count "
        "as misses in the benchmark-wide primary score."
    )
    with gr.Row():
        kind = gr.Dropdown(VIEW_KINDS, value=initial_kind, label="View")
        name = gr.Dropdown(initial_names, value=initial_name, label="Task or slice")
    leaderboard = gr.DataFrame(value=table(initial_kind, initial_name), interactive=False)
    kind.change(view_names, inputs=kind, outputs=name)
    name.change(table, inputs=[kind, name], outputs=leaderboard)
    gr.Markdown(
        "[Benchmark](https://github.com/Hanno-Labs/decision-bench) · "
        "[Reviewed results](https://github.com/Hanno-Labs/decision-bench-results) · "
        "[Documentation](https://ubiquitous-bassoon-zzmjggp.pages.github.io/)"
    )

demo.launch()
