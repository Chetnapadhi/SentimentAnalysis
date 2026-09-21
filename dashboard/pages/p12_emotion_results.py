"""Page 12: Phase 2 fine-grained emotion results."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import render_metric_card, render_page_header, render_research_alert
from dashboard.data import get_project_root, load_emotion_summary
from dashboard.styles import BORDER_LIGHT, TEXT_PRIMARY


def _summary_frame(records: list[dict], dataset: str) -> pd.DataFrame:
    rows = [record for record in records if record["dataset"] == dataset]
    frame = pd.DataFrame(rows)
    # Colab summary.json uses compact names; per-run metrics.json uses the
    # longer names. Normalize both formats before rendering the table.
    field_aliases = {
        "accuracy_with_emoji": "acc_with_emoji",
        "accuracy_without_emoji": "acc_without_emoji",
    }
    for canonical, compact in field_aliases.items():
        if canonical not in frame and compact in frame:
            frame[canonical] = frame[compact]
    frame["model_label"] = frame["experiment"].str.replace("_", " ", regex=False)
    frame["macro_f1_delta_pp"] = (frame["macro_f1"] - frame["majority_macro_f1"]) * 100
    return frame.sort_values(["macro_f1", "accuracy"], ascending=False)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render():
    render_page_header(
        title="Phase 2: Fine-Grained Emotion Study",
        subtitle="Teacher-directed extension from coarse financial sentiment to emotion classification on TweetEval and GoEmotions.",
    )

    records = load_emotion_summary()
    if not records:
        render_research_alert(
            "Phase 2 results are not available. Run the emotion suite on Colab and place results/emotion/summary.json in the project.",
            alert_type="warning",
        )
        return

    datasets = list(dict.fromkeys(record["dataset"] for record in records))
    dataset = st.radio(
        "Dataset",
        datasets,
        format_func=lambda value: "GoEmotions Ekman-6" if value == "goemotions" else "TweetEval Emotion",
        horizontal=True,
    )
    frame = _summary_frame(records, dataset)
    best = frame.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Best Macro F1", _percent(best["macro_f1"]), best["experiment"], is_positive=True)
    with col2:
        render_metric_card("Best Accuracy", _percent(frame["accuracy"].max()), "Held-out test split", is_positive=True)
    with col3:
        render_metric_card("Majority Macro F1", _percent(best["majority_macro_f1"]), "Required baseline", is_positive=False)
    with col4:
        render_metric_card("Emotion Classes", str(int(best["num_labels"])), "Dataset label space", is_positive=True)

    st.markdown("### What the Colab run shows")
    st.markdown(
        f"Fine-tuning the top two {best['backbone']} transformer blocks produces the strongest result in this dataset: **{best['experiment']} reaches {_percent(best['macro_f1'])} Macro F1**. The majority-class reference is only **{_percent(best['majority_macro_f1'])}**, so accuracy is not being presented in isolation."
    )

    chart = go.Figure()
    for mode, color, label in [
        ("frozen", "#2563EB", "Frozen backbone"),
        ("finetune", "#059669", "Top 2 layers fine-tuned"),
    ]:
        subset = frame[frame["mode"] == mode]
        if subset.empty:
            continue
        chart.add_trace(
            go.Bar(
                name=label,
                x=subset["model_label"],
                y=subset["macro_f1"] * 100,
                marker_color=color,
                text=[f"{value * 100:.1f}%" for value in subset["macro_f1"]],
                textposition="outside",
            )
        )
    chart.update_layout(
        barmode="group",
        title="Macro F1 by fusion model and training mode",
        yaxis=dict(title="Macro F1 (%)", range=[0, 100], gridcolor=BORDER_LIGHT),
        xaxis=dict(title="Experiment"),
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_PRIMARY),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=40, r=20, t=55, b=55),
    )
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("### Experiment leaderboard")
    display = frame[["experiment", "mode", "accuracy", "macro_f1", "weighted_f1", "macro_f1_delta_pp"]].copy()
    display = display.rename(columns={
        "experiment": "Experiment",
        "mode": "Mode",
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "macro_f1_delta_pp": "vs majority (pp)",
    })
    for column in ("Accuracy", "Macro F1", "Weighted F1"):
        display[column] = display[column].map(_percent)
    display["vs majority (pp)"] = display["vs majority (pp)"].map(lambda value: f"+{value:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Emoji slice and class-level detail")
    slice_rows = []
    for _, row in frame.iterrows():
        slice_rows.append({
            "Experiment": row["experiment"],
            "With emoji": f"{row['accuracy_with_emoji'] * 100:.1f}% (n={int(row['n_with_emoji'])})",
            "Without emoji": f"{row['accuracy_without_emoji'] * 100:.1f}% (n={int(row['n_without_emoji'])})",
            "Trainable": f"{row['trainable_pct']:.2f}%",
        })
    st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, hide_index=True)

    selected_experiment = st.selectbox("Inspect per-class F1", frame["experiment"].tolist())
    selected = frame[frame["experiment"] == selected_experiment].iloc[0]
    metrics_path = get_project_root() / selected["path"].replace("\\", "/") / "metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        class_report = pd.DataFrame(metrics["classification_report"]).T
        label_names = metrics.get("label_names")
        if not label_names:
            label_names = [
                label for label in class_report.index
                if label not in {"accuracy", "macro avg", "weighted avg"}
            ]
        class_report = class_report.loc[label_names][["precision", "recall", "f1-score", "support"]]
        class_report.columns = ["Precision", "Recall", "F1", "Support"]
        class_report["Precision"] = class_report["Precision"].map(_percent)
        class_report["Recall"] = class_report["Recall"].map(_percent)
        class_report["F1"] = class_report["F1"].map(_percent)
        st.dataframe(class_report, use_container_width=True)
    except (FileNotFoundError, KeyError, TypeError):
        render_research_alert("Per-class metrics are not available for this run; the summary metrics are still valid.", alert_type="warning")

    render_research_alert(
        "Interpretation guardrail: the datasets use different label spaces and test sets. GoEmotions is a single-label Ekman-6 subset created by dropping neutral and ambiguous examples; TweetEval uses its published four-class emotion labels. The emoji/no-emoji slices are descriptive, not causal evidence, and the study is based on one seed.",
        alert_type="info",
    )