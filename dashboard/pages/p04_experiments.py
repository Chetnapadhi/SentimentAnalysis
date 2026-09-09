"""Page 4: Experiment Comparison (Master Table & Rankings)."""

import pandas as pd
import streamlit as st

from dashboard.components import render_page_header, render_metric_card
from dashboard.data import load_master_results, load_model_rankings
from dashboard.visualizations import plot_metric_comparison_bars


def render():
    render_page_header(
        title="Experimental Results",
        subtitle="Controlled comparison across six model configurations evaluated on the canonical test split (N = 11,966).",
    )

    master_df = load_master_results()
    rankings_df = load_model_rankings()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("E3 Macro F1 (Best)", "51.60%", "+5.39 pp over E0 Baseline", is_positive=True)
    with col2:
        render_metric_card("E3 Test Accuracy", "54.20%", "+6.51 pp over E0 Baseline", is_positive=True)
    with col3:
        render_metric_card("E5 Macro F1 (Rank 2)", "50.91%", "+4.70 pp over E0 Baseline", is_positive=True)

    st.markdown("### Interactive Metric Comparison")
    metric_choice = st.selectbox(
        "Select evaluation metric to compare across E0–E5:",
        ["Macro F1", "Accuracy", "Macro Precision", "Macro Recall"],
        index=0,
    )

    metric_mapping = {
        "Macro F1": "macro_f1",
        "Accuracy": "accuracy",
        "Macro Precision": "macro_precision",
        "Macro Recall": "macro_recall",
    }

    selected_col = metric_mapping[metric_choice]
    fig = plot_metric_comparison_bars(master_df, selected_col, metric_choice)
    st.plotly_chart(fig, use_container_width=True)

    # Separate accuracy comparison chart when Macro F1 is selected
    if metric_choice == "Macro F1":
        st.markdown("### Accuracy Comparison")
        fig_acc = plot_metric_comparison_bars(master_df, "accuracy", "Test Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("### Official Master Results Table")
    
    # Leaderboard table formatted with rank and descriptions
    display_df = rankings_df[[
        "rank_by_f1", "experiment", "emoji_init", "fusion_type",
        "accuracy", "macro_precision", "macro_recall", "macro_f1", "delta_f1_pp"
    ]].copy()

    display_df.columns = [
        "Rank", "Experiment", "Emoji Representation", "Fusion Mechanism",
        "Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Delta F1 (pp)"
    ]

    display_df["Accuracy"] = (display_df["Accuracy"] * 100).map("{:.2f}%".format)
    display_df["Macro Precision"] = (display_df["Macro Precision"] * 100).map("{:.2f}%".format)
    display_df["Macro Recall"] = (display_df["Macro Recall"] * 100).map("{:.2f}%".format)
    display_df["Macro-F1"] = (display_df["Macro-F1"] * 100).map("{:.2f}%".format)
    display_df["Delta F1 (pp)"] = display_df["Delta F1 (pp)"].apply(lambda v: f"{v:+.2f} pp")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("### Research Interpretation")
    st.markdown(
        """
        - **E3 achieves the highest Macro-F1 (51.60%) and Accuracy (54.20%)** among all evaluated configurations.
        - **Pretraining Effect under Concat (E1 vs E2):** Pretraining produces a large **+7.05 pp Macro F1 gain** (0.4335 $\to$ 0.5041), preventing random initialization from degrading performance below the text-only baseline (E0: 0.4621).
        - **Pretraining Effect under Attention (E3 vs E4):** Trainable random embeddings outperform TweetEval pretraining by **+1.07 pp Macro F1** (0.5160 vs 0.5053), as task-specific end-to-end tuning adapts directly to financial emojis.
        - **Fusion Architecture Effect (E1 vs E3):** Text-conditioned attention turns random emoji embeddings from a liability (0.4335) into the benchmark leader (0.5160), a **+8.25 pp improvement**.
        """
    )
