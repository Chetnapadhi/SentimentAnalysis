"""Cached data loading module for results, metrics, and dataset summaries."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return Path(__file__).resolve().parent.parent


@st.cache_data
def load_master_results() -> pd.DataFrame:
    """Load master_results.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "master_results.csv"
    return pd.read_csv(path)


@st.cache_data
def load_model_rankings() -> pd.DataFrame:
    """Load model_rankings.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "model_rankings.csv"
    return pd.read_csv(path)


@st.cache_data
def load_per_class_metrics() -> pd.DataFrame:
    """Load per_class_metrics.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "per_class_metrics.csv"
    return pd.read_csv(path)


@st.cache_data
def load_emoji_count_breakdown() -> pd.DataFrame:
    """Load emoji_count_breakdown.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "emoji_count_breakdown.csv"
    return pd.read_csv(path)


@st.cache_data
def load_sentiment_by_emoji_count() -> pd.DataFrame:
    """Load sentiment_by_emoji_count.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "sentiment_by_emoji_count.csv"
    return pd.read_csv(path)


@st.cache_data
def load_transition_summary() -> pd.DataFrame:
    """Load transition_summary.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "transition_summary.csv"
    return pd.read_csv(path)


@st.cache_data
def load_error_transition_cases() -> pd.DataFrame:
    """Load error_transition_cases.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "error_transition_cases.csv"
    return pd.read_csv(path)


@st.cache_data
def load_gate_summary() -> pd.DataFrame:
    """Load gate_summary_table.csv from results/final_analysis/."""
    path = get_project_root() / "results" / "final_analysis" / "gate_summary_table.csv"
    return pd.read_csv(path)


@st.cache_data
def load_gate_raw_json() -> dict:
    """Load results/E5/gate_analysis.json."""
    path = get_project_root() / "results" / "E5" / "gate_analysis.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix(exp: str) -> list[list[float]]:
    """Return normalized confusion matrix for the requested experiment."""
    path = get_project_root() / "results" / exp / "predictions.csv"
    if not path.exists():
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    df = pd.read_csv(path)
    from sklearn.metrics import confusion_matrix
    y_true = df["label"] if "label" in df.columns else df["true_label"]
    y_pred = df["predicted_label"].astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    return cm.tolist()
