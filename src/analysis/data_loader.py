"""Data loader utilities for consolidating E0–E5 results and canonical test data."""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
LABEL_MAP = {"Bearish": 0, "Neutral": 1, "Bullish": 2}


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_canonical_test() -> pd.DataFrame:
    """Load canonical test set from data/processed/canonical/final_test.jsonl."""
    root = get_project_root()
    path = root / "data" / "processed" / "canonical" / "final_test.jsonl"
    df = pd.read_json(path, lines=True)
    return df


def load_all_predictions() -> pd.DataFrame:
    """Load and merge predictions from E0 through E5 with canonical test metadata.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing ground truth and predictions/correctness
        for all 6 experiments.
    """
    root = get_project_root()
    test_df = load_canonical_test()
    N = len(test_df)
    assert N == 11966, f"Expected 11966 test samples, got {N}"

    master_df = test_df.copy()

    # Load E0 predictions
    e0_path = root / "results" / "E0" / "predictions.csv"
    if e0_path.exists():
        e0_df = pd.read_csv(e0_path)
        master_df["e0_pred"] = e0_df["predicted_label"].astype(int)
        master_df["e0_correct"] = master_df["e0_pred"] == master_df["label"]
        if "pred_conf_bearish" in e0_df.columns:
            master_df["e0_prob_bearish"] = e0_df["pred_conf_bearish"]
            master_df["e0_prob_neutral"] = e0_df["pred_conf_neutral"]
            master_df["e0_prob_bullish"] = e0_df["pred_conf_bullish"]

    # Load E1 predictions
    e1_path = root / "results" / "E1" / "predictions.csv"
    if e1_path.exists():
        e1_df = pd.read_csv(e1_path)
        master_df["e1_pred"] = e1_df["predicted_label"].astype(int)
        master_df["e1_correct"] = master_df["e1_pred"] == master_df["label"]

    # Load E2 predictions
    e2_path = root / "results" / "E2" / "predictions.csv"
    if e2_path.exists():
        e2_df = pd.read_csv(e2_path)
        master_df["e2_pred"] = e2_df["predicted_label"].astype(int)
        master_df["e2_correct"] = master_df["e2_pred"] == master_df["label"]
        if "prob_bearish" in e2_df.columns:
            master_df["e2_prob_bearish"] = e2_df["prob_bearish"]
            master_df["e2_prob_neutral"] = e2_df["prob_neutral"]
            master_df["e2_prob_bullish"] = e2_df["prob_bullish"]

    # Load E3 predictions
    e3_path = root / "results" / "E3" / "predictions.csv"
    if e3_path.exists():
        e3_df = pd.read_csv(e3_path)
        master_df["e3_pred"] = e3_df["predicted_label"].astype(int)
        master_df["e3_correct"] = master_df["e3_pred"] == master_df["label"]
        if "pred_conf_bearish" in e3_df.columns:
            master_df["e3_prob_bearish"] = e3_df["pred_conf_bearish"]
            master_df["e3_prob_neutral"] = e3_df["pred_conf_neutral"]
            master_df["e3_prob_bullish"] = e3_df["pred_conf_bullish"]

    # Load E4 predictions
    e4_path = root / "results" / "E4" / "predictions.csv"
    if e4_path.exists():
        e4_df = pd.read_csv(e4_path)
        master_df["e4_pred"] = e4_df["predicted_label"].astype(int)
        master_df["e4_correct"] = master_df["e4_pred"] == master_df["label"]
        if "pred_conf_bearish" in e4_df.columns:
            master_df["e4_prob_bearish"] = e4_df["pred_conf_bearish"]
            master_df["e4_prob_neutral"] = e4_df["pred_conf_neutral"]
            master_df["e4_prob_bullish"] = e4_df["pred_conf_bullish"]

    # Load E5 predictions
    e5_path = root / "results" / "E5" / "predictions.csv"
    if e5_path.exists():
        e5_df = pd.read_csv(e5_path)
        master_df["e5_pred"] = e5_df["predicted_label"].astype(int)
        master_df["e5_correct"] = master_df["e5_pred"] == master_df["label"]
        if "pred_conf_bearish" in e5_df.columns:
            master_df["e5_prob_bearish"] = e5_df["pred_conf_bearish"]
            master_df["e5_prob_neutral"] = e5_df["pred_conf_neutral"]
            master_df["e5_prob_bullish"] = e5_df["pred_conf_bullish"]

    # Emoji presence boolean
    master_df["has_emoji"] = master_df["num_emojis"] > 0

    return master_df


def load_all_metrics() -> dict[str, dict]:
    """Load metrics.json for all six experiments."""
    root = get_project_root()
    all_metrics = {}
    for exp in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        path = root / "results" / exp / "metrics.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_metrics[exp] = json.load(f)
        else:
            raise FileNotFoundError(f"Missing metrics.json for {exp} at {path}")
    return all_metrics


def load_all_confusion_matrices() -> dict[str, np.ndarray]:
    """Load confusion matrix as 3x3 numpy array for all six experiments."""
    root = get_project_root()
    cms = {}
    metrics = load_all_metrics()

    for exp in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        m = metrics[exp]
        if "confusion_matrix" in m:
            cms[exp] = np.array(m["confusion_matrix"])
        else:
            # Check CSV fallback (E1, E2)
            csv_path = root / "results" / exp / "confusion_matrix.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                cms[exp] = df.to_numpy()
            else:
                raise ValueError(f"Could not load confusion matrix for {exp}")
    return cms
