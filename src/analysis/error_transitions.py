"""Error transition analysis: qualitative comparisons between E0, E3, and E5."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.analysis.data_loader import LABEL_NAMES, load_all_predictions


def analyze_error_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify transition categories between E0, E3, and E5.

    Returns
    -------
    summary_df : pd.DataFrame
        Counts and percentages of transitions.
    examples_df : pd.DataFrame
        Representative qualitative cases for inspection.
    """
    N = len(df)

    # Transition boolean masks
    c_e0_wrong_e3_right = (~df["e0_correct"]) & df["e3_correct"]
    c_e0_wrong_e5_right = (~df["e0_correct"]) & df["e5_correct"]
    c_e0_right_e3_wrong = df["e0_correct"] & (~df["e3_correct"])
    c_e0_right_e5_wrong = df["e0_correct"] & (~df["e5_correct"])

    c_e0_wrong_both_right = (~df["e0_correct"]) & df["e3_correct"] & df["e5_correct"]
    c_e0_right_both_wrong = df["e0_correct"] & (~df["e3_correct"]) & (~df["e5_correct"])

    # Build summary
    transitions = [
        ("E0 wrong -> E3 correct (Attention recovery)", c_e0_wrong_e3_right),
        ("E0 wrong -> E5 correct (Gated recovery)", c_e0_wrong_e5_right),
        ("E0 correct -> E3 wrong (Attention regression)", c_e0_right_e3_wrong),
        ("E0 correct -> E5 wrong (Gated regression)", c_e0_right_e5_wrong),
        ("E0 wrong -> Both E3 & E5 correct (Joint recovery)", c_e0_wrong_both_right),
        ("E0 correct -> Both E3 & E5 wrong (Joint regression)", c_e0_right_both_wrong),
    ]

    summary_rows = []
    for name, mask in transitions:
        count_all = int(mask.sum())
        pct_all = (count_all / N) * 100
        count_with_emoji = int((mask & df["has_emoji"]).sum())
        count_no_emoji = int((mask & (~df["has_emoji"])).sum())

        summary_rows.append({
            "transition": name,
            "total_count": count_all,
            "total_pct": pct_all,
            "count_with_emoji": count_with_emoji,
            "count_no_emoji": count_no_emoji,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Collect representative qualitative examples (10 per category)
    example_records = []
    category_tags = [
        ("E0_wrong__E3_correct", c_e0_wrong_e3_right & df["has_emoji"]),
        ("E0_wrong__E5_correct", c_e0_wrong_e5_right & df["has_emoji"]),
        ("E0_correct__E3_wrong", c_e0_right_e3_wrong & df["has_emoji"]),
        ("E0_correct__E5_wrong", c_e0_right_e5_wrong & df["has_emoji"]),
    ]

    for tag, mask in category_tags:
        sub = df[mask].head(10)
        for _, row in sub.iterrows():
            example_records.append({
                "category": tag,
                "id": row["id"],
                "original_text": row["original_text"],
                "text_without_emoji": row["text_without_emoji"],
                "emoji_list": str(row["emoji_list"]),
                "num_emojis": row["num_emojis"],
                "true_label": LABEL_NAMES[row["label"]],
                "e0_prediction": LABEL_NAMES[row["e0_pred"]],
                "e3_prediction": LABEL_NAMES[row["e3_pred"]],
                "e5_prediction": LABEL_NAMES[row["e5_pred"]],
            })

    examples_df = pd.DataFrame(example_records)
    return summary_df, examples_df
