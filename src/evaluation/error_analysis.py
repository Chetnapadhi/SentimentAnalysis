"""E0 error-analysis pipeline.

Operates on ``results/E0/error_analysis.csv`` (and ``predictions.csv``),
which are produced after E0 is run on Colab/T4.

IMPORTANT: Because E0 has only been validated (not fully trained locally —
the full BERT featurization requires a GPU), these artifacts may not exist
yet on a CPU-only machine. This script reports that clearly rather than
fabricating results.

Categories used are DETERMINISTIC and documented, derived only from the
existing emoji metadata / emoji library — no LLM is used to label emojis.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Deterministic emoji sentiment category labels (documented, heuristic-free
# except where the emoji library provides named aliases).
#
# We use the `emoji` library to resolve each emoji to its short-name alias;
# the alias is the deterministic, documented source of meaning. Grouping is
# by explicit keyword matching on the official emoji short-name. This is
# transparent and reproducible, NOT an LLM.
# ---------------------------------------------------------------------------

def emoji_category(emoji_char: str) -> str:
    """Return a broad category for an emoji based on its official short-name."""
    import emoji
    try:
        name = emoji.demojize(emoji_char, delimiters=("", ""))
    except Exception:
        name = ""
    name_lower = name.lower()

    positive = ["joy", "laugh", "smile", "heart_eyes", "fire", "rocket", "money",
                "thumbsup", "partying", "clap", "hundred", "growing", "chart_increasing",
                "green_heart", "relieved", "sunny", "tada", "collision", "gem",
                "muscle", "ok_hand", "crown"]
    negative = ["cry", "sob", "angry", "pensive", "fearful", "disappointed", "rage",
                "poop", "clown", "skull", "rolling_on_the_floor", "facepalm", "eyes_roll",
                "chart_decreasing", "fire_engine", "scream", "broken_heart", "weary",
                "sweat", "frowning"]
    # financial/market
    financial = ["chart", "money", "bank", "coin", "dollar", "euro", "rocket",
                 "airplane", "stock", "candle"]

    if any(k in name_lower for k in financial) and not any(k in name_lower for k in negative):
        return "financial/market"
    if any(k in name_lower for k in positive):
        return "positive"
    if any(k in name_lower for k in negative):
        return "negative"

    # Skin-tone / ZWJ variants: resolve base
    return "other"


def categorize_emojis(emoji_list):
    """Return category tallies for a row's emoji_list."""
    cats = Counter(emoji_category(e) for e in emoji_list)
    # Determine dominant category
    if not cats:
        return "none"
    # Ambiguous if multiple distinct categories present
    if len(cats) > 1 and all(v >= 1 for v in cats.values()):
        if sum(1 for v in cats.values() if v >= 1) >= 2:
            return "mixed"
    return cats.most_common(1)[0][0]


def run_error_analysis(error_csv="results/E0/error_analysis.csv",
                       out_csv="results/E0/e0_error_analysis_ranked.csv"):
    if not os.path.exists(error_csv):
        print(f"E0 error analysis not available: {error_csv} not found.")
        print("E0 has been validated but not fully trained here (needs Colab/T4).")
        print("Run `python run_e0.py` on Colab first, then re-run this analysis.")
        return None

    df = pd.read_csv(error_csv)
    print(f"Loaded {len(df)} E0 misclassified examples from {error_csv}")

    # 1. Errors by emoji count
    emoji_count_buckets = Counter()
    for n in df["num_emojis"]:
        if n == 1:
            emoji_count_buckets["1 emoji"] += 1
        elif n == 2:
            emoji_count_buckets["2 emojis"] += 1
        else:
            emoji_count_buckets["3+ emojis"] += 1
    print("\n=== Errors by emoji count ===")
    for k in ["1 emoji", "2 emojis", "3+ emojis"]:
        print(f"  {k}: {emoji_count_buckets[k]}")

    # 2. Group by emoji category (deterministic, from emoji aliases)
    cats = Counter(df["emoji_list"].apply(categorize_emojis))
    print("\n=== Errors by emoji category ===")
    for k, v in cats.most_common():
        print(f"  {k}: {v}")

    # 3. Add category column + short-text flag + contradiction hint (length-based,
    #    no LLM). Contradiction detection is NOT inferred semantically; we flag
    #    short/ambiguous texts and record emoji presence.
    df["emoji_category"] = df["emoji_list"].apply(categorize_emojis)
    df["token_count"] = df.get("token_count", 0)
    df["is_short"] = df["token_count"] <= 5
    df["error_category"] = df.apply(
        lambda r: (
            "short/ambiguous" if r["is_short"] else
            "mixed_emojis" if r["emoji_category"] == "mixed" else
            "category_" + r["emoji_category"]
        ), axis=1
    )

    # Rank: sort by confidence of true class vs predicted discrepancy
    if "pred_conf_bearish" in df.columns:
        conf_cols = ["pred_conf_bearish", "pred_conf_neutral", "pred_conf_bullish"]
        df["max_conf"] = df[conf_cols].max(axis=1)
    else:
        df["max_conf"] = 0.0

    ranked = df.sort_values(["is_short", "max_conf"], ascending=[False, False])

    # 4. Save ranked table
    out_cols = ["original_text", "text_without_emoji", "emoji_list",
                "label", "predicted_label", "max_conf", "emoji_category", "error_category"]
    out_cols = [c for c in out_cols if c in ranked.columns]
    ranked[out_cols].to_csv(out_csv, index=False)
    print(f"\nSaved ranked error table -> {out_csv}")

    # 5. Representative examples (up to 20)
    print("\n=== Representative examples (for E1/E2 comparison) ===")
    reps = ranked.head(20)
    for _, r in reps.iterrows():
        print(f"  [{r['label']}->{r['predicted_label']}] cat={r.get('emoji_category','?')} "
              f"emoji={r['emoji_list']} | {str(r['text_without_emoji'])[:45]!r}")

    return ranked


if __name__ == "__main__":
    run_error_analysis()
