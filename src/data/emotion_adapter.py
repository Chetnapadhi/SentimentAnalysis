"""Emotion dataset adapters (Phase 2).

Phase 1 of this project classified 3-class StockTwits *sentiment*
(Bearish / Neutral / Bullish) and plateaued near 0.48 accuracy. Phase 2
follows the supervisor's mandate: classify the **sentence emotion** instead,
on a fine-grained emotion label space.

Two corpora are adapted here, both reduced to the SAME canonical schema used
by ``src/data/preprocessing.py`` so that every existing model, training loop
and analysis module works unchanged:

    id, split, original_text, text_without_emoji, emoji_list,
    num_emojis, label, label_name, token_count

GoEmotions -> Ekman-6
---------------------
``google-research-datasets/go_emotions`` (simplified) is 43,410 Reddit
comments with 28 multi-label annotations. We apply the **Ekman grouping**
published with the GoEmotions paper to collapse 27 emotions into 6 families,
drop ``neutral``, and keep only rows that map to exactly ONE Ekman family.
That yields a clean single-label 6-class problem -- ``num_labels = 6``.

TweetEval Emotion
-----------------
``cardiffnlp/tweet_eval`` config ``emotion`` is 3,257 tweets labeled with
4 classes (anger / joy / optimism / sadness). Smaller and coarser, but it is
social-media text and therefore carries ~6x the emoji density of GoEmotions,
so it is the corpus where the emoji branch can actually be exercised.

Emoji density (measured, not assumed) is reported by ``coverage_report`` and
recorded in the manifest, because it bounds how much the emoji branch can
possibly contribute.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from datasets import load_dataset

from src.data.preprocessing import (
    extract_emojis,
    make_id,
    remove_emojis,
    whitespace_token_count,
)

# ---------------------------------------------------------------------------
# GoEmotions -> Ekman grouping (from the GoEmotions paper, Demszky et al. 2020)
# ---------------------------------------------------------------------------

EKMAN_MAP: dict[str, str] = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy", "excitement": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy", "relief": "joy",
    "pride": "joy", "admiration": "joy", "desire": "joy", "caring": "joy",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "curiosity": "surprise",
    "neutral": "neutral",
}

# Fixed label order -> ids 0..5. ``neutral`` is deliberately excluded.
EKMAN6_NAMES: list[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EKMAN6_TO_ID: dict[str, int] = {name: i for i, name in enumerate(EKMAN6_NAMES)}

TWEETEVAL_NAMES: list[str] = ["anger", "joy", "optimism", "sadness"]


# ---------------------------------------------------------------------------
# Canonical row construction
# ---------------------------------------------------------------------------

def _canonical_rows(
    texts: list[str], labels: list[int], label_names: list[str], split: str,
) -> pd.DataFrame:
    """Build the canonical DataFrame shared with Phase 1 preprocessing."""
    rows = []
    for idx, (text, label, lname) in enumerate(zip(texts, labels, label_names)):
        emojis = extract_emojis(text)
        stripped = remove_emojis(text)
        rows.append({
            "id": make_id(text, split, idx),
            "split": split,
            "original_text": text,
            "text_without_emoji": stripped,
            "emoji_list": emojis,
            "num_emojis": len(emojis),
            "label": int(label),
            "label_name": lname,
            "token_count": whitespace_token_count(stripped),
        })
    return pd.DataFrame(rows)


def coverage_report(df: pd.DataFrame) -> dict:
    """Measured emoji coverage + class distribution for one split."""
    n = len(df)
    with_emoji = int((df["num_emojis"] > 0).sum())
    uniq: set[str] = set()
    for lst in df["emoji_list"]:
        uniq.update(lst)
    return {
        "rows": n,
        "rows_with_emoji": with_emoji,
        "emoji_coverage_pct": round(100.0 * with_emoji / max(n, 1), 2),
        "unique_emojis": len(uniq),
        "class_distribution": {
            k: int(v) for k, v in df["label_name"].value_counts().sort_index().items()
        },
    }


# ---------------------------------------------------------------------------
# GoEmotions
# ---------------------------------------------------------------------------

def build_goemotions_ekman6() -> tuple[dict[str, pd.DataFrame], dict]:
    """Adapt GoEmotions to a single-label 6-class Ekman problem.

    Selection rule (applied identically to every split, no leakage):
      1. Map each of the row's 28-way labels through ``EKMAN_MAP``.
      2. Drop rows whose mapped set contains ``neutral``.
      3. Keep rows whose mapped set has exactly ONE family (ambiguous
         multi-family rows are dropped rather than arbitrarily resolved).
    """
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    id2name = ds["train"].features["labels"].feature.names

    splits: dict[str, pd.DataFrame] = {}
    stats: dict = {
        "source": "google-research-datasets/go_emotions (simplified)",
        "label_space": "Ekman-6 single-label",
        "label_mapping": {str(i): n for i, n in enumerate(EKMAN6_NAMES)},
        "selection_rule": (
            "map 28 labels -> Ekman families; drop rows containing neutral; "
            "keep rows mapping to exactly one family"
        ),
        "splits": {},
    }

    for split in ["train", "validation", "test"]:
        texts, labels, lnames = [], [], []
        raw_n = len(ds[split])
        dropped_neutral = dropped_ambiguous = 0

        for row in ds[split]:
            families = {EKMAN_MAP[id2name[i]] for i in row["labels"]}
            if "neutral" in families:
                dropped_neutral += 1
                continue
            if len(families) != 1:
                dropped_ambiguous += 1
                continue
            fam = families.pop()
            texts.append(row["text"])
            labels.append(EKMAN6_TO_ID[fam])
            lnames.append(fam)

        df = _canonical_rows(texts, labels, lnames, split)
        splits[split] = df
        rep = coverage_report(df)
        rep.update({
            "raw_rows": raw_n,
            "dropped_neutral": dropped_neutral,
            "dropped_multi_family": dropped_ambiguous,
        })
        stats["splits"][split] = rep

    return splits, stats


# ---------------------------------------------------------------------------
# TweetEval emotion
# ---------------------------------------------------------------------------

def build_tweeteval_emotion() -> tuple[dict[str, pd.DataFrame], dict]:
    """Adapt TweetEval ``emotion`` (4-class, single-label) to canonical form."""
    ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
    names = ds["train"].features["label"].names

    splits: dict[str, pd.DataFrame] = {}
    stats: dict = {
        "source": "cardiffnlp/tweet_eval (config=emotion)",
        "label_space": "4-class single-label",
        "label_mapping": {str(i): n for i, n in enumerate(names)},
        "selection_rule": "used as published; no rows dropped",
        "splits": {},
    }

    for split in ["train", "validation", "test"]:
        texts = list(ds[split]["text"])
        labels = list(ds[split]["label"])
        lnames = [names[l] for l in labels]
        df = _canonical_rows(texts, labels, lnames, split)
        splits[split] = df
        rep = coverage_report(df)
        rep["raw_rows"] = len(ds[split])
        stats["splits"][split] = rep

    return splits, stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

BUILDERS = {
    "goemotions": build_goemotions_ekman6,
    "tweeteval": build_tweeteval_emotion,
}


def save_emotion_dataset(
    name: str, output_dir: str = "data/processed/canonical_emotion",
) -> dict:
    """Build one emotion corpus and persist canonical JSONL + manifest."""
    if name not in BUILDERS:
        raise ValueError(
            f"Unknown emotion dataset {name!r}; choose from {list(BUILDERS)}"
        )

    splits, stats = BUILDERS[name]()
    os.makedirs(output_dir, exist_ok=True)

    for split, df in splits.items():
        path = os.path.join(output_dir, f"{name}_{split}.jsonl")
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"  saved {len(df):>6} rows -> {path}")

    manifest_path = os.path.join(output_dir, f"{name}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  manifest -> {manifest_path}")
    return stats


def load_emotion_splits(
    name: str, input_dir: str = "data/processed/canonical_emotion",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical emotion splits built by ``save_emotion_dataset``."""
    paths = {
        s: os.path.join(input_dir, f"{name}_{s}.jsonl")
        for s in ["train", "validation", "test"]
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing canonical emotion splits: {missing}. "
            f"Build them first: python -m src.data.emotion_adapter --dataset {name}"
        )
    return tuple(
        pd.read_json(paths[s], lines=True) for s in ["train", "validation", "test"]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["goemotions", "tweeteval", "all"], default="all"
    )
    parser.add_argument("--output-dir", default="data/processed/canonical_emotion")
    args = parser.parse_args()

    targets = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"\n=== Building {name} ===")
        stats = save_emotion_dataset(name, args.output_dir)
        for split, rep in stats["splits"].items():
            print(
                f"  {split:<11} rows={rep['rows']:>6}  "
                f"emoji_coverage={rep['emoji_coverage_pct']:>5.2f}%  "
                f"unique_emoji={rep['unique_emojis']:>4}"
            )
            print(f"              classes={rep['class_distribution']}")
