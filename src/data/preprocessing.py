"""Canonical preprocessing for the StockTwits dataset.

Produces one canonical representation per row with:
    id, split, original_text, text_without_emoji, emoji_list,
    num_emojis, label, label_name, token_count

Emoji extraction uses the ``emoji`` library (Unicode-aware), not a manual
regex. The original text is preserved exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import emoji
import pandas as pd


# ---------------------------------------------------------------------------
# Emoji extraction
# ---------------------------------------------------------------------------

def extract_emojis(text: str) -> list[str]:
    """Return every distinct emoji sequence in ``text`` (Unicode-aware).

    ``emoji.distinct_emoji_list`` treats ZWJ sequences, skin-tone modifiers,
    and variation selectors as single emoji units, which is what we want for
    this emoji-aware project.
    """
    if not isinstance(text, str):
        return []
    return emoji.distinct_emoji_list(text)


def remove_emojis(text: str) -> str:
    """Remove all emoji from ``text``, preserving the rest exactly."""
    if not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace="")


def make_id(original_text: str, split: str, index: int) -> str:
    """Stable, auditable identifier for a row."""
    raw = f"{split}::{index}::{original_text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Canonical preprocessing
# ---------------------------------------------------------------------------

def build_canonical(
    df: pd.DataFrame,
    split: str,
    tokenizer_fn,
) -> pd.DataFrame:
    """Add canonical columns to a labeled StockTwits DataFrame.

    Parameters
    ----------
    df : DataFrame with columns ``text``, ``label``, ``label_name``.
    split : "train", "validation", or "test".
    tokenizer_fn : callable(text) -> int token count.
    """
    rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        original = row["text"]
        emojis = extract_emojis(original)
        stripped = remove_emojis(original)
        # normalize internal whitespace for token counting only, not the text
        rows.append({
            "id": make_id(original, split, int(idx)),
            "split": split,
            "original_text": original,
            "text_without_emoji": stripped,
            "emoji_list": emojis,
            "num_emojis": len(emojis),
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "token_count": tokenizer_fn(stripped),
        })

    return pd.DataFrame(rows)


def save_canonical(df: pd.DataFrame, path: str) -> None:
    """Save canonical dataframe to disk (JSONL to preserve exact text)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(df)} canonical rows -> {path}")


# ---------------------------------------------------------------------------
# Tokenizer (simple whitespace for token_count; BERT tokenizer used later)
# ---------------------------------------------------------------------------

def whitespace_token_count(text: str) -> int:
    return len([t for t in text.split() if t])


def bert_token_count(text: str) -> int:
    """Token count using the actual BERT tokenizer (slow — prefer whitespace)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    return len(tok.encode(text, add_special_tokens=True))


# ---------------------------------------------------------------------------
# Build canonical for the FINAL approved splits
# ---------------------------------------------------------------------------

def build_final_canonical(
    input_dir: str = "data/processed",
    output_dir: str = "data/processed/canonical",
) -> dict[str, pd.DataFrame]:
    """Build canonical final datasets from approved split files.

    Inputs (approved):
      - stocktwits_train_dedup.csv
      - stocktwits_validation_final.csv
      - stocktwits_test_final.csv
    """
    split_files = {
        "train": "stocktwits_train_dedup.csv",
        "validation": "stocktwits_validation_final.csv",
        "test": "stocktwits_test_final.csv",
    }
    result = {}
    for split, fname in split_files.items():
        df = pd.read_csv(os.path.join(input_dir, fname))
        # token_count via BERT tokenizer for consistency with model
        canonical = build_canonical(df, split, whitespace_token_count)
        result[split] = canonical
        save_canonical(canonical, os.path.join(output_dir, f"final_{split}.jsonl"))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed/canonical")
    args = parser.parse_args()

    build_final_canonical(args.input_dir, args.output_dir)
