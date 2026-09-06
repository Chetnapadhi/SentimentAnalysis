"""StockTwits label adapter.

The HuggingFace dataset ``ElKulako/stocktwits-emoji`` exposes only a ``text``
column.  The sentiment labels (Bullish / Neutral / Bearish) are recoverable
from the repository's source text files whose filenames encode the class.

Source file structure
---------------------
Each class has its own file per split::

    train-emoji-bear-unmodified.txt   -> Bearish
    train-emoji-net-unmodified.txt    -> Neutral
    train-emoji-bull-unmodified.txt   -> Bullish
    val_bear.txt                      -> Bearish
    val_net.txt                       -> Neutral
    val_bull.txt                      -> Bullish
    test_emoji_bear.txt               -> Bearish
    test-emoji-net.txt                -> Neutral
    test-emoji-bull.txt               -> Bullish

The adapter matches each HuggingFace row to the corresponding source file
entry by text content, preserving the original split and avoiding any
fabrication of labels.

Label encoding
--------------
0 = Bearish
1 = Neutral
2 = Bullish
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = "ElKulako/stocktwits-emoji"

LABEL_MAP = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

# Source file names keyed by (split, class_name)
SOURCE_FILES: dict[tuple[str, str], str] = {
    ("train", "Bearish"): "train-emoji-bear-unmodified.txt",
    ("train", "Neutral"): "train-emoji-net-unmodified.txt",
    ("train", "Bullish"): "train-emoji-bull-unmodified.txt",
    ("validation", "Bearish"): "val_bear.txt",
    ("validation", "Neutral"): "val_net.txt",
    ("validation", "Bullish"): "val_bull.txt",
    ("test", "Bearish"): "test_emoji_bear.txt",
    ("test", "Neutral"): "test-emoji-net.txt",
    ("test", "Bullish"): "test-emoji-bull.txt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_source_files(cache_dir: str) -> dict[str, str]:
    """Download every source text file and return ``{filename: local_path}``."""
    unique_files = set(SOURCE_FILES.values())
    paths: dict[str, str] = {}
    for fname in unique_files:
        local = hf_hub_download(
            DATASET_PATH,
            fname,
            repo_type="dataset",
            local_dir=cache_dir,
        )
        paths[fname] = local
    return paths


def _build_label_lookup(
    source_paths: dict[str, str],
    split: str,
) -> tuple[dict[str, int], dict[str, list[int]], int]:
    """Build ``text -> label`` for one split.

    Returns
    -------
    first_label : dict[str, int]
        Text to label using first-file-occurrence.
    all_labels : dict[str, list[int]]
        Text to all labels found (for conflict detection).
    conflict_count : int
        Number of texts with more than one distinct label.
    """
    first_label: dict[str, int] = {}
    all_labels: dict[str, list[int]] = defaultdict(list)
    conflict_count = 0

    # Process classes in a fixed order so first-occurrence is deterministic.
    for class_name in ["Bearish", "Neutral", "Bullish"]:
        fname = SOURCE_FILES[(split, class_name)]
        filepath = source_paths[fname]
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                text = line.rstrip("\n").strip()
                if not text:
                    continue
                label_id = LABEL_MAP[class_name]
                all_labels[text].append(label_id)
                if text not in first_label:
                    first_label[text] = label_id

    conflict_count = sum(1 for labels in all_labels.values() if len(set(labels)) > 1)
    return first_label, dict(all_labels), conflict_count


def _match_hf_rows(
    hf_split: Dataset,
    first_label: dict[str, int],
    all_labels: dict[str, list[int]],
    conflict_strategy: str,
) -> dict[str, Any]:
    """Match every HuggingFace row to a source-file label.

    Parameters
    ----------
    conflict_strategy : str
        ``"drop"``  – rows whose text has conflicting labels are excluded.
        ``"first"`` – the first-file-occurrence label is used.

    Returns
    -------
    dict with keys ``matched``, ``dropped``, ``labels``, ``conflict_indices``.
    """
    matched_indices: list[int] = []
    dropped_indices: list[int] = []
    labels: list[int] = []

    for idx, row in enumerate(hf_split):
        text = row["text"].strip()
        if text not in first_label:
            dropped_indices.append(idx)
            continue

        # Check for conflict
        row_labels = all_labels.get(text, [])
        has_conflict = len(set(row_labels)) > 1

        if has_conflict and conflict_strategy == "drop":
            dropped_indices.append(idx)
        else:
            matched_indices.append(idx)
            labels.append(first_label[text])

    return {
        "matched": matched_indices,
        "dropped": dropped_indices,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

def build_labeled_stocktwits(
    cache_dir: str = "data/raw/stocktwits",
    conflict_strategy: str = "drop",
) -> DatasetDict:
    """Build a labeled StockTwits DatasetDict with ``text``, ``label``,
    ``label_name``, and ``split`` columns.

    Parameters
    ----------
    cache_dir : str
        Directory for downloaded source files.
    conflict_strategy : str
        ``"drop"`` (default) removes rows with conflicting labels.
        ``"first"`` assigns the first-file-occurrence label.

    Returns
    -------
    DatasetDict with splits ``train``, ``validation``, ``test``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    source_paths = _download_source_files(cache_dir)

    # Load the HuggingFace dataset (text-only)
    hf_ds = load_dataset(DATASET_PATH)

    report: dict[str, Any] = {
        "source": "ElKulako/stocktwits-emoji",
        "label_source": "Source text files in HuggingFace repository (filename-encoded classes)",
        "label_mapping": {v: k for k, v in LABEL_MAP.items()},
        "conflict_strategy": conflict_strategy,
        "splits": {},
        "total_hf_rows": 0,
        "total_matched": 0,
        "total_dropped": 0,
        "total_conflicts": 0,
    }

    result: dict[str, Dataset] = {}

    for split in ["train", "validation", "test"]:
        first_label, all_labels, conflict_count = _build_label_lookup(
            source_paths, split
        )
        match_result = _match_hf_rows(
            hf_ds[split], first_label, all_labels, conflict_strategy
        )

        # Build new dataset with labels
        matched_indices = match_result["matched"]
        label_list = match_result["labels"]

        texts = [hf_ds[split][i]["text"] for i in matched_indices]
        label_names = [LABEL_NAMES[l] for l in label_list]
        split_names = [split] * len(matched_indices)

        new_ds = Dataset.from_dict({
            "text": texts,
            "label": label_list,
            "label_name": label_names,
            "split": split_names,
        })
        result[split] = new_ds

        # Validation checks
        assert len(new_ds) == len(matched_indices) == len(label_list), (
            "Length mismatch in %s: dataset=%d, indices=%d, labels=%d"
            % (split, len(new_ds), len(matched_indices), len(label_list))
        )
        assert all(0 <= l <= 2 for l in label_list), (
            "Invalid labels found in %s" % split
        )

        class_counts = Counter(label_list)
        emoji_count = sum(
            1 for t in texts if len(__import__("emoji").distinct_emoji_list(t)) > 0
        )

        split_report = {
            "hf_rows": len(hf_ds[split]),
            "matched_rows": len(matched_indices),
            "dropped_rows": len(match_result["dropped"]),
            "conflict_texts": conflict_count,
            "class_distribution": {
                LABEL_NAMES[k]: v for k, v in sorted(class_counts.items())
            },
            "emoji_containing_records": emoji_count,
        }
        report["splits"][split] = split_report
        report["total_hf_rows"] += len(hf_ds[split])
        report["total_matched"] += len(matched_indices)
        report["total_dropped"] += len(match_result["dropped"])
        report["total_conflicts"] += conflict_count

    return DatasetDict(result), report


def save_labeled_dataset(
    ds: DatasetDict,
    output_dir: str = "data/processed",
) -> None:
    """Save the labeled dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for split in ds.keys():
        path = os.path.join(output_dir, f"stocktwits_{split}")
        ds[split].to_csv(f"{path}.csv", index=False)
        print(f"Saved {split}: {len(ds[split])} rows -> {path}.csv")


def save_report(report: dict, output_path: str = "data/inspection/stocktwits_labels.json") -> None:
    """Save the adapter report as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conflict-strategy",
        choices=["drop", "first"],
        default="drop",
        help="How to handle texts with conflicting labels across class files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/stocktwits",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
    )
    parser.add_argument(
        "--report-path",
        default="data/inspection/stocktwits_labels.json",
    )
    args = parser.parse_args()

    ds, report = build_labeled_stocktwits(
        cache_dir=args.cache_dir,
        conflict_strategy=args.conflict_strategy,
    )

    save_labeled_dataset(ds, args.output_dir)
    save_report(report, args.report_path)

    print()
    print("=" * 60)
    print("STOCKTWITS LABEL RECOVERY COMPLETE")
    print("=" * 60)
    for split in ["train", "validation", "test"]:
        r = report["splits"][split]
        print(f"\n{split.upper()}:")
        print(f"  HF rows:       {r['hf_rows']}")
        print(f"  Matched:       {r['matched_rows']}")
        print(f"  Dropped:       {r['dropped_rows']} (conflicting labels)")
        print(f"  Class dist:    {r['class_distribution']}")
        print(f"  Emoji records: {r['emoji_containing_records']}")
    print(f"\nTotal: {report['total_matched']} labeled, {report['total_dropped']} dropped")
