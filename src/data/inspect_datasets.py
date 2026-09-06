"""Inspect the three project datasets independently.

This module deliberately keeps the datasets separate. It produces one JSON
report per source dataset and never concatenates or joins their records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import emoji
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset


DATASETS = {
    "stocktwits": {"path": "ElKulako/stocktwits-emoji", "config": None},
    "youtube": {"path": "bapcon/101k-emojis", "config": None},
    "tweeteval_emoji": {"path": "cardiffnlp/tweet_eval", "config": "emoji"},
}
TEXT_CANDIDATES = ("text", "comment", "content", "sentence", "tweet")
LABEL_CANDIDATES = ("label", "labels", "sentiment")


def _as_dataset_dict(loaded: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"train": loaded})


def _find_column(dataset: Dataset, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in dataset.column_names:
            return candidate
    string_columns = [
        name for name, feature in dataset.features.items() if str(feature) == "Value('string')"
    ]
    return string_columns[0] if string_columns else None


def _label_metadata(dataset: Dataset, label_column: str | None) -> dict[str, Any]:
    if label_column is None:
        return {"column": None, "mapping": {}, "counts": {}}

    feature = dataset.features.get(label_column)
    values = dataset[label_column]
    counts = Counter(str(value) for value in values if value is not None)
    mapping: dict[str, str] = {}
    if isinstance(feature, ClassLabel):
        mapping = {str(index): name for index, name in enumerate(feature.names)}
    return {"column": label_column, "mapping": mapping, "counts": dict(counts)}


def _inspect_split(dataset: Dataset, text_column: str | None) -> dict[str, Any]:
    if text_column is None:
        return {
            "records": dataset.num_rows,
            "emoji_records": None,
            "unique_emojis": [],
            "duplicate_texts": None,
            "missing_texts": None,
        }

    texts = dataset[text_column]
    normalized = [text.strip() if isinstance(text, str) else "" for text in texts]
    emoji_lists = [emoji.distinct_emoji_list(text) for text in normalized]
    non_empty = [text for text in normalized if text]
    text_counts = Counter(non_empty)
    unique_emojis = Counter(item for items in emoji_lists for item in items)
    return {
        "records": dataset.num_rows,
        "emoji_records": sum(bool(items) for items in emoji_lists),
        "emoji_record_fraction": (
            sum(bool(items) for items in emoji_lists) / dataset.num_rows
            if dataset.num_rows
            else 0.0
        ),
        "unique_emojis": unique_emojis.most_common(),
        "duplicate_texts": sum(count - 1 for count in text_counts.values() if count > 1),
        "missing_texts": sum(not text for text in normalized),
        "text_length": {
            "min": min(map(len, normalized), default=0),
            "max": max(map(len, normalized), default=0),
            "mean": sum(map(len, normalized)) / len(normalized) if normalized else 0.0,
        },
    }


def inspect_dataset(name: str, output_dir: Path) -> dict[str, Any]:
    """Download/cache and inspect one dataset without mixing sources."""
    spec = DATASETS[name]
    load_kwargs = {"path": spec["path"]}
    if spec["config"] is not None:
        load_kwargs["name"] = spec["config"]
    loaded = _as_dataset_dict(load_dataset(**load_kwargs))

    first_split = next(iter(loaded.values()))
    text_column = _find_column(first_split, TEXT_CANDIDATES)
    label_column = next(
        (candidate for candidate in LABEL_CANDIDATES if candidate in first_split.column_names),
        None,
    )
    report: dict[str, Any] = {
        "dataset_key": name,
        "dataset_path": spec["path"],
        "dataset_config": spec["config"],
        "columns_by_split": {split: data.column_names for split, data in loaded.items()},
        "features_by_split": {split: str(data.features) for split, data in loaded.items()},
        "text_column": text_column,
        "label": _label_metadata(first_split, label_column),
        "supervised_labels_available": label_column is not None,
        "splits": {
            split: _inspect_split(data, text_column) for split, data in loaded.items()
        },
        "notes": [],
    }
    if name == "stocktwits":
        report["notes"].append(
            "This is intended to be the supervised sentiment source, but the loaded Hugging Face schema exposes text only; sentiment labels require a separately documented raw-file parser."
        )
    elif name == "youtube":
        report["notes"].append(
            "Use for emoji frequency, visualization, and optional external analysis; no sentiment labels are assumed."
        )
    else:
        report["notes"].append(
            "Use only as auxiliary emoji-task data; its labels are not StockTwits sentiment labels."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="Inspect one source or all three independently.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/inspection"),
    )
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        report = inspect_dataset(name, args.output_dir)
        print(
            f"{name}: "
            f"splits={list(report['splits'])}, "
            f"text_column={report['text_column']}, "
            f"labels={report['label']['mapping']}"
        )


if __name__ == "__main__":
    main()