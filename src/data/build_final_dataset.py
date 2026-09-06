"""Build the final versioned experimental dataset for E0-E5.

Pipeline (represents the approved decisions):
1. Load raw labeled StockTwits splits (data/processed/stocktwits_{split}.csv).
2. Keep the raw versions untouched; create *_raw.csv snapshots.
3. Training:
   - drop conflicting-label rows (already done by adapter)
   - normalize text only for duplicate detection
   - deduplicate exact text deterministically (keep first occurrence + provenance)
   - remove the 27 exact train<->test overlapping texts
   -> stocktwits_train_dedup.csv
4. Validation/test: kept unchanged (evaluation policy decided separately);
   validation/test overlap is reported as a documented limitation.
5. Recalculate class weights from the FINAL training set only.
6. Write an experiment manifest with hashes, counts, distributions, decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_key(text: str) -> str:
    """Normalize text for duplicate detection only (collapse whitespace, strip)."""
    return " ".join(str(text).split())


def sha1_of_df(df: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha1()
    for col in columns:
        for value in df[col].astype(str).tolist():
            h.update(value.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build final dataset
# ---------------------------------------------------------------------------

def main() -> None:
    processed = Path("data/processed")
    os.makedirs(processed, exist_ok=True)

    # 1. Load raw labeled splits
    train_raw = pd.read_csv(processed / "stocktwits_train.csv")
    val_raw = pd.read_csv(processed / "stocktwits_validation.csv")
    test_raw = pd.read_csv(processed / "stocktwits_test.csv")

    # 2. Snapshot the raw versions (never overwrite originals)
    for split, df in [("train", train_raw), ("validation", val_raw), ("test", test_raw)]:
        df.to_csv(processed / f"stocktwits_{split}_raw.csv", index=False)
    print("Saved raw snapshots: stocktwits_{train,validation,test}_raw.csv")

    # 3. Build the FINAL training set
    # 3a. The adapter already dropped conflicting labels. Re-assert: each train
    #     row has exactly one valid label (0/1/2).
    assert train_raw["label"].isin([0, 1, 2]).all(), "Invalid label in train"

    train_orig_rows = len(train_raw)

    # 3b. Deduplicate exact text (normalized). Keep first occurrence.
    seen: dict[str, int] = {}
    keep_idx: list[int] = []
    key_to_index: dict[str, int] = {}
    for idx, row in train_raw.iterrows():
        key = text_key(row["text"])
        # Track first row index for each key (for provenance)
        if key not in key_to_index:
            key_to_index[key] = idx
            keep_idx.append(idx)
    train_dedup = train_raw.loc[keep_idx].reset_index(drop=True).copy()
    n_dup_removed = train_orig_rows - len(train_dedup)

    # 3c. Remove the 27 exact train<->test overlapping texts.
    test_text_keys = set(text_key(t) for t in test_raw["text"].astype(str))
    overlap_mask = ~train_dedup["text"].astype(str).apply(lambda t: text_key(t) in test_text_keys)
    n_test_overlap_removed = int((~overlap_mask).sum())
    train_final = train_dedup.loc[overlap_mask].reset_index(drop=True).copy()

    # Add provenance: source file is deterministic from label + we know the dict
    # is derived from the emoji source files. Record the source split + a stable
    # id derived from text.
    train_final = train_final.copy()
    train_final["id"] = train_final["text"].apply(
        lambda t: hashlib.sha1(text_key(t).encode("utf-8")).hexdigest()[:16]
    )

    n_final = len(train_final)

    # 4. Validation/test unchanged (evaluation policy documented separately).
    val_final = val_raw.copy()
    test_final = test_raw.copy()

    # Save versioned outputs
    train_final.to_csv(processed / "stocktwits_train_dedup.csv", index=False)
    val_final.to_csv(processed / "stocktwits_validation_final.csv", index=False)
    test_final.to_csv(processed / "stocktwits_test_final.csv", index=False)

    # 5. Class weights from FINAL training only
    class_counts = Counter(train_final["label"])
    n = len(train_final)
    n_classes = 3
    class_weights = {c: n / (n_classes * class_counts[c]) for c in class_counts}
    class_names = {c: LABEL_NAMES[c] for c in class_counts}

    # 6. Manifest
    manifest = {
        "project": "Emoji-Aware Sentiment Analysis",
        "original_dataset": "ElKulako/stocktwits-emoji",
        "label_recovery_method": "Source text files (filename-encoded classes) via src/data/stocktwits_adapter.py",
        "label_encoding": {"0": "Bearish", "1": "Neutral", "2": "Bullish"},
        "conflict_strategy": "drop (conflicting-label texts removed, never guessed)",
        "deduplication_strategy": "normalize text (collapse whitespace), deduplicate exact text, keep first occurrence + provenance",
        "train_test_overlap_handling": "removed 27 exact train<->test overlapping texts from training",
        "validation_test_overlap_policy": "keep unchanged; 865-text overlap reported and documented as a dataset limitation",
        "random_seed": SEED,
        "row_counts": {
            "hf_train_rows": 211758,
            "train_after_conflict_drop": int(train_orig_rows),
            "train_duplicate_rows_removed_this_step": int(n_dup_removed),
            "train_test_overlap_removed": int(n_test_overlap_removed),
            "train_final": int(n_final),
            "validation_final": int(len(val_final)),
            "test_final": int(len(test_final)),
        },
        "class_distributions": {
            "train_original": {LABEL_NAMES[k]: int(v) for k, v in Counter(train_raw["label"]).items()},
            "train_final": {LABEL_NAMES[k]: int(v) for k, v in class_counts.items()},
            "validation_final": {LABEL_NAMES[k]: int(v) for k, v in Counter(val_final["label"]).items()},
            "test_final": {LABEL_NAMES[k]: int(v) for k, v in Counter(test_final["label"]).items()},
        },
        "class_weights_train_only": {LABEL_NAMES[k]: round(v, 5) for k, v in class_weights.items()},
        "dataset_hashes": {
            "train_final": sha1_of_df(train_final, ["text", "label"]),
            "validation_final": sha1_of_df(val_final, ["text", "label"]),
            "test_final": sha1_of_df(test_final, ["text", "label"]),
        },
        "files_created": [
            "data/processed/stocktwits_train_raw.csv",
            "data/processed/stocktwits_validation_raw.csv",
            "data/processed/stocktwits_test_raw.csv",
            "data/processed/stocktwits_train_dedup.csv",
            "data/processed/stocktwits_validation_final.csv",
            "data/processed/stocktwits_test_final.csv",
        ],
    }

    manifest_path = processed / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Report ----
    print()
    print("=" * 60)
    print("FINAL DATASET CONSTRUCTION REPORT")
    print("=" * 60)
    print("Training deduplication:")
    print(f"  HF train rows:              211758")
    print(f"  After conflict-label drop:   {train_orig_rows} (adapter output)")
    print(f"  Duplicate rows removed:      {n_dup_removed}")
    print(f"  Train<->test overlap removed:{n_test_overlap_removed}")
    print(f"  Final training rows:         {n_final}")
    print(f"  Validation (unchanged):      {len(val_final)}")
    print(f"  Test (unchanged):            {len(test_final)}")
    print()
    print("Class distribution (train original -> final):")
    for c in [0, 1, 2]:
        print(f"  {LABEL_NAMES[c]}: {Counter(train_raw['label']).get(c,0)} -> {class_counts.get(c,0)}")
    print()
    print("Class weights (train-only, final):")
    for c in [0, 1, 2]:
        print(f"  {LABEL_NAMES[c]}: {class_weights.get(c, 0):.4f}")
    print()
    print(f"Manifest written -> {manifest_path}")


if __name__ == "__main__":
    main()
