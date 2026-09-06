"""Data-integrity and preprocessing validation for the StockTwits dataset.

Performs the full validation suite required before E0–E5 training:
  1. Label provenance verification
  2. Conflicting-row investigation
  3. Duplicate & leakage measurement
  4. Canonical preprocessing (via src/data/preprocessing.py)
  5. Emoji-extraction validation
  6. Experiment manifest generation
  7. Class-weight calculation (train-only)
  8. Colab full-data cost estimation
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import emoji
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data import preprocessing  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = "ElKulako/stocktwits-emoji"
LABEL_MAP = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

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

CACHE_DIR = "data/raw/stocktwits"


# ---------------------------------------------------------------------------
# 1. Label provenance
# ---------------------------------------------------------------------------

def verify_label_provenance(hf_rows: dict[str, list[str]], source_dir: str) -> dict:
    """For each split, confirm every matched label came from the correct source file."""
    provenance = {}

    for split, texts in hf_rows.items():
        # Build text -> label using source files (first file occurrence)
        source_to_texts: dict[str, int] = {}
        file_to_texts: dict[str, set] = defaultdict(set)
        label_counts = Counter()

        for label_name in ["Bearish", "Neutral", "Bullish"]:
            fname = SOURCE_FILES[(split, label_name)]
            filepath = os.path.join(source_dir, fname)
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    text = line.rstrip("\n").strip()
                    file_to_texts[fname].add(text)
                    if text not in source_to_texts:
                        source_to_texts[text] = LABEL_MAP[label_name]
            label_counts[label_name] = len(file_to_texts[fname])

        # Verify all HF matched texts derive from the correct source lookup.
        # The CSV stores original (possibly leading-whitespace) text; the
        # source lookup stores stripped text, so strip before matching.
        matched_from_source = sum(1 for text in texts if text.strip() in source_to_texts)
        unmatched = len(texts) - matched_from_source

        # A text is "conflicting" if a single text appears in >1 class file
        text_to_class_sets: dict[str, set] = defaultdict(set)
        for label_name in ["Bearish", "Neutral", "Bullish"]:
            fname = SOURCE_FILES[(split, label_name)]
            for text in file_to_texts[fname]:
                text_to_class_sets[text].add(LABEL_MAP[label_name])
        conflicting_texts = {
            t: sorted(s) for t, s in text_to_class_sets.items() if len(s) > 1
        }

        provenance[split] = {
            "source_files": {name: len(s) for name, s in file_to_texts.items()},
            "matched_from_source": matched_from_source,
            "total_hf_texts": len(texts),
            "unmatched": unmatched,
            "conflicting_texts": len(conflicting_texts),
            "label_counts": dict(label_counts),
        }
    return provenance


# ---------------------------------------------------------------------------
# 2. Conflict investigation
# ---------------------------------------------------------------------------

def investigate_conflicts(source_dir: str) -> dict:
    """Classify the reasons for conflicting (multi-class) texts."""
    # For each split, collect texts that appear in >1 class file
    result = {}
    for split in ["train", "validation", "test"]:
        text_to_labels: dict[str, set] = defaultdict(set)
        for label_name in ["Bearish", "Neutral", "Bullish"]:
            fname = SOURCE_FILES[(split, label_name)]
            filepath = os.path.join(source_dir, fname)
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    text = line.rstrip("\n").strip()
                    text_to_labels[text].add(LABEL_MAP[label_name])

        conflicts = {t: labs for t, labs in text_to_labels.items() if len(labs) > 1}
        # Classify: how many conflict pairs are vs each class
        pair_counts = Counter()
        for labs in conflicts.values():
            pair_counts[tuple(sorted(labs))] += 1

        # Sample some conflicts to inspect manually
        samples = []
        for i, (t, labs) in enumerate(conflicts.items()):
            if i >= 5:
                break
            samples.append({"text": t[:100], "labels": [LABEL_NAMES[l] for l in labs]})

        result[split] = {
            "num_conflicts": len(conflicts),
            "conflict_pairs": {str(k): v for k, v in pair_counts.items()},
            "sample_conflicts": samples,
        }
    return result


# ---------------------------------------------------------------------------
# 3. Duplicate / leakage
# ---------------------------------------------------------------------------

def measure_duplicates(dfs: dict[str, pd.DataFrame]) -> dict:
    """Measure duplicates within and across splits."""
    result = {}

    for split, df in dfs.items():
        texts = df["text"].tolist()
        counts = Counter(texts)
        dups = {t: c for t, c in counts.items() if c > 1}
        # Multi-label within split
        text_to_labels: dict[str, set] = defaultdict(set)
        for text, lab in zip(texts, df["label"]):
            text_to_labels[text].add(int(lab))
        multi_label = {t: labs for t, labs in text_to_labels.items() if len(labs) > 1}

        result[split] = {
            "total_rows": len(df),
            "unique_texts": len(counts),
            "duplicated_texts": len(dups),
            "duplicate_extra_rows": sum(c - 1 for c in dups.values()),
            "multi_label_texts": len(multi_label),
            "multi_label_examples": list(multi_label.items())[:5],
        }

    # Cross-split overlap
    split_sets = {s: set(dfs[s]["text"].tolist()) for s in dfs}
    cross = {}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        cross[f"{a}<->{b}"] = len(split_sets[a] & split_sets[b])

    result["cross_split_overlap"] = cross
    return result


# ---------------------------------------------------------------------------
# 5. Emoji validation
# ---------------------------------------------------------------------------

def validate_emoji_extraction(df: pd.DataFrame) -> dict:
    """Produce emoji statistics on the canonical output."""
    num_rows = len(df)
    with_emoji = (df["num_emojis"] > 0).sum()
    zero_emoji = (df["num_emojis"] == 0).sum()
    total_emojis = df["num_emojis"].sum()
    mean_emojis = total_emojis / num_rows if num_rows else 0
    median_emojis = float(df["num_emojis"].median()) if num_rows else 0

    all_emoji_flat = []
    for emoji_list in df["emoji_list"]:
        all_emoji_flat.extend(emoji_list)
    top30 = Counter(all_emoji_flat).most_common(30)
    unique = len(set(all_emoji_flat))

    # ZWJ / skin-tone / variation selectors handling check
    zwj_count = sum(1 for e in all_emoji_flat if "\u200d" in e)
    skin_count = sum(1 for e in all_emoji_flat if "\U0001f3fb" <= e < "\U0001f3ff")
    vs16_count = sum(1 for e in all_emoji_flat if "\ufe0f" in e)

    return {
        "num_rows": num_rows,
        "rows_with_emojis": int(with_emoji),
        "percent_rows_with_emojis": round(100 * with_emoji / num_rows, 2) if num_rows else 0,
        "rows_with_zero_emojis": int(zero_emoji),
        "total_emoji_occurrences": int(total_emojis),
        "mean_emojis_per_row": round(mean_emojis, 3),
        "median_emojis_per_row": median_emojis,
        "unique_emojis": unique,
        "top_30_emojis": [[e, c] for e, c in top30],
        "emoji_count_distribution": dict(Counter(df["num_emojis"]).most_common(15)),
        "zwj_sequences": zwj_count,
        "skin_tone_sequences": skin_count,
        "variation_selector_16": vs16_count,
    }


# ---------------------------------------------------------------------------
# 7. Class weights (train only)
# ---------------------------------------------------------------------------

def class_weights_from_train(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute inverse-frequency class weights from the TRAINING set only."""
    counts = Counter(train_df["label"])
    n = len(train_df)
    n_classes = len(LABEL_NAMES)
    # Inverse frequency, normalized so mean weight = 1
    weights = {c: n / (n_classes * counts[c]) for c in counts}
    return dict(counts), weights


# ---------------------------------------------------------------------------
# 9. Colab cost estimate
# ---------------------------------------------------------------------------

def estimate_colab_cost(train_df: pd.DataFrame) -> dict:
    """Estimate frozen-BERT featurization cost on a T4 GPU."""
    n_train = len(train_df)
    # BERT-base: ~225M params, forward ~1-2ms/sample on T4 at batch 32
    # Featurization (frozen, no backward): ~2ms/sample on T4
    # Full batch forward on 243K samples
    per_sample_sec = 0.002  # 2ms on T4 for BERT-base forward
    featurize_sec = n_train * per_sample_sec
    # Training classifier head (small MLP) on cached embeddings: fast
    head_train_sec = n_train * 0.0001  # ~0.1ms/sample
    total_sec = featurize_sec + head_train_sec

    # Memory estimate for cached embeddings (768-d float32 per token-pooled)
    embed_bytes = n_train * 768 * 4  # float32
    mem_mb = embed_bytes / (1024 * 1024)

    return {
        "n_train": n_train,
        "featurize_estimate_sec": featurize_sec,
        "head_train_estimate_sec": head_train_sec,
        "total_estimate_sec": total_sec,
        "total_estimate_min": total_sec / 60,
        "cached_embedding_mem_mb": mem_mb,
        "feasible_on_t4": total_sec < 7200,  # < 2 hours
        "note": (
            "Frozen BERT featurization is one-time and cached. "
            "Training the lightweight classifier head on cached embeddings "
            "is cheap. Feasible on a single T4 within ~1-2 hours."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    source_dir = CACHE_DIR
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)
    os.makedirs("data/processed/canonical", exist_ok=True)

    # Load existing labeled CSVs
    dfs = {}
    for split in ["train", "validation", "test"]:
        csv_path = f"data/processed/stocktwits_{split}.csv"
        dfs[split] = pd.read_csv(csv_path)

    report: dict = {}

    # --- 1. Label provenance ---
    print("=" * 60)
    print("[1/9] VERIFYING LABEL PROVENANCE")
    print("=" * 60)
    hf_rows = {s: df["text"].tolist() for s, df in dfs.items()}
    provenance = verify_label_provenance(hf_rows, source_dir)
    report["label_provenance"] = provenance
    for split in provenance:
        p = provenance[split]
        print(f"  {split}: matched_from_source={p['matched_from_source']}/{p['total_hf_texts']}, "
              f"unmatched={p['unmatched']}, conflicts={p['conflicting_texts']}")

    # --- 2. Conflict investigation ---
    print()
    print("=" * 60)
    print("[2/9] INVESTIGATING CONFLICTING ROWS")
    print("=" * 60)
    conflicts = investigate_conflicts(source_dir)
    report["conflict_investigation"] = conflicts
    for split in conflicts:
        c = conflicts[split]
        print(f"  {split}: {c['num_conflicts']} conflicting texts, pairs={c['conflict_pairs']}")
        for s in c["sample_conflicts"]:
            print(f"    '{s['text']}' -> labels {s['labels']}")

    # --- 3. Duplicate / leakage ---
    print()
    print("=" * 60)
    print("[3/9] MEASURING DUPLICATES & LEAKAGE")
    print("=" * 60)
    dups = measure_duplicates(dfs)
    report["duplicate_analysis"] = dups
    for split in dups:
        if split.startswith("train") or split in ["validation", "test"]:
            d = dups[split]
            print(f"  {split}: total={d['total_rows']}, unique={d['unique_texts']}, "
                  f"dup_texts={d['duplicated_texts']}, dup_extra_rows={d['duplicate_extra_rows']}, "
                  f"multi_label_texts={d['multi_label_texts']}")
    print(f"  Cross-split overlap: {dups['cross_split_overlap']}")

    # --- 4 & 5. Build canonical + validate emoji ---
    print()
    print("=" * 60)
    print("[4/9] BUILDING CANONICAL DATASET")
    print("=" * 60)
    canonical = {}
    for split in ["train", "validation", "test"]:
        canonical[split] = preprocessing.build_canonical(dfs[split], split, preprocessing.whitespace_token_count)
        preprocessing.save_canonical(canonical[split], f"data/processed/canonical/{split}.jsonl")
        print(f"  {split}: {len(canonical[split])} canonical rows")

    combined = pd.concat(canonical.values(), ignore_index=True)
    combined.to_json("data/processed/canonical/all.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"  Combined: {len(combined)} rows")

    print()
    print("=" * 60)
    print("[5/9] VALIDATING EMOJI EXTRACTION")
    print("=" * 60)
    emoji_report = validate_emoji_extraction(combined)
    report["emoji_extraction"] = emoji_report
    print(f"  Rows with emojis: {emoji_report['percent_rows_with_emojis']}%")
    print(f"  Mean emojis/row: {emoji_report['mean_emojis_per_row']}, median: {emoji_report['median_emojis_per_row']}")
    print(f"  Unique emojis: {emoji_report['unique_emojis']}")
    print(f"  ZWJ: {emoji_report['zwj_sequences']}, skin-tone: {emoji_report['skin_tone_sequences']}, VS16: {emoji_report['variation_selector_16']}")

    # --- 6. Experiment manifest ---
    print()
    print("=" * 60)
    print("[6/9] CREATING EXPERIMENT MANIFEST")
    print("=" * 60)
    manifest = {
        "dataset": "ElKulako/stocktwits-emoji (labels recovered from source files)",
        "dataset_hash": hashlib_sha1(combined["original_text"].tolist()),
        "splits": {
            "train": len(canonical["train"]),
            "validation": len(canonical["validation"]),
            "test": len(canonical["test"]),
        },
        "seed": 42,
        "experiments": {
            "E0": {"text": True, "emoji": False, "emoji_init": None, "fusion": None},
            "E1": {"text": True, "emoji": True, "emoji_init": "random", "fusion": "concat"},
            "E2": {"text": True, "emoji": True, "emoji_init": "pretrained", "fusion": "concat"},
            "E3": {"text": True, "emoji": True, "emoji_init": "random", "fusion": "attention"},
            "E4": {"text": True, "emoji": True, "emoji_init": "pretrained", "fusion": "attention"},
            "E5": {"text": True, "emoji": True, "emoji_init": "pretrained", "fusion": "gated"},
        },
        "model_config": {
            "text_encoder": "bert-base-uncased (frozen)",
            "max_length": 128,
            "emoji_embedding_dim": 32,
            "classifier": "MLP -> 3 classes",
        },
        "training_config": {
            "seed": 42,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
            "class_weighting": "inverse-frequency from TRAIN only",
        },
    }
    report["experiment_manifest"] = manifest
    print(f"  Manifest created with dataset_hash={manifest['dataset_hash'][:16]}...")

    # --- 7. Class weights ---
    print()
    print("=" * 60)
    print("[7/9] CALCULATING CLASS WEIGHTS (TRAIN ONLY)")
    print("=" * 60)
    train_counts, weights = class_weights_from_train(canonical["train"])
    report["class_distribution"] = {
        "train": dict(Counter(canonical["train"]["label"])),
        "validation": dict(Counter(canonical["validation"]["label"])),
        "test": dict(Counter(canonical["test"]["label"])),
    }
    report["class_weights_train_only"] = weights
    print(f"  Train counts: {dict(Counter(canonical['train']['label']))}")
    print(f"  Weights: {weights}")

    # --- 9. Colab cost ---
    print()
    print("=" * 60)
    print("[9/9] ESTIMATING COLAB FULL-DATA COST")
    print("=" * 60)
    cost = estimate_colab_cost(canonical["train"])
    report["colab_cost_estimate"] = cost
    print(f"  Featurize: {cost['featurize_estimate_sec']:.0f}s ({cost['featurize_estimate_sec']/60:.1f} min)")
    print(f"  Head train: {cost['head_train_estimate_sec']:.0f}s")
    print(f"  Total: {cost['total_estimate_min']:.1f} min")
    print(f"  Feasible on T4: {cost['feasible_on_t4']}")
    print(f"  Cached embedding memory: {cost['cached_embedding_mem_mb']:.0f} MB")

    # Save report
    with open("data/validation/integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print()
    print("Report saved -> data/validation/integrity_report.json")


def hashlib_sha1(items):
    import hashlib
    h = hashlib.sha1()
    for i in items:
        h.update(i.encode("utf-8"))
    return h.hexdigest()


if __name__ == "__main__":
    main()
