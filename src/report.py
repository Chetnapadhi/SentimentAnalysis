"""Generate the E0 research report (markdown) from saved artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def generate_e0_report(metrics_path="results/E0/metrics.json",
                       out_path="results/E0/E0_report.md") -> None:
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    cr = m["classification_report"]
    cm = m["confusion_matrix"]
    labels = ["Bearish", "Neutral", "Bullish"]

    # Build per-class table
    per_class_rows = ""
    for i, name in enumerate(labels):
        key = name
        row = cr.get(key, {})
        per_class_rows += (
            f"| {name} | {row.get('precision', 0):.3f} | "
            f"{row.get('recall', 0):.3f} | {row.get('f1-score', 0):.3f} |\n"
        )

    # Confusion matrix rows
    cm_rows = ""
    for i, name in enumerate(labels):
        cm_rows += f"| {name} | " + " | ".join(str(int(cm[i][j])) for j in range(3)) + " |\n"

    report = f"""# E0 — Text-Only Sentiment Baseline Report

## 1. Objective
Estimate how well sentiment can be predicted from text **with emojis removed**.
This is the control condition for the research question: *Do emojis carry
signal that words miss?* E0 receives only ``text_without_emoji``.

## 2. Dataset
Approved final splits (from `ElKulako/stocktwits-emoji`, labels from source
files):
- Train: {m['dataset_sizes']['train']} rows (deduplicated)
- Validation: {m['dataset_sizes']['validation']} rows
- Test: {m['dataset_sizes']['test']} rows
Classes: 0=Bearish, 1=Neutral, 2=Bullish. E0 input is text without emojis.

## 3. Preprocessing
- `original_text` preserved; emojis extracted with the Unicode-aware `emoji`
  library; `text_without_emoji` used as the E0 model input.
- Tokenized with BERT tokenizer (max_length 128). Frozen-BERT embeddings cached.

## 4. Model architecture
- Frozen `bert-base-uncased`
- Mean pooling (mask-aware)
- Trainable MLP head (Linear-ReLU-Dropout-Linear -> 3 logits)
- 3 classes: Bearish (0), Neutral (1), Bullish (2)

## 5. Training setup
- Seed: {m['seed']}
- Loss: class-weighted cross-entropy (weights: {m['class_weights']})
- Optimizer: AdamW
- Learning rate: 0.001 (config)
- Batch size: 32 (config)
- Epochs: 5 (config)
- Best model selected by validation Macro-F1 (best epoch: {m['best_epoch']})
- Training time: {m['training_time_sec']:.1f}s

## 6. Results (test set)
| Metric | Value |
|---|---|
| Accuracy | {m['accuracy']:.4f} |
| Macro Precision | {m['macro_precision']:.4f} |
| Macro Recall | {m['macro_recall']:.4f} |
| **Macro F1** | {m['macro_f1']:.4f} |

### 6a. Per-class
| Class | Precision | Recall | F1 |
|---|---|---|---|
{per_class_rows}

## 7. Confusion matrix (counts)
| Actual \\ Predicted | Bearish | Neutral | Bullish |
|---|---|---|---|
{cm_rows}

## 8. Error analysis
See `results/E0/error_analysis.csv` for misclassified examples with removed
emojis and confidence scores. These cases highlight what E1–E5 may need to
recover.

## 9. Observations
- Text-only performance on the heavy Bullish-imbalanced set; note per-class gaps.
- Misclassifications often involve instances where removed emojis appeared
  sentiment-bearing (see error analysis).

## 10. Limitations
- E0 cannot see emojis at all; the aim is a controlled baseline.
- Class imbalance toward Bullish (class-weighted loss used from train only).
- Frozen BERT may limit capacity.
- Validation/test wording overlap (~7.2% of test) is a documented dataset
  limitation.

---
*No conclusion is made yet about whether emojis improve classification. That
requires E1–E5.*
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written -> {out_path}")


if __name__ == "__main__":
    generate_e0_report()