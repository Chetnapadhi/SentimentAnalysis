"""E0 training + evaluation script.

Frozen-BERT featurization is precomputed and cached once, then only the
classification head is trained — this is GPU-light and Colab-feasible on 91K
examples. The same cached embeddings are reused for all E0 runs.

Input for E0 is ``text_without_emoji`` (emojis explicitly removed).

GUARDED: requires RUN_E0=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E0=1 to prevent accidental execution
if os.environ.get("RUN_E0") != "1":
    print("GUARDED: E0 requires RUN_E0=1 environment variable.")
    print("Run as: RUN_E0=1 python -m src.train")
    sys.exit(0)

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.text_only import TextOnlyModel
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history
from src.utils.seed import set_seed

LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Featurization (cached frozen-BERT embeddings)
# ---------------------------------------------------------------------------

def featurize(
    texts: list[str],
    tokenizer,
    encoder,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode texts with frozen BERT + mean pooling. Returns (N, hidden)."""
    embeds = []
    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Featurizing"):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()
            embeds.append(pooled)
    return np.concatenate(embeds, axis=0)


def cache_path_for(cache_dir: str, split: str, cfg: dict, hash_str: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeds_{split}_{cfg['text_model']['name'].replace('/', '_')}_{hash_str[:8]}.npy")


# ---------------------------------------------------------------------------
# Training loop (head only)
# ---------------------------------------------------------------------------

def train_head(
    model: nn.Module,
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    val_embeds: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
    class_weights: torch.Tensor,
    out_dir: str,
) -> dict:
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]

    train_ds = TensorDataset(
        torch.tensor(train_embeds, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(val_embeds, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_epoch = -1
    best_val_f1 = -1.0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for emb, lab in train_loader:
            emb, lab = emb.to(device), lab.to(device)
            optimizer.zero_grad()
            logits = model.classifier(emb)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for emb, lab in val_loader:
                emb, lab = emb.to(device), lab.to(device)
                logits = model.classifier(emb)
                vloss = criterion(logits, lab)
                val_loss += vloss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(lab.cpu().numpy().tolist())
        val_loss = val_loss / max(len(val_loader), 1)
        val_acc = accuracy_score(all_true, all_preds)
        val_mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_mf1)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"ValAcc {val_acc:.4f} | "
            f"ValMacroF1 {val_mf1:.4f}"
        )

        if val_mf1 > best_val_f1:
            best_val_f1 = val_mf1
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.classifier.state_dict(),
                    "encoder_config": model.model_name,
                    "best_val_f1": val_mf1,
                    "best_epoch": epoch,
                    "class_weights": class_weights.tolist(),
                },
                os.path.join(out_dir, "best_model.pt"),
            )

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_cfg = cfg["text_model"]
    model_name = text_cfg["name"]
    max_length = text_cfg["max_length"]
    batch_size = cfg["training"]["batch_size"]

    # Output dir
    out_dir = "results/E0"
    os.makedirs(out_dir, exist_ok=True)

    # Save config snapshot for reproducibility
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f)

    # 1. Load canonical final splits
    train_df = pd.read_json("data/processed/canonical/final_train.jsonl", lines=True)
    val_df = pd.read_json("data/processed/canonical/final_validation.jsonl", lines=True)
    test_df = pd.read_json("data/processed/canonical/final_test.jsonl", lines=True)

    # 2. Class weights from final training only
    counts = Counter(train_df["label"])
    class_weights = torch.tensor(
        [counts[c] for c in [0, 1, 2]], dtype=torch.float32
    )
    # inverse frequency normalized
    total = sum(counts.values())
    n_classes = 3
    class_weights = torch.tensor(
        [total / (n_classes * counts[c]) for c in [0, 1, 2]], dtype=torch.float32
    )
    print("Class weights (train-only):", class_weights.tolist())

    # 3. Tokenizer + encoder (frozen)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextOnlyModel(
        model_name=model_name,
        hidden_size=768,
        num_labels=3,
        freeze_encoder=True,
    )
    encoder = model.encoder.to(device)

    # 4. Featurize (cache)
    hash_str = hashlib.sha1(
        (train_df["text_without_emoji"].sum() + val_df["text_without_emoji"].sum()
         + test_df["text_without_emoji"].sum()).encode("utf-8")
    ).hexdigest()

    cache_dir = "results/E0/embeddings_cache"
    train_emb = None
    cache_path = cache_path_for(cache_dir, "train", cfg, hash_str)
    if os.path.exists(cache_path):
        train_emb = np.load(cache_path)
        print("Loaded cached train embeddings")
    else:
        train_emb = featurize(
            train_df["text_without_emoji"].tolist(), tokenizer, encoder,
            max_length, batch_size, device,
        )
        np.save(cache_path, train_emb)
        print("Saved train embeddings cache")

    val_path = cache_path_for(cache_dir, "validation", cfg, hash_str)
    if os.path.exists(val_path):
        val_emb = np.load(val_path)
    else:
        val_emb = featurize(
            val_df["text_without_emoji"].tolist(), tokenizer, encoder,
            max_length, batch_size, device,
        )
        np.save(val_path, val_emb)

    test_path = cache_path_for(cache_dir, "test", cfg, hash_str)
    if os.path.exists(test_path):
        test_emb = np.load(test_path)
    else:
        test_emb = featurize(
            test_df["text_without_emoji"].tolist(), tokenizer, encoder,
            max_length, batch_size, device,
        )
        np.save(test_path, test_emb)

    # 5. Train head
    start = time.time()
    train_result = train_head(
        model,
        train_emb, train_df["label"].to_numpy(),
        val_emb, val_df["label"].to_numpy(),
        cfg, device, class_weights, out_dir,
    )
    train_time = time.time() - start

    # 6. Load best checkpoint + evaluate on test
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"), map_location="cpu")
    model.classifier.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(test_emb, dtype=torch.float32),
        torch.tensor(test_df["label"].to_numpy(), dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for emb, _ in test_loader:
            emb = emb.to(device)
            logits = model.classifier(emb)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = test_df["label"].to_numpy()
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=["Bearish", "Neutral", "Bullish"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print()
    print("=" * 60)
    print("E0 TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro P:    {macro_p:.4f}")
    print(f"Macro R:    {macro_r:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")

    # 7. Save everything
    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "classification_report": per_class,
        "confusion_matrix": cm.tolist(),
        "best_epoch": train_result["best_epoch"],
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "training_time_sec": train_time,
        "dataset_sizes": {
            "train": len(train_df), "validation": len(val_df), "test": len(test_df),
        },
        "class_weights": class_weights.tolist(),
        "seed": seed,
        "dataset_hash": hash_str,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Predictions CSV
    test_df = test_df.copy()
    test_df["predicted_label"] = y_pred
    test_df["pred_conf_bearish"] = y_prob[:, 0]
    test_df["pred_conf_neutral"] = y_prob[:, 1]
    test_df["pred_conf_bullish"] = y_prob[:, 2]
    test_df["correct"] = (y_true == y_pred)
    test_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Plots
    plot_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"))
    plot_training_history(train_result["history"], os.path.join(out_dir, "training_history.png"))

    # Error analysis (50+ misclassified)
    errors = test_df[~test_df["correct"]].copy()
    errors = errors.sort_values("correct").head(200)
    errors.to_csv(os.path.join(out_dir, "error_analysis.csv"), index=False)
    print(f"\nSaved {len(errors)} error examples -> results/E0/error_analysis.csv")

    # 8. Short/medium/long + emoji-count analysis
    print("\nShort/Medium/Long analysis:")
    buckets = pd.cut(
        test_df["token_count"],
        bins=[-1, 5, 15, 10**9],
        labels=["short", "medium", "long"],
    )
    test_df["bucket"] = buckets
    for b in ["short", "medium", "long"]:
        sub = test_df[test_df["bucket"] == b]
        if len(sub):
            sub_f1 = f1_score(sub["label"], sub["predicted_label"], average="macro", zero_division=0)
            sub_acc = accuracy_score(sub["label"], sub["predicted_label"])
            print(f"  {b:7s}: n={len(sub):5d} acc={sub_acc:.4f} macroF1={sub_f1:.4f}")

    print("\nEmoji-count analysis (original emojis):")
    for ec in [1, 2, 3]:
        sub = test_df[test_df["num_emojis"] == ec] if ec < 3 else test_df[test_df["num_emojis"] >= 3]
        if len(sub):
            sub_f1 = f1_score(sub["label"], sub["predicted_label"], average="macro", zero_division=0)
            sub_acc = accuracy_score(sub["label"], sub["predicted_label"])
            print(f"  {ec} emoji(s): n={len(sub):5d} acc={sub_acc:.4f} macroF1={sub_f1:.4f}")

    print(f"\nDone. All E0 artifacts in results/E0/")


if __name__ == "__main__":
    main()
