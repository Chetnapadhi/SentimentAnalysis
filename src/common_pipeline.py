"""Common pipeline module for emoji-sentiment experiments E3–E5.

Extracts duplicated logic from per-experiment training scripts into reusable
functions: data loading, frozen-BERT featurization with caching, emoji
encoding, training loop with early stopping, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.embeddings.emoji_encoder import build_emoji_vocab, encode_emoji_list, save_vocab
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history
from src.utils.seed import set_seed

LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


# ---------------------------------------------------------------------------
# Config & data loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical final train/validation/test splits."""
    train_df = pd.read_json("data/processed/canonical/final_train.jsonl", lines=True)
    val_df = pd.read_json("data/processed/canonical/final_validation.jsonl", lines=True)
    test_df = pd.read_json("data/processed/canonical/final_test.jsonl", lines=True)
    return train_df, val_df, test_df


def build_emoji_vocab_from_train(train_df: pd.DataFrame) -> dict:
    """Build emoji vocab from training split only and save to disk."""
    vocab = build_emoji_vocab(train_df["emoji_list"].tolist())
    os.makedirs("models/emoji_embeddings", exist_ok=True)
    save_vocab(vocab, "models/emoji_embeddings/emoji_vocab.json")
    print(f"Emoji vocab size (train-only): {vocab['vocab_size']}")
    return vocab


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels."""
    from collections import Counter
    counts = Counter(train_df["label"])
    total = sum(counts.values())
    weights = [total / (3.0 * counts[c]) for c in [0, 1, 2]]
    cw = torch.tensor(weights, dtype=torch.float32)
    print("Class weights (train-only):", cw.tolist())
    return cw


# ---------------------------------------------------------------------------
# Frozen-BERT featurization with caching
# ---------------------------------------------------------------------------

def _featurize_texts(
    texts: list[str], tokenizer, encoder, max_length: int,
    batch_size: int, device,
) -> np.ndarray:
    """Encode texts with frozen BERT + mean pooling. Returns (N, 768)."""
    embeds = []
    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Text featurize"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            out = encoder(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeds.append(pooled.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def get_or_cache_bert_embeddings(
    df: pd.DataFrame,
    split_name: str,
    tokenizer,
    encoder,
    max_length: int,
    batch_size: int,
    device,
    cache_dir: str,
) -> np.ndarray:
    """Return cached BERT embeddings or featurize and cache."""
    os.makedirs(cache_dir, exist_ok=True)
    texts = df["text_without_emoji"].tolist()
    text_hash = hashlib.sha1("".join(texts).encode("utf-8")).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"text_{split_name}_{text_hash}.npy")

    if os.path.exists(cache_file):
        print(f"Loaded cached {split_name} embeddings from {cache_file}")
        return np.load(cache_file)

    print(f"Featurizing {split_name} texts ({len(texts)} examples)...")
    embeds = _featurize_texts(texts, tokenizer, encoder, max_length, batch_size, device)
    np.save(cache_file, embeds)
    print(f"Saved {split_name} embeddings cache -> {cache_file}")
    return embeds


# ---------------------------------------------------------------------------
# Emoji encoding
# ---------------------------------------------------------------------------

def encode_emojis(
    df: pd.DataFrame, emoji_to_id: dict, max_emojis: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode emoji lists into padded IDs + masks."""
    ids_list, masks_list = [], []
    for emojis in df["emoji_list"].tolist():
        ids, mask = encode_emoji_list(emojis, emoji_to_id, max_emojis)
        ids_list.append(ids)
        masks_list.append(mask)
    return np.array(ids_list), np.array(masks_list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmojiDataset(Dataset):
    """Dataset producing (text_embed, emoji_ids, emoji_masks, label) tuples."""

    def __init__(
        self,
        text_embeds: np.ndarray,
        emoji_ids: np.ndarray,
        emoji_masks: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
        self.emoji_ids = torch.tensor(emoji_ids, dtype=torch.long)
        self.emoji_masks = torch.tensor(emoji_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.text_embeds[idx],
            self.emoji_ids[idx],
            self.emoji_masks[idx],
            self.labels[idx],
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    model: nn.Module,
    forward_fn,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: dict,
    device,
    class_weights: torch.Tensor,
    out_dir: str,
) -> dict:
    """Generic training loop across all epochs (no early stopping).
    Best checkpoint selected on validation Macro-F1.

    Parameters
    ----------
    forward_fn : callable
        ``forward_fn(model, text_emb, emoji_ids, emoji_masks) -> logits``
    """
    os.makedirs(out_dir, exist_ok=True)

    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 1e-4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [],
    }
    best_val_f1 = -1.0
    best_epoch = -1

    model.to(device)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        for text_emb, e_ids, e_mask, lab in train_loader:
            text_emb, lab = text_emb.to(device), lab.to(device)
            e_ids, e_mask = e_ids.to(device), e_mask.to(device)
            optimizer.zero_grad()
            logits = forward_fn(model, text_emb, e_ids, e_mask)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for text_emb, e_ids, e_mask, lab in val_loader:
                text_emb, lab = text_emb.to(device), lab.to(device)
                e_ids, e_mask = e_ids.to(device), e_mask.to(device)
                logits = forward_fn(model, text_emb, e_ids, e_mask)
                val_loss_sum += criterion(logits, lab).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_true.extend(lab.cpu().numpy().tolist())
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = accuracy_score(all_true, all_preds)
        val_mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_mf1)

        print(
            f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | "
            f"ValMacroF1 {val_mf1:.4f}"
        )

        if val_mf1 > best_val_f1:
            best_val_f1 = val_mf1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": val_mf1,
                    "best_epoch": epoch,
                    "class_weights": class_weights.tolist(),
                },
                os.path.join(out_dir, "best_model.pt"),
            )
            print(f"  ✓ Saved new best checkpoint (Val Macro-F1: {val_mf1:.4f})")

    train_time = time.time() - start_time
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "training_time_sec": train_time,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    forward_fn,
    test_ds: Dataset,
    test_df: pd.DataFrame,
    train_result: dict,
    cfg: dict,
    device,
    out_dir: str,
    experiment_name: str,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
) -> dict:
    """Full evaluation pipeline: metrics, predictions, plots, error analysis.

    Parameters
    ----------
    train_result : dict
        Output of ``train_loop`` (contains history for plotting).
    """
    # Load best checkpoint
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt.get('best_epoch', '?')}")

    model.to(device)
    model.eval()
    batch_size = cfg["training"]["batch_size"]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for text_emb, e_ids, e_mask, _ in test_loader:
            text_emb = text_emb.to(device)
            e_ids, e_mask = e_ids.to(device), e_mask.to(device)
            logits = forward_fn(model, text_emb, e_ids, e_mask)
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
        y_true, y_pred, labels=[0, 1, 2],
        target_names=["Bearish", "Neutral", "Bullish"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print()
    print("=" * 60)
    print(f"{experiment_name} TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro P:    {macro_p:.4f}")
    print(f"Macro R:    {macro_r:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")

    # Save metrics
    seed = cfg["project"]["seed"]
    dataset_sizes = {"test": len(test_df)}
    if train_df is not None:
        dataset_sizes["train"] = len(train_df)
    if val_df is not None:
        dataset_sizes["validation"] = len(val_df)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "classification_report": per_class,
        "confusion_matrix": cm.tolist(),
        "best_epoch": train_result["best_epoch"],
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "training_time_sec": train_result.get("training_time_sec"),
        "dataset_sizes": dataset_sizes,
        "class_weights": ckpt.get("class_weights") if os.path.exists(ckpt_path) else None,
        "seed": seed,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save predictions CSV
    pred_df = test_df.copy()
    pred_df["predicted_label"] = y_pred
    pred_df["pred_conf_bearish"] = y_prob[:, 0]
    pred_df["pred_conf_neutral"] = y_prob[:, 1]
    pred_df["pred_conf_bullish"] = y_prob[:, 2]
    pred_df["correct"] = y_true == y_pred
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Plots
    plot_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"))
    plot_training_history(
        train_result["history"], os.path.join(out_dir, "training_history.png"),
    )

    # Error analysis (top 200 misclassified)
    errors = pred_df[~pred_df["correct"]].copy()
    errors = errors.head(200)
    errors.to_csv(os.path.join(out_dir, "error_analysis.csv"), index=False)
    print(f"\nSaved {len(errors)} error examples -> {out_dir}/error_analysis.csv")
    print(f"Done. All {experiment_name} artifacts in {out_dir}/")

    return metrics


# ---------------------------------------------------------------------------
# Convenience: prepare_experiment
# ---------------------------------------------------------------------------

def prepare_experiment(experiment_name: str, cfg: dict | None = None) -> dict:
    """Run full data preparation pipeline (no BERT featurization).

    Returns dict with cfg, device, train_df, val_df, test_df, vocab,
    class_weights, out_dir.
    """
    if cfg is None:
        cfg = load_config()

    seed = cfg["project"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_splits()
    vocab = build_emoji_vocab_from_train(train_df)
    class_weights = compute_class_weights(train_df)

    out_dir = f"results/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    return {
        "cfg": cfg,
        "device": device,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "vocab": vocab,
        "class_weights": class_weights,
        "out_dir": out_dir,
    }
