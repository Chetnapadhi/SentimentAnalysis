"""Shared pipeline for the Phase 2 emotion experiments (EM0 / EM3 / EM5).

Phase 1 (E0-E5) answered "bullish / bearish / neutral" on StockTwits with a
fully frozen BERT and a cached-embedding head. Phase 2 keeps that protocol for
the *controlled* emoji comparison, but adds the two upgrades the supervisor
asked for:

1. **Partial unfreezing** -- the top ``N`` transformer layers train at a low
   learning rate while the rest of the encoder stays frozen.
2. **Domain-matched backbone** -- ``cardiffnlp/twitter-roberta-base`` instead
   of ``bert-base-uncased``.

Two training modes are provided:

``frozen`` (cached)
    Backbone is frozen, embeddings are computed once and cached, only the
    fusion + head train. Cheap, and it is the mode that makes EM0 / EM3 / EM5
    a *controlled* comparison: the text representation is byte-identical
    across the three, so any delta is attributable to the emoji branch.

``finetune`` (end-to-end)
    Top ``N`` encoder layers + fusion + head train together. This is where the
    accuracy gain comes from, at roughly 100x the compute.

Backbone choice note
--------------------
The supervisor suggested ``cardiffnlp/twitter-roberta-base-emotion``. That
checkpoint is *fine-tuned on TweetEval emotion*, i.e. on the exact task and
test split we evaluate on, so using it would leak the test set and inflate the
result. We therefore use ``cardiffnlp/twitter-roberta-base``, which has the
same Twitter pretraining but no emotion-task supervision.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.emotion_adapter import EKMAN6_NAMES, IRONY_NAMES, TWEETEVAL_NAMES, load_emotion_splits
from src.embeddings.emoji_encoder import build_emoji_vocab, encode_emoji_list, save_vocab
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_LABELS: dict[str, list[str]] = {
    "goemotions": EKMAN6_NAMES,
    "tweeteval": TWEETEVAL_NAMES,
    "irony": IRONY_NAMES,
}

DEFAULT_BACKBONE = "cardiffnlp/twitter-roberta-base"


def label_names_for(dataset: str) -> list[str]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASET_LABELS)}")
    return DATASET_LABELS[dataset]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def compute_class_weights(train_df: pd.DataFrame, num_labels: int) -> torch.Tensor:
    """Inverse-frequency class weights from the training split only."""
    counts = Counter(train_df["label"])
    total = sum(counts.values())
    weights = [
        total / (num_labels * counts[c]) if counts.get(c) else 0.0
        for c in range(num_labels)
    ]
    cw = torch.tensor(weights, dtype=torch.float32)
    print("Class weights (train-only):", [round(w, 3) for w in cw.tolist()])
    return cw


def build_vocab(train_df: pd.DataFrame, dataset: str) -> dict:
    """Emoji vocabulary from the training split only (no eval leakage)."""
    vocab = build_emoji_vocab(train_df["emoji_list"].tolist())
    out = f"models/emoji_embeddings/emoji_vocab_{dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_vocab(vocab, out)
    print(f"Emoji vocab size (train-only): {vocab['vocab_size']} -> {out}")
    return vocab


def unfreeze_top_layers(encoder: nn.Module, n_layers: int) -> list[str]:
    """Freeze the whole encoder, then re-enable the top ``n_layers`` blocks.

    Works for BERT- and RoBERTa-style encoders, whose transformer blocks live
    at ``encoder.encoder.layer``. Returns the names of unfrozen parameters so
    the caller can assert on / log exactly what is trainable.
    """
    for param in encoder.parameters():
        param.requires_grad = False

    if n_layers <= 0:
        return []

    blocks = getattr(getattr(encoder, "encoder", None), "layer", None)
    if blocks is None:
        raise AttributeError(
            f"Cannot locate transformer blocks on {type(encoder).__name__}; "
            "expected encoder.encoder.layer"
        )

    unfrozen: list[str] = []
    for block in blocks[-n_layers:]:
        for name, param in block.named_parameters():
            param.requires_grad = True
            unfrozen.append(name)
    return unfrozen


def save_trainable_checkpoint(model: nn.Module, path: str, **meta) -> int:
    """Persist only the parameters that actually train.

    A full ``state_dict()`` of these models is ~477 MB, almost all of it the
    frozen backbone, which is byte-identical to the public checkpoint and is
    rewritten on every validation improvement. In ``frozen`` mode fewer than
    0.5% of parameters change, so saving everything wastes ~476 MB per run and
    will fill a disk during a multi-run sweep.

    We therefore store only ``requires_grad`` tensors. ``load_trainable_checkpoint``
    restores them onto a freshly constructed model, whose frozen weights come
    from the pretrained backbone and are identical by construction.

    Returns the number of bytes written.
    """
    frozen_params = {n for n, p in model.named_parameters() if not p.requires_grad}
    # Drop frozen parameters; keep trainable ones and every buffer (buffers are
    # not parameters, are tiny, and some modules need them to reload cleanly).
    keep = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name not in frozen_params
    }

    payload = {"trainable_state_dict": keep, "format": "trainable_only", **meta}
    torch.save(payload, path)
    return os.path.getsize(path)


def load_trainable_checkpoint(model: nn.Module, path: str, device) -> dict:
    """Restore a checkpoint written by ``save_trainable_checkpoint``.

    Also accepts a legacy full ``model_state_dict`` payload.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "trainable_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(
            ckpt["trainable_state_dict"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def trainable_report(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 3),
    }


# ---------------------------------------------------------------------------
# Emoji encoding
# ---------------------------------------------------------------------------

def encode_emojis(df: pd.DataFrame, emoji_to_id: dict, max_emojis: int = 8):
    ids_list, masks_list = [], []
    for emojis in df["emoji_list"].tolist():
        ids, mask = encode_emoji_list(emojis, emoji_to_id, max_emojis)
        ids_list.append(ids)
        masks_list.append(mask)
    return np.array(ids_list), np.array(masks_list)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TokenizedEmotionDataset(Dataset):
    """Raw text + emoji ids for end-to-end fine-tuning.

    Tokenization happens in the collate function so each batch is padded to its
    own longest sequence rather than to a global ``max_length``. On CPU this is
    the single biggest speedup available.
    """

    def __init__(
        self,
        texts: list[str],
        emoji_ids: np.ndarray,
        emoji_masks: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.texts = texts
        self.emoji_ids = torch.tensor(emoji_ids, dtype=torch.long)
        self.emoji_masks = torch.tensor(emoji_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.emoji_ids[idx], self.emoji_masks[idx], self.labels[idx]


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        texts, e_ids, e_masks, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.stack(e_ids),
            torch.stack(e_masks),
            torch.stack(labels),
        )

    return collate


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, names: list[str]) -> dict:
    labels = list(range(len(names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=names,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def majority_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, names: list[str]) -> dict:
    """Always-predict-the-training-majority-class reference point.

    Reported alongside every result because these label spaces are heavily
    skewed (GoEmotions Ekman-6 is 57% ``joy``), so raw accuracy on its own is
    not interpretable.
    """
    majority = Counter(train_df["label"]).most_common(1)[0][0]
    y_true = test_df["label"].to_numpy()
    y_pred = np.full_like(y_true, majority)
    m = _metrics(y_true, y_pred, names)
    return {
        "majority_class": names[majority],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
    }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _run_eval(model, loader, forward_fn, device, criterion=None):
    model.eval()
    preds, trues, probs, loss_sum = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            *inputs, labels = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            logits = forward_fn(model, *inputs)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item()
            probs.extend(torch.softmax(logits, dim=-1).cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    return preds, trues, probs, loss_sum / max(len(loader), 1)


def train_model(
    model: nn.Module,
    forward_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device,
    class_weights: torch.Tensor,
    out_dir: str,
    epochs: int,
    head_lr: float,
    encoder_lr: float | None,
    weight_decay: float,
    names: list[str],
    max_grad_norm: float = 1.0,
    checkpoint_meta: dict | None = None,
) -> dict:
    """Train with best-checkpoint selection on validation macro-F1.

    When ``encoder_lr`` is given, encoder parameters that are still trainable
    form their own param group at that (low) learning rate -- discriminative
    fine-tuning, so the pretrained layers move far more slowly than the head.
    """
    os.makedirs(out_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (encoder_params if name.startswith("encoder.") else head_params).append(param)

    groups = [{"params": head_params, "lr": head_lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr or head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)

    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1, best_epoch = -1.0, -1
    model.to(device)
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            *inputs, labels = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            optimizer.zero_grad()
            logits = forward_fn(model, *inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        train_loss = total_loss / max(n_batches, 1)
        preds, trues, _, val_loss = _run_eval(
            model, val_loader, forward_fn, device, criterion
        )
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f} | ValMacroF1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            save_trainable_checkpoint(
                model,
                os.path.join(out_dir, "best_model.pt"),
                best_val_macro_f1=val_f1,
                best_epoch=epoch,
                label_names=names,
                **(checkpoint_meta or {}),
            )
            print(f"  [*] new best checkpoint (Val Macro-F1 {val_f1:.4f})")

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "training_time_sec": round(time.time() - start, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    forward_fn,
    test_loader: DataLoader,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_result: dict,
    *,
    device,
    out_dir: str,
    experiment: str,
    names: list[str],
    extra: dict | None = None,
) -> dict:
    """Load the best checkpoint, score the test split, persist all artifacts."""
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = load_trainable_checkpoint(model, ckpt_path, device)
        print(f"Loaded best checkpoint (epoch {ckpt.get('best_epoch', '?')})")

    model.to(device)
    preds, trues, probs, _ = _run_eval(model, test_loader, forward_fn, device)
    y_true, y_pred = np.array(trues), np.array(preds)

    metrics = _metrics(y_true, y_pred, names)
    metrics.update({
        "experiment": experiment,
        "label_names": names,
        "best_epoch": train_result["best_epoch"],
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "training_time_sec": train_result.get("training_time_sec"),
        "history": train_result["history"],
        "dataset_sizes": {"train": len(train_df), "test": len(test_df)},
        "majority_baseline": majority_baseline(train_df, test_df, names),
    })
    if extra:
        metrics.update(extra)

    # Emoji-stratified accuracy: the number that actually speaks to the
    # project's research question.
    has_emoji = test_df["num_emojis"].to_numpy() > 0
    for subset, mask in (("with_emoji", has_emoji), ("without_emoji", ~has_emoji)):
        if mask.sum() > 0:
            metrics[f"accuracy_{subset}"] = float(accuracy_score(y_true[mask], y_pred[mask]))
            metrics[f"macro_f1_{subset}"] = float(
                f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            )
        metrics[f"n_{subset}"] = int(mask.sum())

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = test_df.copy()
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_name"] = [names[p] for p in y_pred]
    for i, n in enumerate(names):
        pred_df[f"p_{n}"] = np.array(probs)[:, i]
    pred_df["correct"] = y_true == y_pred
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    print()
    print("=" * 64)
    print(f"{experiment} TEST RESULTS")
    print("=" * 64)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    mb = metrics["majority_baseline"]
    print(f"Majority baseline ({mb['majority_class']}): acc {mb['accuracy']:.4f} / macroF1 {mb['macro_f1']:.4f}")
    if "accuracy_with_emoji" in metrics:
        print(
            f"Emoji subset (n={metrics['n_with_emoji']}): acc {metrics['accuracy_with_emoji']:.4f} | "
            f"No-emoji (n={metrics['n_without_emoji']}): acc {metrics['accuracy_without_emoji']:.4f}"
        )
    print(f"Artifacts -> {out_dir}/")
    return metrics


# ---------------------------------------------------------------------------
# Experiment context
# ---------------------------------------------------------------------------

def prepare(dataset: str, experiment: str, seed: int = 42) -> dict:
    """Load splits, build vocab + class weights, create the output directory."""
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = label_names_for(dataset)
    train_df, val_df, test_df = load_emotion_splits(dataset)

    print(f"Device: {device} | dataset={dataset} | classes={len(names)} {names}")
    print(f"Splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    vocab = build_vocab(train_df, dataset)
    class_weights = compute_class_weights(train_df, len(names))
    out_dir = f"results/emotion/{dataset}/{experiment}"
    os.makedirs(out_dir, exist_ok=True)

    return {
        "device": device,
        "names": names,
        "num_labels": len(names),
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "vocab": vocab,
        "class_weights": class_weights,
        "out_dir": out_dir,
        "seed": seed,
    }
