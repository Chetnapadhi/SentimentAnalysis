"""E1 — text + random emoji embedding, concatenation fusion.

MIRRORS E0 exactly (seed, class weights, optimizer, batch, max_length, metrics,
model-selection criterion) except it adds a randomly-initialized trainable
emoji embedding branch fused by concatenation.

Controlled-experiment contract:
- Same train/validation/test examples and labels as E0.
- Text branch receives ONLY ``text_without_emoji``.
- Emoji branch receives ONLY ``emoji_list`` (mean-pooled random embedding).
- ``original_text`` is never passed to BERT.

This script is PREPARED but must NOT be trained until approved.
"""

from __future__ import annotations

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
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.embeddings.emoji_encoder import (
    EmojiEncoder,
    build_emoji_vocab,
    encode_emoji_list,
)
from src.models.concat_fusion import ConcatFusionModel
from src.utils.seed import set_seed

LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


def load_config(path: str = "config.yaml") -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text featurization (reuses E0 frozen-BERT mean-pooling)
# ---------------------------------------------------------------------------

def featurize_texts(texts, tokenizer, encoder, max_length, batch_size, device):
    embeds = []
    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Text featurize"):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            out = encoder(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embeds.append(pooled.cpu().numpy())
    return np.concatenate(embeds, axis=0)


# ---------------------------------------------------------------------------
# Dataset producing (text_embed, emoji_ids, emoji_mask, label)
# ---------------------------------------------------------------------------

class EmojiDataset(Dataset):
    def __init__(self, text_embeds, emoji_ids, emoji_masks, labels):
        self.text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
        self.emoji_ids = torch.tensor(emoji_ids, dtype=torch.long)
        self.emoji_masks = torch.tensor(emoji_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.text_embeds[idx], self.emoji_ids[idx], self.emoji_masks[idx], self.labels[idx])


# ---------------------------------------------------------------------------
# Training (head + emoji embedding only; text frozen)
# ---------------------------------------------------------------------------

def train_e1(
    model: ConcatFusionModel,
    train_ds,
    val_ds,
    cfg: dict,
    device,
    class_weights,
    out_dir: str,
) -> dict:
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 1e-4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimize trainable params only: classifier + emoji embedding
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_epoch = -1
    best_val_f1 = -1.0
    patience = train_cfg.get("early_stop_patience", 5)
    no_improve = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for text_emb, e_ids, e_mask, lab in train_loader:
            text_emb, lab = text_emb.to(device), lab.to(device)
            e_ids, e_mask = e_ids.to(device), e_mask.to(device)
            optimizer.zero_grad()
            # Forward through classifier on concatenated (precomputed text emb + emoji emb)
            emoji_rep = model.emoji_encoder(e_ids, e_mask)
            fused = torch.cat([text_emb, emoji_rep], dim=-1)
            logits = model.classifier(fused)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for text_emb, e_ids, e_mask, lab in val_loader:
                text_emb, lab = text_emb.to(device), lab.to(device)
                e_ids, e_mask = e_ids.to(device), e_mask.to(device)
                emoji_rep = model.emoji_encoder(e_ids, e_mask)
                fused = torch.cat([text_emb, emoji_rep], dim=-1)
                logits = model.classifier(fused)
                val_loss += criterion(logits, lab).item()
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_true.extend(lab.cpu().numpy().tolist())
        val_loss /= max(len(val_loader), 1)
        val_acc = accuracy_score(all_true, all_preds)
        val_mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_mf1)

        print(f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
              f"ValAcc {val_acc:.4f} | ValMacroF1 {val_mf1:.4f}")

        if val_mf1 > best_val_f1:
            best_val_f1 = val_mf1
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "classifier_state": model.classifier.state_dict(),
                "emoji_embedding_state": model.emoji_encoder.embedding.state_dict(),
                "emoji_vocab_size": model.emoji_encoder.vocab_size,
                "best_val_f1": val_mf1,
                "best_epoch": epoch,
                "class_weights": class_weights.tolist(),
            }, os.path.join(out_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return {"best_epoch": best_epoch, "best_val_macro_f1": best_val_f1, "history": history}


# ---------------------------------------------------------------------------
# Main (prepared; do not call until approved)
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
    emoji_cfg = cfg["emoji"]
    emoji_dim = emoji_cfg["embedding_dim"]
    train_cfg = cfg["training"]

    out_dir = "results/E1"
    os.makedirs(out_dir, exist_ok=True)

    # Load canonical final splits (same as E0)
    train_df = pd.read_json("data/processed/canonical/final_train.jsonl", lines=True)
    val_df = pd.read_json("data/processed/canonical/final_validation.jsonl", lines=True)
    test_df = pd.read_json("data/processed/canonical/final_test.jsonl", lines=True)

    # Build emoji vocab from TRAIN only (no test leakage)
    vocab = build_emoji_vocab(train_df["emoji_list"].tolist())
    print(f"Emoji vocab size (train-only): {vocab['vocab_size']}")
    save_vocab_path = os.path.join("models/emoji_embeddings", "emoji_vocab.json")
    os.makedirs(os.path.dirname(save_vocab_path), exist_ok=True)
    from src.embeddings.emoji_encoder import save_vocab
    save_vocab(vocab, save_vocab_path)

    max_emojis = 8

    # Class weights from train only
    counts = Counter(train_df["label"])
    total = sum(counts.values())
    n_classes = 3
    class_weights = torch.tensor(
        [total / (n_classes * counts[c]) for c in [0, 1, 2]], dtype=torch.float32
    )
    print("Class weights (train-only):", class_weights.tolist())

    # Tokenizer + frozen text encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = ConcatFusionModel(
        model_name=model_name,
        text_dim=768,
        emoji_dim=emoji_dim,
        num_labels=n_classes,
        dropout=cfg["classifier_head"]["dropout"],
        classifier_hidden=cfg["classifier_head"]["classifier_hidden"],
        freeze_encoder=True,
        emoji_vocab_size=vocab["vocab_size"],
        max_emojis=max_emojis,
    )
    encoder = text_model.encoder.to(device)

    # Featurize text (cache) — same as E0
    hash_str = hashlib.sha1(
        (train_df["text_without_emoji"].sum() + val_df["text_without_emoji"].sum()
         + test_df["text_without_emoji"].sum()).encode("utf-8")
    ).hexdigest()
    cache_dir = "results/E1/embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)

    def get_emb(split_df, name):
        path = os.path.join(cache_dir, f"text_{name}_{hash_str[:8]}.npy")
        if os.path.exists(path):
            return np.load(path)
        emb = featurize_texts(split_df["text_without_emoji"].tolist(), tokenizer,
                              encoder, max_length, train_cfg["batch_size"], device)
        np.save(path, emb)
        return emb

    X_train = get_emb(train_df, "train")
    X_val = get_emb(val_df, "validation")
    X_test = get_emb(test_df, "test")

    # Encode emojis
    def encode_split(df):
        ids, masks = [], []
        for emojis in df["emoji_list"].tolist():
            e_ids, e_mask = encode_emoji_list(emojis, vocab["emoji_to_id"], max_emojis)
            ids.append(e_ids)
            masks.append(e_mask)
        return np.array(ids), np.array(masks)

    e_train_ids, e_train_mask = encode_split(train_df)
    e_val_ids, e_val_mask = encode_split(val_df)
    e_test_ids, e_test_mask = encode_split(test_df)

    train_ds = EmojiDataset(X_train, e_train_ids, e_train_mask, train_df["label"].to_numpy())
    val_ds = EmojiDataset(X_val, e_val_ids, e_val_mask, val_df["label"].to_numpy())
    test_ds = EmojiDataset(X_test, e_test_ids, e_test_mask, test_df["label"].to_numpy())

    # Train (NOT EXECUTED here — guarded)
    print("\n[E1 PREPARED] Training is NOT executed by default. Call train_e1() after approval.")
    assert False, "E1 training is disabled until approved. Set RUN_E1=1 to enable."


if __name__ == "__main__":
    if os.environ.get("RUN_E1") == "1":
        main()
    else:
        print("E1 is prepared but disabled. Set RUN_E1=1 to train (after approval).")
